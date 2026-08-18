"""Reproducible trainer for focal and contextual masked molecular modeling."""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from spatial_scaling.data.synthetic.context import (
    ContextSelectionConfig,
    SyntheticContextDataset,
)
from spatial_scaling.data.synthetic.dataset import SyntheticExpressionDataset
from spatial_scaling.evaluation.masked_expression import MaskedMetricAccumulator
from spatial_scaling.models.cell_encoder import CellMLP
from spatial_scaling.models.spatial_transformer import SpatialTransformer
from spatial_scaling.spatial.neighborhoods import (
    exact_distance_excluded_knn_indices,
    exact_knn_indices,
    selected_context_distances_um,
    summarize_selected_context_distances_um,
)
from spatial_scaling.training.config import SSLExperimentConfig, config_identity_sha256
from spatial_scaling.training.masking import (
    MaskingConfig,
    build_masked_batch,
    masks_for_cells,
)
from spatial_scaling.training.objective import masked_mse
from spatial_scaling.training.policies import (
    TrainingDataSelection,
    TrainingExecutionPolicy,
    ValidationConvergenceController,
    ValidationPlateauController,
)


def _git_state() -> dict[str, Any]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit_sha": None, "dirty": None}
    return {"commit_sha": sha or None, "dirty": dirty}


def _source_state() -> dict[str, Any]:
    """Hash executable project sources when runs necessarily use a dirty tree."""
    root = Path.cwd()
    paths: list[Path] = []
    for directory in ("src", "scripts", "configs", "slurm"):
        candidate = root / directory
        if candidate.is_dir():
            paths.extend(
                path
                for path in candidate.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts
            )
    paths.extend(
        path
        for name in ("pyproject.toml", "uv.lock")
        if (path := root / name).is_file()
    )
    files = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(set(paths))
    }
    digest = hashlib.sha256()
    for name, value in files.items():
        digest.update(name.encode("utf-8"))
        digest.update(value.encode("ascii"))
    return {"tree_sha256": digest.hexdigest(), "files": files}


def _environment() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "torch", "matplotlib", "pyyaml"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "hostname": platform.node(),
    }


def _select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def _set_reproducibility(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def _prepare_output(path: Path) -> Path:
    """Create a unique staging directory for an atomic completed run."""
    if path.exists():
        raise FileExistsError(f"experiment output path already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    attempt = os.environ.get("SLURM_JOB_ID", f"pid{os.getpid()}")
    staging = path.parent / f".{path.name}.in_progress_{attempt}"
    if staging.exists():
        raise FileExistsError(f"experiment staging directory already exists: {staging}")
    staging.mkdir()
    return staging


def _publish_output(staging: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"experiment output appeared during run: {destination}")
    staging.replace(destination)


def _index_digest(indices: np.ndarray) -> str:
    return hashlib.sha256(indices.astype("<i8", copy=False).tobytes()).hexdigest()


def _stable_seed(*components: object) -> int:
    digest = hashlib.blake2b(digest_size=8)
    for component in components:
        encoded = str(component).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    return int.from_bytes(digest.digest(), "little")


def _evaluation_mask_digest(
    dataset: SyntheticExpressionDataset,
    indices: np.ndarray,
    masking: MaskingConfig,
    *,
    batch_size: int,
) -> str:
    """Hash exact cell identities and fixed evaluation masks in index order."""
    digest = hashlib.sha256()
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        for index in batch_indices:
            example = dataset[int(index)]
            cell_id = example["cell_id"]
            encoded = cell_id.encode("utf-8")
            digest.update(len(encoded).to_bytes(4, "little"))
            digest.update(encoded)
            mask = masks_for_cells(dataset.num_genes, [cell_id], masking, epoch=0)[0]
            digest.update(np.packbits(mask, bitorder="little").tobytes())
    return digest.hexdigest()


def _context_eligibility(
    context: SyntheticContextDataset,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Resolve exact current and common matched eligibility by section."""
    common: list[np.ndarray] = []
    section_rows = []
    current_threshold = context.selection.minimum_distance_um
    common_threshold = context.selection.eligibility_minimum_distance_um
    for section_offset, section_id in enumerate(context.dataset.section_ids):
        common_offsets = context.eligible_offsets(section_offset)
        if current_threshold == common_threshold:
            current_offsets = common_offsets
        else:
            current_offsets = context.eligible_offsets(
                section_offset, minimum_distance_um=current_threshold
            )
        if not len(common_offsets):
            raise ValueError(
                f"section {section_id} has no common eligible focal observations"
            )
        common.append(common_offsets)
        section_rows.append(
            {
                "section_id": section_id,
                "total_observations": context.dataset.cells_per_section,
                "current_threshold_eligible": len(current_offsets),
                "common_threshold_eligible": len(common_offsets),
            }
        )
    return common, {
        "policy": "intersection_at_maximum_distance_threshold",
        "selection_minimum_distance_um": current_threshold,
        "eligibility_minimum_distance_um": common_threshold,
        "total_observations": len(context.dataset),
        "current_threshold_eligible": sum(
            row["current_threshold_eligible"] for row in section_rows
        ),
        "common_threshold_eligible": sum(
            row["common_threshold_eligible"] for row in section_rows
        ),
        "sections": section_rows,
    }


def _filter_evaluation_indices(
    indices: np.ndarray,
    dataset: SyntheticExpressionDataset,
    eligible_by_section: list[np.ndarray],
) -> np.ndarray:
    keep = np.zeros(len(indices), dtype=bool)
    section_offsets = indices // dataset.cells_per_section
    cell_offsets = indices % dataset.cells_per_section
    for section_offset in np.unique(section_offsets):
        rows = section_offsets == section_offset
        keep[rows] = np.isin(
            cell_offsets[rows], eligible_by_section[int(section_offset)]
        )
    return indices[keep]


def _context_distance_summary(
    context: SyntheticContextDataset, indices: np.ndarray
) -> dict[str, Any]:
    """Measure true selected geometry for the exact evaluation observations."""
    distances = []
    section_offsets = indices // context.dataset.cells_per_section
    for section_offset in np.unique(section_offsets):
        queries = (
            indices[section_offsets == section_offset]
            % context.dataset.cells_per_section
        )
        _, coordinates, _ = context.dataset.spatial_section(int(section_offset))
        if context.selection.mode == "nearest":
            selected = exact_knn_indices(coordinates, queries, context.context_size)
        else:
            selected = exact_distance_excluded_knn_indices(
                coordinates,
                queries,
                context.context_size,
                minimum_distance_um=context.selection.minimum_distance_um,
            )
        distances.append(selected_context_distances_um(coordinates, queries, selected))
    values = np.concatenate(distances, axis=0)
    if len(values) != len(indices):
        raise RuntimeError("context-distance observation count is inconsistent")
    return {
        "evaluation_targets": len(indices),
        "selected_context_cells_per_target": context.context_size,
        "distributions": summarize_selected_context_distances_um(values),
    }


def _predict_contextual_batch(
    model: SpatialTransformer,
    context_data: SyntheticContextDataset,
    section_offset: int,
    offsets: np.ndarray,
    masking: MaskingConfig,
    device: torch.device,
    *,
    epoch: int,
) -> tuple[torch.Tensor, Any]:
    context = context_data.context_batch(section_offset, offsets)
    expression = torch.from_numpy(context.focal_expression).to(device)
    batch = build_masked_batch(expression, context.focal_cell_ids, masking, epoch=epoch)
    context_expression = torch.from_numpy(context.context_expression).to(device)
    relative_coordinates = torch.from_numpy(context.relative_coordinates_um).to(device)
    predictions = model(
        batch.masked_expression,
        batch.visibility,
        context_expression,
        relative_coordinates,
    )
    return predictions, batch


def evaluate_model(
    model: CellMLP | SpatialTransformer,
    dataset: SyntheticExpressionDataset,
    indices: np.ndarray,
    *,
    batch_size: int,
    masking: MaskingConfig,
    device: torch.device,
    context_data: SyntheticContextDataset | None = None,
) -> dict[str, float | int]:
    """Evaluate fixed observations with fixed per-cell masks."""
    accumulator = MaskedMetricAccumulator(dataset.gene_metadata)
    model.eval()
    section_offsets = indices // dataset.cells_per_section
    with torch.inference_mode():
        for section_offset in np.unique(section_offsets):
            offsets = indices[section_offsets == section_offset]
            offsets = offsets % dataset.cells_per_section
            for start in range(0, len(offsets), batch_size):
                batch_offsets = offsets[start : start + batch_size]
                if isinstance(model, SpatialTransformer):
                    if context_data is None:
                        raise ValueError(
                            "context_data is required for contextual model"
                        )
                    predictions, batch = _predict_contextual_batch(
                        model,
                        context_data,
                        int(section_offset),
                        batch_offsets,
                        masking,
                        device,
                        epoch=0,
                    )
                else:
                    if context_data is not None:
                        raise ValueError("context_data cannot be used with focal model")
                    values, cell_ids = dataset.section_batch(
                        int(section_offset), batch_offsets
                    )
                    expression = torch.from_numpy(values).to(device)
                    batch = build_masked_batch(expression, cell_ids, masking, epoch=0)
                    predictions = model(batch.masked_expression, batch.visibility)
                accumulator.update(predictions, batch.targets, batch.mask)
    return accumulator.compute()


def _write_history(path: Path, history: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def _plot_history(path: Path, history: list[dict[str, float | int]]) -> None:
    steps = [int(row["step"]) for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    train_steps = [
        step
        for step, row in zip(steps, history, strict=True)
        if row["train_masked_mse"]
    ]
    train_loss = [
        float(row["train_masked_mse"]) for row in history if row["train_masked_mse"]
    ]
    axes[0].plot(train_steps, train_loss, color="black", marker="o", label="training")
    axes[0].plot(
        steps,
        [float(row["validation_masked_mse_all"]) for row in history],
        color="tab:blue",
        marker="o",
        label="validation",
    )
    axes[0].set(
        title="Masked reconstruction", xlabel="Optimization step", ylabel="Masked MSE"
    )
    axes[0].legend()
    colors = {"intrinsic": "tab:orange", "mixed": "tab:green", "spatial": "tab:red"}
    for name, color in colors.items():
        axes[1].plot(
            steps,
            [float(row[f"validation_masked_mse_{name}"]) for row in history],
            color=color,
            marker="o",
            label=name,
        )
    axes[1].set(
        title="Validation by gene class",
        xlabel="Optimization step",
        ylabel="Masked MSE",
    )
    axes[1].legend()
    for axis in axes:
        axis.tick_params(colors="black")
        axis.xaxis.label.set_color("black")
        axis.yaxis.label.set_color("black")
        axis.title.set_color("black")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_experiment(
    config: SSLExperimentConfig,
    *,
    context_selection: ContextSelectionConfig | None = None,
    training_data_selection: TrainingDataSelection | None = None,
    execution_policy: TrainingExecutionPolicy | None = None,
    expected_config_identity_sha256: str | None = None,
) -> dict[str, Any]:
    """Train and evaluate one controlled masking configuration."""
    config.validate()
    selection_was_explicit = context_selection is not None
    selection = context_selection or ContextSelectionConfig()
    selection.validate()
    if training_data_selection is not None:
        training_data_selection.validate()
    if execution_policy is not None:
        execution_policy.validate()
        if (
            execution_policy.name == "fixed_compute_2000_steps"
            and config.training.steps != 2000
        ):
            raise ValueError("fixed-compute configuration must request 2,000 steps")
        if (
            execution_policy.lr_schedule == "validation_plateau_decay"
            and execution_policy.minimum_learning_rate >= config.optimizer.learning_rate
        ):
            raise ValueError("minimum learning rate must be below the base LR")
    if config.condition == "focal" and selection_was_explicit:
        raise ValueError("context selection cannot be supplied for a focal model")
    _set_reproducibility(config.seed)
    destination = Path(config.output_dir)
    output = _prepare_output(destination)
    device = _select_device(config.device)
    train_data = SyntheticExpressionDataset(
        config.data.corpus_path,
        "train",
        cache_sections=config.data.cache_sections,
        section_ids=(
            training_data_selection.active_train_section_ids
            if training_data_selection is not None
            else None
        ),
    )
    evaluation_data = SyntheticExpressionDataset(
        config.data.corpus_path,
        config.evaluation.split,
        cache_sections=config.data.cache_sections,
    )
    if train_data.gene_metadata != evaluation_data.gene_metadata:
        raise ValueError("training and evaluation gene metadata differ")
    train_context: SyntheticContextDataset | None = None
    evaluation_context: SyntheticContextDataset | None = None
    if config.condition != "focal":
        train_context = SyntheticContextDataset(
            train_data,
            context_size=config.context.context_size,
            condition=config.condition,
            shuffle_seed=config.context.shuffle_seed,
            selection=selection,
        )
        evaluation_context = SyntheticContextDataset(
            evaluation_data,
            context_size=config.context.context_size,
            condition=config.condition,
            shuffle_seed=config.context.shuffle_seed,
            selection=selection,
        )
    requested_evaluation_indices = evaluation_data.deterministic_indices(
        config.evaluation.max_cells, config.evaluation.observation_seed
    )
    train_eligible_by_section: list[np.ndarray] | None = None
    eligibility: dict[str, Any] | None = None
    if train_context is not None and evaluation_context is not None:
        train_eligible_by_section, train_eligibility = _context_eligibility(
            train_context
        )
        evaluation_eligible_by_section, evaluation_eligibility = _context_eligibility(
            evaluation_context
        )
        evaluation_indices = _filter_evaluation_indices(
            requested_evaluation_indices,
            evaluation_data,
            evaluation_eligible_by_section,
        )
        if not len(evaluation_indices):
            raise ValueError(
                "no requested evaluation observations are context-eligible"
            )
        eligibility = {
            "train": train_eligibility,
            "evaluation": evaluation_eligibility,
            "requested_evaluation_observations": len(requested_evaluation_indices),
            "retained_evaluation_observations": len(evaluation_indices),
        }
    else:
        evaluation_indices = requested_evaluation_indices
    evaluation_masking = MaskingConfig(
        policy=config.masking.policy,
        fraction=config.masking.fraction,
        seed=config.evaluation.masking_seed,
        mask_value=config.masking.mask_value,
    )
    model_config = config.model_config(train_data.num_genes)
    if config.condition == "focal":
        model = CellMLP(model_config).to(device)
    else:
        model = SpatialTransformer(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
    )
    resolved = config.to_dict()
    extensions: dict[str, Any] = {}
    if selection_was_explicit:
        extensions["context_selection"] = selection.to_dict()
    if training_data_selection is not None:
        extensions["training_data_selection"] = training_data_selection.to_dict()
    if execution_policy is not None:
        extensions["execution_policy"] = execution_policy.to_dict()
    identity_payload = (
        {"experiment_config": resolved, **extensions} if extensions else resolved
    )
    identity = config_identity_sha256(identity_payload)
    if (
        expected_config_identity_sha256 is not None
        and identity != expected_config_identity_sha256
    ):
        raise ValueError("resolved run identity does not match manifest identity")
    if selection_was_explicit:
        resolved["context_selection"] = selection.to_dict()
    if training_data_selection is not None:
        resolved["training_data_selection"] = training_data_selection.to_dict()
    if execution_policy is not None:
        resolved["execution_policy"] = execution_policy.to_dict()
    resolved["resolved_device"] = str(device)
    resolved["config_identity_sha256"] = identity
    resolved["model"]["parameter_count"] = model.parameter_count
    resolved["section_ids"] = {
        "train": list(train_data.section_ids_by_split["train"]),
        "validation": list(train_data.section_ids_by_split["validation"]),
        "test": list(train_data.section_ids_by_split["test"]),
    }
    resolved["active_training_section_ids"] = list(train_data.section_ids)
    (output / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2) + "\n", encoding="utf-8"
    )
    observations = {
        "split": config.evaluation.split,
        "count": len(evaluation_indices),
        "index_sha256": _index_digest(evaluation_indices),
        "observation_seed": config.evaluation.observation_seed,
        "masking_seed": config.evaluation.masking_seed,
        "mask_sha256": _evaluation_mask_digest(
            evaluation_data,
            evaluation_indices,
            evaluation_masking,
            batch_size=config.evaluation.batch_size,
        ),
        "section_ids": list(evaluation_data.section_ids),
        "requested_count": len(requested_evaluation_indices),
        "eligibility": eligibility,
    }
    (output / "evaluation_observations.json").write_text(
        json.dumps(observations, indent=2) + "\n", encoding="utf-8"
    )
    run_started_at = datetime.now(UTC)
    device_information: dict[str, Any] = {"type": device.type}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        device_information.update(
            {
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "compute_capability": [properties.major, properties.minor],
            }
        )
    provenance = {
        "git": _git_state(),
        "source_state": _source_state(),
        "environment": _environment(),
        "device": device_information,
        "corpus_metadata": train_data.metadata,
        "run": {
            "experiment_id": config.experiment_name,
            "config_identity_sha256": identity,
            "started_at_utc": run_started_at.isoformat(),
            "context_selection": (
                selection.to_dict() if config.condition != "focal" else None
            ),
            "training_data_selection": (
                training_data_selection.to_dict()
                if training_data_selection is not None
                else None
            ),
            "execution_policy": (
                execution_policy.to_dict() if execution_policy is not None else None
            ),
        },
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    initial_validation = evaluate_model(
        model,
        evaluation_data,
        evaluation_indices,
        batch_size=config.evaluation.batch_size,
        masking=evaluation_masking,
        device=device,
        context_data=evaluation_context,
    )
    history: list[dict[str, float | int]] = [
        {
            "step": 0,
            "epoch": 0.0,
            "learning_rate": config.optimizer.learning_rate,
            "learning_rate_after_validation": config.optimizer.learning_rate,
            "lr_reduction_event": False,
            "early_stopping_decision": False,
            "train_masked_mse": "",
            **{
                f"validation_{name}": value
                for name, value in initial_validation.items()
                if name.startswith("masked_mse_")
            },
        }
    ]
    interval_losses: list[float] = []
    first_interval_loss: float | None = None
    final_training_loss: float | None = None
    step_times: list[float] = []
    step = 0
    examples_processed = 0
    epochs_completed = 0
    early_stopping_reason = "fixed_steps_completed"
    lr_reduction_events: list[dict[str, float | int]] = []
    exposure_counts = np.zeros(len(train_data.section_ids), dtype=np.int64)
    seen_offsets = [
        np.zeros(train_data.cells_per_section, dtype=bool)
        for _ in train_data.section_ids
    ]
    available_by_section = [
        (
            np.arange(train_data.cells_per_section, dtype=np.int64)
            if train_eligible_by_section is None
            else train_eligible_by_section[section_offset]
        )
        for section_offset in range(len(train_data.section_ids))
    ]
    available_training_observations = sum(map(len, available_by_section))

    def train_batch(section_offset: int, cell_offsets: np.ndarray) -> float:
        nonlocal step, examples_processed
        step += 1
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_started = time.perf_counter()
        model.train()
        cell_offsets = np.asarray(cell_offsets, dtype=np.int64)
        exposure_counts[section_offset] += len(cell_offsets)
        seen_offsets[section_offset][cell_offsets] = True
        examples_processed += len(cell_offsets)
        mask_epoch = (step - 1) // config.training.mask_epoch_steps
        optimizer.zero_grad(set_to_none=True)
        if isinstance(model, SpatialTransformer):
            if train_context is None:
                raise RuntimeError("contextual model lacks contextual training data")
            predictions, batch = _predict_contextual_batch(
                model,
                train_context,
                section_offset,
                cell_offsets,
                config.masking,
                device,
                epoch=mask_epoch,
            )
        else:
            values, cell_ids = train_data.section_batch(section_offset, cell_offsets)
            expression = torch.from_numpy(values).to(device)
            batch = build_masked_batch(
                expression, cell_ids, config.masking, epoch=mask_epoch
            )
            predictions = model(batch.masked_expression, batch.visibility)
        loss = masked_mse(predictions, batch.targets, batch.mask)
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_times.append(time.perf_counter() - step_started)
        value = float(loss.detach().item())
        interval_losses.append(value)
        return value

    def evaluate_and_record(epoch: float) -> dict[str, float | int]:
        nonlocal first_interval_loss, final_training_loss, interval_losses
        mean_train_loss = float(np.mean(interval_losses))
        if first_interval_loss is None:
            first_interval_loss = mean_train_loss
        final_training_loss = mean_train_loss
        current = evaluate_model(
            model,
            evaluation_data,
            evaluation_indices,
            batch_size=config.evaluation.batch_size,
            masking=evaluation_masking,
            device=device,
            context_data=evaluation_context,
        )
        history.append(
            {
                "step": step,
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "learning_rate_after_validation": optimizer.param_groups[0]["lr"],
                "lr_reduction_event": False,
                "early_stopping_decision": False,
                "train_masked_mse": mean_train_loss,
                **{
                    f"validation_{name}": value
                    for name, value in current.items()
                    if name.startswith("masked_mse_")
                },
            }
        )
        interval_losses = []
        return current

    validation = initial_validation
    best_validation = initial_validation
    best_validation_step = 0
    best_validation_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    if execution_policy is not None and execution_policy.sampling == (
        "deterministic_shuffled_epochs"
    ):
        plateau_controller = (
            ValidationPlateauController(
                reduction_factor=execution_policy.lr_reduction_factor,
                reduction_patience=execution_policy.lr_reduction_patience,
                minimum_learning_rate=execution_policy.minimum_learning_rate,
            )
            if execution_policy.lr_schedule == "validation_plateau_decay"
            else None
        )
        convergence_controller = ValidationConvergenceController(
            best_loss=float(initial_validation["masked_mse_all"]),
            improvement_threshold=execution_policy.improvement_threshold,
            early_stopping_patience=execution_policy.early_stopping_patience,
            plateau=plateau_controller,
        )
        reached_absolute_cap = False
        for epoch in range(1, execution_policy.maximum_epochs + 1):
            epoch_rng = np.random.default_rng(
                _stable_seed("data-scaling-epoch", config.seed, epoch)
            )
            section_order = epoch_rng.permutation(len(train_data.section_ids))
            for section_offset_value in section_order:
                section_offset = int(section_offset_value)
                offsets = np.array(
                    available_by_section[section_offset], dtype=np.int64, copy=True
                )
                epoch_rng.shuffle(offsets)
                for start in range(0, len(offsets), config.training.batch_size):
                    if step >= execution_policy.absolute_max_steps:
                        reached_absolute_cap = True
                        break
                    train_batch(
                        section_offset,
                        offsets[start : start + config.training.batch_size],
                    )
                if reached_absolute_cap:
                    break
            if reached_absolute_cap:
                early_stopping_reason = "absolute_max_steps"
                if epoch <= execution_policy.minimum_epochs:
                    raise RuntimeError(
                        "absolute step cap prevented the minimum exposure policy"
                    )
                break
            epochs_completed = epoch
            if epoch == 1:
                unique_seen = sum(int(values.sum()) for values in seen_offsets)
                if unique_seen != available_training_observations:
                    raise RuntimeError(
                        "deterministic epoch failed to expose every focal observation"
                    )
            validation = evaluate_and_record(float(epoch))
            current_loss = float(validation["masked_mse_all"])
            current_lr = float(optimizer.param_groups[0]["lr"])
            decision = convergence_controller.update(
                validation_loss=current_loss,
                current_learning_rate=current_lr,
            )
            if decision.improved:
                best_validation = validation
                best_validation_step = step
                best_validation_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
            if decision.lr_reduced:
                if plateau_controller is None:
                    raise RuntimeError("LR reduction lacks plateau controller")
                next_lr = decision.learning_rate
                for group in optimizer.param_groups:
                    group["lr"] = next_lr
                event = {
                    "epoch": epoch,
                    "step": step,
                    "validation_masked_mse_all": current_loss,
                    "previous_learning_rate": current_lr,
                    "new_learning_rate": next_lr,
                }
                lr_reduction_events.append(event)
                history[-1]["learning_rate_after_validation"] = next_lr
                history[-1]["lr_reduction_event"] = True
            if epoch >= execution_policy.minimum_epochs and decision.should_stop:
                early_stopping_reason = "early_stopping_patience"
                history[-1]["early_stopping_decision"] = True
                break
        else:
            early_stopping_reason = "maximum_epochs"
        if execution_policy.restore_best_validation:
            model.load_state_dict(best_state)
            final_validation = evaluate_model(
                model,
                evaluation_data,
                evaluation_indices,
                batch_size=config.evaluation.batch_size,
                masking=evaluation_masking,
                device=device,
                context_data=evaluation_context,
            )
        else:
            final_validation = validation
    else:
        sampling_rng = np.random.default_rng(config.seed)
        fixed_steps = (
            execution_policy.fixed_steps
            if execution_policy is not None
            else config.training.steps
        )
        for _ in range(fixed_steps):
            section_offset = int(sampling_rng.integers(len(train_data.section_ids)))
            sampling_population = available_by_section[section_offset]
            population_size = len(sampling_population)
            cell_offsets = sampling_rng.choice(
                sampling_population,
                size=config.training.batch_size,
                replace=config.training.batch_size > population_size,
            )
            train_batch(section_offset, cell_offsets)
            should_evaluate = (
                step % config.training.evaluation_interval == 0 or step == fixed_steps
            )
            if should_evaluate:
                validation = evaluate_and_record(
                    examples_processed / available_training_observations
                )
        final_validation = validation
        best_validation = min(
            (initial_validation, validation),
            key=lambda values: float(values["masked_mse_all"]),
        )
        if best_validation is validation:
            best_validation_step = step
            best_validation_epoch = examples_processed / available_training_observations
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started
    peak_device_memory_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    peak_device_memory_reserved_bytes = (
        int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
    )
    optimization_elapsed_seconds = float(sum(step_times))
    run_ended_at = datetime.now(UTC)
    context_tokens_per_example = (
        config.context.context_size + 1 if config.condition != "focal" else 1
    )
    unique_observations_seen = sum(int(values.sum()) for values in seen_offsets)
    effective_passes = examples_processed / available_training_observations
    section_exposure = []
    for section_offset, section_id in enumerate(train_data.section_ids):
        available_offsets = available_by_section[section_offset]
        section_exposure.append(
            {
                "section_id": section_id,
                "available_observations": len(available_offsets),
                "examples_processed": int(exposure_counts[section_offset]),
                "unique_observations_seen": int(
                    seen_offsets[section_offset][available_offsets].sum()
                ),
            }
        )
    summary: dict[str, Any] = {
        "experiment_name": config.experiment_name,
        "experiment_id": config.experiment_name,
        "config_identity_sha256": identity,
        "parameter_count": model.parameter_count,
        "optimization_steps": step,
        "training_batch_size": config.training.batch_size,
        "examples_processed": examples_processed,
        "effective_training_examples": examples_processed,
        "available_training_observations": available_training_observations,
        "unique_training_observations_seen": unique_observations_seen,
        "unique_observation_fraction": unique_observations_seen
        / available_training_observations,
        "effective_passes": effective_passes,
        "epochs_completed": epochs_completed,
        "early_stopping_reason": early_stopping_reason,
        "lr_reduction_events": lr_reduction_events,
        "num_lr_reductions": len(lr_reduction_events),
        "final_learning_rate": float(optimizer.param_groups[0]["lr"]),
        "best_validation_step": best_validation_step,
        "best_validation_epoch": best_validation_epoch,
        "best_validation": best_validation,
        "minimum_exposure_required": bool(
            execution_policy is not None
            and execution_policy.sampling == "deterministic_shuffled_epochs"
        ),
        "minimum_exposure_satisfied": unique_observations_seen
        == available_training_observations,
        "section_exposure": section_exposure,
        "training_data_selection": (
            training_data_selection.to_dict()
            if training_data_selection is not None
            else None
        ),
        "execution_policy": (
            execution_policy.to_dict() if execution_policy is not None else None
        ),
        "context_tokens_per_example": context_tokens_per_example,
        "context_tokens_processed": examples_processed * context_tokens_per_example,
        "elapsed_seconds": elapsed_seconds,
        "run_elapsed_seconds": elapsed_seconds,
        "optimization_elapsed_seconds": optimization_elapsed_seconds,
        "training_step_time_seconds_mean": float(np.mean(step_times)),
        "training_step_time_seconds_median": float(np.median(step_times)),
        "examples_per_second": examples_processed / optimization_elapsed_seconds,
        "context_tokens_per_second": examples_processed
        * context_tokens_per_example
        / optimization_elapsed_seconds,
        "peak_device_memory_bytes": peak_device_memory_bytes,
        "peak_device_memory_allocated_bytes": peak_device_memory_bytes,
        "peak_device_memory_reserved_bytes": peak_device_memory_reserved_bytes,
        "device": str(device),
        "started_at_utc": run_started_at.isoformat(),
        "ended_at_utc": run_ended_at.isoformat(),
        "initial_validation": initial_validation,
        "final_validation": final_validation,
        "first_training_interval_masked_mse": first_interval_loss,
        "final_training_interval_masked_mse": final_training_loss,
        "validation_relative_improvement": 1.0
        - float(final_validation["masked_mse_all"])
        / float(initial_validation["masked_mse_all"]),
        "evaluation_observations": observations,
        "context_selection": (
            selection.to_dict() if config.condition != "focal" else None
        ),
        "context_eligibility": eligibility,
        "context_distance_summary": (
            _context_distance_summary(evaluation_context, evaluation_indices)
            if evaluation_context is not None
            else None
        ),
    }
    (output / "metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _write_history(output / "history.csv", history)
    _plot_history(output / "loss_curves.png", history)
    provenance["run"]["ended_at_utc"] = run_ended_at.isoformat()
    provenance["run"]["elapsed_seconds"] = elapsed_seconds
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    _publish_output(output, destination)
    return summary
