"""Minimal reproducible trainer for focal-cell masked molecular modeling."""

from __future__ import annotations

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
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from spatial_scaling.data.synthetic.dataset import SyntheticExpressionDataset
from spatial_scaling.evaluation.masked_expression import MaskedMetricAccumulator
from spatial_scaling.models.cell_encoder import CellMLP
from spatial_scaling.training.config import SSLExperimentConfig
from spatial_scaling.training.masking import MaskingConfig, build_masked_batch
from spatial_scaling.training.objective import masked_mse


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


def _prepare_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"experiment output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _index_digest(indices: np.ndarray) -> str:
    return hashlib.sha256(indices.astype("<i8", copy=False).tobytes()).hexdigest()


def evaluate_model(
    model: CellMLP,
    dataset: SyntheticExpressionDataset,
    indices: np.ndarray,
    *,
    batch_size: int,
    masking: MaskingConfig,
    device: torch.device,
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
                values, cell_ids = dataset.section_batch(
                    int(section_offset), offsets[start : start + batch_size]
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


def run_experiment(config: SSLExperimentConfig) -> dict[str, Any]:
    """Train and evaluate one controlled masking configuration."""
    config.validate()
    _set_reproducibility(config.seed)
    output = Path(config.output_dir)
    _prepare_output(output)
    device = _select_device(config.device)
    train_data = SyntheticExpressionDataset(
        config.data.corpus_path,
        "train",
        cache_sections=config.data.cache_sections,
    )
    evaluation_data = SyntheticExpressionDataset(
        config.data.corpus_path,
        config.evaluation.split,
        cache_sections=config.data.cache_sections,
    )
    if train_data.gene_metadata != evaluation_data.gene_metadata:
        raise ValueError("training and evaluation gene metadata differ")
    evaluation_indices = evaluation_data.deterministic_indices(
        config.evaluation.max_cells, config.evaluation.observation_seed
    )
    evaluation_masking = MaskingConfig(
        policy=config.masking.policy,
        fraction=config.masking.fraction,
        seed=config.evaluation.masking_seed,
        mask_value=config.masking.mask_value,
    )
    model = CellMLP(config.model_config(train_data.num_genes)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
    )
    resolved = config.to_dict()
    resolved["resolved_device"] = str(device)
    resolved["model"]["parameter_count"] = model.parameter_count
    resolved["section_ids"] = {
        "train": list(train_data.section_ids_by_split["train"]),
        "validation": list(train_data.section_ids_by_split["validation"]),
        "test": list(train_data.section_ids_by_split["test"]),
    }
    (output / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2) + "\n", encoding="utf-8"
    )
    observations = {
        "split": config.evaluation.split,
        "count": len(evaluation_indices),
        "index_sha256": _index_digest(evaluation_indices),
        "observation_seed": config.evaluation.observation_seed,
        "masking_seed": config.evaluation.masking_seed,
        "section_ids": list(evaluation_data.section_ids),
    }
    (output / "evaluation_observations.json").write_text(
        json.dumps(observations, indent=2) + "\n", encoding="utf-8"
    )
    provenance = {
        "git": _git_state(),
        "environment": _environment(),
        "corpus_metadata": train_data.metadata,
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    started = time.perf_counter()
    initial_validation = evaluate_model(
        model,
        evaluation_data,
        evaluation_indices,
        batch_size=config.evaluation.batch_size,
        masking=evaluation_masking,
        device=device,
    )
    history: list[dict[str, float | int]] = [
        {
            "step": 0,
            "train_masked_mse": "",
            **{
                f"validation_{name}": value
                for name, value in initial_validation.items()
                if name.startswith("masked_mse_")
            },
        }
    ]
    sampling_rng = np.random.default_rng(config.seed)
    interval_losses: list[float] = []
    first_interval_loss: float | None = None
    final_training_loss: float | None = None
    for step in range(1, config.training.steps + 1):
        model.train()
        section_offset = int(sampling_rng.integers(len(train_data.section_ids)))
        cell_offsets = sampling_rng.choice(
            train_data.cells_per_section,
            size=config.training.batch_size,
            replace=config.training.batch_size > train_data.cells_per_section,
        )
        values, cell_ids = train_data.section_batch(section_offset, cell_offsets)
        expression = torch.from_numpy(values).to(device)
        mask_epoch = (step - 1) // config.training.mask_epoch_steps
        batch = build_masked_batch(
            expression, cell_ids, config.masking, epoch=mask_epoch
        )
        optimizer.zero_grad(set_to_none=True)
        predictions = model(batch.masked_expression, batch.visibility)
        loss = masked_mse(predictions, batch.targets, batch.mask)
        loss.backward()
        optimizer.step()
        interval_losses.append(float(loss.detach().item()))
        should_evaluate = (
            step % config.training.evaluation_interval == 0
            or step == config.training.steps
        )
        if should_evaluate:
            mean_train_loss = float(np.mean(interval_losses))
            if first_interval_loss is None:
                first_interval_loss = mean_train_loss
            final_training_loss = mean_train_loss
            validation = evaluate_model(
                model,
                evaluation_data,
                evaluation_indices,
                batch_size=config.evaluation.batch_size,
                masking=evaluation_masking,
                device=device,
            )
            history.append(
                {
                    "step": step,
                    "train_masked_mse": mean_train_loss,
                    **{
                        f"validation_{name}": value
                        for name, value in validation.items()
                        if name.startswith("masked_mse_")
                    },
                }
            )
            interval_losses = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started
    final_validation = validation
    summary: dict[str, Any] = {
        "experiment_name": config.experiment_name,
        "parameter_count": model.parameter_count,
        "optimization_steps": config.training.steps,
        "examples_processed": config.training.steps * config.training.batch_size,
        "elapsed_seconds": elapsed_seconds,
        "device": str(device),
        "initial_validation": initial_validation,
        "final_validation": final_validation,
        "first_training_interval_masked_mse": first_interval_loss,
        "final_training_interval_masked_mse": final_training_loss,
        "validation_relative_improvement": 1.0
        - float(final_validation["masked_mse_all"])
        / float(initial_validation["masked_mse_all"]),
        "evaluation_observations": observations,
    }
    (output / "metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _write_history(output / "history.csv", history)
    _plot_history(output / "loss_curves.png", history)
    return summary
