"""Manifest-driven Pilot v0 independent-training-section scaling."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from spatial_scaling.training.config import (
    SSLExperimentConfig,
    config_identity_sha256,
    load_experiment_config,
)
from spatial_scaling.training.policies import (
    TrainingDataSelection,
    TrainingExecutionPolicy,
)


@dataclass(frozen=True)
class DataScalingEntry:
    """One immutable model run on one nested training-section prefix."""

    task_id: int
    phase: str
    scaling_axis: str
    world: str
    model_family: str
    training_protocol: str
    n_train_sections: int
    n_train_cells: int
    condition: str
    seed: int
    experiment_id: str
    output_dir: str
    config_identity_sha256: str
    resolved_config: dict[str, Any]
    training_data_selection: dict[str, Any]
    execution_policy: dict[str, Any]


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _stable_seed(*components: object) -> int:
    digest = hashlib.blake2b(digest_size=8)
    for component in components:
        encoded = str(component).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    return int.from_bytes(digest.digest(), "little")


def _read_spec(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)
    if not isinstance(spec, dict):
        raise TypeError("data-scaling specification must be a mapping")
    required = {
        "sweep_name",
        "base_configs",
        "output_root",
        "corpora",
        "subset_selection_seed",
        "scale_points_sections",
        "seeds",
        "evaluation",
        "masking",
        "policies",
        "sweeps",
    }
    missing = required - set(spec)
    if missing:
        raise ValueError(f"data-scaling specification lacks {sorted(missing)}")
    if spec["scale_points_sections"] != [1, 2, 4, 8, 16, 32, 48]:
        raise ValueError("scale points must be exactly 1,2,4,8,16,32,48 sections")
    if spec["seeds"] != [101, 202, 303]:
        raise ValueError("model seeds must be exactly 101, 202, and 303")
    return spec


def _corpus_definition(corpus_path: str | Path) -> tuple[tuple[str, ...], int]:
    metadata_path = Path(corpus_path) / "metadata.json"
    values = json.loads(metadata_path.read_text(encoding="utf-8"))
    train_ids = tuple(values["split_definition"]["train"])
    cells_per_section = values["cells_per_section"]
    if len(train_ids) != 48 or cells_per_section != 8192:
        raise ValueError(
            "Pilot v0 data scaling requires 48 training sections of 8,192 cells"
        )
    return train_ids, cells_per_section


def ordered_training_sections(
    train_section_ids: tuple[str, ...], subset_selection_seed: int
) -> tuple[str, ...]:
    """Return the one deterministic ordering used by every N and model seed."""
    if len(train_section_ids) != len(set(train_section_ids)):
        raise ValueError("training section identifiers must be unique")
    rng = np.random.default_rng(
        _stable_seed("pilot-v0-data-scaling-subsets", subset_selection_seed)
    )
    order = rng.permutation(len(train_section_ids))
    return tuple(train_section_ids[int(index)] for index in order)


def nested_training_subsets(
    train_section_ids: tuple[str, ...],
    scale_points: list[int],
    subset_selection_seed: int,
) -> dict[int, tuple[str, ...]]:
    ordered = ordered_training_sections(train_section_ids, subset_selection_seed)
    result = {point: ordered[:point] for point in scale_points}
    for smaller, larger in pairwise(scale_points):
        if result[smaller] != result[larger][:smaller]:
            raise RuntimeError("generated training subsets are not nested")
    return result


def _entry_id(
    sweep_name: str,
    phase: str,
    world: str,
    n_sections: int,
    condition: str,
    seed: int,
) -> str:
    return (
        f"{sweep_name}__{phase}__{world}__nsec{n_sections:02d}__"
        f"{condition}__seed{seed:04d}"
    )


def data_scaling_entries(
    spec_path: str | Path,
    *,
    phases: set[str] | None = None,
    n_train_sections: set[int] | None = None,
) -> list[DataScalingEntry]:
    """Resolve the controlled grid without launching computation."""
    spec = _read_spec(spec_path)
    corpus_definitions = {
        world: _corpus_definition(path) for world, path in spec["corpora"].items()
    }
    reference_ids, cells_per_section = corpus_definitions["positive"]
    for world, (section_ids, section_cells) in corpus_definitions.items():
        if section_ids != reference_ids or section_cells != cells_per_section:
            raise ValueError(f"corpus {world!r} does not share the fixed split")
    scale_points = spec["scale_points_sections"]
    ordered = ordered_training_sections(reference_ids, spec["subset_selection_seed"])
    subsets = nested_training_subsets(
        reference_ids, scale_points, spec["subset_selection_seed"]
    )
    entries: list[DataScalingEntry] = []
    for sweep in spec["sweeps"]:
        phase = sweep["phase"]
        if phases is not None and phase not in phases:
            continue
        world = sweep["world"]
        model_family = sweep["model_family"]
        protocol_name = sweep["training_protocol"]
        configured_policy = dict(spec["policies"][protocol_name])
        configured_policy.update(sweep.get("policy_overrides", {}))
        policy = TrainingExecutionPolicy.from_dict(configured_policy)
        points = sweep.get("scale_points_sections", scale_points)
        seeds = sweep.get("seeds", spec["seeds"])
        for point in points:
            if point not in scale_points:
                raise ValueError(f"phase {phase!r} uses an undeclared scale point")
            if n_train_sections is not None and point not in n_train_sections:
                continue
            selection = TrainingDataSelection(
                scaling_axis="N_train_sections",
                subset_selection_seed=spec["subset_selection_seed"],
                ordered_train_section_ids=ordered,
                active_train_section_ids=subsets[point],
            )
            for condition in sweep["conditions"]:
                if model_family == "contextual" and condition not in {
                    "spatial",
                    "shuffled",
                }:
                    raise ValueError("contextual sweeps require spatial/shuffled arms")
                if model_family == "focal" and condition != "focal":
                    raise ValueError("focal sweeps require the focal condition")
                for seed in seeds:
                    experiment_id = _entry_id(
                        spec["sweep_name"], phase, world, point, condition, seed
                    )
                    output_dir = str(Path(spec["output_root"]) / experiment_id)
                    overrides = [
                        f"experiment_name={experiment_id}",
                        f"output_dir={output_dir}",
                        f"seed={seed}",
                        f"condition={condition}",
                        f"data.corpus_path={spec['corpora'][world]}",
                        f"context.shuffle_seed={seed}",
                        f"masking.policy={spec['masking']['policy']}",
                        f"masking.fraction={spec['masking']['fraction']}",
                        f"masking.seed={seed}",
                        f"evaluation.max_cells={spec['evaluation']['max_cells']}",
                        f"evaluation.observation_seed={spec['evaluation']['observation_seed']}",
                        f"evaluation.masking_seed={spec['evaluation']['masking_seed']}",
                    ]
                    if model_family == "contextual":
                        context_size = spec["context_size"]
                        overrides.extend(
                            (
                                f"context.context_size={context_size}",
                                f"model.context_size={context_size}",
                            )
                        )
                    if policy.fixed_steps is not None:
                        overrides.append(f"training.steps={policy.fixed_steps}")
                    config = load_experiment_config(
                        spec["base_configs"][model_family], overrides
                    )
                    resolved = config.to_dict()
                    identity_payload = {
                        "experiment_config": resolved,
                        "training_data_selection": selection.to_dict(),
                        "execution_policy": policy.to_dict(),
                    }
                    entries.append(
                        DataScalingEntry(
                            task_id=len(entries),
                            phase=phase,
                            scaling_axis="N_train_sections",
                            world=world,
                            model_family=model_family,
                            training_protocol=policy.name,
                            n_train_sections=point,
                            n_train_cells=point * cells_per_section,
                            condition=condition,
                            seed=seed,
                            experiment_id=experiment_id,
                            output_dir=output_dir,
                            config_identity_sha256=config_identity_sha256(
                                identity_payload
                            ),
                            resolved_config=resolved,
                            training_data_selection=selection.to_dict(),
                            execution_policy=policy.to_dict(),
                        )
                    )
    entries = [
        DataScalingEntry(**{**asdict(entry), "task_id": task_id})
        for task_id, entry in enumerate(entries)
    ]
    identifiers = [entry.experiment_id for entry in entries]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("data-scaling grid contains duplicate experiment IDs")
    return entries


def write_manifest(
    spec_path: str | Path,
    output_path: str | Path,
    *,
    phases: set[str] | None = None,
    n_train_sections: set[int] | None = None,
) -> dict[str, Any]:
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"manifest already exists: {destination}")
    entries = data_scaling_entries(
        spec_path, phases=phases, n_train_sections=n_train_sections
    )
    if not entries:
        raise ValueError("manifest filters selected no data-scaling experiments")
    spec = _read_spec(spec_path)
    manifest = {
        "schema_version": "pilot-data-scaling-manifest-v1",
        "sweep_name": spec["sweep_name"],
        "source_spec": str(spec_path),
        "spec_sha256": _canonical_sha256(spec),
        "git_sha": _git_sha(),
        "provenance": spec.get("provenance"),
        "filters": {
            "phases": sorted(phases) if phases is not None else None,
            "n_train_sections": (
                sorted(n_train_sections) if n_train_sections is not None else None
            ),
        },
        "task_count": len(entries),
        "entries": [asdict(entry) for entry in entries],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_manifest_entry(
    manifest_path: str | Path, task_id: int
) -> tuple[DataScalingEntry, dict[str, Any]]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "pilot-data-scaling-manifest-v1":
        raise ValueError("unsupported data-scaling manifest schema")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or manifest.get("task_count") != len(entries):
        raise ValueError("manifest task count is inconsistent")
    if not isinstance(task_id, int) or isinstance(task_id, bool):
        raise TypeError("task_id must be an integer")
    if not 0 <= task_id < len(entries):
        raise IndexError(f"task_id must be between 0 and {len(entries) - 1}")
    entry = DataScalingEntry(**entries[task_id])
    if entry.task_id != task_id:
        raise ValueError("manifest array index does not match task_id")
    config = SSLExperimentConfig.from_dict(entry.resolved_config)
    selection = TrainingDataSelection.from_dict(entry.training_data_selection)
    policy = TrainingExecutionPolicy.from_dict(entry.execution_policy)
    identity = config_identity_sha256(
        {
            "experiment_config": config.to_dict(),
            "training_data_selection": selection.to_dict(),
            "execution_policy": policy.to_dict(),
        }
    )
    if identity != entry.config_identity_sha256:
        raise ValueError("manifest resolved configuration identity is invalid")
    return entry, manifest
