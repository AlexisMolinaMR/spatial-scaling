"""Manifest-driven Pilot v0 spatial-context scaling experiments."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from spatial_scaling.data.synthetic.context import ContextSelectionConfig
from spatial_scaling.training.config import (
    SSLExperimentConfig,
    config_identity_sha256,
    load_experiment_config,
)

CONDITIONS = ("shuffled", "spatial")


@dataclass(frozen=True)
class ContextScalingEntry:
    """One array task and its fully resolved scientific configuration."""

    task_id: int
    phase: str
    world: str
    masking_regime: str
    masking_policy: str
    masking_fraction: float
    context_size: int
    condition: str
    seed: int
    experiment_id: str
    output_dir: str
    config_identity_sha256: str
    resolved_config: dict[str, Any]
    context_selection: dict[str, Any] | None = None


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


def _read_spec(path: str | Path) -> dict[str, Any]:
    spec_path = Path(path)
    with spec_path.open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise TypeError("context-scaling specification must be a mapping")
    required = {
        "sweep_name",
        "base_config",
        "output_root",
        "corpora",
        "seeds",
        "conditions",
        "evaluation",
        "sweeps",
    }
    missing = required - set(values)
    if missing:
        raise ValueError(f"context-scaling specification lacks {sorted(missing)}")
    if tuple(values["conditions"]) != CONDITIONS:
        raise ValueError(f"conditions must be exactly {CONDITIONS}")
    seeds = values["seeds"]
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds)
    ):
        raise ValueError("seeds must be a non-empty list of integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    sweeps = values["sweeps"]
    if not isinstance(sweeps, list) or not sweeps:
        raise ValueError("sweeps must be a non-empty list")
    return values


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _entry_id(
    sweep_name: str,
    world: str,
    regime: str,
    context_size: int,
    condition: str,
    seed: int,
    minimum_distance_um: float | None = None,
) -> str:
    identifier = (
        f"{sweep_name}__{world}__{regime}__k{context_size:04d}__"
        f"{condition}__seed{seed:04d}"
    )
    if minimum_distance_um is not None:
        distance = f"{minimum_distance_um:g}".replace(".", "p")
        identifier += f"__dmin{distance}um"
    return identifier


def _validate_sweep(sweep: dict[str, Any], corpora: dict[str, str]) -> None:
    required = {"phase", "world", "masking", "context_sizes"}
    missing = required - set(sweep)
    if missing:
        raise ValueError(f"sweep lacks {sorted(missing)}")
    if sweep["world"] not in corpora:
        raise ValueError(f"unknown corpus world: {sweep['world']!r}")
    masking = sweep["masking"]
    if not isinstance(masking, dict) or set(masking) != {
        "regime",
        "policy",
        "fraction",
    }:
        raise ValueError("masking must define regime, policy, and fraction")
    context_sizes = sweep["context_sizes"]
    if not isinstance(context_sizes, list) or not context_sizes:
        raise ValueError("context_sizes must be a non-empty list")
    for context_size in context_sizes:
        if (
            not isinstance(context_size, int)
            or isinstance(context_size, bool)
            or context_size <= 0
        ):
            raise ValueError("every configured K must be a positive integer")
    if len(set(context_sizes)) != len(context_sizes):
        raise ValueError(f"duplicate K in phase {sweep['phase']!r}")
    distances = sweep.get("minimum_distances_um")
    if distances is not None:
        if not isinstance(distances, list) or not distances:
            raise ValueError("minimum_distances_um must be a non-empty list")
        parsed = []
        for value in distances:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("minimum distance values must be numeric")
            parsed.append(float(value))
        if len(set(parsed)) != len(parsed):
            raise ValueError(f"duplicate d_min in phase {sweep['phase']!r}")
        eligibility = sweep.get("eligibility_minimum_distance_um", max(parsed))
        for value in parsed:
            ContextSelectionConfig(
                mode="distance_exclusion",
                minimum_distance_um=value,
                eligibility_minimum_distance_um=eligibility,
            ).validate()
    elif "eligibility_minimum_distance_um" in sweep:
        raise ValueError(
            "eligibility_minimum_distance_um requires minimum_distances_um"
        )
    if "selection" in sweep:
        if distances is not None:
            raise ValueError(
                "selection and minimum_distances_um are mutually exclusive"
            )
        ContextSelectionConfig.from_dict(sweep["selection"])


def context_scaling_entries(
    spec_path: str | Path,
    *,
    phases: set[str] | None = None,
    include_context_sizes: set[int] | None = None,
    exclude_context_sizes: set[int] | None = None,
    include_minimum_distances_um: set[float] | None = None,
    exclude_minimum_distances_um: set[float] | None = None,
) -> list[ContextScalingEntry]:
    """Resolve a deterministic, optionally filtered experiment grid."""
    spec = _read_spec(spec_path)
    corpora = spec["corpora"]
    if not isinstance(corpora, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in corpora.items()
    ):
        raise ValueError("corpora must map world names to paths")
    evaluation = spec["evaluation"]
    if not isinstance(evaluation, dict):
        raise TypeError("evaluation must be a mapping")
    all_entries: list[ContextScalingEntry] = []
    for sweep in spec["sweeps"]:
        if not isinstance(sweep, dict):
            raise TypeError("every sweep must be a mapping")
        _validate_sweep(sweep, corpora)
        phase = str(sweep["phase"])
        if phases is not None and phase not in phases:
            continue
        phase_seeds = sweep.get("seeds", spec["seeds"])
        training = sweep.get("training", {})
        if not isinstance(training, dict):
            raise TypeError("optional sweep training overrides must be a mapping")
        if "minimum_distances_um" in sweep:
            selection_values: list[dict[str, Any] | None] = [
                ContextSelectionConfig(
                    mode="distance_exclusion",
                    minimum_distance_um=float(distance),
                    eligibility_minimum_distance_um=float(
                        sweep.get(
                            "eligibility_minimum_distance_um",
                            max(sweep["minimum_distances_um"]),
                        )
                    ),
                ).to_dict()
                for distance in sweep["minimum_distances_um"]
            ]
        elif "selection" in sweep:
            selection_values = [
                ContextSelectionConfig.from_dict(sweep["selection"]).to_dict()
            ]
        else:
            selection_values = [None]
        for context_size in sweep["context_sizes"]:
            if (
                include_context_sizes is not None
                and context_size not in include_context_sizes
            ):
                continue
            if (
                exclude_context_sizes is not None
                and context_size in exclude_context_sizes
            ):
                continue
            for context_selection in selection_values:
                minimum_distance_um = (
                    None
                    if context_selection is None
                    else float(context_selection["minimum_distance_um"])
                )
                if (
                    include_minimum_distances_um is not None
                    and minimum_distance_um not in include_minimum_distances_um
                ):
                    continue
                if (
                    exclude_minimum_distances_um is not None
                    and minimum_distance_um in exclude_minimum_distances_um
                ):
                    continue
                for condition in spec["conditions"]:
                    for seed in phase_seeds:
                        masking = sweep["masking"]
                        experiment_id = _entry_id(
                            spec["sweep_name"],
                            sweep["world"],
                            masking["regime"],
                            context_size,
                            condition,
                            seed,
                            minimum_distance_um=(
                                minimum_distance_um
                                if "minimum_distances_um" in sweep
                                else None
                            ),
                        )
                        output_dir = str(Path(spec["output_root"]) / experiment_id)
                        overrides = [
                            f"experiment_name={experiment_id}",
                            f"output_dir={output_dir}",
                            f"seed={seed}",
                            f"condition={condition}",
                            f"data.corpus_path={corpora[sweep['world']]}",
                            f"context.context_size={context_size}",
                            f"context.shuffle_seed={seed}",
                            f"model.context_size={context_size}",
                            f"masking.policy={masking['policy']}",
                            f"masking.fraction={masking['fraction']}",
                            f"masking.seed={seed}",
                            f"evaluation.max_cells={evaluation['max_cells']}",
                            f"evaluation.observation_seed={evaluation['observation_seed']}",
                            f"evaluation.masking_seed={seed}",
                        ]
                        for key, value in training.items():
                            overrides.append(f"training.{key}={value}")
                        if "evaluation_max_cells" in sweep:
                            overrides.append(
                                f"evaluation.max_cells={sweep['evaluation_max_cells']}"
                            )
                        config = load_experiment_config(spec["base_config"], overrides)
                        resolved = config.to_dict()
                        identity_payload = (
                            {
                                "experiment_config": resolved,
                                "context_selection": context_selection,
                            }
                            if context_selection is not None
                            else resolved
                        )
                        all_entries.append(
                            ContextScalingEntry(
                                task_id=len(all_entries),
                                phase=phase,
                                world=sweep["world"],
                                masking_regime=masking["regime"],
                                masking_policy=masking["policy"],
                                masking_fraction=float(masking["fraction"]),
                                context_size=context_size,
                                condition=condition,
                                seed=seed,
                                experiment_id=experiment_id,
                                output_dir=output_dir,
                                config_identity_sha256=config_identity_sha256(
                                    identity_payload
                                ),
                                resolved_config=resolved,
                                context_selection=context_selection,
                            )
                        )
    entries = [
        ContextScalingEntry(**{**asdict(entry), "task_id": task_id})
        for task_id, entry in enumerate(all_entries)
    ]
    experiment_ids = [entry.experiment_id for entry in entries]
    if len(experiment_ids) != len(set(experiment_ids)):
        raise ValueError("generated experiment grid contains duplicate experiment IDs")
    return entries


def write_manifest(
    spec_path: str | Path,
    output_path: str | Path,
    *,
    phases: set[str] | None = None,
    include_context_sizes: set[int] | None = None,
    exclude_context_sizes: set[int] | None = None,
    include_minimum_distances_um: set[float] | None = None,
    exclude_minimum_distances_um: set[float] | None = None,
) -> dict[str, Any]:
    """Write one immutable array-index mapping without launching work."""
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"manifest already exists: {destination}")
    spec_path = Path(spec_path)
    entries = context_scaling_entries(
        spec_path,
        phases=phases,
        include_context_sizes=include_context_sizes,
        exclude_context_sizes=exclude_context_sizes,
        include_minimum_distances_um=include_minimum_distances_um,
        exclude_minimum_distances_um=exclude_minimum_distances_um,
    )
    if not entries:
        raise ValueError("manifest filters selected no experiments")
    spec = _read_spec(spec_path)
    manifest = {
        "schema_version": "pilot-context-scaling-manifest-v1",
        "sweep_name": spec["sweep_name"],
        "source_spec": str(spec_path),
        "spec_sha256": _canonical_sha256(spec),
        "git_sha": _git_sha(),
        "filters": {
            "phases": sorted(phases) if phases is not None else None,
            "include_context_sizes": (
                sorted(include_context_sizes)
                if include_context_sizes is not None
                else None
            ),
            "exclude_context_sizes": (
                sorted(exclude_context_sizes)
                if exclude_context_sizes is not None
                else None
            ),
            "include_minimum_distances_um": (
                sorted(include_minimum_distances_um)
                if include_minimum_distances_um is not None
                else None
            ),
            "exclude_minimum_distances_um": (
                sorted(exclude_minimum_distances_um)
                if exclude_minimum_distances_um is not None
                else None
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
) -> tuple[ContextScalingEntry, dict[str, Any]]:
    """Load and validate exactly one stable array mapping."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "pilot-context-scaling-manifest-v1":
        raise ValueError("unsupported context-scaling manifest schema")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or manifest.get("task_count") != len(entries):
        raise ValueError("manifest task count is inconsistent")
    if not isinstance(task_id, int) or isinstance(task_id, bool):
        raise TypeError("task_id must be an integer")
    if not 0 <= task_id < len(entries):
        raise IndexError(f"task_id must be between 0 and {len(entries) - 1}")
    entry = ContextScalingEntry(**entries[task_id])
    if entry.task_id != task_id:
        raise ValueError("manifest array index does not match recorded task_id")
    config = SSLExperimentConfig.from_dict(entry.resolved_config)
    identity_payload = (
        {
            "experiment_config": config.to_dict(),
            "context_selection": entry.context_selection,
        }
        if entry.context_selection is not None
        else config.to_dict()
    )
    identity = config_identity_sha256(identity_payload)
    if identity != entry.config_identity_sha256:
        raise ValueError("manifest resolved configuration identity is invalid")
    return entry, manifest
