"""Preflight final frontier grids and historical endpoint reuse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spatial_scaling.evaluation.converged_data_frontier import (
    validate_endpoint_reuse,
    validate_final_grids,
)
from spatial_scaling.experiments.data_scaling import (
    data_scaling_entries,
    load_manifest_entry,
)
from spatial_scaling.training.policies import TrainingDataSelection


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("configs/pilot/data_scaling_converged_frontier_v0.yaml"),
    )
    parser.add_argument(
        "--spatial-endpoint-spec",
        type=Path,
        default=Path("configs/pilot/data_scaling_spatial_90pass_convergence_v0.yaml"),
    )
    parser.add_argument(
        "--shuffled-endpoint-spec",
        type=Path,
        default=Path("configs/pilot/data_scaling_endpoint_lr_policy_v0.yaml"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "results/pilot_data_scaling_converged_frontier_v0/manifests/new_runs.json"
        ),
    )
    args = parser.parse_args()
    logical = data_scaling_entries(args.spec)
    new = data_scaling_entries(args.spec, n_train_sections={2, 4, 8, 16, 32})
    validate_final_grids(logical, new)
    manifest_entries = [
        load_manifest_entry(args.manifest, task_id)[0] for task_id in range(30)
    ]
    for manifest_entry, expected in zip(manifest_entries, new, strict=True):
        if (
            manifest_entry.experiment_id != expected.experiment_id
            or manifest_entry.config_identity_sha256 != expected.config_identity_sha256
            or manifest_entry.resolved_config != expected.resolved_config
            or TrainingDataSelection.from_dict(
                manifest_entry.training_data_selection
            ).to_dict()
            != TrainingDataSelection.from_dict(
                expected.training_data_selection
            ).to_dict()
            or manifest_entry.execution_policy != expected.execution_policy
        ):
            raise ValueError("immutable manifest differs from the resolved 30-run grid")
    spatial = data_scaling_entries(args.spatial_endpoint_spec)
    shuffled = [
        entry
        for entry in data_scaling_entries(args.shuffled_endpoint_spec)
        if entry.training_protocol == "validation_plateau_lr_decay"
        and entry.condition == "shuffled"
    ]
    result = validate_endpoint_reuse(logical, spatial, shuffled)
    result["logical_runs"] = len(logical)
    result["new_runs"] = len(new)
    result["manifest"] = str(args.manifest)
    result["manifest_task_count"] = len(manifest_entries)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
