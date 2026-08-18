"""Summarize the final converged Pilot v0 N-train frontier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spatial_scaling.evaluation.converged_data_frontier import (
    summarize_converged_data_frontier,
)
from spatial_scaling.experiments.data_scaling import data_scaling_entries


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
        "--capped-spec",
        type=Path,
        default=Path("configs/pilot/data_scaling_v0.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pilot_data_scaling_converged_frontier_v0/final_summary"),
    )
    args = parser.parse_args()
    logical = data_scaling_entries(args.spec)
    new = data_scaling_entries(args.spec, n_train_sections={2, 4, 8, 16, 32})
    spatial = data_scaling_entries(args.spatial_endpoint_spec)
    shuffled = [
        entry
        for entry in data_scaling_entries(args.shuffled_endpoint_spec)
        if entry.training_protocol == "validation_plateau_lr_decay"
        and entry.condition == "shuffled"
    ]
    capped = [
        entry
        for entry in data_scaling_entries(args.capped_spec)
        if entry.phase == "primary_positive_85"
    ]
    result = summarize_converged_data_frontier(
        logical, new, spatial, shuffled, capped, args.output
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
