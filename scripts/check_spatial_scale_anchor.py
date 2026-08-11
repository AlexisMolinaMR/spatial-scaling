"""Gate distance-exclusion scaling on the completed K=32 anchor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spatial_scaling.evaluation.spatial_scale_calibration import (
    check_distance_zero_anchor,
)
from spatial_scaling.experiments.context_scaling import context_scaling_entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("configs/pilot/spatial_scale_calibration_v0.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/pilot_spatial_scale_calibration_v0/anchor_d0_reproduction.json"
        ),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"anchor output already exists: {args.output}")
    new_entries = context_scaling_entries(
        args.spec,
        phases={"distance_positive_85"},
        include_minimum_distances_um={0.0},
    )
    reference_entries = context_scaling_entries(
        "configs/pilot/context_scaling_v0.yaml",
        phases={"primary_85"},
        include_context_sizes={32},
    )
    result = check_distance_zero_anchor(new_entries, reference_entries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
