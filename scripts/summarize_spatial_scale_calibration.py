"""Aggregate completed Pilot v0 spatial-scale calibration runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spatial_scaling.evaluation.spatial_scale_calibration import (
    reused_low_k_entries,
    summarize_spatial_scale_calibration,
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
        default=Path("results/pilot_spatial_scale_calibration_v0/final_summary"),
    )
    args = parser.parse_args()
    entries = context_scaling_entries(args.spec)
    result = summarize_spatial_scale_calibration(
        entries, args.output, reused_entries=reused_low_k_entries()
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
