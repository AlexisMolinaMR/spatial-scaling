"""Aggregate completed Pilot v0 independent-section scaling runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spatial_scaling.evaluation.data_scaling import summarize_data_scaling
from spatial_scaling.experiments.data_scaling import data_scaling_entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec", type=Path, default=Path("configs/pilot/data_scaling_v0.yaml")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pilot_data_scaling_v0/final_summary"),
    )
    args = parser.parse_args()
    entries = data_scaling_entries(
        args.spec,
        phases={
            "primary_positive_85",
            "required_null_85",
            "fixed_compute_positive_85",
            "focal_reference_85",
        },
    )
    result = summarize_data_scaling(entries, args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
