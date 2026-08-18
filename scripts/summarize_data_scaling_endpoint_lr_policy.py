"""Summarize the Pilot v0 endpoint learning-rate policy diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spatial_scaling.evaluation.lr_policy_diagnostic import (
    summarize_lr_policy_diagnostic,
)
from spatial_scaling.experiments.data_scaling import data_scaling_entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("configs/pilot/data_scaling_endpoint_lr_policy_v0.yaml"),
    )
    parser.add_argument(
        "--historical-spec",
        type=Path,
        default=Path("configs/pilot/data_scaling_endpoint_convergence_v0.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pilot_data_scaling_endpoint_lr_policy_v0/final_summary"),
    )
    args = parser.parse_args()
    current = data_scaling_entries(args.spec)
    historical = data_scaling_entries(args.historical_spec)
    result = summarize_lr_policy_diagnostic(current, historical, args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
