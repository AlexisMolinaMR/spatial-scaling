"""Create the focal-cell masking calibration table and figure."""

from __future__ import annotations

import argparse
from pathlib import Path

from spatial_scaling.evaluation.ssl_summary import summarize_runs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    summarize_runs(args.runs, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
