"""Aggregate the complete matched Pilot v0 spatial falsification matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

from spatial_scaling.evaluation.spatial_falsification import (
    summarize_spatial_falsification,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/pilot_spatial_falsification"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_dirs = sorted(path.parent for path in args.root.rglob("metrics.json"))
    output = args.output or args.root / "summary"
    summarize_spatial_falsification(run_dirs, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
