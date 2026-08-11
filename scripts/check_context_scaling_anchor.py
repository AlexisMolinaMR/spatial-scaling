"""Enforce the validated K=256 stop condition before extending the sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spatial_scaling.evaluation.context_scaling import check_k256_anchor


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("results/pilot_spatial_falsification/summary/paired_deltas.csv"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"anchor report already exists: {args.output}")
    result = check_k256_anchor(args.new, args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
