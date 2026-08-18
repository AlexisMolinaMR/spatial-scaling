"""Generate an immutable N-train scaling array manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spatial_scaling.experiments.data_scaling import write_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec", type=Path, default=Path("configs/pilot/data_scaling_v0.yaml")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", action="append")
    parser.add_argument("--n-sections", action="append", type=int)
    args = parser.parse_args()
    manifest = write_manifest(
        args.spec,
        args.output,
        phases=set(args.phase) if args.phase else None,
        n_train_sections=set(args.n_sections) if args.n_sections else None,
    )
    print(
        json.dumps({key: manifest[key] for key in ("task_count", "filters")}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
