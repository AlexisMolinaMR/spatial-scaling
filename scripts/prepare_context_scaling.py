"""Generate a stable, recorded SLURM array manifest from one sweep spec."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spatial_scaling.experiments.context_scaling import write_manifest


def _integers(values: list[str] | None) -> set[int] | None:
    return None if values is None else {int(value) for value in values}


def _floats(values: list[str] | None) -> set[float] | None:
    return None if values is None else {float(value) for value in values}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec", type=Path, default=Path("configs/pilot/context_scaling_v0.yaml")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", action="append")
    parser.add_argument("--include-k", action="append")
    parser.add_argument("--exclude-k", action="append")
    parser.add_argument("--include-d-min", action="append")
    parser.add_argument("--exclude-d-min", action="append")
    args = parser.parse_args()
    manifest = write_manifest(
        args.spec,
        args.output,
        phases=set(args.phase) if args.phase else None,
        include_context_sizes=_integers(args.include_k),
        exclude_context_sizes=_integers(args.exclude_k),
        include_minimum_distances_um=_floats(args.include_d_min),
        exclude_minimum_distances_um=_floats(args.exclude_d_min),
    )
    print(
        json.dumps({key: manifest[key] for key in ("task_count", "filters")}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
