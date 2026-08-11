"""Aggregate selected completed Pilot v0 context-scaling phases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spatial_scaling.evaluation.context_scaling import summarize_context_scaling
from spatial_scaling.experiments.context_scaling import context_scaling_entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec", type=Path, default=Path("configs/pilot/context_scaling_v0.yaml")
    )
    parser.add_argument("--phase", action="append", required=True)
    parser.add_argument("--include-k", action="append", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    phases = set(args.phase)
    context_sizes = set(args.include_k) if args.include_k else None
    entries = context_scaling_entries(
        args.spec, phases=phases, include_context_sizes=context_sizes
    )
    scientific_entries = [
        entry for entry in entries if entry.phase != "feasibility_2048"
    ]
    if not scientific_entries:
        raise ValueError("summary requires at least one three-seed scientific phase")
    radii = sorted({entry.context_size for entry in scientific_entries})
    result = summarize_context_scaling(
        scientific_entries,
        args.output,
        radius_corpus_path="outputs/pilot_v0_synthetic_spatial",
        radius_context_sizes=radii,
    )
    print(
        json.dumps(
            {
                "runs": len(result["runs"]),
                "paired_deltas": len(result["paired_deltas"]),
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
