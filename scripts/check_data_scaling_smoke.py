"""Gate the full Pilot v0 N-train grids on actual N=1/N=48 exposure."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from spatial_scaling.experiments.data_scaling import data_scaling_entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec", type=Path, default=Path("configs/pilot/data_scaling_v0.yaml")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pilot_data_scaling_v0/smoke_validation.json"),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"smoke validation already exists: {args.output}")
    entries = data_scaling_entries(args.spec, phases={"smoke_positive_85"})
    if {entry.n_train_sections for entry in entries} != {1, 48}:
        raise ValueError("smoke gate requires exactly N=1 and N=48")
    rows = []
    for entry in entries:
        metrics_path = Path(entry.output_dir) / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        expected_examples = entry.n_train_cells
        expected_steps = expected_examples // metrics["training_batch_size"]
        failures = []
        if metrics["parameter_count"] != 121024:
            failures.append("contextual parameter count changed")
        if metrics["examples_processed"] != expected_examples:
            failures.append("one complete pass was not processed")
        if metrics["optimization_steps"] != expected_steps:
            failures.append("optimizer step count does not equal one complete pass")
        if metrics["unique_training_observations_seen"] != expected_examples:
            failures.append("not every available focal observation was seen")
        if not metrics["minimum_exposure_satisfied"]:
            failures.append("minimum exposure was not satisfied")
        if metrics["effective_passes"] != 1.0:
            failures.append("smoke did not execute exactly one effective pass")
        if len(metrics["section_exposure"]) != entry.n_train_sections:
            failures.append("section exposure row count is incorrect")
        for section in metrics["section_exposure"]:
            if section["unique_observations_seen"] != section["available_observations"]:
                failures.append(f"incomplete section exposure: {section['section_id']}")
        final_loss = float(metrics["final_validation"]["masked_mse_all"])
        if not math.isfinite(final_loss):
            failures.append("validation loss is non-finite")
        rows.append(
            {
                "n_train_sections": entry.n_train_sections,
                "n_train_cells": entry.n_train_cells,
                "optimization_steps": metrics["optimization_steps"],
                "examples_processed": metrics["examples_processed"],
                "unique_training_observations_seen": metrics[
                    "unique_training_observations_seen"
                ],
                "effective_passes": metrics["effective_passes"],
                "final_validation_masked_mse_all": final_loss,
                "evaluation_index_sha256": metrics["evaluation_observations"][
                    "index_sha256"
                ],
                "evaluation_mask_sha256": metrics["evaluation_observations"][
                    "mask_sha256"
                ],
                "failures": failures,
            }
        )
    if len({row["evaluation_index_sha256"] for row in rows}) != 1:
        raise RuntimeError("validation observations differ between N=1 and N=48")
    if len({row["evaluation_mask_sha256"] for row in rows}) != 1:
        raise RuntimeError("validation masks differ between N=1 and N=48")
    failures = [failure for row in rows for failure in row["failures"]]
    result = {"passed": not failures, "runs": rows, "failures": failures}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if failures:
        raise RuntimeError("data-scaling smoke validation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
