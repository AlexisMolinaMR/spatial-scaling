"""Auditable aggregation for matched spatial-versus-shuffled Pilot v0 runs."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

from spatial_scaling.data.synthetic.dataset import GENE_CLASSES

METRIC_GROUPS = ("all", *GENE_CLASSES)


def _read_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "resolved_config.json"
    if not metrics_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"incomplete run artifacts: {run_dir}")
    return (
        json.loads(metrics_path.read_text(encoding="utf-8")),
        json.loads(config_path.read_text(encoding="utf-8")),
    )


def _matched_pair_values(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": config["seed"],
        "corpus_path": config["data"]["corpus_path"],
        "context": config["context"],
        "model": config["model"],
        "masking": config["masking"],
        "optimizer": config["optimizer"],
        "training": config["training"],
        "evaluation": config["evaluation"],
        "section_ids": config["section_ids"],
    }


def summarize_spatial_falsification(
    run_dirs: list[Path], output_dir: Path
) -> dict[str, Any]:
    """Write per-run losses, paired deltas, and three-seed summaries."""
    rows: list[dict[str, Any]] = []
    pairs: dict[tuple[str, str, int], dict[str, tuple[dict, dict]]] = {}
    for run_dir in run_dirs:
        metrics, config = _read_run(run_dir)
        parts = run_dir.parts
        try:
            root_index = parts.index("pilot_spatial_falsification")
            world, regime, condition = parts[root_index + 1 : root_index + 4]
        except (ValueError, IndexError) as error:
            raise ValueError(f"cannot infer matrix condition from {run_dir}") from error
        seed = int(config["seed"])
        final = metrics["final_validation"]
        row = {
            "world": world,
            "regime": regime,
            "condition": condition,
            "seed": seed,
            "parameter_count": metrics["parameter_count"],
            "train_masked_mse": metrics["final_training_interval_masked_mse"],
            **{
                f"validation_masked_mse_{name}": final[f"masked_mse_{name}"]
                for name in METRIC_GROUPS
            },
            "elapsed_seconds": metrics["elapsed_seconds"],
            "examples_per_second": metrics["examples_per_second"],
            "peak_device_memory_bytes": metrics["peak_device_memory_bytes"],
            "evaluation_index_sha256": metrics["evaluation_observations"][
                "index_sha256"
            ],
            "evaluation_mask_sha256": metrics["evaluation_observations"]["mask_sha256"],
            "run_dir": str(run_dir),
        }
        rows.append(row)
        if condition in {"spatial", "shuffled"}:
            pairs.setdefault((world, regime, seed), {})[condition] = (metrics, config)

    paired_rows: list[dict[str, Any]] = []
    for (world, regime, seed), conditions in sorted(pairs.items()):
        if set(conditions) != {"spatial", "shuffled"}:
            raise ValueError(f"incomplete matched pair: {(world, regime, seed)}")
        spatial_metrics, spatial_config = conditions["spatial"]
        shuffled_metrics, shuffled_config = conditions["shuffled"]
        if _matched_pair_values(spatial_config) != _matched_pair_values(
            shuffled_config
        ):
            raise ValueError(
                f"uncontrolled matched-pair difference: {(world, regime, seed)}"
            )
        spatial_observations = spatial_metrics["evaluation_observations"]
        shuffled_observations = shuffled_metrics["evaluation_observations"]
        if spatial_observations != shuffled_observations:
            raise ValueError(f"evaluation observations differ: {(world, regime, seed)}")
        if spatial_metrics["parameter_count"] != shuffled_metrics["parameter_count"]:
            raise ValueError(f"parameter counts differ: {(world, regime, seed)}")
        spatial_final = spatial_metrics["final_validation"]
        shuffled_final = shuffled_metrics["final_validation"]
        paired_rows.append(
            {
                "world": world,
                "regime": regime,
                "seed": seed,
                "parameter_count": spatial_metrics["parameter_count"],
                **{
                    f"delta_space_{name}": shuffled_final[f"masked_mse_{name}"]
                    - spatial_final[f"masked_mse_{name}"]
                    for name in METRIC_GROUPS
                },
            }
        )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in paired_rows:
        grouped.setdefault((str(row["world"]), str(row["regime"])), []).append(row)
    summaries: list[dict[str, Any]] = []
    for (world, regime), group in sorted(grouped.items()):
        if len(group) < 3:
            raise ValueError(f"fewer than three matched seeds: {(world, regime)}")
        summary: dict[str, Any] = {
            "world": world,
            "regime": regime,
            "num_seeds": len(group),
        }
        for name in METRIC_GROUPS:
            values = [float(row[f"delta_space_{name}"]) for row in group]
            summary[f"delta_space_{name}_mean"] = statistics.mean(values)
            summary[f"delta_space_{name}_sample_std"] = statistics.stdev(values)
            summary[f"delta_space_{name}_positive_seeds"] = sum(
                value > 0 for value in values
            )
        summaries.append(summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("run_metrics.csv", "paired_deltas.csv", "delta_summary.csv"):
        if (output_dir / filename).exists():
            raise FileExistsError(
                f"summary output already exists: {output_dir / filename}"
            )
    _write_csv(output_dir / "run_metrics.csv", rows)
    _write_csv(output_dir / "paired_deltas.csv", paired_rows)
    _write_csv(output_dir / "delta_summary.csv", summaries)
    result = {
        "runs": rows,
        "paired_deltas": paired_rows,
        "delta_summaries": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty summary: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
