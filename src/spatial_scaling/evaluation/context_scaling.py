"""Paired scientific and computational summaries for context scaling."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from spatial_scaling.data.synthetic.dataset import (
    GENE_CLASSES,
    SyntheticExpressionDataset,
)
from spatial_scaling.experiments.context_scaling import ContextScalingEntry
from spatial_scaling.spatial.neighborhoods import (
    kth_neighbor_radii_um,
    summarize_radii_um,
)

METRIC_GROUPS = ("all", *GENE_CLASSES)

ANCHOR_ABSOLUTE_TOLERANCES = {
    "all": 0.04,
    "intrinsic": 0.02,
    "mixed": 0.04,
    "spatial": 0.06,
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_k256_anchor(
    new_paired_csv: str | Path,
    reference_paired_csv: str | Path,
) -> dict[str, Any]:
    """Compare K=256 seed-level deltas to the validated positive 85% anchor."""
    with Path(new_paired_csv).open(newline="", encoding="utf-8") as handle:
        new_rows = list(csv.DictReader(handle))
    with Path(reference_paired_csv).open(newline="", encoding="utf-8") as handle:
        reference_rows = list(csv.DictReader(handle))
    new = {
        int(row["seed"]): row
        for row in new_rows
        if row["world"] == "positive"
        and row["masking_regime"] == "high_random_085"
        and int(row["context_size"]) == 256
    }
    reference = {
        int(row["seed"]): row
        for row in reference_rows
        if row["world"] == "positive" and row["regime"] == "high_random_085"
    }
    if set(new) != {101, 202, 303} or set(reference) != {101, 202, 303}:
        raise ValueError("anchor comparison requires seeds 101, 202, and 303")
    comparisons = []
    failures = []
    for seed in sorted(new):
        if int(new[seed]["parameter_count"]) != int(reference[seed]["parameter_count"]):
            failures.append(f"seed {seed}: parameter count changed")
        row: dict[str, Any] = {"seed": seed}
        for group, tolerance in ANCHOR_ABSOLUTE_TOLERANCES.items():
            observed = float(new[seed][f"delta_space_{group}"])
            expected = float(reference[seed][f"delta_space_{group}"])
            difference = observed - expected
            row[f"observed_delta_space_{group}"] = observed
            row[f"reference_delta_space_{group}"] = expected
            row[f"difference_{group}"] = difference
            row[f"absolute_tolerance_{group}"] = tolerance
            if abs(difference) > tolerance:
                failures.append(
                    f"seed {seed} {group}: |{difference:.6g}| > {tolerance}"
                )
        comparisons.append(row)
    positive_spatial_seeds = sum(
        float(row["delta_space_spatial"]) > 0 for row in new.values()
    )
    if positive_spatial_seeds != 3:
        failures.append(
            f"spatial delta positive in {positive_spatial_seeds}/3 seeds, expected 3/3"
        )
    return {
        "passed": not failures,
        "context_size": 256,
        "world": "positive",
        "masking_regime": "high_random_085",
        "positive_spatial_seeds": positive_spatial_seeds,
        "comparisons": comparisons,
        "failures": failures,
    }


def _matched_config(config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    for key in (
        "condition",
        "experiment_name",
        "output_dir",
        "config_identity_sha256",
    ):
        result.pop(key, None)
    return result


def paired_delta_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Calculate deltas from matched seed-level runs, never averaged arms."""
    pairs: dict[tuple[str, str, int, int], dict[str, dict[str, Any]]] = defaultdict(
        dict
    )
    for row in run_rows:
        key = (
            str(row["world"]),
            str(row["masking_regime"]),
            int(row["context_size"]),
            int(row["seed"]),
        )
        condition = str(row["condition"])
        if condition in pairs[key]:
            raise ValueError(
                f"duplicate condition in matched pair: {(*key, condition)}"
            )
        pairs[key][condition] = row
    result: list[dict[str, Any]] = []
    for key, conditions in sorted(pairs.items()):
        if set(conditions) != {"shuffled", "spatial"}:
            raise ValueError(f"incomplete matched pair: {key}")
        shuffled = conditions["shuffled"]
        spatial = conditions["spatial"]
        for field in (
            "parameter_count",
            "optimization_steps",
            "training_batch_size",
            "effective_training_examples",
            "evaluation_index_sha256",
            "evaluation_mask_sha256",
            "validation_observation_count",
        ):
            if shuffled[field] != spatial[field]:
                raise ValueError(f"matched pair differs in {field}: {key}")
        if shuffled["matched_config"] != spatial["matched_config"]:
            raise ValueError(f"uncontrolled matched-pair config difference: {key}")
        world, regime, context_size, seed = key
        result.append(
            {
                "world": world,
                "masking_regime": regime,
                "context_size": context_size,
                "seed": seed,
                "parameter_count": spatial["parameter_count"],
                **{
                    f"delta_space_{group}": shuffled[f"validation_masked_mse_{group}"]
                    - spatial[f"validation_masked_mse_{group}"]
                    for group in METRIC_GROUPS
                },
            }
        )
    return result


def summarize_paired_deltas(
    deltas: list[dict[str, Any]], *, required_seeds: int
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in deltas:
        grouped[
            (
                str(row["world"]),
                str(row["masking_regime"]),
                int(row["context_size"]),
            )
        ].append(row)
    summaries: list[dict[str, Any]] = []
    for (world, regime, context_size), rows in sorted(grouped.items()):
        if len(rows) != required_seeds:
            raise ValueError(
                f"expected {required_seeds} matched seeds for "
                f"{(world, regime, context_size)}; got {len(rows)}"
            )
        summary: dict[str, Any] = {
            "world": world,
            "masking_regime": regime,
            "context_size": context_size,
            "num_seeds": len(rows),
        }
        for group in METRIC_GROUPS:
            values = [float(row[f"delta_space_{group}"]) for row in rows]
            summary[f"delta_space_{group}_mean"] = statistics.mean(values)
            summary[f"delta_space_{group}_sample_std"] = (
                statistics.stdev(values) if len(values) >= 2 else None
            )
            summary[f"delta_space_{group}_positive_seeds"] = sum(
                value > 0 for value in values
            )
        summaries.append(summary)
    return summaries


def evaluation_radius_rows(
    corpus_path: str | Path,
    *,
    split: str,
    max_cells: int,
    observation_seed: int,
    context_sizes: list[int],
) -> tuple[list[dict[str, Any]], str]:
    """Measure true-geometry Kth-neighbor radii for every evaluation target."""
    dataset = SyntheticExpressionDataset(corpus_path, split, cache_sections=1)
    indices = dataset.deterministic_indices(max_cells, observation_seed)
    digest = hashlib.sha256(indices.astype("<i8", copy=False).tobytes()).hexdigest()
    section_offsets = indices // dataset.cells_per_section
    by_k: dict[int, list[np.ndarray]] = {value: [] for value in context_sizes}
    for section_offset in np.unique(section_offsets):
        query_offsets = (
            indices[section_offsets == section_offset] % dataset.cells_per_section
        )
        _, coordinates, _ = dataset.spatial_section(int(section_offset))
        section_radii = kth_neighbor_radii_um(coordinates, query_offsets, context_sizes)
        for context_size, values in section_radii.items():
            by_k[context_size].append(values)
    rows = []
    for context_size in sorted(by_k):
        values = np.concatenate(by_k[context_size])
        if len(values) != len(indices):
            raise RuntimeError("physical-radius target count is inconsistent")
        rows.append(
            {
                "context_size": context_size,
                "evaluation_targets": len(values),
                "evaluation_index_sha256": digest,
                **summarize_radii_um(values),
            }
        )
    return rows, digest


def run_metric_rows(entries: list[ContextScalingEntry]) -> list[dict[str, Any]]:
    """Load completed run artifacts into auditable flat metric records."""
    rows: list[dict[str, Any]] = []
    for entry in entries:
        run_dir = Path(entry.output_dir)
        metrics = _read_json(run_dir / "metrics.json")
        config = _read_json(run_dir / "resolved_config.json")
        provenance = _read_json(run_dir / "provenance.json")
        if metrics.get("experiment_id") != entry.experiment_id:
            raise ValueError(f"experiment identity mismatch: {run_dir}")
        if metrics.get("config_identity_sha256") != entry.config_identity_sha256:
            raise ValueError(f"configuration identity mismatch: {run_dir}")
        observations = metrics["evaluation_observations"]
        final = metrics["final_validation"]
        selection = metrics.get("context_selection") or {
            "mode": "nearest",
            "minimum_distance_um": 0.0,
            "eligibility_minimum_distance_um": 0.0,
        }
        eligibility = metrics.get("context_eligibility") or {}
        distance_summary = metrics.get("context_distance_summary") or {}
        distance_rows = {
            row["selected_distance_statistic"]: row
            for row in distance_summary.get("distributions", [])
        }
        row = {
            "phase": entry.phase,
            "world": entry.world,
            "masking_regime": entry.masking_regime,
            "masking_policy": entry.masking_policy,
            "masking_fraction": entry.masking_fraction,
            "context_size": entry.context_size,
            "context_selection_mode": selection["mode"],
            "minimum_distance_um": selection["minimum_distance_um"],
            "eligibility_minimum_distance_um": selection[
                "eligibility_minimum_distance_um"
            ],
            "condition": entry.condition,
            "seed": entry.seed,
            "experiment_id": entry.experiment_id,
            "config_identity_sha256": entry.config_identity_sha256,
            "git_sha": provenance["git"]["commit_sha"],
            "git_dirty": provenance["git"]["dirty"],
            "parameter_count": metrics["parameter_count"],
            "optimization_steps": metrics["optimization_steps"],
            "training_batch_size": metrics["training_batch_size"],
            "effective_training_examples": metrics["effective_training_examples"],
            "train_masked_mse": metrics["final_training_interval_masked_mse"],
            **{
                f"validation_masked_mse_{group}": final[f"masked_mse_{group}"]
                for group in METRIC_GROUPS
            },
            "evaluation_index_sha256": observations["index_sha256"],
            "evaluation_mask_sha256": observations["mask_sha256"],
            "validation_observation_count": observations["count"],
            "requested_validation_observation_count": observations.get(
                "requested_count", observations["count"]
            ),
            "train_current_threshold_eligible": eligibility.get("train", {}).get(
                "current_threshold_eligible"
            ),
            "train_total_observations": eligibility.get("train", {}).get(
                "total_observations"
            ),
            "train_common_threshold_eligible": eligibility.get("train", {}).get(
                "common_threshold_eligible"
            ),
            "evaluation_current_threshold_eligible": eligibility.get(
                "evaluation", {}
            ).get("current_threshold_eligible"),
            "evaluation_total_observations": eligibility.get("evaluation", {}).get(
                "total_observations"
            ),
            "evaluation_common_threshold_eligible": eligibility.get(
                "evaluation", {}
            ).get("common_threshold_eligible"),
            **{
                f"actual_{statistic}_selected_distance_{quantile}": values[quantile]
                for statistic, values in distance_rows.items()
                for quantile in ("median_um", "p25_um", "p75_um", "p05_um", "p95_um")
            },
            "run_elapsed_seconds": metrics["run_elapsed_seconds"],
            "optimization_elapsed_seconds": metrics["optimization_elapsed_seconds"],
            "training_step_time_seconds_mean": metrics[
                "training_step_time_seconds_mean"
            ],
            "training_step_time_seconds_median": metrics[
                "training_step_time_seconds_median"
            ],
            "examples_per_second": metrics["examples_per_second"],
            "context_tokens_per_second": metrics["context_tokens_per_second"],
            "peak_device_memory_allocated_bytes": metrics[
                "peak_device_memory_allocated_bytes"
            ],
            "peak_device_memory_reserved_bytes": metrics[
                "peak_device_memory_reserved_bytes"
            ],
            "device_name": provenance["device"].get("name"),
            "slurm_job_id": provenance["environment"].get("slurm_job_id"),
            "started_at_utc": metrics["started_at_utc"],
            "ended_at_utc": metrics["ended_at_utc"],
            "run_dir": str(run_dir),
            "matched_config": _matched_config(config),
        }
        rows.append(row)
    return rows


_run_rows = run_metric_rows


def _mean_sd(rows: list[dict[str, Any]], key: str) -> tuple[float, float | None]:
    values = [float(row[key]) for row in rows]
    return statistics.mean(values), statistics.stdev(values) if len(
        values
    ) > 1 else None


def computational_summary_rows(
    run_rows: list[dict[str, Any]], *, phase: str
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        if row["phase"] == phase:
            grouped[int(row["context_size"])].append(row)
    result = []
    metric_keys = (
        "run_elapsed_seconds",
        "training_step_time_seconds_median",
        "peak_device_memory_allocated_bytes",
        "peak_device_memory_reserved_bytes",
        "examples_per_second",
        "context_tokens_per_second",
    )
    for context_size, rows in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "context_size": context_size,
            "num_runs": len(rows),
            "training_batch_size": rows[0]["training_batch_size"],
            "optimization_steps": rows[0]["optimization_steps"],
        }
        for key in metric_keys:
            mean, sample_std = _mean_sd(rows, key)
            summary[f"{key}_mean"] = mean
            summary[f"{key}_sample_std"] = sample_std
        result.append(summary)
    return result


def _plot_delta_vs_k(
    path: Path,
    summaries: list[dict[str, Any]],
    *,
    world: str,
    regime: str,
    groups: tuple[str, ...],
    title: str,
) -> None:
    selected = sorted(
        (
            row
            for row in summaries
            if row["world"] == world and row["masking_regime"] == regime
        ),
        key=lambda row: row["context_size"],
    )
    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    colors = {
        "all": "black",
        "intrinsic": "tab:orange",
        "mixed": "tab:green",
        "spatial": "tab:red",
    }
    for group in groups:
        axis.errorbar(
            [row["context_size"] for row in selected],
            [row[f"delta_space_{group}_mean"] for row in selected],
            yerr=[row[f"delta_space_{group}_sample_std"] for row in selected],
            color=colors[group],
            marker="o",
            capsize=3,
            label=group,
        )
    axis.axhline(0.0, color="0.6", linewidth=1)
    axis.set_xscale("log", base=2)
    axis.set(
        xlabel="Context size K (log2 scale)",
        ylabel="Paired spatial advantage Δspace (masked MSE)",
        title=title,
    )
    axis.set_xticks([row["context_size"] for row in selected])
    axis.set_xticklabels([str(row["context_size"]) for row in selected])
    axis.legend(title="Gene class")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_delta_vs_radius(
    path: Path,
    summaries: list[dict[str, Any]],
    radii: list[dict[str, Any]],
) -> None:
    selected = {
        int(row["context_size"]): row
        for row in summaries
        if row["world"] == "positive" and row["masking_regime"] == "high_random_085"
    }
    radius_by_k = {int(row["context_size"]): row for row in radii}
    context_sizes = sorted(selected)
    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    colors = {"intrinsic": "tab:orange", "mixed": "tab:green", "spatial": "tab:red"}
    for group, color in colors.items():
        axis.errorbar(
            [radius_by_k[k]["median_radius_um"] for k in context_sizes],
            [selected[k][f"delta_space_{group}_mean"] for k in context_sizes],
            yerr=[
                selected[k][f"delta_space_{group}_sample_std"] for k in context_sizes
            ],
            color=color,
            marker="o",
            capsize=3,
            label=group,
        )
    axis.axhline(0.0, color="0.6", linewidth=1)
    axis.axvline(200.0, color="tab:blue", linestyle="--", label="RBF scale (200 µm)")
    axis.set(
        xlabel="Median true Kth-neighbor radius (µm)",
        ylabel="Paired spatial advantage Δspace (masked MSE)",
        title="Spatial advantage versus physical radius",
    )
    axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_raw_losses(path: Path, run_rows: list[dict[str, Any]]) -> None:
    selected = [row for row in run_rows if row["phase"] == "primary_85"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for axis, group in zip(axes, ("all", "spatial"), strict=True):
        grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            grouped[(int(row["context_size"]), str(row["condition"]))].append(row)
        for condition, color in (("shuffled", "tab:gray"), ("spatial", "tab:blue")):
            context_sizes = sorted({key[0] for key in grouped if key[1] == condition})
            means = []
            errors = []
            for context_size in context_sizes:
                mean, sample_std = _mean_sd(
                    grouped[(context_size, condition)],
                    f"validation_masked_mse_{group}",
                )
                means.append(mean)
                errors.append(sample_std)
            axis.errorbar(
                context_sizes,
                means,
                yerr=errors,
                marker="o",
                capsize=3,
                color=color,
                label=condition,
            )
        axis.set_xscale("log", base=2)
        axis.set(
            xlabel="Context size K (log2 scale)",
            ylabel="Validation masked MSE",
            title=f"{group.capitalize()} loss",
        )
        axis.set_xticks(context_sizes)
        axis.set_xticklabels([str(value) for value in context_sizes])
        axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_computational(path: Path, rows: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    specifications = (
        ("run_elapsed_seconds_mean", "Run time (s)", "Runtime"),
        ("training_step_time_seconds_median_mean", "Median step time (s)", "Step time"),
        (
            "peak_device_memory_allocated_bytes_mean",
            "Peak allocated GPU memory (GiB)",
            "GPU memory",
        ),
        ("examples_per_second_mean", "Training examples/s", "Throughput"),
    )
    x = [row["context_size"] for row in rows]
    for axis, (key, ylabel, title) in zip(axes.flat, specifications, strict=True):
        values = [row[key] for row in rows]
        if key == "peak_device_memory_allocated_bytes_mean":
            values = [value / 2**30 for value in values]
        axis.plot(x, values, color="black", marker="o")
        axis.set_xscale("log", base=2)
        axis.set(xlabel="Context size K (log2 scale)", ylabel=ylabel, title=title)
        axis.set_xticks(x)
        axis.set_xticklabels([str(value) for value in x])
    figure.savefig(path, dpi=180)
    plt.close(figure)


def summarize_context_scaling(
    entries: list[ContextScalingEntry],
    output_dir: str | Path,
    *,
    radius_corpus_path: str | Path,
    radius_context_sizes: list[int],
    evaluation_split: str = "validation",
    evaluation_max_cells: int = 8192,
    observation_seed: int = 991,
    required_seeds: int = 3,
) -> dict[str, Any]:
    """Validate runs and write auditable tables plus unsmoothed figures."""
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"summary output already exists: {output}")
    output.mkdir(parents=True)
    run_rows = _run_rows(entries)
    deltas = paired_delta_rows(run_rows)
    summaries = summarize_paired_deltas(deltas, required_seeds=required_seeds)
    radius_rows, evaluation_hash = evaluation_radius_rows(
        radius_corpus_path,
        split=evaluation_split,
        max_cells=evaluation_max_cells,
        observation_seed=observation_seed,
        context_sizes=radius_context_sizes,
    )
    for row in run_rows:
        if (
            row["phase"] != "feasibility_2048"
            and row["evaluation_index_sha256"] != evaluation_hash
        ):
            raise ValueError("run and physical-radius evaluation targets differ")
    computational = computational_summary_rows(run_rows, phase="primary_85")
    serializable_runs = [
        {key: value for key, value in row.items() if key != "matched_config"}
        for row in run_rows
    ]
    _write_csv(output / "run_metrics.csv", serializable_runs)
    _write_csv(output / "paired_deltas.csv", deltas)
    _write_csv(output / "delta_summary.csv", summaries)
    _write_csv(output / "physical_radius_summary.csv", radius_rows)
    _write_csv(output / "computational_scaling.csv", computational)
    _plot_delta_vs_k(
        output / "positive_85_delta_vs_k.png",
        summaries,
        world="positive",
        regime="high_random_085",
        groups=GENE_CLASSES,
        title="Spatial advantage versus context size",
    )
    _plot_delta_vs_radius(
        output / "positive_85_delta_vs_radius.png", summaries, radius_rows
    )
    _plot_raw_losses(output / "positive_85_raw_loss_vs_k.png", run_rows)
    if any(row["phase"] == "null_85" for row in run_rows):
        _plot_delta_vs_k(
            output / "null_85_delta_vs_k.png",
            summaries,
            world="null",
            regime="high_random_085",
            groups=METRIC_GROUPS,
            title="Null-world spatial advantage",
        )
    if any(row["phase"] == "secondary_30" for row in run_rows):
        _plot_delta_vs_k(
            output / "positive_30_delta_vs_k.png",
            summaries,
            world="positive",
            regime="random_030",
            groups=GENE_CLASSES,
            title="30% masking spatial advantage",
        )
    _plot_computational(output / "computational_scaling.png", computational)
    result = {
        "runs": serializable_runs,
        "paired_deltas": deltas,
        "delta_summaries": summaries,
        "physical_radius_summaries": radius_rows,
        "computational_summaries": computational,
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result
