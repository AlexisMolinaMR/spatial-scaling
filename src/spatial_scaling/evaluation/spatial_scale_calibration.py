"""Aggregation for low-K and distance-exclusion spatial calibration."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from spatial_scaling.data.synthetic.dataset import GENE_CLASSES
from spatial_scaling.evaluation.context_scaling import run_metric_rows
from spatial_scaling.experiments.context_scaling import (
    ContextScalingEntry,
    context_scaling_entries,
)

METRIC_GROUPS = ("all", *GENE_CLASSES)
DISTANCE_ZERO_ABSOLUTE_TOLERANCES = {
    "all": 0.04,
    "intrinsic": 0.02,
    "mixed": 0.04,
    "spatial": 0.06,
}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def calibration_paired_delta_rows(
    run_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pair arms at seed level with phase and physical geometry in the key."""
    pairs: dict[tuple[str, str, str, int, float, int], dict[str, dict[str, Any]]] = (
        defaultdict(dict)
    )
    for row in run_rows:
        key = (
            str(row["phase"]),
            str(row["world"]),
            str(row["masking_regime"]),
            int(row["context_size"]),
            float(row["minimum_distance_um"]),
            int(row["seed"]),
        )
        condition = str(row["condition"])
        if condition in pairs[key]:
            raise ValueError(
                f"duplicate condition in matched pair: {(*key, condition)}"
            )
        pairs[key][condition] = row

    result = []
    matched_fields = (
        "parameter_count",
        "optimization_steps",
        "training_batch_size",
        "effective_training_examples",
        "evaluation_index_sha256",
        "evaluation_mask_sha256",
        "validation_observation_count",
        "requested_validation_observation_count",
        "train_common_threshold_eligible",
        "evaluation_common_threshold_eligible",
    )
    for key, conditions in sorted(pairs.items()):
        if set(conditions) != {"shuffled", "spatial"}:
            raise ValueError(f"incomplete matched pair: {key}")
        shuffled = conditions["shuffled"]
        spatial = conditions["spatial"]
        for field in matched_fields:
            if shuffled.get(field) != spatial.get(field):
                raise ValueError(f"matched pair differs in {field}: {key}")
        if shuffled["matched_config"] != spatial["matched_config"]:
            raise ValueError(f"uncontrolled matched-pair config difference: {key}")
        phase, world, regime, context_size, minimum_distance_um, seed = key
        row = {
            "phase": phase,
            "world": world,
            "masking_regime": regime,
            "context_size": context_size,
            "minimum_distance_um": minimum_distance_um,
            "seed": seed,
            "parameter_count": spatial["parameter_count"],
            "validation_observation_count": spatial["validation_observation_count"],
            "train_common_threshold_eligible": spatial.get(
                "train_common_threshold_eligible"
            ),
            "evaluation_common_threshold_eligible": spatial.get(
                "evaluation_common_threshold_eligible"
            ),
            "reused": bool(spatial.get("reused", False)),
            **{
                f"delta_space_{group}": shuffled[f"validation_masked_mse_{group}"]
                - spatial[f"validation_masked_mse_{group}"]
                for group in METRIC_GROUPS
            },
        }
        for statistic in ("nearest", "median", "farthest"):
            for quantile in ("median_um", "p25_um", "p75_um", "p05_um", "p95_um"):
                field = f"actual_{statistic}_selected_distance_{quantile}"
                if field in spatial:
                    if shuffled.get(field) != spatial[field]:
                        raise ValueError(
                            f"matched pair differs in realized geometry {field}: {key}"
                        )
                    row[field] = spatial[field]
        result.append(row)
    return result


def calibration_delta_summary_rows(
    paired_rows: list[dict[str, Any]], *, required_seeds: int = 3
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int, float], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in paired_rows:
        key = (
            str(row["phase"]),
            str(row["world"]),
            str(row["masking_regime"]),
            int(row["context_size"]),
            float(row["minimum_distance_um"]),
        )
        grouped[key].append(row)
    result = []
    for key, rows in sorted(grouped.items()):
        if len(rows) != required_seeds:
            raise ValueError(f"expected {required_seeds} matched seeds for {key}")
        phase, world, regime, context_size, minimum_distance_um = key
        summary: dict[str, Any] = {
            "phase": phase,
            "world": world,
            "masking_regime": regime,
            "context_size": context_size,
            "minimum_distance_um": minimum_distance_um,
            "num_seeds": len(rows),
        }
        for group in METRIC_GROUPS:
            values = [float(row[f"delta_space_{group}"]) for row in rows]
            summary[f"delta_space_{group}_mean"] = statistics.mean(values)
            summary[f"delta_space_{group}_sample_std"] = statistics.stdev(values)
            summary[f"delta_space_{group}_positive_seeds"] = sum(
                value > 0 for value in values
            )
        realized = [
            float(row["actual_median_selected_distance_median_um"])
            for row in rows
            if row.get("actual_median_selected_distance_median_um") is not None
        ]
        if realized:
            summary["actual_median_selected_distance_median_um_mean"] = statistics.mean(
                realized
            )
        result.append(summary)
    return result


def check_distance_zero_anchor(
    new_entries: list[ContextScalingEntry],
    reference_entries: list[ContextScalingEntry],
) -> dict[str, Any]:
    """Gate distance scaling on reproduction of completed nearest-K=32 runs."""
    new_rows = run_metric_rows(new_entries)
    reference_rows = run_metric_rows(reference_entries)
    new = {(row["condition"], row["seed"]): row for row in new_rows}
    reference = {(row["condition"], row["seed"]): row for row in reference_rows}
    expected = {
        (condition, seed)
        for condition in ("shuffled", "spatial")
        for seed in (101, 202, 303)
    }
    if set(new) != expected or set(reference) != expected:
        raise ValueError(
            "distance-zero anchor requires both conditions and three seeds"
        )
    failures = []
    comparisons = []
    for key in sorted(expected):
        observed = new[key]
        prior = reference[key]
        row: dict[str, Any] = {"condition": key[0], "seed": key[1]}
        for field in (
            "parameter_count",
            "optimization_steps",
            "training_batch_size",
            "evaluation_index_sha256",
            "evaluation_mask_sha256",
            "validation_observation_count",
        ):
            if observed[field] != prior[field]:
                failures.append(f"{key}: anchor differs in {field}")
        for common_field, total_field in (
            ("train_common_threshold_eligible", "train_total_observations"),
            (
                "evaluation_common_threshold_eligible",
                "evaluation_total_observations",
            ),
        ):
            if observed.get(common_field) != observed.get(total_field):
                failures.append(
                    f"{key}: common eligibility excludes observations in {common_field}"
                )
        for group, tolerance in DISTANCE_ZERO_ABSOLUTE_TOLERANCES.items():
            field = f"validation_masked_mse_{group}"
            difference = float(observed[field]) - float(prior[field])
            row[f"observed_{field}"] = observed[field]
            row[f"reference_{field}"] = prior[field]
            row[f"difference_{group}"] = difference
            row[f"absolute_tolerance_{group}"] = tolerance
            if abs(difference) > tolerance:
                failures.append(f"{key} {group}: |{difference:.6g}| > {tolerance}")
        comparisons.append(row)
    new_paired = calibration_paired_delta_rows(new_rows)
    reference_for_pairing = []
    for row in reference_rows:
        copied = dict(row)
        copied["phase"] = "distance_positive_85"
        reference_for_pairing.append(copied)
    reference_paired = calibration_paired_delta_rows(reference_for_pairing)
    observed_by_seed = {row["seed"]: row for row in new_paired}
    prior_by_seed = {row["seed"]: row for row in reference_paired}
    delta_comparisons = []
    for seed in (101, 202, 303):
        row = {"seed": seed}
        for group, tolerance in DISTANCE_ZERO_ABSOLUTE_TOLERANCES.items():
            difference = (
                observed_by_seed[seed][f"delta_space_{group}"]
                - prior_by_seed[seed][f"delta_space_{group}"]
            )
            row[f"observed_delta_space_{group}"] = observed_by_seed[seed][
                f"delta_space_{group}"
            ]
            row[f"reference_delta_space_{group}"] = prior_by_seed[seed][
                f"delta_space_{group}"
            ]
            row[f"difference_{group}"] = difference
            if abs(difference) > tolerance:
                failures.append(
                    f"seed {seed} delta {group}: |{difference:.6g}| > {tolerance}"
                )
        delta_comparisons.append(row)
    return {
        "passed": not failures,
        "failures": failures,
        "run_comparisons": comparisons,
        "paired_delta_comparisons": delta_comparisons,
    }


def _plot_delta(
    path: Path,
    paired: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    *,
    phase: str,
    x_key: str,
    xlabel: str,
    title: str,
    log2: bool = False,
) -> None:
    selected_pairs = [row for row in paired if row["phase"] == phase]
    selected_summaries = sorted(
        (row for row in summaries if row["phase"] == phase),
        key=lambda row: float(row[x_key]),
    )
    figure, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    colors = {"intrinsic": "tab:orange", "mixed": "tab:green", "spatial": "tab:red"}
    for group, color in colors.items():
        axis.scatter(
            [row[x_key] for row in selected_pairs],
            [row[f"delta_space_{group}"] for row in selected_pairs],
            color=color,
            alpha=0.35,
            s=20,
        )
        axis.errorbar(
            [row[x_key] for row in selected_summaries],
            [row[f"delta_space_{group}_mean"] for row in selected_summaries],
            yerr=[row[f"delta_space_{group}_sample_std"] for row in selected_summaries],
            color=color,
            marker="o",
            capsize=3,
            label=group,
        )
    axis.axhline(0, color="0.6", linewidth=1)
    if log2:
        axis.set_xscale("log", base=2)
        ticks = [row[x_key] for row in selected_summaries]
        axis.set_xticks(ticks)
        axis.set_xticklabels([str(int(value)) for value in ticks])
    axis.set(
        xlabel=xlabel,
        ylabel="Paired spatial advantage Δspace (masked MSE)",
        title=title,
    )
    axis.legend(title="Gene class")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_realized_distance(
    path: Path,
    paired: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
) -> None:
    x_key = "actual_median_selected_distance_median_um"
    selected_pairs = [row for row in paired if row["phase"] == "distance_positive_85"]
    selected_summaries = sorted(
        (row for row in summaries if row["phase"] == "distance_positive_85"),
        key=lambda row: row["actual_median_selected_distance_median_um_mean"],
    )
    figure, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    colors = {"intrinsic": "tab:orange", "mixed": "tab:green", "spatial": "tab:red"}
    for group, color in colors.items():
        axis.scatter(
            [row[x_key] for row in selected_pairs],
            [row[f"delta_space_{group}"] for row in selected_pairs],
            color=color,
            alpha=0.35,
            s=20,
        )
        axis.errorbar(
            [
                row["actual_median_selected_distance_median_um_mean"]
                for row in selected_summaries
            ],
            [row[f"delta_space_{group}_mean"] for row in selected_summaries],
            yerr=[row[f"delta_space_{group}_sample_std"] for row in selected_summaries],
            color=color,
            marker="o",
            capsize=3,
            label=group,
        )
    axis.axhline(0, color="0.6", linewidth=1)
    axis.axvline(200, color="tab:blue", linestyle="--", label="RBF scale (200 µm)")
    axis.set(
        xlabel="Median selected-cell context distance (µm)",
        ylabel="Paired spatial advantage Δspace (masked MSE)",
        title="Spatial advantage versus realized context distance",
    )
    axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_raw_distance_losses(path: Path, rows: list[dict[str, Any]]) -> None:
    selected = [row for row in rows if row["phase"] == "distance_positive_85"]
    grouped: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row in selected:
        grouped[(row["condition"], float(row["minimum_distance_um"]))].append(
            float(row["validation_masked_mse_spatial"])
        )
    figure, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    colors = {"spatial": "tab:blue", "shuffled": "tab:purple"}
    for condition in ("spatial", "shuffled"):
        distances = sorted(distance for arm, distance in grouped if arm == condition)
        means = [statistics.mean(grouped[(condition, value)]) for value in distances]
        standard_deviations = [
            statistics.stdev(grouped[(condition, value)]) for value in distances
        ]
        axis.errorbar(
            distances,
            means,
            yerr=standard_deviations,
            color=colors[condition],
            marker="o",
            capsize=3,
            label=condition,
        )
        axis.scatter(
            [
                row["minimum_distance_um"]
                for row in selected
                if row["condition"] == condition
            ],
            [
                row["validation_masked_mse_spatial"]
                for row in selected
                if row["condition"] == condition
            ],
            color=colors[condition],
            alpha=0.3,
            s=18,
        )
    axis.set(
        xlabel="Minimum eligible context distance d_min (µm)",
        ylabel="Spatial-gene validation masked MSE",
        title="Raw spatial-gene loss versus distance exclusion",
    )
    axis.legend(title="Condition")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def summarize_spatial_scale_calibration(
    entries: list[ContextScalingEntry],
    output_dir: str | Path,
    *,
    reused_entries: list[ContextScalingEntry],
) -> dict[str, Any]:
    """Aggregate new runs and explicitly recorded compatible reused runs."""
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"summary output already exists: {destination}")
    destination.mkdir(parents=True)
    new_rows = run_metric_rows(entries)
    reused_rows = run_metric_rows(reused_entries)
    for row in new_rows:
        row["reused"] = False
        row["reuse_source_experiment_id"] = None
    for row in reused_rows:
        row["phase"] = "low_k_positive_85"
        row["reused"] = True
        row["reuse_source_experiment_id"] = row["experiment_id"]
    run_rows = new_rows + reused_rows
    paired = calibration_paired_delta_rows(run_rows)
    summaries = calibration_delta_summary_rows(paired)

    run_csv_rows = [
        {key: value for key, value in row.items() if key != "matched_config"}
        for row in run_rows
    ]
    _write_csv(destination / "run_metrics.csv", run_csv_rows)
    _write_csv(destination / "paired_deltas.csv", paired)
    _write_csv(destination / "delta_summary.csv", summaries)
    distance_rows = [
        row
        for row in paired
        if row["phase"] in {"distance_positive_85", "distance_null_85"}
    ]
    _write_csv(destination / "context_distance_distributions.csv", distance_rows)
    eligibility_rows = [
        {
            key: row.get(key)
            for key in (
                "phase",
                "world",
                "minimum_distance_um",
                "seed",
                "validation_observation_count",
                "train_common_threshold_eligible",
                "evaluation_common_threshold_eligible",
            )
        }
        for row in paired
        if row["phase"].startswith("distance_")
    ]
    _write_csv(destination / "eligibility_counts.csv", eligibility_rows)
    reuse_provenance = [
        {
            "source_experiment_id": row["experiment_id"],
            "context_size": row["context_size"],
            "condition": row["condition"],
            "seed": row["seed"],
            "git_sha": row["git_sha"],
            "config_identity_sha256": row["config_identity_sha256"],
            "run_dir": row["run_dir"],
        }
        for row in reused_rows
    ]
    (destination / "reuse_provenance.json").write_text(
        json.dumps(reuse_provenance, indent=2) + "\n", encoding="utf-8"
    )

    _plot_delta(
        destination / "low_k_delta_vs_k.png",
        paired,
        summaries,
        phase="low_k_positive_85",
        x_key="context_size",
        xlabel="Context size K (log2 scale)",
        title="Low-K spatial advantage",
        log2=True,
    )
    _plot_delta(
        destination / "distance_positive_delta_vs_d_min.png",
        paired,
        summaries,
        phase="distance_positive_85",
        x_key="minimum_distance_um",
        xlabel="Minimum eligible context distance d_min (µm)",
        title="Distance-exclusion spatial advantage",
    )
    _plot_realized_distance(
        destination / "distance_positive_delta_vs_realized_distance.png",
        paired,
        summaries,
    )
    _plot_raw_distance_losses(
        destination / "distance_positive_raw_spatial_loss.png", run_rows
    )
    _plot_delta(
        destination / "distance_null_delta_vs_d_min.png",
        paired,
        summaries,
        phase="distance_null_85",
        x_key="minimum_distance_um",
        xlabel="Minimum eligible context distance d_min (µm)",
        title="Null distance-exclusion control",
    )
    result = {
        "new_runs": len(new_rows),
        "reused_runs": len(reused_rows),
        "matched_pairs": len(paired),
        "output_dir": str(destination),
    }
    (destination / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def reused_low_k_entries() -> list[ContextScalingEntry]:
    """Resolve the validated K=32/64 positive 85% source runs."""
    return context_scaling_entries(
        "configs/pilot/context_scaling_v0.yaml",
        phases={"primary_85"},
        include_context_sizes={32, 64},
    )
