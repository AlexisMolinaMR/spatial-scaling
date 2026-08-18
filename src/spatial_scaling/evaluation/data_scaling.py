"""Paired aggregation and plots for Pilot v0 training-data scaling."""

from __future__ import annotations

import copy
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from spatial_scaling.data.synthetic.dataset import GENE_CLASSES
from spatial_scaling.experiments.data_scaling import DataScalingEntry
from spatial_scaling.training.policies import TrainingDataSelection

METRIC_GROUPS = ("all", *GENE_CLASSES)


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
    fields: list[str] = []
    for row in rows:
        fields.extend(field for field in row if field not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _matched_config(config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    for key in ("condition", "experiment_name", "output_dir", "config_identity_sha256"):
        result.pop(key, None)
    return result


def run_metric_rows(entries: list[DataScalingEntry]) -> list[dict[str, Any]]:
    """Flatten completed runs while validating their manifest identity."""
    rows = []
    for entry in entries:
        run_dir = Path(entry.output_dir)
        metrics = _read_json(run_dir / "metrics.json")
        config = _read_json(run_dir / "resolved_config.json")
        provenance = _read_json(run_dir / "provenance.json")
        if metrics["experiment_id"] != entry.experiment_id:
            raise ValueError(f"experiment identity mismatch: {run_dir}")
        if metrics["config_identity_sha256"] != entry.config_identity_sha256:
            raise ValueError(f"configuration identity mismatch: {run_dir}")
        selection = TrainingDataSelection.from_dict(
            metrics["training_data_selection"]
        ).to_dict()
        expected_selection = TrainingDataSelection.from_dict(
            entry.training_data_selection
        ).to_dict()
        if selection != expected_selection:
            raise ValueError(f"training subset mismatch: {run_dir}")
        if len(selection["active_train_section_ids"]) != entry.n_train_sections:
            raise ValueError(f"training section count mismatch: {run_dir}")
        if metrics["available_training_observations"] != entry.n_train_cells:
            raise ValueError(f"training cell count mismatch: {run_dir}")
        observations = metrics["evaluation_observations"]
        final = metrics["final_validation"]
        rows.append(
            {
                "phase": entry.phase,
                "scaling_axis": entry.scaling_axis,
                "world": entry.world,
                "model_family": entry.model_family,
                "training_protocol": entry.training_protocol,
                "n_train_sections": entry.n_train_sections,
                "n_train_cells": entry.n_train_cells,
                "condition": entry.condition,
                "seed": entry.seed,
                "experiment_id": entry.experiment_id,
                "parameter_count": metrics["parameter_count"],
                **{
                    f"validation_masked_mse_{group}": final[f"masked_mse_{group}"]
                    for group in METRIC_GROUPS
                },
                "optimization_steps": metrics["optimization_steps"],
                "examples_processed": metrics["examples_processed"],
                "unique_training_observations_seen": metrics[
                    "unique_training_observations_seen"
                ],
                "unique_observation_fraction": metrics["unique_observation_fraction"],
                "effective_passes": metrics["effective_passes"],
                "epochs_completed": metrics["epochs_completed"],
                "early_stopping_reason": metrics["early_stopping_reason"],
                "run_elapsed_seconds": metrics["run_elapsed_seconds"],
                "optimization_elapsed_seconds": metrics["optimization_elapsed_seconds"],
                "examples_per_second": metrics["examples_per_second"],
                "peak_device_memory_bytes": metrics["peak_device_memory_bytes"],
                "evaluation_index_sha256": observations["index_sha256"],
                "evaluation_mask_sha256": observations["mask_sha256"],
                "validation_observation_count": observations["count"],
                "subset_selection_seed": selection["subset_selection_seed"],
                "active_train_section_ids": "|".join(
                    selection["active_train_section_ids"]
                ),
                "ordered_train_section_ids": "|".join(
                    selection["ordered_train_section_ids"]
                ),
                "git_sha": provenance["git"]["commit_sha"],
                "git_dirty": provenance["git"]["dirty"],
                "slurm_job_id": provenance["environment"].get("slurm_job_id"),
                "run_dir": str(run_dir),
                "matched_config": _matched_config(config),
                "execution_policy": metrics["execution_policy"],
                "training_data_selection": selection,
            }
        )
    return rows


def paired_delta_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute shuffled-minus-spatial losses within seed before averaging."""
    pairs: dict[tuple[str, str, str, int, int], dict[str, dict[str, Any]]] = (
        defaultdict(dict)
    )
    for row in run_rows:
        if row["condition"] not in {"shuffled", "spatial"}:
            continue
        key = (
            str(row["phase"]),
            str(row["world"]),
            str(row["training_protocol"]),
            int(row["n_train_sections"]),
            int(row["seed"]),
        )
        if row["condition"] in pairs[key]:
            raise ValueError(f"duplicate matched arm: {(*key, row['condition'])}")
        pairs[key][str(row["condition"])] = row
    result = []
    for key, conditions in sorted(pairs.items()):
        if set(conditions) != {"shuffled", "spatial"}:
            raise ValueError(f"incomplete matched pair: {key}")
        shuffled = conditions["shuffled"]
        spatial = conditions["spatial"]
        for field in (
            "parameter_count",
            "n_train_cells",
            "evaluation_index_sha256",
            "evaluation_mask_sha256",
            "validation_observation_count",
            "subset_selection_seed",
            "active_train_section_ids",
            "ordered_train_section_ids",
            "execution_policy",
            "training_data_selection",
            "matched_config",
        ):
            if shuffled[field] != spatial[field]:
                raise ValueError(f"matched pair differs in {field}: {key}")
        phase, world, protocol, n_sections, seed = key
        result.append(
            {
                "phase": phase,
                "world": world,
                "training_protocol": protocol,
                "n_train_sections": n_sections,
                "n_train_cells": spatial["n_train_cells"],
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


def delta_summary_rows(paired: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in paired:
        grouped[
            (
                str(row["phase"]),
                str(row["world"]),
                str(row["training_protocol"]),
                int(row["n_train_sections"]),
            )
        ].append(row)
    summaries = []
    for key, rows in sorted(grouped.items()):
        if len(rows) != 3:
            raise ValueError(f"expected three matched seeds for {key}; got {len(rows)}")
        phase, world, protocol, n_sections = key
        summary: dict[str, Any] = {
            "phase": phase,
            "world": world,
            "training_protocol": protocol,
            "n_train_sections": n_sections,
            "n_train_cells": rows[0]["n_train_cells"],
            "num_seeds": len(rows),
        }
        for group in METRIC_GROUPS:
            values = [float(row[f"delta_space_{group}"]) for row in rows]
            summary[f"delta_space_{group}_mean"] = statistics.mean(values)
            summary[f"delta_space_{group}_sample_std"] = statistics.stdev(values)
            summary[f"delta_space_{group}_positive_seeds"] = sum(
                value > 0 for value in values
            )
        summaries.append(summary)
    return summaries


def arm_summary_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[
            (
                str(row["phase"]),
                str(row["training_protocol"]),
                str(row["condition"]),
                int(row["n_train_sections"]),
            )
        ].append(row)
    result = []
    for key, rows in sorted(grouped.items()):
        phase, protocol, condition, n_sections = key
        summary: dict[str, Any] = {
            "phase": phase,
            "training_protocol": protocol,
            "condition": condition,
            "n_train_sections": n_sections,
            "n_train_cells": rows[0]["n_train_cells"],
            "num_seeds": len(rows),
        }
        for group in METRIC_GROUPS:
            values = [float(row[f"validation_masked_mse_{group}"]) for row in rows]
            summary[f"validation_masked_mse_{group}_mean"] = statistics.mean(values)
            summary[f"validation_masked_mse_{group}_sample_std"] = (
                statistics.stdev(values) if len(values) > 1 else None
            )
        for field in (
            "optimization_steps",
            "examples_processed",
            "run_elapsed_seconds",
            "effective_passes",
        ):
            values = [float(row[field]) for row in rows]
            summary[f"{field}_mean"] = statistics.mean(values)
            summary[f"{field}_sample_std"] = (
                statistics.stdev(values) if len(values) > 1 else None
            )
        result.append(summary)
    return result


def _set_observed_log2_axis(axis: Any, points: list[int]) -> None:
    axis.set_xscale("log", base=2)
    axis.set_xticks(points)
    axis.set_xticklabels([str(point) for point in points])


def _plot_losses(
    path: Path, summaries: list[dict[str, Any]], *, x_field: str, xlabel: str
) -> None:
    selected = [row for row in summaries if row["phase"] == "primary_positive_85"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    colors = {"spatial": "tab:blue", "shuffled": "tab:purple"}
    for axis, group, title in zip(
        axes, ("all", "spatial"), ("Overall", "Spatial genes"), strict=True
    ):
        for condition in ("spatial", "shuffled"):
            rows = sorted(
                (row for row in selected if row["condition"] == condition),
                key=lambda row: row[x_field],
            )
            axis.errorbar(
                [row[x_field] for row in rows],
                [row[f"validation_masked_mse_{group}_mean"] for row in rows],
                yerr=[row[f"validation_masked_mse_{group}_sample_std"] for row in rows],
                marker="o",
                capsize=3,
                color=colors[condition],
                label=condition,
            )
        axis.set(xlabel=xlabel, ylabel="Validation masked MSE", title=title)
        if x_field == "n_train_sections":
            _set_observed_log2_axis(axis, [1, 2, 4, 8, 16, 32, 48])
        else:
            _set_observed_log2_axis(
                axis, [8192, 16384, 32768, 65536, 131072, 262144, 393216]
            )
        axis.legend(title="Context")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_deltas(
    path: Path,
    paired: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    *,
    phase: str,
    title: str,
) -> None:
    selected_pairs = [row for row in paired if row["phase"] == phase]
    selected = sorted(
        (row for row in summaries if row["phase"] == phase),
        key=lambda row: row["n_train_sections"],
    )
    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    colors = {"intrinsic": "tab:orange", "mixed": "tab:green", "spatial": "tab:red"}
    for group, color in colors.items():
        axis.scatter(
            [row["n_train_sections"] for row in selected_pairs],
            [row[f"delta_space_{group}"] for row in selected_pairs],
            color=color,
            alpha=0.3,
            s=18,
        )
        axis.errorbar(
            [row["n_train_sections"] for row in selected],
            [row[f"delta_space_{group}_mean"] for row in selected],
            yerr=[row[f"delta_space_{group}_sample_std"] for row in selected],
            color=color,
            marker="o",
            capsize=3,
            label=group,
        )
    axis.axhline(0, color="0.6", linewidth=1)
    _set_observed_log2_axis(
        axis, sorted({int(row["n_train_sections"]) for row in selected})
    )
    axis.set(
        xlabel="Independent training sections",
        ylabel="Paired spatial advantage Δspace (masked MSE)",
        title=title,
    )
    axis.legend(title="Gene class")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_compute(path: Path, summaries: list[dict[str, Any]]) -> None:
    selected = [
        row
        for row in summaries
        if row["phase"] == "primary_positive_85" and row["condition"] == "spatial"
    ]
    selected.sort(key=lambda row: row["n_train_sections"])
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    fields = (
        ("optimization_steps_mean", "Optimizer steps"),
        ("examples_processed_mean", "Examples processed"),
        ("run_elapsed_seconds_mean", "Wall-clock runtime (s)"),
    )
    for axis, (field, ylabel) in zip(axes, fields, strict=True):
        axis.plot(
            [row["n_train_sections"] for row in selected],
            [row[field] for row in selected],
            color="tab:blue",
            marker="o",
        )
        _set_observed_log2_axis(axis, [1, 2, 4, 8, 16, 32, 48])
        axis.set(xlabel="Independent training sections", ylabel=ylabel)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_training_histories(path: Path, run_rows: list[dict[str, Any]]) -> None:
    selected = [
        row
        for row in run_rows
        if row["phase"] == "primary_positive_85"
        and row["seed"] == 101
        and row["n_train_sections"] in {1, 8, 48}
    ]
    figure, axis = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    colors = {1: "tab:orange", 8: "tab:green", 48: "tab:blue"}
    for row in selected:
        with (Path(row["run_dir"]) / "history.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            history = list(csv.DictReader(handle))
        axis.plot(
            [float(item["epoch"]) for item in history],
            [float(item["validation_masked_mse_all"]) for item in history],
            color=colors[int(row["n_train_sections"])],
            linestyle="-" if row["condition"] == "spatial" else "--",
            marker="o",
            label=f"N={row['n_train_sections']} {row['condition']}",
        )
    axis.set(
        xlabel="Complete training-data passes",
        ylabel="Overall validation masked MSE",
        title="Representative convergence histories (seed 101)",
    )
    axis.legend(ncol=2)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_fixed_compute(path: Path, summaries: list[dict[str, Any]]) -> None:
    selected = [
        row
        for row in summaries
        if row["phase"] in {"primary_positive_85", "fixed_compute_positive_85"}
        and row["n_train_sections"] in {1, 8, 48}
    ]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    colors = {"spatial": "tab:blue", "shuffled": "tab:purple"}
    for axis, group in zip(axes, ("all", "spatial"), strict=True):
        for phase, linestyle, label_prefix in (
            ("primary_positive_85", "-", "converged"),
            ("fixed_compute_positive_85", "--", "2,000 steps"),
        ):
            for condition in ("spatial", "shuffled"):
                rows = sorted(
                    (
                        row
                        for row in selected
                        if row["phase"] == phase and row["condition"] == condition
                    ),
                    key=lambda row: row["n_train_sections"],
                )
                axis.plot(
                    [row["n_train_sections"] for row in rows],
                    [row[f"validation_masked_mse_{group}_mean"] for row in rows],
                    color=colors[condition],
                    linestyle=linestyle,
                    marker="o",
                    label=f"{label_prefix}: {condition}",
                )
        _set_observed_log2_axis(axis, [1, 8, 48])
        axis.set(
            xlabel="Independent training sections",
            ylabel="Validation masked MSE",
            title="Overall" if group == "all" else "Spatial genes",
        )
    axes[1].legend(fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def summarize_data_scaling(
    entries: list[DataScalingEntry], output_dir: str | Path
) -> dict[str, Any]:
    """Validate, aggregate, and plot completed non-smoke data-scaling runs."""
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"summary output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    run_rows = run_metric_rows(entries)
    contextual = [row for row in run_rows if row["model_family"] == "contextual"]
    paired = paired_delta_rows(contextual)
    delta_summaries = delta_summary_rows(paired)
    arm_summaries = arm_summary_rows(run_rows)
    serializable_runs = [
        {
            key: value
            for key, value in row.items()
            if key
            not in {"matched_config", "execution_policy", "training_data_selection"}
        }
        for row in run_rows
    ]
    _write_csv(output / "run_metrics.csv", serializable_runs)
    _write_csv(output / "arm_summary.csv", arm_summaries)
    _write_csv(output / "paired_deltas.csv", paired)
    _write_csv(output / "delta_summary.csv", delta_summaries)
    focal = [row for row in serializable_runs if row["condition"] == "focal"]
    if focal:
        _write_csv(output / "focal_reference.csv", focal)
    _plot_losses(
        output / "validation_loss_vs_training_sections.png",
        arm_summaries,
        x_field="n_train_sections",
        xlabel="Independent training sections",
    )
    _plot_losses(
        output / "validation_loss_vs_training_cells.png",
        arm_summaries,
        x_field="n_train_cells",
        xlabel="Available training focal cells (equal-size synthetic sections)",
    )
    _plot_deltas(
        output / "spatial_advantage_vs_training_sections.png",
        paired,
        delta_summaries,
        phase="primary_positive_85",
        title="Spatial advantage versus training-data scale",
    )
    _plot_deltas(
        output / "null_spatial_advantage_vs_training_sections.png",
        paired,
        delta_summaries,
        phase="required_null_85",
        title="Null spatial advantage versus training-data scale",
    )
    _plot_compute(output / "compute_vs_training_sections.png", arm_summaries)
    _plot_training_histories(output / "convergence_diagnostics.png", run_rows)
    _plot_fixed_compute(output / "fixed_compute_sensitivity.png", arm_summaries)
    result = {
        "runs": len(run_rows),
        "matched_pairs": len(paired),
        "delta_summaries": len(delta_summaries),
        "output_dir": str(output),
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result
