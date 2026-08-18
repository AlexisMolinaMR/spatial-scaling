"""Strict aggregation for the final converged Pilot v0 N-train frontier."""

from __future__ import annotations

import copy
import json
import statistics
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from spatial_scaling.evaluation.lr_policy_diagnostic import (
    METRIC_GROUPS,
    _load_run,
    _write_csv,
    convergence_status,
)
from spatial_scaling.experiments.data_scaling import DataScalingEntry

SCALE_POINTS = (1, 2, 4, 8, 16, 32, 48)
INTERMEDIATE_POINTS = (2, 4, 8, 16, 32)
ENDPOINT_POINTS = (1, 48)
SEEDS = (101, 202, 303)
CONDITIONS = ("spatial", "shuffled")
EXPECTED_INDEX_HASH = "530140cafbb259c2342228d1bbf444115324526ebd981cc490074d10c3594288"
EXPECTED_MASK_HASH = "cc16a3750dbb261031c06d9b7f7912c4d4464a341bd16311f06a2f3eba5f95e6"
TRAINING_CODE_PATHS = (
    "src/spatial_scaling/data/synthetic/context.py",
    "src/spatial_scaling/data/synthetic/dataset.py",
    "src/spatial_scaling/models/cell_encoder.py",
    "src/spatial_scaling/models/spatial_transformer.py",
    "src/spatial_scaling/spatial/neighborhoods.py",
    "src/spatial_scaling/spatial/shuffling.py",
    "src/spatial_scaling/training/config.py",
    "src/spatial_scaling/training/masking.py",
    "src/spatial_scaling/training/objective.py",
    "src/spatial_scaling/training/policies.py",
    "src/spatial_scaling/training/trainer.py",
)


def _key(entry_or_run: Any) -> tuple[int, str, int]:
    if isinstance(entry_or_run, DataScalingEntry):
        return (
            entry_or_run.n_train_sections,
            entry_or_run.condition,
            entry_or_run.seed,
        )
    return (
        int(entry_or_run["n_train_sections"]),
        str(entry_or_run["condition"]),
        int(entry_or_run["seed"]),
    )


def _expected_keys(points: tuple[int, ...]) -> set[tuple[int, str, int]]:
    return {
        (n_sections, condition, seed)
        for n_sections in points
        for condition in CONDITIONS
        for seed in SEEDS
    }


def validate_final_grids(
    logical_entries: list[DataScalingEntry], new_entries: list[DataScalingEntry]
) -> None:
    """Require 42 logical runs and exactly 30 non-endpoint training runs."""
    logical_keys = {_key(entry) for entry in logical_entries}
    new_keys = {_key(entry) for entry in new_entries}
    if len(logical_entries) != 42 or logical_keys != _expected_keys(SCALE_POINTS):
        raise ValueError("final logical grid must contain exactly 42 runs")
    if len(new_entries) != 30 or new_keys != _expected_keys(INTERMEDIATE_POINTS):
        raise ValueError("new-run grid must contain exactly 30 intermediate runs")
    if len({entry.experiment_id for entry in logical_entries}) != 42:
        raise ValueError("final logical grid contains duplicate experiment IDs")
    if len({entry.experiment_id for entry in new_entries}) != 30:
        raise ValueError("new-run grid contains duplicate experiment IDs")
    if any(entry.n_train_sections in ENDPOINT_POINTS for entry in new_entries):
        raise ValueError("new-run grid accidentally schedules an endpoint")
    if {entry.training_protocol for entry in logical_entries} != {
        "validation_plateau_lr_decay"
    }:
        raise ValueError("final grid changed the frozen policy identity")


def _scientific_config(config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    result.pop("experiment_name", None)
    result.pop("output_dir", None)
    return result


def _policy_without_safety_caps(policy: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(policy)
    result.pop("maximum_epochs", None)
    result.pop("absolute_max_steps", None)
    return result


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _load_sources(
    entries: list[DataScalingEntry], provenance_kind: str
) -> list[dict[str, Any]]:
    runs = []
    for entry in entries:
        run = _load_run(entry)
        run["source_entry"] = entry
        run["source_provenance"] = _read_json(
            Path(entry.output_dir) / "provenance.json"
        )
        run["reuse_status"] = provenance_kind
        runs.append(run)
    return runs


def _validate_source_against_logical(
    run: dict[str, Any], expected: DataScalingEntry
) -> dict[str, Any]:
    key = _key(run)
    metrics = run["metrics"]
    source_policy = run["execution_policy"]
    expected_policy = expected.execution_policy
    if run["training_data_selection"] != expected.training_data_selection:
        raise ValueError(f"training subset differs from final logical run: {key}")
    if _scientific_config(run["resolved_config"]) != _scientific_config(
        expected.resolved_config
    ):
        raise ValueError(f"scientific configuration differs at {key}")
    if run["reuse_status"] == "reused_shuffled_60pass":
        if source_policy.get("maximum_epochs") != 60:
            raise ValueError(
                f"reused shuffled source is not the validated policy: {key}"
            )
        if source_policy.get("absolute_max_steps") != 800000:
            raise ValueError(f"reused shuffled step ceiling changed: {key}")
        if _policy_without_safety_caps(source_policy) != _policy_without_safety_caps(
            expected_policy
        ):
            raise ValueError(f"reused shuffled scheduler semantics differ: {key}")
    elif source_policy != expected_policy:
        raise ValueError(f"frozen 90-pass policy differs at {key}")
    observations = metrics["evaluation_observations"]
    if observations["index_sha256"] != EXPECTED_INDEX_HASH:
        raise ValueError(f"evaluation observation hash changed at {key}")
    if observations["mask_sha256"] != EXPECTED_MASK_HASH:
        raise ValueError(f"evaluation mask hash changed at {key}")
    if observations["count"] != 8192:
        raise ValueError(f"evaluation observation count changed at {key}")
    if metrics["parameter_count"] != 121024:
        raise ValueError(f"model parameter count changed at {key}")
    if metrics["available_training_observations"] != expected.n_train_cells:
        raise ValueError(f"available training-cell count changed at {key}")
    if metrics["unique_observation_fraction"] != 1.0:
        raise ValueError(f"minimum exposure was not satisfied at {key}")
    status = convergence_status(metrics)
    if status != "EARLY_STOP_CONVERGED":
        raise ValueError(f"frontier run is not converged at {key}: {status}")
    final_pass = int(run["history"][-1]["epoch"])
    if final_pass >= int(source_policy["maximum_epochs"]):
        raise ValueError(f"frontier run stopped at its safety cap: {key}")
    source_files = run["source_provenance"]["source_state"]["files"]
    missing = [path for path in TRAINING_CODE_PATHS if path not in source_files]
    if missing:
        raise ValueError(f"source provenance lacks training files at {key}: {missing}")
    return {
        "n_train_sections": key[0],
        "condition": key[1],
        "seed": key[2],
        "logical_experiment_id": expected.experiment_id,
        "source_experiment_id": run["experiment_id"],
        "source_run_dir": run["run_dir"],
        "reuse_status": run["reuse_status"],
        "source_maximum_epochs": source_policy["maximum_epochs"],
        "logical_maximum_epochs": expected_policy["maximum_epochs"],
        "source_absolute_max_steps": source_policy["absolute_max_steps"],
        "logical_absolute_max_steps": expected_policy["absolute_max_steps"],
        "safety_cap_equivalence_rule": (
            "same_exact_policy"
            if source_policy == expected_policy
            else "same_scheduler_and_early_stop_semantics; source converged before cap"
        ),
        "convergence_status": status,
        "stop_pass": final_pass,
        "coverage_fraction": metrics["unique_observation_fraction"],
        "evaluation_index_sha256": observations["index_sha256"],
        "evaluation_mask_sha256": observations["mask_sha256"],
        "source_git_sha": run["source_provenance"]["git"]["commit_sha"],
        "source_tree_sha256": run["source_provenance"]["source_state"]["tree_sha256"],
    }


def _validate_training_code(runs: list[dict[str, Any]]) -> dict[str, str]:
    reference = runs[0]["source_provenance"]["source_state"]["files"]
    result = {path: reference[path] for path in TRAINING_CODE_PATHS}
    for run in runs[1:]:
        files = run["source_provenance"]["source_state"]["files"]
        for path, digest in result.items():
            if files.get(path) != digest:
                raise ValueError(
                    f"training-code hash changed for {path} at {_key(run)}"
                )
    return result


def validate_endpoint_reuse(
    logical_entries: list[DataScalingEntry],
    spatial_endpoint_entries: list[DataScalingEntry],
    shuffled_endpoint_entries: list[DataScalingEntry],
) -> dict[str, Any]:
    """Preflight the twelve historical endpoints before new-job submission."""
    expected_by_key = {_key(entry): entry for entry in logical_entries}
    expected_endpoint_keys = _expected_keys(ENDPOINT_POINTS)
    if not expected_endpoint_keys.issubset(expected_by_key):
        raise ValueError("logical grid lacks endpoint expectations")
    if {_key(entry) for entry in spatial_endpoint_entries} != {
        (n, "spatial", seed) for n in ENDPOINT_POINTS for seed in SEEDS
    }:
        raise ValueError("spatial endpoint reuse grid is incomplete")
    if {_key(entry) for entry in shuffled_endpoint_entries} != {
        (n, "shuffled", seed) for n in ENDPOINT_POINTS for seed in SEEDS
    }:
        raise ValueError("shuffled endpoint reuse grid is incomplete")
    runs = [
        *_load_sources(spatial_endpoint_entries, "reused_spatial_90pass"),
        *_load_sources(shuffled_endpoint_entries, "reused_shuffled_60pass"),
    ]
    rows = [
        _validate_source_against_logical(run, expected_by_key[_key(run)])
        for run in sorted(runs, key=_key)
    ]
    hashes = _validate_training_code(runs)
    return {
        "reused_runs": len(rows),
        "all_converged": all(
            row["convergence_status"] == "EARLY_STOP_CONVERGED" for row in rows
        ),
        "evaluation_index_sha256": EXPECTED_INDEX_HASH,
        "evaluation_mask_sha256": EXPECTED_MASK_HASH,
        "training_code_hashes": hashes,
        "runs": rows,
    }


def _run_summary(run: dict[str, Any], expected: DataScalingEntry) -> dict[str, Any]:
    metrics = run["metrics"]
    observations = metrics["evaluation_observations"]
    final = metrics["final_validation"]
    final_pass = int(run["history"][-1]["epoch"])
    provenance = run["source_provenance"]
    row: dict[str, Any] = {
        "n_train_sections": expected.n_train_sections,
        "n_train_cells": expected.n_train_cells,
        "condition": expected.condition,
        "seed": expected.seed,
        "logical_experiment_id": expected.experiment_id,
        "source_experiment_id": run["experiment_id"],
        "reuse_status": run["reuse_status"],
        "run_dir": run["run_dir"],
        "parameter_count": metrics["parameter_count"],
        "best_pass": metrics["best_validation_epoch"],
        "stop_pass": final_pass,
        "best_step": metrics["best_validation_step"],
        "stop_step": metrics["optimization_steps"],
        "convergence_status": convergence_status(metrics),
        "stop_reason": metrics["early_stopping_reason"],
        "num_lr_reductions": metrics["num_lr_reductions"],
        "lr_reduction_passes": "|".join(
            str(event["epoch"]) for event in metrics["lr_reduction_events"]
        ),
        "lr_reduction_events_json": json.dumps(
            metrics["lr_reduction_events"], separators=(",", ":")
        ),
        "final_learning_rate": metrics["final_learning_rate"],
        "optimization_steps": metrics["optimization_steps"],
        "examples_processed": metrics["examples_processed"],
        "available_training_observations": metrics["available_training_observations"],
        "unique_training_observations_seen": metrics[
            "unique_training_observations_seen"
        ],
        "coverage_fraction": metrics["unique_observation_fraction"],
        "effective_passes": metrics["effective_passes"],
        "run_elapsed_seconds": metrics["run_elapsed_seconds"],
        "optimization_elapsed_seconds": metrics["optimization_elapsed_seconds"],
        "examples_per_second": metrics["examples_per_second"],
        "peak_device_memory_bytes": metrics["peak_device_memory_bytes"],
        "active_train_section_ids": "|".join(
            run["training_data_selection"]["active_train_section_ids"]
        ),
        "evaluation_index_sha256": observations["index_sha256"],
        "evaluation_mask_sha256": observations["mask_sha256"],
        "source_config_identity_sha256": metrics["config_identity_sha256"],
        "git_sha": provenance["git"]["commit_sha"],
        "source_tree_sha256": provenance["source_state"]["tree_sha256"],
    }
    for group in METRIC_GROUPS:
        row[f"validation_masked_mse_{group}"] = final[f"masked_mse_{group}"]
        row[f"best_recorded_validation_masked_mse_{group}"] = metrics[
            "best_validation"
        ][f"masked_mse_{group}"]
    return row


def paired_delta_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute shuffled-minus-spatial within every N and seed."""
    grouped: dict[tuple[int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in run_rows:
        key = (int(row["n_train_sections"]), int(row["seed"]))
        if row["condition"] in grouped[key]:
            raise ValueError(f"duplicate condition in paired frontier run: {key}")
        grouped[key][str(row["condition"])] = row
    result = []
    for key, arms in sorted(grouped.items()):
        if set(arms) != set(CONDITIONS):
            raise ValueError(f"incomplete paired frontier run: {key}")
        spatial, shuffled = arms["spatial"], arms["shuffled"]
        result.append(
            {
                "n_train_sections": key[0],
                "n_train_cells": spatial["n_train_cells"],
                "seed": key[1],
                **{
                    f"delta_space_{group}": shuffled[f"validation_masked_mse_{group}"]
                    - spatial[f"validation_masked_mse_{group}"]
                    for group in METRIC_GROUPS
                },
            }
        )
    return result


def _arm_summary_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[(int(row["n_train_sections"]), str(row["condition"]))].append(row)
    result = []
    for key, rows in sorted(grouped.items()):
        if len(rows) != 3:
            raise ValueError(f"frontier arm lacks three seeds: {key}")
        summary: dict[str, Any] = {
            "n_train_sections": key[0],
            "n_train_cells": rows[0]["n_train_cells"],
            "condition": key[1],
            "num_seeds": 3,
        }
        for group in METRIC_GROUPS:
            values = [float(row[f"validation_masked_mse_{group}"]) for row in rows]
            summary[f"validation_masked_mse_{group}_mean"] = statistics.mean(values)
            summary[f"validation_masked_mse_{group}_sample_std"] = statistics.stdev(
                values
            )
        for field in (
            "stop_pass",
            "optimization_steps",
            "examples_processed",
            "run_elapsed_seconds",
            "num_lr_reductions",
        ):
            values = [float(row[field]) for row in rows]
            summary[f"{field}_mean"] = statistics.mean(values)
            summary[f"{field}_sample_std"] = statistics.stdev(values)
        result.append(summary)
    return result


def _delta_summary_rows(paired: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in paired:
        grouped[int(row["n_train_sections"])].append(row)
    result = []
    for n_sections, rows in sorted(grouped.items()):
        if len(rows) != 3:
            raise ValueError(f"delta summary lacks three seeds at N={n_sections}")
        summary: dict[str, Any] = {
            "n_train_sections": n_sections,
            "n_train_cells": rows[0]["n_train_cells"],
            "num_seeds": 3,
        }
        for group in METRIC_GROUPS:
            values = [float(row[f"delta_space_{group}"]) for row in rows]
            summary[f"delta_space_{group}_mean"] = statistics.mean(values)
            summary[f"delta_space_{group}_sample_std"] = statistics.stdev(values)
            summary[f"delta_space_{group}_positive_seeds"] = sum(
                value > 0 for value in values
            )
        result.append(summary)
    return result


def _combined_summary_rows(
    arms: list[dict[str, Any]], deltas: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    arm_map = {
        (int(row["n_train_sections"]), str(row["condition"])): row for row in arms
    }
    delta_map = {int(row["n_train_sections"]): row for row in deltas}
    result = []
    for n_sections in SCALE_POINTS:
        spatial = arm_map[(n_sections, "spatial")]
        shuffled = arm_map[(n_sections, "shuffled")]
        delta = delta_map[n_sections]
        row: dict[str, Any] = {
            "n_train_sections": n_sections,
            "n_train_cells": spatial["n_train_cells"],
        }
        for group in METRIC_GROUPS:
            for condition, source in (("spatial", spatial), ("shuffled", shuffled)):
                row[f"{condition}_{group}_mean"] = source[
                    f"validation_masked_mse_{group}_mean"
                ]
                row[f"{condition}_{group}_sample_std"] = source[
                    f"validation_masked_mse_{group}_sample_std"
                ]
            for suffix in ("mean", "sample_std", "positive_seeds"):
                row[f"delta_space_{group}_{suffix}"] = delta[
                    f"delta_space_{group}_{suffix}"
                ]
        result.append(row)
    return result


def adjacent_improvement_rows(
    arm_summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Observed adjacent absolute and percentage improvements, without a fit."""
    by_key = {
        (int(row["n_train_sections"]), str(row["condition"])): row
        for row in arm_summaries
    }
    rows = []
    for start, end in pairwise(SCALE_POINTS):
        row: dict[str, Any] = {
            "from_n_sections": start,
            "to_n_sections": end,
            "from_n_cells": start * 8192,
            "to_n_cells": end * 8192,
        }
        for condition, group, label in (
            ("spatial", "all", "spatial_overall"),
            ("shuffled", "all", "shuffled_overall"),
            ("spatial", "spatial", "spatial_gene_spatial_arm"),
            ("shuffled", "spatial", "spatial_gene_shuffled_arm"),
        ):
            initial = by_key[(start, condition)][f"validation_masked_mse_{group}_mean"]
            final = by_key[(end, condition)][f"validation_masked_mse_{group}_mean"]
            improvement = initial - final
            row[f"{label}_absolute_improvement"] = improvement
            row[f"{label}_percentage_improvement"] = 100.0 * improvement / initial
        rows.append(row)
    return rows


def _capped_comparison_rows(
    final_rows: list[dict[str, Any]], capped_runs: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_points = {1, 8, 48}
    final = {
        _key(row): row
        for row in final_rows
        if int(row["n_train_sections"]) in selected_points
    }
    old = {
        _key(run): run
        for run in capped_runs
        if run["n_train_sections"] in selected_points
    }
    if set(final) != set(old):
        raise ValueError("capped and converged comparison grids differ")
    raw_rows = []
    for key in sorted(final):
        for group in METRIC_GROUPS:
            old_loss = old[key]["metrics"]["final_validation"][f"masked_mse_{group}"]
            new_loss = final[key][f"validation_masked_mse_{group}"]
            raw_rows.append(
                {
                    "n_train_sections": key[0],
                    "condition": key[1],
                    "seed": key[2],
                    "gene_group": group,
                    "capped_10pass_loss": old_loss,
                    "converged_loss": new_loss,
                    "converged_minus_capped": new_loss - old_loss,
                }
            )
    old_pair_input = []
    for key, run in old.items():
        old_pair_input.append(
            {
                "n_train_sections": key[0],
                "n_train_cells": key[0] * 8192,
                "condition": key[1],
                "seed": key[2],
                **{
                    f"validation_masked_mse_{group}": run["metrics"][
                        "final_validation"
                    ][f"masked_mse_{group}"]
                    for group in METRIC_GROUPS
                },
            }
        )
    old_paired = {
        (row["n_train_sections"], row["seed"]): row
        for row in paired_delta_rows(old_pair_input)
    }
    new_paired = {
        (row["n_train_sections"], row["seed"]): row
        for row in paired_delta_rows(list(final.values()))
    }
    delta_rows = []
    for key in sorted(old_paired):
        delta_rows.append(
            {
                "n_train_sections": key[0],
                "seed": key[1],
                **{
                    f"capped_delta_space_{group}": old_paired[key][
                        f"delta_space_{group}"
                    ]
                    for group in METRIC_GROUPS
                },
                **{
                    f"converged_delta_space_{group}": new_paired[key][
                        f"delta_space_{group}"
                    ]
                    for group in METRIC_GROUPS
                },
                **{
                    f"converged_minus_capped_delta_space_{group}": new_paired[key][
                        f"delta_space_{group}"
                    ]
                    - old_paired[key][f"delta_space_{group}"]
                    for group in METRIC_GROUPS
                },
            }
        )
    return raw_rows, delta_rows


def _set_n_axis(axis: Any, *, cells: bool = False) -> None:
    axis.set_xscale("log", base=2)
    values = [point * 8192 for point in SCALE_POINTS] if cells else list(SCALE_POINTS)
    axis.set_xticks(values)
    axis.set_xticklabels([str(value) for value in values], rotation=20 if cells else 0)


def _plot_losses(path: Path, arms: list[dict[str, Any]], *, cells: bool) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    colors = {"spatial": "tab:blue", "shuffled": "tab:purple"}
    x_field = "n_train_cells" if cells else "n_train_sections"
    for axis, group, title in zip(
        axes, ("all", "spatial"), ("Overall", "Spatial genes"), strict=True
    ):
        for condition in CONDITIONS:
            rows = sorted(
                (row for row in arms if row["condition"] == condition),
                key=lambda row: row[x_field],
            )
            axis.errorbar(
                [row[x_field] for row in rows],
                [row[f"validation_masked_mse_{group}_mean"] for row in rows],
                yerr=[row[f"validation_masked_mse_{group}_sample_std"] for row in rows],
                color=colors[condition],
                marker="o",
                capsize=3,
                label=condition,
            )
        _set_n_axis(axis, cells=cells)
        axis.set(
            xlabel=(
                "Available training cells (equal-size synthetic sections)"
                if cells
                else "Independent training sections"
            ),
            ylabel="Converged validation masked MSE",
            title=title,
        )
        axis.legend(title="Context")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_delta(
    path: Path, deltas: list[dict[str, Any]], groups: tuple[str, ...]
) -> None:
    colors = {
        "all": "black",
        "intrinsic": "tab:orange",
        "mixed": "tab:green",
        "spatial": "tab:red",
    }
    figure, axis = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    for group in groups:
        axis.errorbar(
            [row["n_train_sections"] for row in deltas],
            [row[f"delta_space_{group}_mean"] for row in deltas],
            yerr=[row[f"delta_space_{group}_sample_std"] for row in deltas],
            color=colors[group],
            marker="o",
            capsize=3,
            label=group,
        )
    axis.axhline(0.0, color="0.65", linewidth=1)
    _set_n_axis(axis)
    axis.set(
        xlabel="Independent training sections",
        ylabel="Paired Δspace (shuffled − spatial MSE)",
    )
    axis.legend(title="Gene group")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_spatial_raw(path: Path, arms: list[dict[str, Any]]) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    colors = {"spatial": "tab:blue", "shuffled": "tab:purple"}
    for condition in CONDITIONS:
        rows = sorted(
            (row for row in arms if row["condition"] == condition),
            key=lambda row: row["n_train_sections"],
        )
        axis.errorbar(
            [row["n_train_sections"] for row in rows],
            [row["validation_masked_mse_spatial_mean"] for row in rows],
            yerr=[row["validation_masked_mse_spatial_sample_std"] for row in rows],
            color=colors[condition],
            marker="o",
            capsize=3,
            label=condition,
        )
    _set_n_axis(axis)
    axis.set(
        xlabel="Independent training sections",
        ylabel="Spatial-gene converged validation masked MSE",
    )
    axis.legend(title="Context")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_diagnostics(path: Path, arms: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), constrained_layout=True)
    fields = (
        ("stop_pass_mean", "Stop pass"),
        ("num_lr_reductions_mean", "LR reductions"),
        ("examples_processed_mean", "Examples processed"),
    )
    colors = {"spatial": "tab:blue", "shuffled": "tab:purple"}
    for axis, (field, ylabel) in zip(axes, fields, strict=True):
        for condition in CONDITIONS:
            rows = sorted(
                (row for row in arms if row["condition"] == condition),
                key=lambda row: row["n_train_sections"],
            )
            axis.errorbar(
                [row["n_train_sections"] for row in rows],
                [row[field] for row in rows],
                yerr=[row[f"{field.removesuffix('_mean')}_sample_std"] for row in rows],
                color=colors[condition],
                marker="o",
                capsize=3,
                label=condition,
            )
        _set_n_axis(axis)
        axis.set(xlabel="Independent training sections", ylabel=ylabel)
    axes[0].legend(title="Context")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_compute(path: Path, arms: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), constrained_layout=True)
    fields = (
        ("run_elapsed_seconds_mean", "Wall-clock runtime (s)"),
        ("optimization_steps_mean", "Optimizer steps"),
        ("examples_processed_mean", "Examples processed"),
    )
    colors = {"spatial": "tab:blue", "shuffled": "tab:purple"}
    for axis, (field, ylabel) in zip(axes, fields, strict=True):
        for condition in CONDITIONS:
            rows = sorted(
                (row for row in arms if row["condition"] == condition),
                key=lambda row: row["n_train_sections"],
            )
            axis.plot(
                [row["n_train_sections"] for row in rows],
                [row[field] for row in rows],
                color=colors[condition],
                marker="o",
                label=condition,
            )
        _set_n_axis(axis)
        axis.set(xlabel="Independent training sections", ylabel=ylabel)
    axes[0].legend(title="Context")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_report(
    path: Path,
    summary_rows: list[dict[str, Any]],
    adjacent: list[dict[str, Any]],
    total_new_gpu_seconds: float,
) -> None:
    lines = [
        "# Final converged N_train frontier",
        "",
        "No scaling function or exponent is fitted.",
        "",
        "| N sections | N cells | spatial overall | shuffled overall | overall Δ | intrinsic Δ | mixed Δ | spatial Δ |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {n_train_sections} | {n_train_cells} | {spatial_all_mean:.6f} ± "
            "{spatial_all_sample_std:.6f} | {shuffled_all_mean:.6f} ± "
            "{shuffled_all_sample_std:.6f} | {delta_space_all_mean:.6f} ± "
            "{delta_space_all_sample_std:.6f} | {delta_space_intrinsic_mean:.6f} ± "
            "{delta_space_intrinsic_sample_std:.6f} | {delta_space_mixed_mean:.6f} ± "
            "{delta_space_mixed_sample_std:.6f} | {delta_space_spatial_mean:.6f} ± "
            "{delta_space_spatial_sample_std:.6f} |".format(**row)
        )
    lines.extend(
        (
            "",
            "## Observed adjacent overall improvements",
            "",
            "| N transition | spatial absolute (%) | shuffled absolute (%) |",
            "|---:|---:|---:|",
        )
    )
    for row in adjacent:
        lines.append(
            f"| {row['from_n_sections']}→{row['to_n_sections']} | "
            f"{row['spatial_overall_absolute_improvement']:.6f} "
            f"({row['spatial_overall_percentage_improvement']:.3f}%) | "
            f"{row['shuffled_overall_absolute_improvement']:.6f} "
            f"({row['shuffled_overall_percentage_improvement']:.3f}%) |"
        )
    lines.extend(
        (
            "",
            f"New-run GPU wall time summed across tasks: {total_new_gpu_seconds / 3600:.3f} hours.",
            "",
        )
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def summarize_converged_data_frontier(
    logical_entries: list[DataScalingEntry],
    new_entries: list[DataScalingEntry],
    spatial_endpoint_entries: list[DataScalingEntry],
    shuffled_endpoint_entries: list[DataScalingEntry],
    capped_entries: list[DataScalingEntry],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Validate all 42 logical runs and write the empirical frontier."""
    validate_final_grids(logical_entries, new_entries)
    if {_key(entry) for entry in spatial_endpoint_entries} != {
        (n, "spatial", seed) for n in ENDPOINT_POINTS for seed in SEEDS
    }:
        raise ValueError("spatial endpoint reuse grid is incomplete")
    if {_key(entry) for entry in shuffled_endpoint_entries} != {
        (n, "shuffled", seed) for n in ENDPOINT_POINTS for seed in SEEDS
    }:
        raise ValueError("shuffled endpoint reuse grid is incomplete")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"summary output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    runs = [
        *_load_sources(new_entries, "new_intermediate_run"),
        *_load_sources(spatial_endpoint_entries, "reused_spatial_90pass"),
        *_load_sources(shuffled_endpoint_entries, "reused_shuffled_60pass"),
    ]
    by_key = {_key(run): run for run in runs}
    if len(runs) != 42 or set(by_key) != _expected_keys(SCALE_POINTS):
        raise ValueError("physical sources do not cover the 42 logical runs")
    expected_by_key = {_key(entry): entry for entry in logical_entries}
    reuse_manifest = [
        _validate_source_against_logical(run, expected_by_key[key])
        for key, run in sorted(by_key.items())
    ]
    training_hashes = _validate_training_code(runs)
    run_rows = [
        _run_summary(run, expected_by_key[key]) for key, run in sorted(by_key.items())
    ]
    paired = paired_delta_rows(run_rows)
    if len(paired) != 21:
        raise ValueError("paired frontier must contain 21 N-by-seed rows")
    arms = _arm_summary_rows(run_rows)
    delta_summaries = _delta_summary_rows(paired)
    summaries = _combined_summary_rows(arms, delta_summaries)
    adjacent = adjacent_improvement_rows(arms)
    capped_runs = [_load_run(entry, historical=True) for entry in capped_entries]
    capped_raw, capped_delta = _capped_comparison_rows(run_rows, capped_runs)

    ordered_sections = runs[0]["training_data_selection"]["ordered_train_section_ids"]
    subset_manifest = [
        {
            "n_train_sections": n_sections,
            "n_train_cells": n_sections * 8192,
            "subset_selection_seed": 314159,
            "active_train_section_ids": "|".join(ordered_sections[:n_sections]),
        }
        for n_sections in SCALE_POINTS
    ]
    _write_csv(output / "run_level.csv", run_rows)
    _write_csv(output / "paired_deltas.csv", paired)
    _write_csv(output / "arm_summary.csv", arms)
    _write_csv(output / "delta_summary.csv", delta_summaries)
    _write_csv(output / "frontier_summary.csv", summaries)
    _write_csv(output / "adjacent_improvements.csv", adjacent)
    _write_csv(output / "reuse_provenance.csv", reuse_manifest)
    _write_csv(output / "nested_subset_manifest.csv", subset_manifest)
    _write_csv(output / "capped_vs_converged_losses.csv", capped_raw)
    _write_csv(output / "capped_vs_converged_deltas.csv", capped_delta)
    (output / "reuse_provenance.json").write_text(
        json.dumps(reuse_manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output / "nested_subset_manifest.json").write_text(
        json.dumps(subset_manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output / "training_code_hashes.json").write_text(
        json.dumps(training_hashes, indent=2) + "\n", encoding="utf-8"
    )

    _plot_losses(output / "validation_loss_vs_n_sections.png", arms, cells=False)
    _plot_losses(output / "validation_loss_vs_n_cells.png", arms, cells=True)
    _plot_delta(
        output / "delta_space_vs_n_sections.png", delta_summaries, METRIC_GROUPS
    )
    _plot_spatial_raw(output / "spatial_gene_raw_losses.png", arms)
    _plot_delta(
        output / "intrinsic_delta_vs_n_sections.png",
        delta_summaries,
        ("intrinsic",),
    )
    _plot_diagnostics(output / "convergence_vs_n_sections.png", arms)
    _plot_compute(output / "compute_vs_n_sections.png", arms)

    new_gpu_seconds = sum(
        float(row["run_elapsed_seconds"])
        for row in run_rows
        if row["reuse_status"] == "new_intermediate_run"
    )
    _write_report(output / "report.md", summaries, adjacent, new_gpu_seconds)
    result = {
        "schema_version": "pilot-data-scaling-converged-frontier-summary-v1",
        "logical_runs": 42,
        "new_runs": 30,
        "reused_runs": 12,
        "matched_pairs": 21,
        "scale_points_sections": list(SCALE_POINTS),
        "scale_points_cells": [point * 8192 for point in SCALE_POINTS],
        "all_runs_converged": all(
            row["convergence_status"] == "EARLY_STOP_CONVERGED" for row in run_rows
        ),
        "all_runs_full_coverage": all(
            row["coverage_fraction"] == 1.0 for row in run_rows
        ),
        "evaluation_index_sha256": EXPECTED_INDEX_HASH,
        "evaluation_mask_sha256": EXPECTED_MASK_HASH,
        "parameter_count": 121024,
        "subset_selection_seed": 314159,
        "ordered_train_section_ids": ordered_sections,
        "new_gpu_wall_time_seconds_sum": new_gpu_seconds,
        "new_gpu_wall_time_hours_sum": new_gpu_seconds / 3600.0,
        "no_scaling_law_fitted": True,
        "output_dir": str(output),
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result
