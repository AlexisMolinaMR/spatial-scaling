"""Spatial-only 90-pass endpoint convergence diagnostic for Pilot v0."""

from __future__ import annotations

import copy
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from spatial_scaling.evaluation.lr_policy_diagnostic import (
    HISTORY_PREFIX,
    METRIC_GROUPS,
    _load_run,
    _write_csv,
    convergence_status,
    tail_improvements,
)
from spatial_scaling.experiments.data_scaling import DataScalingEntry

PASS_POINTS = (10, 30, 60)
SEEDS = (101, 202, 303)
N_SECTIONS = (1, 48)


def post_final_reduction_tail(
    history: list[dict[str, Any]], group: str
) -> dict[str, Any]:
    """Summarize every validation interval after the final LR reduction."""
    reduction_passes = [
        int(row["epoch"]) for row in history if row["lr_reduction_event"]
    ]
    if not reduction_passes:
        return {
            "last_reduction_pass": None,
            "checks_after_reduction": 0,
            "passes_after_reduction": 0,
            "mean_improvement": None,
            "final_3_mean_improvement": None,
            "max_absolute_improvement": None,
        }
    last_reduction = reduction_passes[-1]
    by_pass = {int(row["epoch"]): row for row in history}
    final_pass = max(by_pass)
    field = f"{HISTORY_PREFIX}{group}"
    values = []
    for epoch in range(last_reduction + 1, final_pass + 1):
        if epoch - 1 not in by_pass or epoch not in by_pass:
            raise ValueError("post-reduction validation passes are not consecutive")
        values.append(by_pass[epoch - 1][field] - by_pass[epoch][field])
    return {
        "last_reduction_pass": last_reduction,
        "checks_after_reduction": len(values),
        "passes_after_reduction": final_pass - last_reduction,
        "mean_improvement": statistics.mean(values) if values else None,
        "final_3_mean_improvement": (
            statistics.mean(values[-3:]) if len(values) >= 3 else None
        ),
        "max_absolute_improvement": max(map(abs, values)) if values else None,
    }


def restart_reproduction_rows(
    current: list[dict[str, Any]], historical_spatial: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Compare restarted histories exactly through the historical pass-60 cap."""
    keys = ("n_train_sections", "seed")
    current_by_key = {tuple(run[key] for key in keys): run for run in current}
    old_by_key = {tuple(run[key] for key in keys): run for run in historical_spatial}
    if set(current_by_key) != set(old_by_key):
        raise ValueError("current and historical spatial endpoint grids differ")
    rows = []
    for key in sorted(current_by_key):
        run = current_by_key[key]
        old = old_by_key[key]
        if run["training_data_selection"] != old["training_data_selection"]:
            raise ValueError(f"historical training subset mismatch: {key}")
        if run["evaluation_observations"] != old["evaluation_observations"]:
            raise ValueError(f"historical evaluation mismatch: {key}")
        historical_passes = sorted(old["history_by_pass"])
        if historical_passes != list(range(61)):
            raise ValueError(
                f"historical spatial run does not cover passes 0-60: {key}"
            )
        if not set(historical_passes).issubset(run["history_by_pass"]):
            raise ValueError(f"restarted run ends before pass 60: {key}")
        for epoch in historical_passes:
            new_row = run["history_by_pass"][epoch]
            old_row = old["history_by_pass"][epoch]
            row: dict[str, Any] = {
                "n_train_sections": key[0],
                "seed": key[1],
                "pass": epoch,
                "difference_learning_rate": (
                    new_row["learning_rate"] - old_row["learning_rate"]
                ),
                "difference_learning_rate_after_validation": (
                    new_row["learning_rate_after_validation"]
                    - old_row["learning_rate_after_validation"]
                ),
                "lr_reduction_event_matches": (
                    new_row["lr_reduction_event"] == old_row["lr_reduction_event"]
                ),
                "early_stopping_decision_matches": (
                    new_row["early_stopping_decision"]
                    == old_row["early_stopping_decision"]
                ),
            }
            for group in METRIC_GROUPS:
                field = f"{HISTORY_PREFIX}{group}"
                row[f"difference_{group}"] = new_row[field] - old_row[field]
            rows.append(row)
    return rows


def _scientific_config(config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    for field in ("experiment_name", "output_dir"):
        result.pop(field, None)
    return result


def _policy_without_caps(policy: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(policy)
    result.pop("maximum_epochs", None)
    result.pop("absolute_max_steps", None)
    return result


def _validate_grid(entries: list[DataScalingEntry]) -> None:
    expected = {(n, "spatial", seed) for n in N_SECTIONS for seed in SEEDS}
    actual = {
        (entry.n_train_sections, entry.condition, entry.seed) for entry in entries
    }
    if len(entries) != 6 or actual != expected:
        raise ValueError("90-pass grid must contain exactly six spatial endpoint runs")
    if len({entry.experiment_id for entry in entries}) != 6:
        raise ValueError("90-pass grid contains duplicate experiment IDs")
    if {entry.training_protocol for entry in entries} != {
        "validation_plateau_lr_decay"
    }:
        raise ValueError("90-pass grid changed the plateau-decay policy identity")


def _validate_integrity(
    current: list[dict[str, Any]],
    historical_spatial: list[dict[str, Any]],
    historical_shuffled: list[dict[str, Any]],
) -> dict[str, Any]:
    current_by_key = {(run["n_train_sections"], run["seed"]): run for run in current}
    spatial_by_key = {
        (run["n_train_sections"], run["seed"]): run for run in historical_spatial
    }
    shuffled_by_key = {
        (run["n_train_sections"], run["seed"]): run for run in historical_shuffled
    }
    expected = {(n, seed) for n in N_SECTIONS for seed in SEEDS}
    if set(current_by_key) != expected:
        raise ValueError("current spatial grid is incomplete")
    if set(spatial_by_key) != expected or set(shuffled_by_key) != expected:
        raise ValueError("historical plateau endpoint grid is incomplete")
    for key in sorted(expected):
        run = current_by_key[key]
        old = spatial_by_key[key]
        shuffled = shuffled_by_key[key]
        if run["training_data_selection"] != old["training_data_selection"]:
            raise ValueError(f"90-pass subset changed: {key}")
        if run["evaluation_observations"] != old["evaluation_observations"]:
            raise ValueError(f"90-pass evaluation observations changed: {key}")
        if _scientific_config(run["resolved_config"]) != _scientific_config(
            old["resolved_config"]
        ):
            raise ValueError(f"90-pass scientific configuration changed: {key}")
        if _policy_without_caps(run["execution_policy"]) != _policy_without_caps(
            old["execution_policy"]
        ):
            raise ValueError(f"90-pass scheduler policy changed: {key}")
        if run["execution_policy"]["maximum_epochs"] != 90:
            raise ValueError("current safety cap is not 90 passes")
        if old["execution_policy"]["maximum_epochs"] != 60:
            raise ValueError("historical safety cap is not 60 passes")
        if shuffled["metrics"]["early_stopping_reason"] != ("early_stopping_patience"):
            raise ValueError(f"paired shuffled endpoint is not converged: {key}")
        if shuffled["training_data_selection"] != run["training_data_selection"]:
            raise ValueError(f"paired shuffled subset changed: {key}")
        if shuffled["evaluation_observations"] != run["evaluation_observations"]:
            raise ValueError(f"paired shuffled evaluation changed: {key}")
    if any(run["metrics"]["unique_observation_fraction"] != 1.0 for run in current):
        raise ValueError("a 90-pass run did not achieve full training coverage")
    if {run["metrics"]["parameter_count"] for run in current} != {121024}:
        raise ValueError("model parameter count changed")
    hashes = {
        (
            run["metrics"]["evaluation_observations"]["index_sha256"],
            run["metrics"]["evaluation_observations"]["mask_sha256"],
        )
        for run in current
    }
    if len(hashes) != 1:
        raise ValueError("validation observation or mask hashes changed")
    index_hash, mask_hash = next(iter(hashes))
    reference = current[0]["training_data_selection"]
    return {
        "execution_mode": "restart_from_zero",
        "checkpoint_resume_available": False,
        "evaluation_index_sha256": index_hash,
        "evaluation_mask_sha256": mask_hash,
        "all_runs_full_coverage": True,
        "parameter_count": 121024,
        "subset_selection_seed": reference["subset_selection_seed"],
        "n1_sections": reference["ordered_train_section_ids"][:1],
        "n48_sections": reference["ordered_train_section_ids"],
    }


def _run_summary(run: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics = run["metrics"]
    completed = [row for row in run["history"] if row["epoch"] > 0]
    final_row = completed[-1]
    final_pass = int(final_row["epoch"])
    events = metrics["lr_reduction_events"]
    policy = run["execution_policy"]
    batches_per_pass = math.ceil(
        metrics["available_training_observations"] / metrics["training_batch_size"]
    )
    extra_passes = max(0, final_pass - 60)
    summary: dict[str, Any] = {
        key: run[key]
        for key in (
            "n_train_sections",
            "n_train_cells",
            "condition",
            "seed",
            "experiment_id",
            "run_dir",
        )
    }
    summary.update(
        {
            "execution_mode": "restart_from_zero",
            "final_pass": final_pass,
            "final_step": final_row["step"],
            "best_pass": metrics["best_validation_epoch"],
            "best_step": metrics["best_validation_step"],
            "early_stopping_reason": metrics["early_stopping_reason"],
            "convergence_status": convergence_status(metrics),
            "initial_learning_rate": run["history"][0]["learning_rate"],
            "num_lr_reductions": metrics["num_lr_reductions"],
            "lr_reduction_events_json": json.dumps(events, separators=(",", ":")),
            "final_learning_rate": metrics["final_learning_rate"],
            "minimum_learning_rate": policy["minimum_learning_rate"],
            "lr_floor_reached": metrics["final_learning_rate"]
            <= policy["minimum_learning_rate"],
            "optimization_steps": metrics["optimization_steps"],
            "examples_processed": metrics["examples_processed"],
            "passes_newly_executed_after_60": extra_passes,
            "optimizer_steps_after_60": extra_passes * batches_per_pass,
            "examples_processed_after_60": extra_passes
            * metrics["available_training_observations"],
            "estimated_optimization_seconds_after_60": extra_passes
            * batches_per_pass
            * metrics["training_step_time_seconds_mean"],
            "available_training_observations": metrics[
                "available_training_observations"
            ],
            "unique_training_observations_seen": metrics[
                "unique_training_observations_seen"
            ],
            "coverage_fraction": metrics["unique_observation_fraction"],
            "effective_passes": metrics["effective_passes"],
            "run_elapsed_seconds": metrics["run_elapsed_seconds"],
            "optimization_elapsed_seconds": metrics["optimization_elapsed_seconds"],
            "examples_per_second": metrics["examples_per_second"],
            "peak_device_memory_bytes": metrics["peak_device_memory_bytes"],
            "parameter_count": metrics["parameter_count"],
            "evaluation_index_sha256": metrics["evaluation_observations"][
                "index_sha256"
            ],
            "evaluation_mask_sha256": metrics["evaluation_observations"]["mask_sha256"],
            "active_train_section_ids": "|".join(
                run["training_data_selection"]["active_train_section_ids"]
            ),
        }
    )
    tail_rows = []
    for group in METRIC_GROUPS:
        field = f"{HISTORY_PREFIX}{group}"
        tail = tail_improvements(run["history"], group)
        post = post_final_reduction_tail(run["history"], group)
        for epoch in PASS_POINTS:
            summary[f"pass{epoch}_loss_{group}"] = run["history_by_pass"][epoch][field]
        summary[f"final_pass_loss_{group}"] = final_row[field]
        summary[f"best_restored_loss_{group}"] = metrics["final_validation"][
            f"masked_mse_{group}"
        ]
        summary[f"minimum_observed_loss_{group}"] = min(row[field] for row in completed)
        summary[f"minimum_observed_pass_{group}"] = min(
            completed, key=lambda row: row[field]
        )["epoch"]
        summary[f"final_3_mean_improvement_{group}"] = tail["final_3_mean"]
        summary[f"final_5_mean_improvement_{group}"] = tail["final_5_mean"]
        summary[f"final_5_max_absolute_improvement_{group}"] = tail[
            "final_5_max_absolute"
        ]
        for key, value in post.items():
            summary[f"post_final_lr_{key}_{group}"] = value
        for item in tail["rows"]:
            tail_rows.append(
                {
                    "n_train_sections": run["n_train_sections"],
                    "seed": run["seed"],
                    "gene_group": group,
                    **item,
                }
            )
    return summary, tail_rows


def _trajectory_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "n_train_sections": run["n_train_sections"],
            "seed": run["seed"],
            "pass": row["epoch"],
            "step": row["step"],
            "learning_rate": row["learning_rate"],
            "learning_rate_after_validation": row["learning_rate_after_validation"],
            "lr_reduction_event": row["lr_reduction_event"],
            "early_stopping_decision": row["early_stopping_decision"],
            **{
                f"validation_masked_mse_{group}": row[f"{HISTORY_PREFIX}{group}"]
                for group in METRIC_GROUPS
            },
        }
        for run in runs
        for row in run["history"]
    ]


def _lr_event_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "n_train_sections": run["n_train_sections"],
            "seed": run["seed"],
            "event_index": index,
            **event,
        }
        for run in runs
        for index, event in enumerate(run["metrics"]["lr_reduction_events"], start=1)
    ]


def paired_delta_rows(
    spatial_runs: list[dict[str, Any]], shuffled_runs: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pair spatial trajectories to matched then fixed converged shuffled loss."""
    shuffled_by_key = {
        (run["n_train_sections"], run["seed"]): run for run in shuffled_runs
    }
    trajectories: list[dict[str, Any]] = []
    endpoints: list[dict[str, Any]] = []
    for spatial in sorted(
        spatial_runs, key=lambda run: (run["n_train_sections"], run["seed"])
    ):
        key = (spatial["n_train_sections"], spatial["seed"])
        shuffled = shuffled_by_key[key]
        shuffled_final_pass = max(shuffled["history_by_pass"])
        for spatial_row in spatial["history"]:
            epoch = int(spatial_row["epoch"])
            if epoch in shuffled["history_by_pass"]:
                shuffled_values = {
                    group: shuffled["history_by_pass"][epoch][
                        f"{HISTORY_PREFIX}{group}"
                    ]
                    for group in METRIC_GROUPS
                }
                source = "matched_pass_history"
            else:
                shuffled_values = {
                    group: shuffled["metrics"]["final_validation"][
                        f"masked_mse_{group}"
                    ]
                    for group in METRIC_GROUPS
                }
                source = "fixed_converged_shuffled_endpoint"
            trajectories.append(
                {
                    "n_train_sections": key[0],
                    "seed": key[1],
                    "pass": epoch,
                    "shuffled_source": source,
                    "shuffled_history_final_pass": shuffled_final_pass,
                    **{
                        f"delta_space_{group}": shuffled_values[group]
                        - spatial_row[f"{HISTORY_PREFIX}{group}"]
                        for group in METRIC_GROUPS
                    },
                }
            )
        for point in (*PASS_POINTS, "terminal"):
            if point == "terminal":
                spatial_values = {
                    group: spatial["metrics"]["final_validation"][f"masked_mse_{group}"]
                    for group in METRIC_GROUPS
                }
                spatial_pass = spatial["metrics"]["best_validation_epoch"]
                shuffled_values = {
                    group: shuffled["metrics"]["final_validation"][
                        f"masked_mse_{group}"
                    ]
                    for group in METRIC_GROUPS
                }
                source = "fixed_converged_shuffled_endpoint"
            else:
                spatial_pass = point
                spatial_values = {
                    group: spatial["history_by_pass"][point][f"{HISTORY_PREFIX}{group}"]
                    for group in METRIC_GROUPS
                }
                if point in shuffled["history_by_pass"]:
                    shuffled_values = {
                        group: shuffled["history_by_pass"][point][
                            f"{HISTORY_PREFIX}{group}"
                        ]
                        for group in METRIC_GROUPS
                    }
                    source = "matched_pass_history"
                else:
                    shuffled_values = {
                        group: shuffled["metrics"]["final_validation"][
                            f"masked_mse_{group}"
                        ]
                        for group in METRIC_GROUPS
                    }
                    source = "fixed_converged_shuffled_endpoint"
            endpoints.append(
                {
                    "n_train_sections": key[0],
                    "seed": key[1],
                    "point": point,
                    "spatial_pass": spatial_pass,
                    "shuffled_source": source,
                    "shuffled_final_pass": shuffled_final_pass,
                    **{
                        f"spatial_loss_{group}": spatial_values[group]
                        for group in METRIC_GROUPS
                    },
                    **{
                        f"shuffled_loss_{group}": shuffled_values[group]
                        for group in METRIC_GROUPS
                    },
                    **{
                        f"delta_space_{group}": shuffled_values[group]
                        - spatial_values[group]
                        for group in METRIC_GROUPS
                    },
                }
            )
    return trajectories, endpoints


def _delta_summary_rows(endpoints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in endpoints:
        grouped[(row["n_train_sections"], row["point"])].append(row)
    order = {10: 0, 30: 1, 60: 2, "terminal": 3}
    result = []
    for key, rows in sorted(
        grouped.items(), key=lambda item: (item[0][0], order[item[0][1]])
    ):
        if len(rows) != 3:
            raise ValueError(f"delta endpoint lacks three seeds: {key}")
        summary: dict[str, Any] = {
            "n_train_sections": key[0],
            "point": key[1],
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


def _plot_metric(path: Path, runs: list[dict[str, Any]], group: str) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    colors = {101: "tab:blue", 202: "tab:orange", 303: "tab:green"}
    field = f"{HISTORY_PREFIX}{group}"
    for axis, n_sections in zip(axes, N_SECTIONS, strict=True):
        for run in [run for run in runs if run["n_train_sections"] == n_sections]:
            passes = [row["epoch"] for row in run["history"]]
            values = [row[field] for row in run["history"]]
            axis.plot(
                passes, values, color=colors[run["seed"]], label=f"seed {run['seed']}"
            )
            events = [row for row in run["history"] if row["lr_reduction_event"]]
            axis.scatter(
                [row["epoch"] for row in events],
                [row[field] for row in events],
                color=colors[run["seed"]],
                marker="x",
                s=45,
            )
        axis.set(
            xlabel="Complete training-data passes",
            ylabel=f"{group.capitalize()} validation masked MSE",
            title=f"N={n_sections}",
        )
        axis.legend(fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_learning_rates(path: Path, runs: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    colors = {101: "tab:blue", 202: "tab:orange", 303: "tab:green"}
    for axis, n_sections in zip(axes, N_SECTIONS, strict=True):
        for run in [run for run in runs if run["n_train_sections"] == n_sections]:
            axis.step(
                [row["epoch"] for row in run["history"]],
                [row["learning_rate_after_validation"] for row in run["history"]],
                where="post",
                color=colors[run["seed"]],
                label=f"seed {run['seed']}",
            )
        axis.set_yscale("log")
        axis.set(
            xlabel="Complete training-data passes",
            ylabel="Learning rate (log scale)",
            title=f"N={n_sections}",
        )
        axis.legend(fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_deltas(path: Path, rows: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    colors = {
        "all": "black",
        "intrinsic": "tab:orange",
        "mixed": "tab:green",
        "spatial": "tab:red",
    }
    for axis, n_sections in zip(axes, N_SECTIONS, strict=True):
        selected = [row for row in rows if row["n_train_sections"] == n_sections]
        for group, color in colors.items():
            by_pass: dict[int, list[float]] = defaultdict(list)
            for row in selected:
                by_pass[int(row["pass"])].append(row[f"delta_space_{group}"])
            passes = sorted(by_pass)
            axis.plot(
                passes,
                [statistics.mean(by_pass[epoch]) for epoch in passes],
                color=color,
                label=group,
            )
        axis.axhline(0, color="0.7", linewidth=1)
        axis.set(
            xlabel="Complete spatial-training passes",
            ylabel="Paired Δspace (shuffled − spatial MSE)",
            title=f"N={n_sections}",
        )
        axis.legend(fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def summarize_spatial_90pass_convergence(
    entries: list[DataScalingEntry],
    historical_entries: list[DataScalingEntry],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Validate and aggregate the six restarted 90-pass spatial runs."""
    _validate_grid(entries)
    historical_plateau = [
        entry
        for entry in historical_entries
        if entry.training_protocol == "validation_plateau_lr_decay"
    ]
    historical_spatial_entries = [
        entry for entry in historical_plateau if entry.condition == "spatial"
    ]
    historical_shuffled_entries = [
        entry for entry in historical_plateau if entry.condition == "shuffled"
    ]
    if len(historical_spatial_entries) != 6 or len(historical_shuffled_entries) != 6:
        raise ValueError("historical plateau diagnostic lacks endpoint arms")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"summary output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    current = [_load_run(entry) for entry in entries]
    historical_spatial = [_load_run(entry) for entry in historical_spatial_entries]
    historical_shuffled = [_load_run(entry) for entry in historical_shuffled_entries]
    integrity = _validate_integrity(current, historical_spatial, historical_shuffled)
    reproduction = restart_reproduction_rows(current, historical_spatial)
    numeric_fields = (
        "difference_learning_rate",
        "difference_learning_rate_after_validation",
        *(f"difference_{group}" for group in METRIC_GROUPS),
    )
    reproduction_max = max(
        abs(float(row[field])) for row in reproduction for field in numeric_fields
    )
    event_match = all(
        row["lr_reduction_event_matches"] and row["early_stopping_decision_matches"]
        for row in reproduction
    )
    if reproduction_max > 1e-6 or not event_match:
        raise ValueError("restarted trajectory failed exact pass-60 reproduction")
    run_summaries, tail_rows = [], []
    for run in current:
        summary, tails = _run_summary(run)
        run_summaries.append(summary)
        tail_rows.extend(tails)
    trajectories = _trajectory_rows(current)
    lr_events = _lr_event_rows(current)
    delta_trajectories, delta_endpoints = paired_delta_rows(
        current, historical_shuffled
    )
    delta_summaries = _delta_summary_rows(delta_endpoints)
    _write_csv(output / "run_summary.csv", run_summaries)
    _write_csv(output / "validation_trajectories.csv", trajectories)
    _write_csv(output / "lr_reduction_events.csv", lr_events)
    _write_csv(output / "tail_improvements.csv", tail_rows)
    _write_csv(output / "restart_pass60_reproduction.csv", reproduction)
    _write_csv(output / "delta_trajectories.csv", delta_trajectories)
    _write_csv(output / "delta_endpoints.csv", delta_endpoints)
    _write_csv(output / "delta_summary.csv", delta_summaries)
    _plot_metric(output / "overall_validation_vs_pass.png", current, "all")
    _plot_metric(output / "spatial_gene_validation_vs_pass.png", current, "spatial")
    _plot_metric(output / "intrinsic_gene_validation_vs_pass.png", current, "intrinsic")
    _plot_learning_rates(output / "learning_rate_vs_pass.png", current)
    _plot_deltas(output / "delta_space_vs_pass.png", delta_trajectories)
    statuses = defaultdict(int)
    for row in run_summaries:
        statuses[row["convergence_status"]] += 1
    result = {
        "schema_version": "pilot-data-scaling-spatial-90pass-summary-v1",
        "runs": 6,
        "execution_mode": "restart_from_zero",
        "checkpoint_resume_available": False,
        "pass60_reproduction_max_absolute_difference": reproduction_max,
        "pass60_scheduler_events_match": event_match,
        "reproduction_tolerance": 1e-6,
        "convergence_status_counts": dict(statuses),
        "integrity": integrity,
        "classification_rule": (
            "patience termination => EARLY_STOP_CONVERGED; maximum pass cap => "
            "CAP_REACHED_STILL_IMPROVING because no new effective-flat threshold "
            "is introduced; non-finite => UNSTABLE; other termination => OTHER"
        ),
        "output_dir": str(output),
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result
