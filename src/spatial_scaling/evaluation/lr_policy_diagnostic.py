"""Endpoint learning-rate policy comparison for Pilot v0 data scaling."""

from __future__ import annotations

import copy
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from spatial_scaling.data.synthetic.dataset import GENE_CLASSES
from spatial_scaling.experiments.data_scaling import DataScalingEntry
from spatial_scaling.training.policies import TrainingDataSelection

METRIC_GROUPS = ("all", *GENE_CLASSES)
HISTORY_PREFIX = "validation_masked_mse_"
POLICIES = ("prolonged_constant_lr", "validation_plateau_lr_decay")
CONVERGENCE_STATUSES = (
    "EARLY_STOP_CONVERGED",
    "CAP_REACHED_EFFECTIVELY_FLAT",
    "CAP_REACHED_STILL_IMPROVING",
    "UNSTABLE",
    "OTHER",
)


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
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _parse_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid serialized boolean: {value!r}")


def _read_history(path: Path, *, require_policy_events: bool) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    if not raw:
        raise ValueError(f"empty training history: {path}")
    rows = []
    for raw_row in raw:
        row: dict[str, Any] = {
            "step": int(raw_row["step"]),
            "epoch": float(raw_row["epoch"]),
            "learning_rate": float(raw_row["learning_rate"]),
            **{
                f"{HISTORY_PREFIX}{group}": float(raw_row[f"{HISTORY_PREFIX}{group}"])
                for group in METRIC_GROUPS
            },
        }
        if require_policy_events:
            for field in (
                "learning_rate_after_validation",
                "lr_reduction_event",
                "early_stopping_decision",
            ):
                if field not in raw_row:
                    raise ValueError(f"history lacks {field}: {path}")
            row["learning_rate_after_validation"] = float(
                raw_row["learning_rate_after_validation"]
            )
            row["lr_reduction_event"] = _parse_bool(raw_row["lr_reduction_event"])
            row["early_stopping_decision"] = _parse_bool(
                raw_row["early_stopping_decision"]
            )
        else:
            row["learning_rate_after_validation"] = row["learning_rate"]
            row["lr_reduction_event"] = False
            row["early_stopping_decision"] = False
        rows.append(row)
    epochs = [row["epoch"] for row in rows]
    if epochs != sorted(epochs) or len(epochs) != len(set(epochs)):
        raise ValueError(f"history passes are not strictly increasing: {path}")
    return rows


def convergence_status(metrics: dict[str, Any]) -> str:
    """Classify termination without introducing a post-hoc flatness threshold."""
    reason = metrics["early_stopping_reason"]
    final = float(metrics["final_validation"]["masked_mse_all"])
    if not math.isfinite(final):
        return "UNSTABLE"
    if reason == "early_stopping_patience":
        return "EARLY_STOP_CONVERGED"
    if reason == "maximum_epochs":
        return "CAP_REACHED_STILL_IMPROVING"
    if reason == "absolute_max_steps":
        return "OTHER"
    return "OTHER"


def tail_improvements(
    history: list[dict[str, Any]], group: str, *, maximum_intervals: int = 5
) -> dict[str, Any]:
    """Return consecutive final validation improvements for one metric group."""
    if maximum_intervals <= 0:
        raise ValueError("maximum_intervals must be positive")
    completed = [row for row in history if row["epoch"] > 0]
    if len(completed) < maximum_intervals:
        raise ValueError("too few completed passes for requested tail")
    by_pass = {int(row["epoch"]): row for row in history}
    final_pass = int(completed[-1]["epoch"])
    field = f"{HISTORY_PREFIX}{group}"
    rows = []
    for epoch in range(final_pass - maximum_intervals + 1, final_pass + 1):
        if epoch - 1 not in by_pass:
            raise ValueError("tail passes are not consecutive")
        rows.append(
            {
                "pass": epoch,
                "improvement": by_pass[epoch - 1][field] - by_pass[epoch][field],
            }
        )
    values = [row["improvement"] for row in rows]
    return {
        "rows": rows,
        "final_3_mean": statistics.mean(values[-3:]),
        "final_5_mean": statistics.mean(values),
        "final_5_max_absolute": max(map(abs, values)),
    }


def post_last_reduction_tail(
    history: list[dict[str, Any]], group: str
) -> dict[str, Any]:
    """Summarize up to five intervals strictly after the final LR reduction."""
    reduction_passes = [
        int(row["epoch"]) for row in history if row["lr_reduction_event"]
    ]
    if not reduction_passes:
        return {"count": 0, "mean": None, "max_absolute": None}
    last_reduction = reduction_passes[-1]
    by_pass = {int(row["epoch"]): row for row in history}
    final_pass = max(by_pass)
    field = f"{HISTORY_PREFIX}{group}"
    values = [
        by_pass[epoch - 1][field] - by_pass[epoch][field]
        for epoch in range(last_reduction + 1, final_pass + 1)
        if epoch - 1 in by_pass
    ][-5:]
    return {
        "count": len(values),
        "mean": statistics.mean(values) if values else None,
        "max_absolute": max(map(abs, values)) if values else None,
    }


def _load_run(entry: DataScalingEntry, *, historical: bool = False) -> dict[str, Any]:
    run_dir = Path(entry.output_dir)
    metrics = _read_json(run_dir / "metrics.json")
    history = _read_history(
        run_dir / "history.csv", require_policy_events=not historical
    )
    selection = TrainingDataSelection.from_dict(
        metrics["training_data_selection"]
    ).to_dict()
    expected_selection = TrainingDataSelection.from_dict(
        entry.training_data_selection
    ).to_dict()
    if selection != expected_selection:
        raise ValueError(f"training selection mismatch: {run_dir}")
    if metrics["experiment_id"] != entry.experiment_id:
        raise ValueError(f"experiment identity mismatch: {run_dir}")
    if metrics["config_identity_sha256"] != entry.config_identity_sha256:
        raise ValueError(f"configuration identity mismatch: {run_dir}")
    return {
        "policy": entry.training_protocol,
        "n_train_sections": entry.n_train_sections,
        "n_train_cells": entry.n_train_cells,
        "condition": entry.condition,
        "seed": entry.seed,
        "experiment_id": entry.experiment_id,
        "run_dir": str(run_dir),
        "resolved_config": entry.resolved_config,
        "execution_policy": entry.execution_policy,
        "training_data_selection": selection,
        "evaluation_observations": metrics["evaluation_observations"],
        "metrics": metrics,
        "history": history,
        "history_by_pass": {int(row["epoch"]): row for row in history},
    }


def overlapping_history_reproduction(
    current: list[dict[str, Any]], historical: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Compare constant-LR histories to every available historical pass."""
    keys = ("n_train_sections", "condition", "seed")
    current_by_key = {
        tuple(run[key] for key in keys): run
        for run in current
        if run["policy"] == "prolonged_constant_lr"
    }
    historical_by_key = {tuple(run[key] for key in keys): run for run in historical}
    if set(current_by_key) != set(historical_by_key):
        raise ValueError("constant and historical endpoint grids do not match")
    rows = []
    for key in sorted(current_by_key):
        run = current_by_key[key]
        old = historical_by_key[key]
        if run["training_data_selection"] != old["training_data_selection"]:
            raise ValueError(f"historical subset mismatch: {key}")
        if run["evaluation_observations"] != old["evaluation_observations"]:
            raise ValueError(f"historical validation mismatch: {key}")
        overlap = sorted(set(run["history_by_pass"]) & set(old["history_by_pass"]))
        if overlap != sorted(old["history_by_pass"]):
            raise ValueError(
                f"constant run does not cover full historical history: {key}"
            )
        for epoch in overlap:
            row = {
                "n_train_sections": key[0],
                "condition": key[1],
                "seed": key[2],
                "pass": epoch,
            }
            for group in METRIC_GROUPS:
                field = f"{HISTORY_PREFIX}{group}"
                row[f"difference_{group}"] = (
                    run["history_by_pass"][epoch][field]
                    - old["history_by_pass"][epoch][field]
                )
            rows.append(row)
    return rows


def pre_reduction_reproduction(current: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare plateau runs to constant runs through their first LR event."""
    grouped: dict[tuple[int, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for run in current:
        grouped[(run["n_train_sections"], run["condition"], run["seed"])][
            run["policy"]
        ] = run
    rows = []
    for key, policies in sorted(grouped.items()):
        if set(policies) != set(POLICIES):
            raise ValueError(f"incomplete policy pair: {key}")
        constant = policies["prolonged_constant_lr"]
        plateau = policies["validation_plateau_lr_decay"]
        first_events = [
            int(row["epoch"]) for row in plateau["history"] if row["lr_reduction_event"]
        ]
        if not first_events:
            comparison_end = min(
                max(constant["history_by_pass"]), max(plateau["history_by_pass"])
            )
        else:
            comparison_end = first_events[0]
        for epoch in range(comparison_end + 1):
            if epoch not in constant["history_by_pass"]:
                raise ValueError(f"constant history ends before LR comparison: {key}")
            row = {
                "n_train_sections": key[0],
                "condition": key[1],
                "seed": key[2],
                "pass": epoch,
                "first_lr_reduction_pass": first_events[0] if first_events else None,
            }
            for group in METRIC_GROUPS:
                field = f"{HISTORY_PREFIX}{group}"
                row[f"difference_{group}"] = (
                    plateau["history_by_pass"][epoch][field]
                    - constant["history_by_pass"][epoch][field]
                )
            rows.append(row)
    return rows


def _run_summary(run: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics = run["metrics"]
    completed = [row for row in run["history"] if row["epoch"] > 0]
    final_row = completed[-1]
    events = metrics["lr_reduction_events"]
    summary: dict[str, Any] = {
        key: run[key]
        for key in (
            "policy",
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
            "final_pass": int(final_row["epoch"]),
            "final_step": final_row["step"],
            "best_pass": metrics["best_validation_epoch"],
            "best_step": metrics["best_validation_step"],
            "early_stopping_reason": metrics["early_stopping_reason"],
            "convergence_status": convergence_status(metrics),
            "num_lr_reductions": metrics["num_lr_reductions"],
            "lr_reduction_events_json": json.dumps(events, separators=(",", ":")),
            "final_learning_rate": metrics["final_learning_rate"],
            "optimization_steps": metrics["optimization_steps"],
            "examples_processed": metrics["examples_processed"],
            "available_training_observations": metrics[
                "available_training_observations"
            ],
            "unique_training_observations_seen": metrics[
                "unique_training_observations_seen"
            ],
            "coverage_fraction": metrics["unique_observation_fraction"],
            "effective_passes": metrics["effective_passes"],
            "run_elapsed_seconds": metrics["run_elapsed_seconds"],
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
        post = post_last_reduction_tail(run["history"], group)
        summary[f"pass10_loss_{group}"] = run["history_by_pass"][10][field]
        summary[f"pass30_loss_{group}"] = (
            run["history_by_pass"][30][field] if 30 in run["history_by_pass"] else None
        )
        summary[f"final_pass_loss_{group}"] = final_row[field]
        summary[f"best_restored_loss_{group}"] = metrics["final_validation"][
            f"masked_mse_{group}"
        ]
        summary[f"final_3_mean_improvement_{group}"] = tail["final_3_mean"]
        summary[f"final_5_mean_improvement_{group}"] = tail["final_5_mean"]
        summary[f"final_5_max_absolute_improvement_{group}"] = tail[
            "final_5_max_absolute"
        ]
        summary[f"post_last_lr_reduction_intervals_{group}"] = post["count"]
        summary[f"post_last_lr_reduction_mean_improvement_{group}"] = post["mean"]
        summary[f"post_last_lr_reduction_max_absolute_{group}"] = post["max_absolute"]
        for item in tail["rows"]:
            tail_rows.append(
                {
                    "policy": run["policy"],
                    "n_train_sections": run["n_train_sections"],
                    "condition": run["condition"],
                    "seed": run["seed"],
                    "gene_group": group,
                    **item,
                }
            )
    return summary, tail_rows


def _trajectory_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "policy": run["policy"],
            "n_train_sections": run["n_train_sections"],
            "condition": run["condition"],
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
    rows = []
    for run in runs:
        for event_index, event in enumerate(
            run["metrics"]["lr_reduction_events"], start=1
        ):
            rows.append(
                {
                    "policy": run["policy"],
                    "n_train_sections": run["n_train_sections"],
                    "condition": run["condition"],
                    "seed": run["seed"],
                    "event_index": event_index,
                    **event,
                }
            )
    return rows


def _policy_comparison_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in summaries:
        grouped[(row["n_train_sections"], row["condition"], row["seed"])][
            row["policy"]
        ] = row
    result = []
    fields = (
        "best_restored_loss_all",
        "final_pass_loss_all",
        "final_pass",
        "examples_processed",
        "run_elapsed_seconds",
        "final_3_mean_improvement_all",
        "final_5_mean_improvement_all",
        "convergence_status",
        "num_lr_reductions",
        "final_learning_rate",
    )
    for key, policies in sorted(grouped.items()):
        if set(policies) != set(POLICIES):
            raise ValueError(f"incomplete policy comparison: {key}")
        row: dict[str, Any] = {
            "n_train_sections": key[0],
            "condition": key[1],
            "seed": key[2],
        }
        for policy in POLICIES:
            for field in fields:
                row[f"{policy}_{field}"] = policies[policy][field]
        result.append(row)
    return result


def _arm_summary_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        grouped[(row["policy"], row["n_train_sections"], row["condition"])].append(row)
    result = []
    for key, rows in sorted(grouped.items()):
        if len(rows) != 3:
            raise ValueError(f"expected three seeds for arm {key}")
        summary: dict[str, Any] = {
            "policy": key[0],
            "n_train_sections": key[1],
            "condition": key[2],
            "num_seeds": 3,
        }
        for group in METRIC_GROUPS:
            values = [float(row[f"best_restored_loss_{group}"]) for row in rows]
            summary[f"best_restored_loss_{group}_mean"] = statistics.mean(values)
            summary[f"best_restored_loss_{group}_sample_std"] = statistics.stdev(values)
        for field in (
            "final_pass",
            "optimization_steps",
            "examples_processed",
            "run_elapsed_seconds",
            "num_lr_reductions",
            "final_3_mean_improvement_all",
            "final_5_mean_improvement_all",
        ):
            values = [float(row[field]) for row in rows]
            summary[f"{field}_mean"] = statistics.mean(values)
            summary[f"{field}_sample_std"] = statistics.stdev(values)
        summary["converged_seeds"] = sum(
            row["convergence_status"] == "EARLY_STOP_CONVERGED" for row in rows
        )
        result.append(summary)
    return result


def _delta_rows(
    runs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for run in runs:
        grouped[(run["policy"], run["n_train_sections"], run["seed"])][
            run["condition"]
        ] = run
    trajectories, endpoints = [], []
    for key, arms in sorted(grouped.items()):
        if set(arms) != {"spatial", "shuffled"}:
            raise ValueError(f"incomplete context pair: {key}")
        spatial, shuffled = arms["spatial"], arms["shuffled"]
        common = sorted(
            set(spatial["history_by_pass"]) & set(shuffled["history_by_pass"])
        )
        for epoch in common:
            trajectories.append(
                {
                    "policy": key[0],
                    "n_train_sections": key[1],
                    "seed": key[2],
                    "pass": epoch,
                    **{
                        f"delta_space_{group}": shuffled["history_by_pass"][epoch][
                            f"{HISTORY_PREFIX}{group}"
                        ]
                        - spatial["history_by_pass"][epoch][f"{HISTORY_PREFIX}{group}"]
                        for group in METRIC_GROUPS
                    },
                }
            )
        endpoint: dict[str, Any] = {
            "policy": key[0],
            "n_train_sections": key[1],
            "seed": key[2],
            "last_common_pass": max(common),
            "spatial_final_pass": max(spatial["history_by_pass"]),
            "shuffled_final_pass": max(shuffled["history_by_pass"]),
        }
        for group in METRIC_GROUPS:
            field = f"{HISTORY_PREFIX}{group}"
            for epoch in (10, 30):
                endpoint[f"pass{epoch}_delta_space_{group}"] = (
                    shuffled["history_by_pass"][epoch][field]
                    - spatial["history_by_pass"][epoch][field]
                    if epoch in common
                    else None
                )
            endpoint[f"terminal_restored_delta_space_{group}"] = (
                shuffled["metrics"]["final_validation"][f"masked_mse_{group}"]
                - spatial["metrics"]["final_validation"][f"masked_mse_{group}"]
            )
        endpoints.append(endpoint)
    return trajectories, endpoints


def _matched_config(config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    for field in ("experiment_name", "output_dir", "condition"):
        result.pop(field, None)
    return result


def _validate_integrity(runs: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[(run["n_train_sections"], run["condition"], run["seed"])].append(run)
    for key, policy_runs in grouped.items():
        if {run["policy"] for run in policy_runs} != set(POLICIES):
            raise ValueError(f"policy pair incomplete: {key}")
        reference = policy_runs[0]
        for run in policy_runs[1:]:
            if run["training_data_selection"] != reference["training_data_selection"]:
                raise ValueError(f"policy subsets differ: {key}")
            if run["evaluation_observations"] != reference["evaluation_observations"]:
                raise ValueError(f"policy validation observations differ: {key}")
            if _matched_config(run["resolved_config"]) != _matched_config(
                reference["resolved_config"]
            ):
                raise ValueError(f"scientific configuration differs by policy: {key}")
    hashes = {
        (
            run["metrics"]["evaluation_observations"]["index_sha256"],
            run["metrics"]["evaluation_observations"]["mask_sha256"],
        )
        for run in runs
    }
    if len(hashes) != 1:
        raise ValueError("endpoint validation hashes are not globally fixed")
    if any(run["metrics"]["unique_observation_fraction"] != 1.0 for run in runs):
        raise ValueError("an endpoint run did not satisfy full exposure")
    if {run["metrics"]["parameter_count"] for run in runs} != {121024}:
        raise ValueError("endpoint model parameter count changed")
    index_hash, mask_hash = next(iter(hashes))
    return {
        "evaluation_index_sha256": index_hash,
        "evaluation_mask_sha256": mask_hash,
        "all_runs_full_coverage": True,
        "parameter_count": 121024,
        "subset_selection_seed": runs[0]["training_data_selection"][
            "subset_selection_seed"
        ],
        "n1_sections": runs[0]["training_data_selection"]["ordered_train_section_ids"][
            :1
        ],
        "n48_sections": runs[0]["training_data_selection"]["ordered_train_section_ids"],
    }


def _plot_metric(path: Path, runs: list[dict[str, Any]], group: str) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    colors = {"spatial": "tab:blue", "shuffled": "tab:purple"}
    linestyles = {"prolonged_constant_lr": "-", "validation_plateau_lr_decay": "--"}
    for axis, n_sections in zip(axes, (1, 48), strict=True):
        for policy in POLICIES:
            for condition in ("spatial", "shuffled"):
                selected = [
                    run
                    for run in runs
                    if run["policy"] == policy
                    and run["condition"] == condition
                    and run["n_train_sections"] == n_sections
                ]
                by_pass: dict[int, list[float]] = defaultdict(list)
                for run in selected:
                    axis.plot(
                        [row["epoch"] for row in run["history"]],
                        [row[f"{HISTORY_PREFIX}{group}"] for row in run["history"]],
                        color=colors[condition],
                        linestyle=linestyles[policy],
                        alpha=0.22,
                    )
                    for row in run["history"]:
                        by_pass[int(row["epoch"])].append(
                            row[f"{HISTORY_PREFIX}{group}"]
                        )
                passes = sorted(by_pass)
                axis.plot(
                    passes,
                    [statistics.mean(by_pass[p]) for p in passes],
                    color=colors[condition],
                    linestyle=linestyles[policy],
                    linewidth=2,
                    label=f"{condition}, {policy}",
                )
        axis.set(
            xlabel="Complete training-data passes",
            ylabel=f"{group.capitalize()} validation masked MSE",
            title=f"N={n_sections}",
        )
        axis.legend(fontsize=7)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_learning_rates(path: Path, runs: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    colors = {"spatial": "tab:blue", "shuffled": "tab:purple"}
    for axis, n_sections in zip(axes, (1, 48), strict=True):
        selected = [
            run
            for run in runs
            if run["policy"] == "validation_plateau_lr_decay"
            and run["n_train_sections"] == n_sections
        ]
        for run in selected:
            axis.step(
                [row["epoch"] for row in run["history"]],
                [row["learning_rate_after_validation"] for row in run["history"]],
                where="post",
                color=colors[run["condition"]],
                alpha=0.65,
                label=f"{run['condition']} seed {run['seed']}",
            )
        axis.set_yscale("log")
        axis.set(
            xlabel="Complete training-data passes",
            ylabel="Learning rate (log scale)",
            title=f"N={n_sections}",
        )
        axis.legend(fontsize=7)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_deltas(path: Path, rows: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    colors = {
        "all": "black",
        "intrinsic": "tab:orange",
        "mixed": "tab:green",
        "spatial": "tab:red",
    }
    for axis, (policy, n_sections) in zip(
        axes.flat,
        ((policy, n) for policy in POLICIES for n in (1, 48)),
        strict=True,
    ):
        selected = [
            row
            for row in rows
            if row["policy"] == policy and row["n_train_sections"] == n_sections
        ]
        for group, color in colors.items():
            by_pass: dict[int, list[float]] = defaultdict(list)
            for row in selected:
                by_pass[int(row["pass"])].append(row[f"delta_space_{group}"])
            passes = sorted(by_pass)
            axis.plot(
                passes,
                [statistics.mean(by_pass[p]) for p in passes],
                color=color,
                label=group,
            )
        axis.axhline(0, color="0.7", linewidth=1)
        axis.set(
            xlabel="Complete training-data passes",
            ylabel="Paired Δspace (shuffled − spatial MSE)",
            title=f"{policy}, N={n_sections}",
        )
        axis.legend(fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_compute(path: Path, arm_rows: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fields = (
        ("final_pass_mean", "Passes executed"),
        ("examples_processed_mean", "Examples processed"),
        ("run_elapsed_seconds_mean", "Wall-clock runtime (s)"),
        ("best_restored_loss_all_mean", "Best overall validation MSE"),
    )
    labels = ["N1 shuffled", "N1 spatial", "N48 shuffled", "N48 spatial"]
    x = list(range(len(labels)))
    width = 0.36
    for axis, (field, ylabel) in zip(axes.flat, fields, strict=True):
        for offset, policy in zip((-width / 2, width / 2), POLICIES, strict=True):
            policy_rows = {
                (row["n_train_sections"], row["condition"]): row
                for row in arm_rows
                if row["policy"] == policy
            }
            values = [
                policy_rows[(n, condition)][field]
                for n, condition in (
                    (1, "shuffled"),
                    (1, "spatial"),
                    (48, "shuffled"),
                    (48, "spatial"),
                )
            ]
            axis.bar(
                [value + offset for value in x],
                values,
                width,
                label=policy,
            )
        axis.set_xticks(x, labels, rotation=20, ha="right")
        axis.set_ylabel(ylabel)
        axis.legend(fontsize=7)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def summarize_lr_policy_diagnostic(
    entries: list[DataScalingEntry],
    historical_entries: list[DataScalingEntry],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Validate and summarize the fixed 24-run endpoint policy comparison."""
    if len(entries) != 24:
        raise ValueError(f"LR-policy grid must contain 24 runs; got {len(entries)}")
    expected = {
        (policy, n, condition, seed)
        for policy in POLICIES
        for n in (1, 48)
        for condition in ("shuffled", "spatial")
        for seed in (101, 202, 303)
    }
    actual = {
        (entry.training_protocol, entry.n_train_sections, entry.condition, entry.seed)
        for entry in entries
    }
    if actual != expected:
        raise ValueError("LR-policy grid does not match the required matrix")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"summary output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    runs = [_load_run(entry) for entry in entries]
    historical = [_load_run(entry, historical=True) for entry in historical_entries]
    integrity = _validate_integrity(runs)
    historical_reproduction = overlapping_history_reproduction(runs, historical)
    scheduler_reproduction = pre_reduction_reproduction(runs)
    run_summaries, tail_rows = [], []
    for run in runs:
        summary, tails = _run_summary(run)
        run_summaries.append(summary)
        tail_rows.extend(tails)
    trajectories = _trajectory_rows(runs)
    lr_events = _lr_event_rows(runs)
    policy_comparisons = _policy_comparison_rows(run_summaries)
    arm_summaries = _arm_summary_rows(run_summaries)
    delta_trajectories, delta_endpoints = _delta_rows(runs)
    _write_csv(output / "run_summary.csv", run_summaries)
    _write_csv(output / "tail_improvements.csv", tail_rows)
    _write_csv(output / "validation_trajectories.csv", trajectories)
    if lr_events:
        _write_csv(output / "lr_reduction_events.csv", lr_events)
    else:
        (output / "lr_reduction_events.csv").write_text(
            "policy,n_train_sections,condition,seed,event_index,epoch,step,"
            "validation_masked_mse_all,previous_learning_rate,new_learning_rate\n",
            encoding="utf-8",
        )
    _write_csv(output / "policy_comparisons.csv", policy_comparisons)
    _write_csv(output / "policy_arm_summary.csv", arm_summaries)
    _write_csv(output / "delta_trajectories.csv", delta_trajectories)
    _write_csv(output / "delta_endpoints.csv", delta_endpoints)
    _write_csv(output / "constant_historical_reproduction.csv", historical_reproduction)
    _write_csv(
        output / "plateau_pre_reduction_reproduction.csv", scheduler_reproduction
    )
    _plot_metric(output / "overall_validation_vs_pass.png", runs, "all")
    _plot_metric(output / "spatial_gene_validation_vs_pass.png", runs, "spatial")
    _plot_metric(output / "intrinsic_gene_validation_vs_pass.png", runs, "intrinsic")
    _plot_learning_rates(output / "learning_rate_vs_pass.png", runs)
    _plot_deltas(output / "delta_space_vs_pass.png", delta_trajectories)
    _plot_compute(output / "compute_to_frontier.png", arm_summaries)
    historical_max = max(
        abs(float(row[f"difference_{group}"]))
        for row in historical_reproduction
        for group in METRIC_GROUPS
    )
    scheduler_max = max(
        abs(float(row[f"difference_{group}"]))
        for row in scheduler_reproduction
        for group in METRIC_GROUPS
    )
    statuses = {status: 0 for status in CONVERGENCE_STATUSES}
    for row in run_summaries:
        statuses[row["convergence_status"]] += 1
    result = {
        "schema_version": "pilot-data-scaling-endpoint-lr-policy-summary-v1",
        "runs": len(runs),
        "policy_pairs": len(policy_comparisons),
        "matched_context_pairs": len(delta_endpoints),
        "convergence_status_counts": statuses,
        "historical_reproduction_max_absolute_difference": historical_max,
        "plateau_pre_reduction_max_absolute_difference": scheduler_max,
        "reproduction_tolerance": 1e-6,
        "historical_reproduction_within_tolerance": historical_max <= 1e-6,
        "plateau_pre_reduction_within_tolerance": scheduler_max <= 1e-6,
        "classification_rule": (
            "patience termination => EARLY_STOP_CONVERGED; maximum pass cap => "
            "CAP_REACHED_STILL_IMPROVING because no independent pre-existing "
            "effective-flat rule is introduced; non-finite => UNSTABLE; other "
            "termination => OTHER"
        ),
        "integrity": integrity,
        "output_dir": str(output),
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result
