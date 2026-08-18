"""Endpoint convergence diagnostics for Pilot v0 training-data scaling."""

from __future__ import annotations

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
HISTORY_PREFIX = "validation_masked_mse_"
CONVERGENCE_STATUSES = {
    "EARLY_STOP_CONVERGED",
    "CAP_REACHED_STILL_IMPROVING",
    "CAP_REACHED_EFFECTIVELY_FLAT",
    "UNSTABLE_OR_OTHER",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _read_history(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    if not raw:
        raise ValueError(f"empty training history: {path}")
    fields = ("step", "epoch", *(f"{HISTORY_PREFIX}{g}" for g in METRIC_GROUPS))
    history = [{field: float(row[field]) for field in fields} for row in raw]
    epochs = [row["epoch"] for row in history]
    if epochs != sorted(epochs) or len(epochs) != len(set(epochs)):
        raise ValueError(f"history passes are not strictly increasing: {path}")
    return history


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


def tail_improvement_summary(
    history: list[dict[str, float]], group: str
) -> dict[str, Any]:
    """Calculate final-pass improvements without smoothing or fitting."""
    field = f"{HISTORY_PREFIX}{group}"
    completed = [row for row in history if row["epoch"] > 0]
    if len(completed) < 5:
        raise ValueError("tail diagnostics require at least five completed passes")
    by_epoch = {int(row["epoch"]): row for row in history}
    final_epoch = int(completed[-1]["epoch"])
    improvements = []
    for epoch in range(final_epoch - 4, final_epoch + 1):
        if epoch - 1 not in by_epoch or epoch not in by_epoch:
            raise ValueError("tail diagnostics require consecutive complete passes")
        improvements.append(
            {
                "pass": epoch,
                "improvement": by_epoch[epoch - 1][field] - by_epoch[epoch][field],
            }
        )
    values = [item["improvement"] for item in improvements]
    return {
        "improvements": improvements,
        "final_3_mean_improvement": statistics.mean(values[-3:]),
        "final_5_mean_improvement": statistics.mean(values),
        "final_5_max_absolute_improvement": max(map(abs, values)),
    }


def convergence_status(metrics: dict[str, Any]) -> str:
    """Map trainer termination to the conservative endpoint classification."""
    reason = metrics["early_stopping_reason"]
    if reason == "early_stopping_patience":
        status = "EARLY_STOP_CONVERGED"
    elif reason == "maximum_epochs":
        # No independent effective-flat rule existed before this diagnostic.
        status = "CAP_REACHED_STILL_IMPROVING"
    else:
        status = "UNSTABLE_OR_OTHER"
    if status not in CONVERGENCE_STATUSES:
        raise RuntimeError("invalid convergence classification")
    return status


def pass_reproduction_rows(
    current: list[dict[str, Any]], historical: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Compare pass-10 metrics for matched endpoint runs."""
    keys = ("n_train_sections", "condition", "seed")
    old_by_key = {tuple(row[key] for key in keys): row for row in historical}
    rows = []
    for row in current:
        key = tuple(row[name] for name in keys)
        if key not in old_by_key:
            raise ValueError(f"missing historical endpoint run: {key}")
        old = old_by_key[key]
        if row["training_data_selection"] != old["training_data_selection"]:
            raise ValueError(f"endpoint subset differs from historical run: {key}")
        if row["evaluation_observations"] != old["evaluation_observations"]:
            raise ValueError(f"endpoint validation observations differ: {key}")
        new_pass10 = row["history_by_pass"].get(10)
        old_pass10 = old["history_by_pass"].get(10)
        if new_pass10 is None or old_pass10 is None:
            raise ValueError(f"pass 10 missing from reproduction comparison: {key}")
        result = {
            "n_train_sections": key[0],
            "condition": key[1],
            "seed": key[2],
            "current_experiment_id": row["experiment_id"],
            "historical_experiment_id": old["experiment_id"],
            "historical_run_dir": old["run_dir"],
        }
        for group in METRIC_GROUPS:
            field = f"{HISTORY_PREFIX}{group}"
            result[f"current_pass10_{group}"] = new_pass10[field]
            result[f"historical_pass10_{group}"] = old_pass10[field]
            result[f"difference_{group}"] = new_pass10[field] - old_pass10[field]
        rows.append(result)
    return rows


def _load_run(entry: DataScalingEntry) -> dict[str, Any]:
    run_dir = Path(entry.output_dir)
    metrics = _read_json(run_dir / "metrics.json")
    history = _read_history(run_dir / "history.csv")
    selection = TrainingDataSelection.from_dict(
        metrics["training_data_selection"]
    ).to_dict()
    expected = TrainingDataSelection.from_dict(entry.training_data_selection).to_dict()
    if selection != expected:
        raise ValueError(f"training selection mismatch: {run_dir}")
    if metrics["experiment_id"] != entry.experiment_id:
        raise ValueError(f"experiment identity mismatch: {run_dir}")
    if metrics["config_identity_sha256"] != entry.config_identity_sha256:
        raise ValueError(f"configuration identity mismatch: {run_dir}")
    history_by_pass = {int(row["epoch"]): row for row in history}
    return {
        "n_train_sections": entry.n_train_sections,
        "n_train_cells": entry.n_train_cells,
        "condition": entry.condition,
        "seed": entry.seed,
        "experiment_id": entry.experiment_id,
        "run_dir": str(run_dir),
        "metrics": metrics,
        "history": history,
        "history_by_pass": history_by_pass,
        "training_data_selection": selection,
        "evaluation_observations": metrics["evaluation_observations"],
    }


def _run_summary(run: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics = run["metrics"]
    completed = [row for row in run["history"] if row["epoch"] > 0]
    final_row = completed[-1]
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
            "final_pass": int(final_row["epoch"]),
            "final_step": int(final_row["step"]),
            "best_pass": metrics["best_validation_epoch"],
            "best_step": metrics["best_validation_step"],
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
            "early_stopping_reason": metrics["early_stopping_reason"],
            "convergence_status": convergence_status(metrics),
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
        tail = tail_improvement_summary(run["history"], group)
        summary[f"pass10_loss_{group}"] = run["history_by_pass"][10][field]
        summary[f"final_pass_loss_{group}"] = final_row[field]
        summary[f"best_restored_loss_{group}"] = metrics["final_validation"][
            f"masked_mse_{group}"
        ]
        summary[f"final_3_mean_improvement_{group}"] = tail["final_3_mean_improvement"]
        summary[f"final_5_mean_improvement_{group}"] = tail["final_5_mean_improvement"]
        summary[f"final_5_max_absolute_improvement_{group}"] = tail[
            "final_5_max_absolute_improvement"
        ]
        for item in tail["improvements"]:
            tail_rows.append(
                {
                    "n_train_sections": run["n_train_sections"],
                    "condition": run["condition"],
                    "seed": run["seed"],
                    "experiment_id": run["experiment_id"],
                    "gene_group": group,
                    **item,
                }
            )
    return summary, tail_rows


def _trajectory_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        for item in run["history"]:
            rows.append(
                {
                    "n_train_sections": run["n_train_sections"],
                    "condition": run["condition"],
                    "seed": run["seed"],
                    "experiment_id": run["experiment_id"],
                    "pass": item["epoch"],
                    "step": int(item["step"]),
                    **{
                        f"validation_masked_mse_{group}": item[
                            f"{HISTORY_PREFIX}{group}"
                        ]
                        for group in METRIC_GROUPS
                    },
                }
            )
    return rows


def _paired_delta_trajectories(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for run in runs:
        grouped[(run["n_train_sections"], run["seed"])][run["condition"]] = run
    rows = []
    for (n_sections, seed), arms in sorted(grouped.items()):
        if set(arms) != {"spatial", "shuffled"}:
            raise ValueError(f"incomplete matched endpoint pair: {(n_sections, seed)}")
        spatial, shuffled = arms["spatial"], arms["shuffled"]
        common_passes = sorted(
            set(spatial["history_by_pass"]) & set(shuffled["history_by_pass"])
        )
        for epoch in common_passes:
            rows.append(
                {
                    "n_train_sections": n_sections,
                    "seed": seed,
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
    return rows


def _terminal_delta_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for run in runs:
        grouped[(run["n_train_sections"], run["seed"])][run["condition"]] = run
    rows = []
    for (n_sections, seed), arms in sorted(grouped.items()):
        spatial, shuffled = arms["spatial"], arms["shuffled"]
        row: dict[str, Any] = {"n_train_sections": n_sections, "seed": seed}
        for group in METRIC_GROUPS:
            field = f"{HISTORY_PREFIX}{group}"
            row[f"pass10_delta_space_{group}"] = (
                shuffled["history_by_pass"][10][field]
                - spatial["history_by_pass"][10][field]
            )
            row[f"terminal_restored_delta_space_{group}"] = (
                shuffled["metrics"]["final_validation"][f"masked_mse_{group}"]
                - spatial["metrics"]["final_validation"][f"masked_mse_{group}"]
            )
        rows.append(row)
    return rows


def _plot_arm_trajectories(
    path: Path, runs: list[dict[str, Any]], group: str, ylabel: str
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    colors = {"spatial": "tab:blue", "shuffled": "tab:purple"}
    for axis, n_sections in zip(axes, (1, 48), strict=True):
        for run in runs:
            if run["n_train_sections"] != n_sections:
                continue
            axis.plot(
                [row["epoch"] for row in run["history"]],
                [row[f"{HISTORY_PREFIX}{group}"] for row in run["history"]],
                color=colors[run["condition"]],
                alpha=0.75,
                label=f"{run['condition']} seed {run['seed']}",
            )
        axis.axvline(10, color="0.6", linestyle=":", linewidth=1)
        axis.set(
            xlabel="Complete training-data passes",
            ylabel=ylabel,
            title=f"N={n_sections}",
        )
        axis.legend(fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_deltas(path: Path, rows: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    colors = {
        "all": "black",
        "intrinsic": "tab:orange",
        "mixed": "tab:green",
        "spatial": "tab:red",
    }
    for axis, n_sections in zip(axes, (1, 48), strict=True):
        selected = [row for row in rows if row["n_train_sections"] == n_sections]
        for group, color in colors.items():
            for seed in (101, 202, 303):
                seed_rows = [row for row in selected if row["seed"] == seed]
                axis.plot(
                    [row["pass"] for row in seed_rows],
                    [row[f"delta_space_{group}"] for row in seed_rows],
                    color=color,
                    alpha=0.35,
                    linewidth=1,
                )
            by_pass: dict[int, list[float]] = defaultdict(list)
            for row in selected:
                by_pass[int(row["pass"])].append(row[f"delta_space_{group}"])
            axis.plot(
                sorted(by_pass),
                [statistics.mean(by_pass[p]) for p in sorted(by_pass)],
                color=color,
                linewidth=2,
                label=group,
            )
        axis.axvline(10, color="0.6", linestyle=":", linewidth=1)
        axis.axhline(0, color="0.7", linewidth=1)
        axis.set(
            xlabel="Complete training-data passes",
            ylabel="Paired Δspace (shuffled − spatial MSE)",
            title=f"N={n_sections}",
        )
        axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def summarize_endpoint_convergence(
    entries: list[DataScalingEntry],
    historical_entries: list[DataScalingEntry],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Validate, compare, aggregate, and plot all 12 endpoint runs."""
    if len(entries) != 12:
        raise ValueError(
            f"endpoint grid must contain exactly 12 runs; got {len(entries)}"
        )
    expected = {
        (n, condition, seed)
        for n in (1, 48)
        for condition in ("shuffled", "spatial")
        for seed in (101, 202, 303)
    }
    actual = {(e.n_train_sections, e.condition, e.seed) for e in entries}
    if actual != expected:
        raise ValueError("endpoint grid does not match the required 12-run matrix")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"summary output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    runs = [_load_run(entry) for entry in entries]
    historical = [_load_run(entry) for entry in historical_entries]
    run_summaries, tails = [], []
    for run in runs:
        summary, run_tails = _run_summary(run)
        run_summaries.append(summary)
        tails.extend(run_tails)
    trajectories = _trajectory_rows(runs)
    deltas = _paired_delta_trajectories(runs)
    terminal_deltas = _terminal_delta_rows(runs)
    reproduction = pass_reproduction_rows(runs, historical)
    _write_csv(output / "endpoint_run_summary.csv", run_summaries)
    _write_csv(output / "tail_improvements.csv", tails)
    _write_csv(output / "validation_trajectories.csv", trajectories)
    _write_csv(output / "paired_delta_trajectories.csv", deltas)
    _write_csv(output / "paired_terminal_deltas.csv", terminal_deltas)
    _write_csv(output / "pass10_reproduction.csv", reproduction)
    _plot_arm_trajectories(
        output / "endpoint_overall_validation_vs_pass.png",
        runs,
        "all",
        "Overall validation masked MSE",
    )
    _plot_arm_trajectories(
        output / "endpoint_spatial_gene_validation_vs_pass.png",
        runs,
        "spatial",
        "Spatial-gene validation masked MSE",
    )
    _plot_deltas(output / "endpoint_delta_space_vs_pass.png", deltas)
    maximum_reproduction_difference = max(
        abs(float(row[f"difference_{group}"]))
        for row in reproduction
        for group in METRIC_GROUPS
    )
    statuses = {status: 0 for status in sorted(CONVERGENCE_STATUSES)}
    for row in run_summaries:
        statuses[row["convergence_status"]] += 1
    result = {
        "schema_version": "pilot-data-scaling-endpoint-convergence-summary-v1",
        "runs": len(runs),
        "matched_pairs": len(terminal_deltas),
        "convergence_status_counts": statuses,
        "classification_rule": (
            "early_stopping_patience => EARLY_STOP_CONVERGED; maximum_epochs => "
            "CAP_REACHED_STILL_IMPROVING because no independent pre-existing "
            "effective-flat rule exists; other termination => UNSTABLE_OR_OTHER"
        ),
        "maximum_absolute_pass10_reproduction_difference": maximum_reproduction_difference,
        "pass10_reproduction_tolerance": 1e-6,
        "pass10_reproduction_within_tolerance": maximum_reproduction_difference <= 1e-6,
        "historical_results": "results/pilot_data_scaling_v0/runs",
        "output_dir": str(output),
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result
