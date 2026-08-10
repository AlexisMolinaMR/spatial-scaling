"""Aggregate controlled focal-cell masking runs into auditable artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def summarize_runs(run_dirs: list[Path], output_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    observation_hash: str | None = None
    controlled: dict[str, object] | None = None
    for run_dir in run_dirs:
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        config = json.loads(
            (run_dir / "resolved_config.json").read_text(encoding="utf-8")
        )
        current_controlled = {
            "seed": config["seed"],
            "model": config["model"],
            "optimizer": config["optimizer"],
            "training": config["training"],
            "evaluation": config["evaluation"],
            "train_sections": config["section_ids"]["train"],
        }
        if controlled is None:
            controlled = current_controlled
        elif current_controlled != controlled:
            raise ValueError(f"uncontrolled calibration difference in {run_dir}")
        current_hash = metrics["evaluation_observations"]["index_sha256"]
        if observation_hash is None:
            observation_hash = current_hash
        elif current_hash != observation_hash:
            raise ValueError("evaluation observations differ across calibration runs")
        final = metrics["final_validation"]
        rows.append(
            {
                "policy": config["masking"]["policy"],
                "fraction": config["masking"]["fraction"],
                "training_masked_mse": metrics["final_training_interval_masked_mse"],
                "validation_masked_mse_all": final["masked_mse_all"],
                "validation_masked_mse_intrinsic": final["masked_mse_intrinsic"],
                "validation_masked_mse_mixed": final["masked_mse_mixed"],
                "validation_masked_mse_spatial": final["masked_mse_spatial"],
                "validation_relative_improvement": metrics[
                    "validation_relative_improvement"
                ],
                "run_dir": str(run_dir),
            }
        )
    rows.sort(key=lambda item: float(item["fraction"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "calibration_metrics.csv"
    if csv_path.exists():
        raise FileExistsError(f"summary output already exists: {csv_path}")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "calibration_metadata.json").write_text(
        json.dumps(
            {
                "evaluation_index_sha256": observation_hash,
                "controlled_configuration": controlled,
                "runs": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    colors = {
        "all": "black",
        "intrinsic": "tab:orange",
        "mixed": "tab:green",
        "spatial": "tab:red",
    }
    for gene_class, color in colors.items():
        axis.plot(
            [100 * float(row["fraction"]) for row in rows],
            [float(row[f"validation_masked_mse_{gene_class}"]) for row in rows],
            marker="o",
            color=color,
            label=gene_class,
        )
    axis.set(
        title="Focal-cell masking calibration",
        xlabel="Masked genes (%)",
        ylabel="Validation masked MSE",
    )
    axis.legend()
    axis.tick_params(colors="black")
    axis.xaxis.label.set_color("black")
    axis.yaxis.label.set_color("black")
    axis.title.set_color("black")
    figure.savefig(output_dir / "calibration_losses.png", dpi=160)
    plt.close(figure)
    return rows
