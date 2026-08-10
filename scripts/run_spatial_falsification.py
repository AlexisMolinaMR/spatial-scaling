"""Run one traceable task from the matched Pilot v0 falsification matrix."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from spatial_scaling.training.config import load_experiment_config
from spatial_scaling.training.trainer import run_experiment


@dataclass(frozen=True)
class MatrixEntry:
    task_id: int
    world: str
    policy: str
    fraction: float
    regime: str
    condition: str
    seed: int

    @property
    def corpus_path(self) -> str:
        suffix = "spatial" if self.world == "positive" else "null"
        return f"outputs/pilot_v0_synthetic_{suffix}"

    @property
    def config_path(self) -> Path:
        name = (
            "ssl_focal.yaml"
            if self.condition == "focal"
            else ("ssl_spatial_transformer.yaml")
        )
        return Path("configs/pilot") / name

    @property
    def output_dir(self) -> str:
        return (
            "results/pilot_spatial_falsification/"
            f"{self.world}/{self.regime}/{self.condition}/seed_{self.seed}"
        )

    @property
    def experiment_name(self) -> str:
        return (
            f"pilot_spatial_{self.world}_{self.regime}_{self.condition}_"
            f"seed_{self.seed}"
        )


def matrix_entries() -> list[MatrixEntry]:
    entries: list[MatrixEntry] = []
    seeds = (101, 202, 303)
    positive_regimes = (
        ("random", 0.30, "random_030"),
        ("high_random", 0.85, "high_random_085"),
    )
    for policy, fraction, regime in positive_regimes:
        for condition in ("focal", "shuffled", "spatial"):
            for seed in seeds:
                entries.append(
                    MatrixEntry(
                        task_id=len(entries),
                        world="positive",
                        policy=policy,
                        fraction=fraction,
                        regime=regime,
                        condition=condition,
                        seed=seed,
                    )
                )
    for condition in ("shuffled", "spatial"):
        for seed in seeds:
            entries.append(
                MatrixEntry(
                    task_id=len(entries),
                    world="null",
                    policy="high_random",
                    fraction=0.85,
                    regime="high_random_085",
                    condition=condition,
                    seed=seed,
                )
            )
    return entries


def _overrides(entry: MatrixEntry) -> list[str]:
    return [
        f"experiment_name={entry.experiment_name}",
        f"output_dir={entry.output_dir}",
        f"seed={entry.seed}",
        f"condition={entry.condition}",
        f"data.corpus_path={entry.corpus_path}",
        f"context.shuffle_seed={entry.seed}",
        f"masking.policy={entry.policy}",
        f"masking.fraction={entry.fraction}",
        f"masking.seed={entry.seed}",
        "evaluation.max_cells=8192",
        "evaluation.observation_seed=991",
        f"evaluation.masking_seed={entry.seed}",
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--print-matrix", action="store_true")
    args = parser.parse_args()
    entries = matrix_entries()
    if args.print_matrix:
        print(json.dumps([asdict(entry) for entry in entries], indent=2))
        return 0
    if args.task_id is None or not 0 <= args.task_id < len(entries):
        parser.error(f"--task-id must be between 0 and {len(entries) - 1}")
    entry = entries[args.task_id]
    config = load_experiment_config(entry.config_path, _overrides(entry))
    summary = run_experiment(config)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
