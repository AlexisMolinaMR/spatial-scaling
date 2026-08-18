"""Run one immutable Pilot v0 N-train scaling task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spatial_scaling.experiments.data_scaling import load_manifest_entry
from spatial_scaling.training.config import SSLExperimentConfig
from spatial_scaling.training.policies import (
    TrainingDataSelection,
    TrainingExecutionPolicy,
)
from spatial_scaling.training.trainer import run_experiment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    args = parser.parse_args()
    entry, manifest = load_manifest_entry(args.manifest, args.task_id)
    summary = run_experiment(
        SSLExperimentConfig.from_dict(entry.resolved_config),
        training_data_selection=TrainingDataSelection.from_dict(
            entry.training_data_selection
        ),
        execution_policy=TrainingExecutionPolicy.from_dict(entry.execution_policy),
        expected_config_identity_sha256=entry.config_identity_sha256,
    )
    print(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "manifest_spec_sha256": manifest["spec_sha256"],
                "task_id": entry.task_id,
                "experiment_id": entry.experiment_id,
                "summary": summary,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
