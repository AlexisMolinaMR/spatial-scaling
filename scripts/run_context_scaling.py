"""Run one immutable task from a Pilot v0 context-scaling manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spatial_scaling.data.synthetic.context import ContextSelectionConfig
from spatial_scaling.experiments.context_scaling import load_manifest_entry
from spatial_scaling.training.config import SSLExperimentConfig
from spatial_scaling.training.trainer import run_experiment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    args = parser.parse_args()
    entry, manifest = load_manifest_entry(args.manifest, args.task_id)
    config = SSLExperimentConfig.from_dict(entry.resolved_config)
    selection = (
        ContextSelectionConfig.from_dict(entry.context_selection)
        if entry.context_selection is not None
        else None
    )
    summary = run_experiment(
        config,
        context_selection=selection,
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
