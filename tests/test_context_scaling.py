from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from spatial_scaling.evaluation.context_scaling import (
    check_k256_anchor,
    paired_delta_rows,
    summarize_paired_deltas,
)
from spatial_scaling.evaluation.spatial_scale_calibration import (
    calibration_delta_summary_rows,
    calibration_paired_delta_rows,
)
from spatial_scaling.experiments.context_scaling import (
    context_scaling_entries,
    load_manifest_entry,
    write_manifest,
)
from spatial_scaling.models.spatial_transformer import (
    SpatialTransformer,
    SpatialTransformerConfig,
)
from spatial_scaling.spatial.neighborhoods import (
    kth_neighbor_radii_um,
    summarize_radii_um,
)

SPEC = Path("configs/pilot/context_scaling_v0.yaml")
CALIBRATION_SPEC = Path("configs/pilot/spatial_scale_calibration_v0.yaml")


def test_context_grid_has_expected_combinations_and_unique_ids() -> None:
    entries = context_scaling_entries(SPEC)
    counts = {
        phase: sum(entry.phase == phase for entry in entries)
        for phase in {entry.phase for entry in entries}
    }
    assert counts == {
        "primary_85": 36,
        "null_85": 18,
        "secondary_30": 18,
        "feasibility_2048": 2,
    }
    assert len({entry.experiment_id for entry in entries}) == len(entries)
    assert [entry.task_id for entry in entries] == list(range(len(entries)))
    repeated = context_scaling_entries(SPEC)
    assert [entry.experiment_id for entry in entries] == [
        entry.experiment_id for entry in repeated
    ]


def test_spatial_calibration_grid_has_low_k_and_distance_combinations() -> None:
    entries = context_scaling_entries(CALIBRATION_SPEC)
    counts = {
        phase: sum(entry.phase == phase for entry in entries)
        for phase in {entry.phase for entry in entries}
    }
    assert counts == {
        "low_k_positive_85": 30,
        "distance_positive_85": 48,
        "distance_null_85": 18,
    }
    assert len(entries) == len({entry.experiment_id for entry in entries}) == 96
    low_k = [entry for entry in entries if entry.phase == "low_k_positive_85"]
    assert {entry.context_size for entry in low_k} == {1, 2, 4, 8, 16}
    distances = [entry for entry in entries if entry.phase == "distance_positive_85"]
    assert {entry.context_selection["minimum_distance_um"] for entry in distances} == {
        0.0,
        50.0,
        100.0,
        150.0,
        200.0,
        300.0,
        400.0,
        600.0,
    }
    assert {
        entry.context_selection["eligibility_minimum_distance_um"]
        for entry in distances
    } == {600.0}
    repeated = context_scaling_entries(CALIBRATION_SPEC)
    assert [entry.experiment_id for entry in entries] == [
        entry.experiment_id for entry in repeated
    ]


def test_distance_manifest_preserves_selection_identity(tmp_path: Path) -> None:
    path = tmp_path / "distance_manifest.json"
    manifest = write_manifest(
        CALIBRATION_SPEC,
        path,
        phases={"distance_positive_85"},
        include_minimum_distances_um={0.0},
    )
    assert manifest["task_count"] == 6
    for task_id in range(6):
        entry, _ = load_manifest_entry(path, task_id)
        assert entry.context_selection == {
            "mode": "distance_exclusion",
            "minimum_distance_um": 0.0,
            "eligibility_minimum_distance_um": 600.0,
        }


def test_distance_grid_pairs_are_matched_except_condition() -> None:
    entries = context_scaling_entries(
        CALIBRATION_SPEC,
        phases={"distance_positive_85"},
        include_minimum_distances_um={200.0},
    )
    assert len(entries) == 6
    by_seed = {}
    for entry in entries:
        by_seed.setdefault(entry.seed, {})[entry.condition] = entry
    for pair in by_seed.values():
        assert set(pair) == {"shuffled", "spatial"}
        assert pair["shuffled"].context_selection == pair["spatial"].context_selection
        left = dict(pair["shuffled"].resolved_config)
        right = dict(pair["spatial"].resolved_config)
        for values in (left, right):
            values.pop("condition")
            values.pop("experiment_name")
            values.pop("output_dir")
        assert left == right


def test_positive_and_null_metadata_are_distinct_and_pairs_are_matched() -> None:
    entries = context_scaling_entries(
        SPEC, phases={"primary_85", "null_85"}, include_context_sizes={256}
    )
    assert {entry.world for entry in entries} == {"positive", "null"}
    assert {entry.resolved_config["data"]["corpus_path"] for entry in entries} == {
        "outputs/pilot_v0_synthetic_spatial",
        "outputs/pilot_v0_synthetic_null",
    }
    by_key = {}
    for entry in entries:
        key = (entry.world, entry.masking_regime, entry.context_size, entry.seed)
        by_key.setdefault(key, {})[entry.condition] = entry
    for pair in by_key.values():
        assert set(pair) == {"shuffled", "spatial"}
        left = dict(pair["shuffled"].resolved_config)
        right = dict(pair["spatial"].resolved_config)
        for values in (left, right):
            values.pop("condition")
            values.pop("experiment_name")
            values.pop("output_dir")
        assert left == right


def test_manifest_array_index_is_recorded_and_stable(tmp_path: Path) -> None:
    path = tmp_path / "anchor_manifest.json"
    first = write_manifest(
        SPEC,
        path,
        phases={"primary_85"},
        include_context_sizes={256},
    )
    assert first["task_count"] == 6
    for task_id in range(6):
        entry, manifest = load_manifest_entry(path, task_id)
        assert entry.task_id == task_id
        assert manifest["entries"][task_id]["experiment_id"] == entry.experiment_id
    with pytest.raises(FileExistsError, match="already exists"):
        write_manifest(SPEC, path, phases={"primary_85"})


def test_k_changes_shape_but_not_parameter_count() -> None:
    base = SpatialTransformerConfig(num_genes=256, context_size=32)
    small = SpatialTransformer(base)
    large = SpatialTransformer(replace(base, context_size=1024))
    assert small.config.context_size == 32
    assert large.config.context_size == 1024
    assert small.parameter_count == large.parameter_count == 121024


def test_kth_neighbor_radius_and_summary_on_toy_geometry() -> None:
    coordinates = np.asarray(
        [[0.0, 0.0], [3.0, 0.0], [0.0, 4.0], [6.0, 0.0], [0.0, 8.0]]
    )
    radii = kth_neighbor_radii_um(coordinates, np.asarray([0, 1]), [1, 2, 3])
    np.testing.assert_allclose(radii[1], [3.0, 3.0])
    np.testing.assert_allclose(radii[2], [4.0, 3.0])
    np.testing.assert_allclose(radii[3], [6.0, 5.0])
    summary = summarize_radii_um(np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert summary == {
        "median_radius_um": 3.0,
        "p25_radius_um": 2.0,
        "p75_radius_um": 4.0,
        "p05_radius_um": pytest.approx(1.2),
        "p95_radius_um": pytest.approx(4.8),
    }


def _run_row(condition: str, seed: int, loss: float) -> dict[str, object]:
    return {
        "world": "positive",
        "masking_regime": "high_random_085",
        "context_size": 64,
        "seed": seed,
        "condition": condition,
        "parameter_count": 121024,
        "optimization_steps": 2000,
        "training_batch_size": 32,
        "effective_training_examples": 64000,
        "evaluation_index_sha256": "observations",
        "evaluation_mask_sha256": f"mask-{seed}",
        "validation_observation_count": 8192,
        "matched_config": {"seed": seed, "context_size": 64},
        **{
            f"validation_masked_mse_{group}": loss
            for group in ("all", "intrinsic", "mixed", "spatial")
        },
    }


def test_paired_delta_is_seed_level_then_aggregated() -> None:
    rows = []
    for seed, spatial_loss, shuffled_loss in (
        (101, 1.0, 1.3),
        (202, 2.0, 2.5),
        (303, 4.0, 4.7),
    ):
        rows.extend(
            [
                _run_row("spatial", seed, spatial_loss),
                _run_row("shuffled", seed, shuffled_loss),
            ]
        )
    paired = paired_delta_rows(rows)
    assert [row["delta_space_spatial"] for row in paired] == pytest.approx(
        [0.3, 0.5, 0.7]
    )
    summary = summarize_paired_deltas(paired, required_seeds=3)[0]
    assert summary["delta_space_spatial_mean"] == pytest.approx(0.5)
    assert summary["delta_space_spatial_sample_std"] == pytest.approx(0.2)
    assert summary["delta_space_spatial_positive_seeds"] == 3


def test_calibration_pairing_includes_phase_and_distance() -> None:
    rows = []
    for distance in (0.0, 200.0):
        for seed, spatial_loss, shuffled_loss in (
            (101, 1.0, 1.3),
            (202, 1.1, 1.5),
            (303, 1.2, 1.7),
        ):
            for condition, loss in (
                ("spatial", spatial_loss),
                ("shuffled", shuffled_loss),
            ):
                row = _run_row(condition, seed, loss)
                row.update(
                    {
                        "phase": "distance_positive_85",
                        "minimum_distance_um": distance,
                        "requested_validation_observation_count": 8192,
                        "train_common_threshold_eligible": 100,
                        "evaluation_common_threshold_eligible": 20,
                        "reused": False,
                    }
                )
                rows.append(row)
    paired = calibration_paired_delta_rows(rows)
    assert len(paired) == 6
    summaries = calibration_delta_summary_rows(paired)
    assert len(summaries) == 2
    assert {row["minimum_distance_um"] for row in summaries} == {0.0, 200.0}


def test_computational_metrics_are_json_serializable() -> None:
    metrics = {
        "run_elapsed_seconds": 12.5,
        "training_step_time_seconds_mean": 0.01,
        "training_step_time_seconds_median": 0.009,
        "peak_device_memory_allocated_bytes": 123456,
        "examples_per_second": 3200.0,
        "context_tokens_per_second": 105600.0,
    }
    assert json.loads(json.dumps(metrics)) == metrics


def test_anchor_check_uses_seed_level_reference(tmp_path: Path) -> None:
    new = tmp_path / "new.csv"
    reference = tmp_path / "reference.csv"
    new.write_text(
        "world,masking_regime,context_size,seed,parameter_count,"
        "delta_space_all,delta_space_intrinsic,delta_space_mixed,delta_space_spatial\n"
        + "".join(
            f"positive,high_random_085,256,{seed},121024,0.22,0.0,0.22,0.45\n"
            for seed in (101, 202, 303)
        ),
        encoding="utf-8",
    )
    reference.write_text(
        "world,regime,seed,parameter_count,delta_space_all,"
        "delta_space_intrinsic,delta_space_mixed,delta_space_spatial\n"
        + "".join(
            f"positive,high_random_085,{seed},121024,0.22,0.0,0.22,0.45\n"
            for seed in (101, 202, 303)
        ),
        encoding="utf-8",
    )
    assert check_k256_anchor(new, reference)["passed"]
