from __future__ import annotations

import csv
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import yaml

from spatial_scaling.data.synthetic.config import SyntheticConfig
from spatial_scaling.data.synthetic.context import SyntheticContextDataset
from spatial_scaling.data.synthetic.dataset import SyntheticExpressionDataset
from spatial_scaling.data.synthetic.generator import SyntheticCorpusGenerator
from spatial_scaling.data.synthetic.storage import (
    prepare_output_directory,
    write_corpus_metadata,
    write_gene_ground_truth,
    write_section,
)
from spatial_scaling.evaluation.converged_data_frontier import (
    adjacent_improvement_rows,
    validate_final_grids,
)
from spatial_scaling.evaluation.converged_data_frontier import (
    paired_delta_rows as converged_frontier_paired_delta_rows,
)
from spatial_scaling.evaluation.data_scaling import (
    delta_summary_rows,
    paired_delta_rows,
)
from spatial_scaling.evaluation.endpoint_convergence import (
    convergence_status,
    pass_reproduction_rows,
    tail_improvement_summary,
)
from spatial_scaling.evaluation.lr_policy_diagnostic import (
    overlapping_history_reproduction,
    post_last_reduction_tail,
)
from spatial_scaling.evaluation.spatial_90pass_convergence import (
    paired_delta_rows as spatial_90pass_paired_delta_rows,
)
from spatial_scaling.evaluation.spatial_90pass_convergence import (
    post_final_reduction_tail,
    restart_reproduction_rows,
)
from spatial_scaling.experiments.data_scaling import (
    data_scaling_entries,
    load_manifest_entry,
    nested_training_subsets,
    ordered_training_sections,
    write_manifest,
)
from spatial_scaling.models.spatial_transformer import (
    SpatialTransformer,
    SpatialTransformerConfig,
)
from spatial_scaling.training.config import (
    DataConfig,
    EvaluationConfig,
    OptimizerConfig,
    SSLExperimentConfig,
    TrainLoopConfig,
)
from spatial_scaling.training.masking import MaskingConfig
from spatial_scaling.training.policies import (
    TrainingDataSelection,
    TrainingExecutionPolicy,
    ValidationConvergenceController,
    ValidationPlateauController,
)
from spatial_scaling.training.trainer import run_experiment

PRODUCTION_SPEC = Path("configs/pilot/data_scaling_v0.yaml")
ENDPOINT_SPEC = Path("configs/pilot/data_scaling_endpoint_convergence_v0.yaml")
LR_POLICY_SPEC = Path("configs/pilot/data_scaling_endpoint_lr_policy_v0.yaml")
SPATIAL_90PASS_SPEC = Path(
    "configs/pilot/data_scaling_spatial_90pass_convergence_v0.yaml"
)
CONVERGED_FRONTIER_SPEC = Path("configs/pilot/data_scaling_converged_frontier_v0.yaml")
SCALE_POINTS = [1, 2, 4, 8, 16, 32, 48]


def _metadata_only_corpus(path: Path) -> Path:
    path.mkdir()
    train = [f"section_{index:04d}" for index in range(48)]
    validation = [f"section_{index:04d}" for index in range(48, 56)]
    test = [f"section_{index:04d}" for index in range(56, 64)]
    metadata = {
        "split_definition": {
            "train": train,
            "validation": validation,
            "test": test,
        },
        "cells_per_section": 8192,
    }
    (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return path


def _toy_spec(tmp_path: Path) -> Path:
    positive = _metadata_only_corpus(tmp_path / "positive")
    null = _metadata_only_corpus(tmp_path / "null")
    values = yaml.safe_load(PRODUCTION_SPEC.read_text(encoding="utf-8"))
    values["corpora"] = {"positive": str(positive), "null": str(null)}
    values["output_root"] = str(tmp_path / "runs")
    path = tmp_path / "data_scaling.yaml"
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return path


def _toy_endpoint_spec(tmp_path: Path) -> Path:
    positive = _metadata_only_corpus(tmp_path / "endpoint-positive")
    null = _metadata_only_corpus(tmp_path / "endpoint-null")
    values = yaml.safe_load(ENDPOINT_SPEC.read_text(encoding="utf-8"))
    values["corpora"] = {"positive": str(positive), "null": str(null)}
    values["output_root"] = str(tmp_path / "endpoint-runs")
    path = tmp_path / "endpoint.yaml"
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return path


def _toy_lr_policy_spec(tmp_path: Path) -> Path:
    positive = _metadata_only_corpus(tmp_path / "lr-positive")
    null = _metadata_only_corpus(tmp_path / "lr-null")
    values = yaml.safe_load(LR_POLICY_SPEC.read_text(encoding="utf-8"))
    values["corpora"] = {"positive": str(positive), "null": str(null)}
    values["output_root"] = str(tmp_path / "lr-runs")
    path = tmp_path / "lr-policy.yaml"
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return path


def _toy_spatial_90pass_spec(tmp_path: Path) -> Path:
    positive = _metadata_only_corpus(tmp_path / "spatial90-positive")
    null = _metadata_only_corpus(tmp_path / "spatial90-null")
    values = yaml.safe_load(SPATIAL_90PASS_SPEC.read_text(encoding="utf-8"))
    values["corpora"] = {"positive": str(positive), "null": str(null)}
    values["output_root"] = str(tmp_path / "spatial90-runs")
    path = tmp_path / "spatial90.yaml"
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return path


def _toy_converged_frontier_spec(tmp_path: Path) -> Path:
    positive = _metadata_only_corpus(tmp_path / "frontier-positive")
    null = _metadata_only_corpus(tmp_path / "frontier-null")
    values = yaml.safe_load(CONVERGED_FRONTIER_SPEC.read_text(encoding="utf-8"))
    values["corpora"] = {"positive": str(positive), "null": str(null)}
    values["output_root"] = str(tmp_path / "frontier-runs")
    path = tmp_path / "converged-frontier.yaml"
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return path


def test_nested_subsets_are_exact_deterministic_and_seed_separate() -> None:
    train_ids = tuple(f"section_{index:04d}" for index in range(48))
    first = nested_training_subsets(train_ids, SCALE_POINTS, 314159)
    repeated = nested_training_subsets(train_ids, SCALE_POINTS, 314159)
    changed = nested_training_subsets(train_ids, SCALE_POINTS, 314160)
    assert first == repeated
    assert first != changed
    assert [len(first[point]) for point in SCALE_POINTS] == SCALE_POINTS
    for smaller, larger in pairwise(SCALE_POINTS):
        assert first[smaller] == first[larger][:smaller]
    assert set(first[48]) == set(train_ids)
    assert ordered_training_sections(train_ids, 314159) == first[48]


def test_training_selection_survives_json_round_trip() -> None:
    section_ids = tuple(f"section_{index:04d}" for index in range(4))
    selection = TrainingDataSelection(
        scaling_axis="N_train_sections",
        subset_selection_seed=314159,
        ordered_train_section_ids=section_ids,
        active_train_section_ids=section_ids[:2],
    )
    serialized = json.loads(json.dumps(selection.to_dict()))
    assert TrainingDataSelection.from_dict(serialized).to_dict() == selection.to_dict()


def test_grid_has_required_pairs_and_distinct_policies(tmp_path: Path) -> None:
    entries = data_scaling_entries(_toy_spec(tmp_path))
    counts = {
        phase: sum(entry.phase == phase for entry in entries)
        for phase in {entry.phase for entry in entries}
    }
    assert counts == {
        "primary_positive_85": 42,
        "required_null_85": 18,
        "fixed_compute_positive_85": 18,
        "focal_reference_85": 21,
        "smoke_positive_85": 2,
    }
    primary = [entry for entry in entries if entry.phase == "primary_positive_85"]
    by_key = {}
    for entry in primary:
        by_key.setdefault((entry.n_train_sections, entry.seed), set()).add(
            entry.condition
        )
        assert entry.n_train_cells == entry.n_train_sections * 8192
        assert entry.scaling_axis == "N_train_sections"
        assert len(entry.training_data_selection["active_train_section_ids"]) == (
            entry.n_train_sections
        )
    assert all(conditions == {"spatial", "shuffled"} for conditions in by_key.values())
    assert {entry.training_protocol for entry in primary} == {"converged_data_frontier"}
    fixed = [entry for entry in entries if entry.phase == "fixed_compute_positive_85"]
    assert {entry.training_protocol for entry in fixed} == {"fixed_compute_2000_steps"}
    assert {entry.execution_policy["fixed_steps"] for entry in fixed} == {2000}


def test_training_seed_does_not_change_subset_and_split_cannot_leak(
    tmp_path: Path,
) -> None:
    entries = data_scaling_entries(
        _toy_spec(tmp_path), phases={"primary_positive_85"}, n_train_sections={8}
    )
    memberships = {
        tuple(entry.training_data_selection["active_train_section_ids"])
        for entry in entries
    }
    assert len(memberships) == 1
    membership = next(iter(memberships))
    assert all(int(section.split("_")[1]) < 48 for section in membership)
    assert not set(membership) & {
        *(f"section_{index:04d}" for index in range(48, 56)),
        *(f"section_{index:04d}" for index in range(56, 64)),
    }


def test_manifest_identity_and_experiment_ids_are_stable(tmp_path: Path) -> None:
    spec = _toy_spec(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = write_manifest(spec, path, phases={"required_null_85"})
    assert manifest["task_count"] == 18
    loaded = [load_manifest_entry(path, task_id)[0] for task_id in range(18)]
    assert [entry.task_id for entry in loaded] == list(range(18))
    repeated = data_scaling_entries(spec, phases={"required_null_85"})
    assert [entry.experiment_id for entry in loaded] == [
        entry.experiment_id for entry in repeated
    ]


def _small_config() -> SyntheticConfig:
    return SyntheticConfig.from_dict(
        {
            "master_seed": 91,
            "num_sections": 4,
            "cells_per_section": 64,
            "num_genes": 24,
            "lambda_s": 1.0,
            "latent": {"intrinsic_dim": 4, "spatial_dim": 4},
            "spatial_field": {
                "correlation_length_um": 120.0,
                "num_fourier_features": 12,
            },
            "gene_classes": {"intrinsic": 8, "mixed": 8, "spatial": 8},
            "splits": {"train": 2, "validation": 1, "test": 1},
            "qc": {"max_sections": 4},
        }
    )


@pytest.fixture(scope="module")
def small_corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    generator = SyntheticCorpusGenerator(_small_config())
    output = prepare_output_directory(tmp_path_factory.mktemp("data-scale") / "corpus")
    write_corpus_metadata(output, generator)
    write_gene_ground_truth(output, generator)
    for section in generator.generate_sections():
        write_section(output, section)
    return output


def test_active_dataset_and_context_cannot_access_excluded_sections(
    small_corpus: Path,
) -> None:
    full = SyntheticExpressionDataset(small_corpus, "train")
    active_id, excluded_id = full.section_ids
    active = SyntheticExpressionDataset(
        small_corpus, "train", section_ids=(active_id,), cache_sections=1
    )
    assert active.section_ids == (active_id,)
    assert len(active) == active.cells_per_section
    context = SyntheticContextDataset(
        active, context_size=8, condition="spatial", shuffle_seed=101
    )
    batch = context.context_batch(0, np.asarray([0, 1, 2]))
    assert batch.section_id == active_id
    assert all(
        cell_id.startswith(active_id)
        for row in batch.context_cell_ids
        for cell_id in row
    )
    assert all(
        not cell_id.startswith(excluded_id)
        for row in batch.context_cell_ids
        for cell_id in row
    )
    with pytest.raises(ValueError, match="outside the fixed"):
        SyntheticExpressionDataset(
            small_corpus,
            "train",
            section_ids=(
                SyntheticExpressionDataset(small_corpus, "validation").section_ids[0],
            ),
        )


def _selection(small_corpus: Path, count: int) -> TrainingDataSelection:
    train_ids = SyntheticExpressionDataset(small_corpus, "train").section_ids
    return TrainingDataSelection(
        scaling_axis="N_train_sections",
        subset_selection_seed=7,
        ordered_train_section_ids=train_ids,
        active_train_section_ids=train_ids[:count],
    )


def _tiny_focal_config(corpus: Path, output: Path) -> SSLExperimentConfig:
    return SSLExperimentConfig(
        experiment_name=output.name,
        output_dir=str(output),
        seed=13,
        device="cpu",
        condition="focal",
        data=DataConfig(corpus_path=str(corpus), cache_sections=1),
        model={"hidden_dims": [16], "activation": "relu", "dropout": 0.0},
        masking=MaskingConfig(policy="high_random", fraction=0.85, seed=13),
        optimizer=OptimizerConfig(learning_rate=1e-3, weight_decay=0.0),
        training=TrainLoopConfig(
            batch_size=16, steps=4, evaluation_interval=2, mask_epoch_steps=2
        ),
        evaluation=EvaluationConfig(
            batch_size=32,
            max_cells=32,
            observation_seed=991,
            masking_seed=992,
        ),
    )


def test_minimum_exposure_policy_and_fixed_validation_across_n(
    small_corpus: Path, tmp_path: Path
) -> None:
    policy = TrainingExecutionPolicy(
        name="smoke_minimum_exposure",
        sampling="deterministic_shuffled_epochs",
        minimum_epochs=1,
        maximum_epochs=1,
        validation_interval_epochs=1,
        early_stopping_patience=1,
        improvement_threshold=0.0,
        absolute_max_steps=100,
    )
    one = run_experiment(
        _tiny_focal_config(small_corpus, tmp_path / "one"),
        training_data_selection=_selection(small_corpus, 1),
        execution_policy=policy,
    )
    two = run_experiment(
        _tiny_focal_config(small_corpus, tmp_path / "two"),
        training_data_selection=_selection(small_corpus, 2),
        execution_policy=policy,
    )
    for metrics in (one, two):
        assert metrics["minimum_exposure_satisfied"]
        assert metrics["unique_observation_fraction"] == 1.0
        assert metrics["effective_passes"] == 1.0
        assert metrics["early_stopping_reason"] == "maximum_epochs"
    assert one["evaluation_observations"] == two["evaluation_observations"]
    assert one["parameter_count"] == two["parameter_count"]


def _paired_row(condition: str, seed: int, n_sections: int, loss: float) -> dict:
    selection = {
        "active_train_section_ids": [f"section_{i}" for i in range(n_sections)]
    }
    return {
        "phase": "primary_positive_85",
        "world": "positive",
        "training_protocol": "converged_data_frontier",
        "n_train_sections": n_sections,
        "n_train_cells": n_sections * 8192,
        "condition": condition,
        "seed": seed,
        "parameter_count": 121024,
        "evaluation_index_sha256": "observations",
        "evaluation_mask_sha256": "masks",
        "validation_observation_count": 8192,
        "subset_selection_seed": 314159,
        "active_train_section_ids": "sections",
        "ordered_train_section_ids": "order",
        "execution_policy": {"name": "converged_data_frontier"},
        "training_data_selection": selection,
        "matched_config": {"seed": seed, "n": n_sections},
        **{
            f"validation_masked_mse_{group}": loss
            for group in ("all", "intrinsic", "mixed", "spatial")
        },
    }


def test_paired_delta_is_computed_before_seed_summary() -> None:
    rows = []
    for seed, spatial, shuffled in ((101, 1.0, 1.3), (202, 1.1, 1.5), (303, 1.2, 1.7)):
        rows.extend(
            (
                _paired_row("spatial", seed, 8, spatial),
                _paired_row("shuffled", seed, 8, shuffled),
            )
        )
    paired = paired_delta_rows(rows)
    assert [row["delta_space_spatial"] for row in paired] == pytest.approx(
        [0.3, 0.4, 0.5]
    )
    summary = delta_summary_rows(paired)[0]
    assert summary["delta_space_spatial_mean"] == pytest.approx(0.4)
    assert summary["delta_space_spatial_sample_std"] == pytest.approx(0.1)
    assert summary["delta_space_spatial_positive_seeds"] == 3


def test_contextual_parameter_count_is_fixed_across_n() -> None:
    counts = {
        SpatialTransformer(
            SpatialTransformerConfig(num_genes=256, context_size=32)
        ).parameter_count
        for _ in SCALE_POINTS
    }
    assert counts == {121024}


def test_endpoint_grid_is_exact_unique_and_changes_only_cap_metadata(
    tmp_path: Path,
) -> None:
    endpoint = data_scaling_entries(_toy_endpoint_spec(tmp_path))
    assert len(endpoint) == 12
    assert len({entry.experiment_id for entry in endpoint}) == 12
    assert {
        (entry.n_train_sections, entry.condition, entry.seed) for entry in endpoint
    } == {
        (n, condition, seed)
        for n in (1, 48)
        for condition in ("shuffled", "spatial")
        for seed in (101, 202, 303)
    }
    assert {entry.execution_policy["maximum_epochs"] for entry in endpoint} == {30}
    assert {entry.execution_policy["absolute_max_steps"] for entry in endpoint} == {
        400000
    }


def test_endpoint_manifest_links_historical_capped_experiment(tmp_path: Path) -> None:
    manifest = write_manifest(
        _toy_endpoint_spec(tmp_path), tmp_path / "endpoint-manifest.json"
    )
    assert manifest["provenance"]["historical_manifest"] == (
        "results/pilot_data_scaling_v0/manifests/primary.json"
    )
    assert manifest["provenance"]["historical_phase"] == "primary_positive_85"


def test_endpoint_cap_does_not_change_subsets_or_validation_definition(
    tmp_path: Path,
) -> None:
    old = data_scaling_entries(
        _toy_spec(tmp_path),
        phases={"primary_positive_85"},
        n_train_sections={1, 48},
    )
    endpoint = data_scaling_entries(_toy_endpoint_spec(tmp_path))
    old_by_key = {
        (entry.n_train_sections, entry.condition, entry.seed): entry for entry in old
    }
    for entry in endpoint:
        key = (entry.n_train_sections, entry.condition, entry.seed)
        historical = old_by_key[key]
        assert entry.training_data_selection == historical.training_data_selection
        assert (
            entry.resolved_config["evaluation"]
            == historical.resolved_config["evaluation"]
        )
        assert entry.resolved_config["masking"] == historical.resolved_config["masking"]
        assert entry.resolved_config["model"] == historical.resolved_config["model"]


def test_endpoint_status_serialization_is_conservative() -> None:
    assert (
        convergence_status({"early_stopping_reason": "early_stopping_patience"})
        == "EARLY_STOP_CONVERGED"
    )
    assert (
        convergence_status({"early_stopping_reason": "maximum_epochs"})
        == "CAP_REACHED_STILL_IMPROVING"
    )
    assert (
        convergence_status({"early_stopping_reason": "absolute_max_steps"})
        == "UNSTABLE_OR_OTHER"
    )


def test_convergence_tail_calculations() -> None:
    history = [
        {
            "epoch": float(epoch),
            "validation_masked_mse_all": 1.0 - 0.01 * epoch,
        }
        for epoch in range(8)
    ]
    tail = tail_improvement_summary(history, "all")
    assert [row["pass"] for row in tail["improvements"]] == [3, 4, 5, 6, 7]
    assert [row["improvement"] for row in tail["improvements"]] == pytest.approx(
        [0.01] * 5
    )
    assert tail["final_3_mean_improvement"] == pytest.approx(0.01)
    assert tail["final_5_mean_improvement"] == pytest.approx(0.01)
    assert tail["final_5_max_absolute_improvement"] == pytest.approx(0.01)


def test_historical_pass10_reproduction_comparison() -> None:
    selection = {
        "scaling_axis": "N_train_sections",
        "subset_selection_seed": 314159,
        "ordered_train_section_ids": ["section_0013"],
        "active_train_section_ids": ["section_0013"],
    }
    observations = {"index_sha256": "indices", "mask_sha256": "masks"}
    base = {
        "n_train_sections": 1,
        "condition": "spatial",
        "seed": 101,
        "training_data_selection": selection,
        "evaluation_observations": observations,
        "run_dir": "old",
    }
    metric_row = {
        f"validation_masked_mse_{group}": 0.5
        for group in ("all", "intrinsic", "mixed", "spatial")
    }
    current = [{**base, "experiment_id": "new", "history_by_pass": {10: metric_row}}]
    historical = [{**base, "experiment_id": "old", "history_by_pass": {10: metric_row}}]
    comparison = pass_reproduction_rows(current, historical)[0]
    assert comparison["difference_all"] == 0.0
    assert comparison["difference_spatial"] == 0.0


def test_plateau_policy_serialization_and_lr_floor() -> None:
    policy = TrainingExecutionPolicy(
        name="validation_plateau_lr_decay",
        sampling="deterministic_shuffled_epochs",
        lr_schedule="validation_plateau_decay",
        minimum_epochs=2,
        maximum_epochs=60,
        validation_interval_epochs=1,
        early_stopping_patience=8,
        improvement_threshold=1e-4,
        absolute_max_steps=800000,
        lr_reduction_factor=0.3,
        lr_reduction_patience=3,
        minimum_learning_rate=1e-5,
    )
    serialized = json.loads(json.dumps(policy.to_dict()))
    assert TrainingExecutionPolicy.from_dict(serialized).to_dict() == policy.to_dict()
    controller = ValidationPlateauController(0.3, 1, 1e-5)
    lr, reduced = controller.update(improved=False, current_learning_rate=2e-5)
    assert reduced
    assert lr == 1e-5
    lr, reduced = controller.update(improved=False, current_learning_rate=lr)
    assert not reduced
    assert lr == 1e-5


def test_plateau_and_early_stop_interaction_is_deterministic() -> None:
    controller = ValidationConvergenceController(
        best_loss=1.0,
        improvement_threshold=0.01,
        early_stopping_patience=3,
        plateau=ValidationPlateauController(0.5, 2, 0.25),
    )
    lr = 1.0
    decisions = []
    for loss in (0.995,) * 7:
        decision = controller.update(validation_loss=loss, current_learning_rate=lr)
        decisions.append(decision)
        lr = decision.learning_rate
    assert [decision.lr_reduced for decision in decisions] == [
        False,
        True,
        False,
        True,
        False,
        False,
        False,
    ]
    assert not decisions[1].should_stop
    assert not decisions[3].should_stop
    assert decisions[-1].should_stop
    assert lr == 0.25


def test_lr_policy_grid_has_24_unique_policy_identified_runs(tmp_path: Path) -> None:
    entries = data_scaling_entries(_toy_lr_policy_spec(tmp_path))
    assert len(entries) == 24
    assert len({entry.experiment_id for entry in entries}) == 24
    assert {
        (entry.training_protocol, entry.n_train_sections, entry.condition, entry.seed)
        for entry in entries
    } == {
        (policy, n, condition, seed)
        for policy in ("prolonged_constant_lr", "validation_plateau_lr_decay")
        for n in (1, 48)
        for condition in ("shuffled", "spatial")
        for seed in (101, 202, 303)
    }
    assert all(
        ("constant" in entry.experiment_id)
        if entry.training_protocol == "prolonged_constant_lr"
        else ("plateau_decay" in entry.experiment_id)
        for entry in entries
    )


def test_lr_policies_preserve_subsets_masks_and_validation(tmp_path: Path) -> None:
    entries = data_scaling_entries(_toy_lr_policy_spec(tmp_path))
    grouped = {}
    for entry in entries:
        grouped.setdefault(
            (entry.n_train_sections, entry.condition, entry.seed), []
        ).append(entry)
    for pair in grouped.values():
        assert len(pair) == 2
        assert pair[0].training_data_selection == pair[1].training_data_selection
        assert (
            pair[0].resolved_config["evaluation"]
            == pair[1].resolved_config["evaluation"]
        )
        assert pair[0].resolved_config["masking"] == pair[1].resolved_config["masking"]
        assert pair[0].resolved_config["model"] == pair[1].resolved_config["model"]


def test_tail_after_last_lr_transition() -> None:
    history = [
        {
            "epoch": float(epoch),
            "lr_reduction_event": epoch == 3,
            "validation_masked_mse_all": 1.0 - 0.1 * epoch,
        }
        for epoch in range(7)
    ]
    tail = post_last_reduction_tail(history, "all")
    assert tail["count"] == 3
    assert tail["mean"] == pytest.approx(0.1)
    assert tail["max_absolute"] == pytest.approx(0.1)


def test_full_historical_overlap_reproduction_parser() -> None:
    selection = {"active": ["section_0013"]}
    observations = {"index_sha256": "index", "mask_sha256": "mask"}
    rows = {
        epoch: {
            f"validation_masked_mse_{group}": 1.0 - epoch * 0.01
            for group in ("all", "intrinsic", "mixed", "spatial")
        }
        for epoch in range(4)
    }
    base = {
        "n_train_sections": 1,
        "condition": "spatial",
        "seed": 101,
        "training_data_selection": selection,
        "evaluation_observations": observations,
        "history_by_pass": rows,
    }
    current = [{**base, "policy": "prolonged_constant_lr"}]
    historical = [{**base, "policy": "converged_data_frontier"}]
    result = overlapping_history_reproduction(current, historical)
    assert len(result) == 4
    assert max(abs(row["difference_all"]) for row in result) == 0.0


def test_trainer_records_plateau_events_without_same_event_stop(
    small_corpus: Path, tmp_path: Path
) -> None:
    policy = TrainingExecutionPolicy(
        name="validation_plateau_lr_decay",
        sampling="deterministic_shuffled_epochs",
        lr_schedule="validation_plateau_decay",
        minimum_epochs=1,
        maximum_epochs=3,
        validation_interval_epochs=1,
        early_stopping_patience=2,
        improvement_threshold=100.0,
        absolute_max_steps=100,
        lr_reduction_factor=0.5,
        lr_reduction_patience=1,
        minimum_learning_rate=1e-5,
    )
    metrics = run_experiment(
        _tiny_focal_config(small_corpus, tmp_path / "plateau-events"),
        training_data_selection=_selection(small_corpus, 1),
        execution_policy=policy,
    )
    assert metrics["epochs_completed"] == 3
    assert metrics["early_stopping_reason"] == "maximum_epochs"
    assert metrics["num_lr_reductions"] == 3
    assert [event["epoch"] for event in metrics["lr_reduction_events"]] == [1, 2, 3]
    assert metrics["minimum_exposure_satisfied"]
    history = list(csv.DictReader((tmp_path / "plateau-events" / "history.csv").open()))
    assert all(row["lr_reduction_event"] == "True" for row in history[1:])
    assert all(row["early_stopping_decision"] == "False" for row in history)


def test_spatial_90pass_grid_is_exact_and_contains_no_shuffled_runs(
    tmp_path: Path,
) -> None:
    entries = data_scaling_entries(_toy_spatial_90pass_spec(tmp_path))
    assert len(entries) == 6
    assert len({entry.experiment_id for entry in entries}) == 6
    assert {
        (entry.n_train_sections, entry.condition, entry.seed) for entry in entries
    } == {(n, "spatial", seed) for n in (1, 48) for seed in (101, 202, 303)}
    assert {entry.execution_policy["maximum_epochs"] for entry in entries} == {90}
    assert {entry.execution_policy["absolute_max_steps"] for entry in entries} == {
        1200000
    }
    assert {entry.training_protocol for entry in entries} == {
        "validation_plateau_lr_decay"
    }


def test_spatial_90pass_changes_only_caps_and_preserves_data_definitions(
    tmp_path: Path,
) -> None:
    old_values = yaml.safe_load(LR_POLICY_SPEC.read_text(encoding="utf-8"))
    new_values = yaml.safe_load(SPATIAL_90PASS_SPEC.read_text(encoding="utf-8"))
    old_policy = dict(old_values["policies"]["validation_plateau_lr_decay"])
    new_policy = dict(new_values["policies"]["validation_plateau_lr_decay"])
    assert old_policy.pop("maximum_epochs") == 60
    assert new_policy.pop("maximum_epochs") == 90
    assert old_policy.pop("absolute_max_steps") == 800000
    assert new_policy.pop("absolute_max_steps") == 1200000
    assert new_policy == old_policy
    for field in (
        "subset_selection_seed",
        "scale_points_sections",
        "seeds",
        "context_size",
        "masking",
        "evaluation",
        "corpora",
        "base_configs",
    ):
        assert new_values[field] == old_values[field]
    assert new_values["provenance"]["execution_mode"] == "restart_from_zero"


def test_spatial_90pass_subsets_and_validation_match_prior_policy(
    tmp_path: Path,
) -> None:
    positive = _metadata_only_corpus(tmp_path / "shared-positive")
    null = _metadata_only_corpus(tmp_path / "shared-null")
    paths = {}
    for name, source in (
        ("old", LR_POLICY_SPEC),
        ("new", SPATIAL_90PASS_SPEC),
    ):
        values = yaml.safe_load(source.read_text(encoding="utf-8"))
        values["corpora"] = {"positive": str(positive), "null": str(null)}
        values["output_root"] = str(tmp_path / f"{name}-runs")
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
        paths[name] = path
    old = {
        (entry.n_train_sections, entry.seed): entry
        for entry in data_scaling_entries(paths["old"])
        if entry.training_protocol == "validation_plateau_lr_decay"
        and entry.condition == "spatial"
    }
    new = data_scaling_entries(paths["new"])
    for entry in new:
        historical = old[(entry.n_train_sections, entry.seed)]
        assert entry.training_data_selection == historical.training_data_selection
        assert (
            entry.resolved_config["evaluation"]
            == historical.resolved_config["evaluation"]
        )
        assert entry.resolved_config["masking"] == historical.resolved_config["masking"]
        assert entry.resolved_config["model"] == historical.resolved_config["model"]


def test_post_final_lr_reduction_tail_uses_all_available_checks() -> None:
    history = [
        {
            "epoch": float(epoch),
            "lr_reduction_event": epoch == 3,
            "validation_masked_mse_all": 1.0 - 0.01 * epoch,
        }
        for epoch in range(9)
    ]
    result = post_final_reduction_tail(history, "all")
    assert result["last_reduction_pass"] == 3
    assert result["checks_after_reduction"] == 5
    assert result["passes_after_reduction"] == 5
    assert result["mean_improvement"] == pytest.approx(0.01)
    assert result["final_3_mean_improvement"] == pytest.approx(0.01)
    assert result["max_absolute_improvement"] == pytest.approx(0.01)


def test_spatial_90pass_restart_reproduction_includes_scheduler_state() -> None:
    rows = {}
    for epoch in range(61):
        rows[epoch] = {
            "epoch": float(epoch),
            "learning_rate": 3e-4,
            "learning_rate_after_validation": 3e-4,
            "lr_reduction_event": False,
            "early_stopping_decision": False,
            **{
                f"validation_masked_mse_{group}": 1.0 - epoch * 0.001
                for group in ("all", "intrinsic", "mixed", "spatial")
            },
        }
    base = {
        "n_train_sections": 1,
        "seed": 101,
        "training_data_selection": {"active": ["section_0013"]},
        "evaluation_observations": {"index_sha256": "index", "mask_sha256": "mask"},
        "history_by_pass": rows,
    }
    reproduction = restart_reproduction_rows([{**base}], [{**base}])
    assert len(reproduction) == 61
    assert max(abs(row["difference_all"]) for row in reproduction) == 0.0
    assert all(row["lr_reduction_event_matches"] for row in reproduction)


def test_spatial_90pass_delta_pairing_freezes_converged_shuffled_endpoint() -> None:
    def metric_row(epoch: int, loss: float) -> dict:
        return {
            "epoch": float(epoch),
            **{
                f"validation_masked_mse_{group}": loss
                for group in ("all", "intrinsic", "mixed", "spatial")
            },
        }

    spatial_history = [
        metric_row(epoch, loss)
        for epoch, loss in (
            (0, 1.0),
            (1, 0.7),
            (2, 0.6),
            (10, 0.5),
            (30, 0.4),
            (60, 0.3),
        )
    ]
    shuffled_history = [
        metric_row(epoch, loss)
        for epoch, loss in ((0, 1.0), (1, 0.9), (10, 0.8), (30, 0.75))
    ]
    spatial = {
        "n_train_sections": 1,
        "seed": 101,
        "history": spatial_history,
        "history_by_pass": {int(row["epoch"]): row for row in spatial_history},
        "metrics": {
            "best_validation_epoch": 60,
            "final_validation": {
                f"masked_mse_{group}": 0.3
                for group in ("all", "intrinsic", "mixed", "spatial")
            },
        },
    }
    shuffled = {
        "n_train_sections": 1,
        "seed": 101,
        "history": shuffled_history,
        "history_by_pass": {int(row["epoch"]): row for row in shuffled_history},
        "metrics": {
            "final_validation": {
                f"masked_mse_{group}": 0.8
                for group in ("all", "intrinsic", "mixed", "spatial")
            },
        },
    }
    trajectories, _ = spatial_90pass_paired_delta_rows([spatial], [shuffled])
    by_pass = {row["pass"]: row for row in trajectories}
    assert by_pass[1]["delta_space_all"] == pytest.approx(0.2)
    assert by_pass[1]["shuffled_source"] == "matched_pass_history"
    assert by_pass[2]["delta_space_all"] == pytest.approx(0.2)
    assert by_pass[2]["shuffled_source"] == ("fixed_converged_shuffled_endpoint")


def test_converged_frontier_has_42_logical_and_30_new_unique_runs(
    tmp_path: Path,
) -> None:
    spec = _toy_converged_frontier_spec(tmp_path)
    logical = data_scaling_entries(spec)
    new = data_scaling_entries(spec, n_train_sections={2, 4, 8, 16, 32})
    validate_final_grids(logical, new)
    assert len(logical) == 42
    assert len(new) == 30
    assert {entry.n_train_sections for entry in new} == {2, 4, 8, 16, 32}
    assert not {entry.n_train_sections for entry in new} & {1, 48}
    assert len({entry.experiment_id for entry in new}) == 30


def test_converged_frontier_freezes_90pass_scheduler_and_subsets(
    tmp_path: Path,
) -> None:
    positive = _metadata_only_corpus(tmp_path / "frontier-shared-positive")
    null = _metadata_only_corpus(tmp_path / "frontier-shared-null")
    paths = {}
    for name, source in (
        ("frontier", CONVERGED_FRONTIER_SPEC),
        ("endpoint", SPATIAL_90PASS_SPEC),
    ):
        values = yaml.safe_load(source.read_text(encoding="utf-8"))
        values["corpora"] = {"positive": str(positive), "null": str(null)}
        values["output_root"] = str(tmp_path / f"{name}-runs")
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
        paths[name] = path
    frontier = data_scaling_entries(paths["frontier"])
    endpoint = {
        (entry.n_train_sections, entry.seed): entry
        for entry in data_scaling_entries(paths["endpoint"])
    }
    assert {
        json.dumps(entry.execution_policy, sort_keys=True) for entry in frontier
    } == {json.dumps(next(iter(endpoint.values())).execution_policy, sort_keys=True)}
    for entry in frontier:
        if entry.n_train_sections in {1, 48} and entry.condition == "spatial":
            old = endpoint[(entry.n_train_sections, entry.seed)]
            assert entry.training_data_selection == old.training_data_selection
            assert (
                entry.resolved_config["evaluation"] == old.resolved_config["evaluation"]
            )
            assert entry.resolved_config["masking"] == old.resolved_config["masking"]
            assert entry.resolved_config["model"] == old.resolved_config["model"]


def test_converged_frontier_paired_delta_is_seed_level() -> None:
    rows = []
    for seed, spatial, shuffled in ((101, 0.3, 0.6), (202, 0.4, 0.65), (303, 0.5, 0.7)):
        for condition, loss in (("spatial", spatial), ("shuffled", shuffled)):
            rows.append(
                {
                    "n_train_sections": 8,
                    "n_train_cells": 65536,
                    "condition": condition,
                    "seed": seed,
                    **{
                        f"validation_masked_mse_{group}": loss
                        for group in ("all", "intrinsic", "mixed", "spatial")
                    },
                }
            )
    paired = converged_frontier_paired_delta_rows(rows)
    assert [row["delta_space_all"] for row in paired] == pytest.approx([0.3, 0.25, 0.2])


def test_adjacent_improvements_report_absolute_and_percentage() -> None:
    rows = []
    for n_sections, spatial, shuffled in (
        (1, 1.0, 2.0),
        (2, 0.8, 1.5),
        (4, 0.7, 1.25),
        (8, 0.6, 1.0),
        (16, 0.55, 0.9),
        (32, 0.5, 0.8),
        (48, 0.45, 0.75),
    ):
        for condition, loss in (("spatial", spatial), ("shuffled", shuffled)):
            rows.append(
                {
                    "n_train_sections": n_sections,
                    "condition": condition,
                    "validation_masked_mse_all_mean": loss,
                    "validation_masked_mse_spatial_mean": loss * 2,
                }
            )
    adjacent = adjacent_improvement_rows(rows)
    assert len(adjacent) == 6
    assert adjacent[0]["spatial_overall_absolute_improvement"] == pytest.approx(0.2)
    assert adjacent[0]["spatial_overall_percentage_improvement"] == pytest.approx(20.0)
    assert adjacent[0]["shuffled_overall_absolute_improvement"] == pytest.approx(0.5)
    assert adjacent[0]["shuffled_overall_percentage_improvement"] == pytest.approx(25.0)
