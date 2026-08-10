from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from spatial_scaling.data.synthetic.config import SyntheticConfig
from spatial_scaling.data.synthetic.dataset import (
    GeneMetadata,
    SyntheticExpressionDataset,
)
from spatial_scaling.data.synthetic.generator import SyntheticCorpusGenerator
from spatial_scaling.data.synthetic.storage import (
    prepare_output_directory,
    write_corpus_metadata,
    write_gene_ground_truth,
    write_section,
)
from spatial_scaling.evaluation.masked_expression import MaskedMetricAccumulator
from spatial_scaling.models.cell_encoder import CellMLP, CellMLPConfig
from spatial_scaling.training.config import (
    DataConfig,
    EvaluationConfig,
    OptimizerConfig,
    SSLExperimentConfig,
    TrainLoopConfig,
)
from spatial_scaling.training.masking import (
    MaskingConfig,
    build_masked_batch,
    mask_for_cell,
    masks_for_cells,
)
from spatial_scaling.training.objective import masked_mse
from spatial_scaling.training.trainer import run_experiment


def _small_config(lambda_s: float) -> SyntheticConfig:
    return SyntheticConfig.from_dict(
        {
            "master_seed": 31,
            "num_sections": 4,
            "cells_per_section": 64,
            "num_genes": 24,
            "lambda_s": lambda_s,
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


def _write_small_corpus(path: Path, lambda_s: float) -> Path:
    generator = SyntheticCorpusGenerator(_small_config(lambda_s))
    output = prepare_output_directory(path)
    write_corpus_metadata(output, generator)
    write_gene_ground_truth(output, generator)
    for section in generator.generate_sections():
        write_section(output, section)
    return output


@pytest.fixture(scope="module")
def corpora(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("ssl-corpora")
    return {
        "positive": _write_small_corpus(root / "positive", 1.0),
        "null": _write_small_corpus(root / "null", 0.0),
    }


def test_dataset_respects_disjoint_section_splits(corpora: dict[str, Path]) -> None:
    datasets = {
        split: SyntheticExpressionDataset(corpora["positive"], split)
        for split in ("train", "validation", "test")
    }
    section_sets = {
        name: set(dataset.section_ids) for name, dataset in datasets.items()
    }
    assert len(datasets["train"]) == 128
    assert len(datasets["validation"]) == len(datasets["test"]) == 64
    assert section_sets["train"].isdisjoint(section_sets["validation"])
    assert section_sets["train"].isdisjoint(section_sets["test"])
    assert section_sets["validation"].isdisjoint(section_sets["test"])
    example = datasets["validation"][0]
    assert example["split"] == "validation"
    assert example["section_id"] in section_sets["validation"]
    assert example["expression"].shape == (24,)
    assert len(datasets["train"].gene_metadata.gene_classes) == 24


@pytest.mark.parametrize("world", ["positive", "null"])
def test_same_loader_supports_positive_and_null_corpora(
    corpora: dict[str, Path], world: str
) -> None:
    dataset = SyntheticExpressionDataset(corpora[world], "test")
    assert dataset[3]["expression"].dtype == np.float32
    assert set(dataset.gene_metadata.gene_classes) == {
        "intrinsic",
        "mixed",
        "spatial",
    }


def test_masked_values_are_absent_and_explicitly_identified() -> None:
    expression = torch.tensor([[0.0, 2.0, 3.0, 4.0], [5.0, 0.0, 7.0, 8.0]])
    config = MaskingConfig(fraction=0.50, seed=7)
    batch = build_masked_batch(expression, ["cell-a", "cell-b"], config)
    torch.testing.assert_close(batch.targets, expression)
    assert torch.all(batch.masked_expression[batch.mask] == 0.0)
    assert torch.all(~batch.visibility[batch.mask])
    assert torch.all(batch.visibility[~batch.mask])
    assert torch.any((batch.masked_expression == 0.0) & batch.visibility)


def test_loss_is_computed_only_on_masked_entries() -> None:
    targets = torch.tensor([[1.0, 2.0, 3.0]])
    predictions = torch.tensor([[2.0, 200.0, -300.0]])
    mask = torch.tensor([[True, False, False]])
    assert masked_mse(predictions, targets, mask).item() == pytest.approx(1.0)


@pytest.mark.parametrize("fraction", [0.15, 0.30, 0.50, 0.70])
def test_random_mask_fraction_is_exact_up_to_rounding(fraction: float) -> None:
    mask = mask_for_cell(
        256, "section_0001_cell_0000001", MaskingConfig(fraction=fraction, seed=8)
    )
    assert mask.mean() == pytest.approx(round(256 * fraction) / 256)


def test_masking_is_seeded_and_independent_of_batch_order() -> None:
    cells = ["cell-a", "cell-b", "cell-c"]
    config = MaskingConfig(fraction=0.30, seed=11)
    first = masks_for_cells(32, cells, config)
    repeated = masks_for_cells(32, cells, config)
    changed = masks_for_cells(32, cells, replace(config, seed=12))
    reordered_cells = [cells[2], cells[0], cells[1]]
    reordered = masks_for_cells(32, reordered_cells, config)
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, changed)
    by_cell = dict(zip(cells, first, strict=True))
    for cell_id, mask in zip(reordered_cells, reordered, strict=True):
        np.testing.assert_array_equal(mask, by_cell[cell_id])


def test_masks_are_nested_across_random_calibration_fractions() -> None:
    low = mask_for_cell(64, "cell", MaskingConfig(fraction=0.15, seed=4))
    high = mask_for_cell(64, "cell", MaskingConfig(fraction=0.70, seed=4))
    assert np.all(high[low])


def test_gene_class_metric_aggregation() -> None:
    metadata = GeneMetadata(
        gene_ids=tuple(f"gene-{index}" for index in range(6)),
        gene_classes=("intrinsic", "intrinsic", "mixed", "mixed", "spatial", "spatial"),
    )
    accumulator = MaskedMetricAccumulator(metadata)
    predictions = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    targets = torch.zeros_like(predictions)
    mask = torch.ones_like(predictions, dtype=torch.bool)
    accumulator.update(predictions, targets, mask)
    metrics = accumulator.compute()
    assert metrics["masked_mse_all"] == pytest.approx(91 / 6)
    assert metrics["masked_mse_intrinsic"] == pytest.approx(2.5)
    assert metrics["masked_mse_mixed"] == pytest.approx(12.5)
    assert metrics["masked_mse_spatial"] == pytest.approx(30.5)


def test_model_output_shape_and_focal_only_interface() -> None:
    model = CellMLP(CellMLPConfig(num_genes=24, hidden_dims=(16, 8)))
    expression = torch.randn(5, 24)
    visibility = torch.ones_like(expression, dtype=torch.bool)
    assert model(expression, visibility).shape == (5, 24)
    assert model.parameter_count == 24 * 2 * 16 + 16 + 16 * 8 + 8 + 8 * 24 + 24


@pytest.mark.parametrize(
    "config,message",
    [
        (MaskingConfig(fraction=0.0), "strictly between"),
        (MaskingConfig(fraction=1.0), "strictly between"),
        (MaskingConfig(policy="unknown"), "policy must be"),
        (MaskingConfig(policy="high_random", fraction=0.30), "requires fraction"),
    ],
)
def test_invalid_masking_configurations_fail_clearly(
    config: MaskingConfig, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        config.validate()


def _tiny_experiment(corpus: Path, output: Path, seed: int) -> SSLExperimentConfig:
    return SSLExperimentConfig(
        experiment_name="tiny-reproducibility",
        output_dir=str(output),
        seed=seed,
        device="cpu",
        data=DataConfig(corpus_path=str(corpus), cache_sections=1),
        model={"hidden_dims": [16], "activation": "relu", "dropout": 0.0},
        masking=MaskingConfig(fraction=0.50, seed=seed),
        optimizer=OptimizerConfig(learning_rate=1e-3, weight_decay=0.0),
        training=TrainLoopConfig(
            batch_size=16, steps=4, evaluation_interval=2, mask_epoch_steps=2
        ),
        evaluation=EvaluationConfig(
            batch_size=32,
            max_cells=32,
            observation_seed=41,
            masking_seed=42,
        ),
    )


def test_training_and_fixed_evaluation_are_reproducible(
    corpora: dict[str, Path], tmp_path: Path
) -> None:
    left = run_experiment(_tiny_experiment(corpora["positive"], tmp_path / "left", 9))
    right = run_experiment(_tiny_experiment(corpora["positive"], tmp_path / "right", 9))
    assert left["evaluation_observations"] == right["evaluation_observations"]
    assert left["initial_validation"] == right["initial_validation"]
    assert left["final_validation"] == right["final_validation"]
    assert (
        left["final_training_interval_masked_mse"]
        == right["final_training_interval_masked_mse"]
    )
    changed_seed = run_experiment(
        _tiny_experiment(corpora["positive"], tmp_path / "changed-seed", 10)
    )
    assert changed_seed["final_validation"] != left["final_validation"]
    assert (
        changed_seed["final_training_interval_masked_mse"]
        != left["final_training_interval_masked_mse"]
    )


def test_validation_and_test_observation_selection_is_fraction_independent(
    corpora: dict[str, Path],
) -> None:
    for split in ("validation", "test"):
        dataset = SyntheticExpressionDataset(corpora["positive"], split)
        first = dataset.deterministic_indices(17, seed=55)
        second = dataset.deterministic_indices(17, seed=55)
        np.testing.assert_array_equal(first, second)
