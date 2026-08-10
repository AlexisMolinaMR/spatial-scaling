from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

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
from spatial_scaling.models.spatial_transformer import (
    SpatialTransformer,
    SpatialTransformerConfig,
)
from spatial_scaling.spatial.neighborhoods import exact_knn_indices
from spatial_scaling.training.config import (
    ContextConfig,
    DataConfig,
    EvaluationConfig,
    OptimizerConfig,
    SSLExperimentConfig,
    TrainLoopConfig,
)
from spatial_scaling.training.masking import MaskingConfig, build_masked_batch
from spatial_scaling.training.trainer import _evaluation_mask_digest, run_experiment


@pytest.fixture(scope="module")
def corpora(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("spatial-context-corpora")
    result = {}
    for world, lambda_s in (("positive", 1.0), ("null", 0.0)):
        config = SyntheticConfig.from_dict(
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
        generator = SyntheticCorpusGenerator(config)
        output = prepare_output_directory(root / world)
        write_corpus_metadata(output, generator)
        write_gene_ground_truth(output, generator)
        for section in generator.generate_sections():
            write_section(output, section)
        result[world] = output
    return result


def test_exact_knn_is_local_deterministic_and_excludes_focal() -> None:
    coordinates = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [4.0, 0.0], [0.0, 8.0]],
        dtype=np.float32,
    )
    first = exact_knn_indices(coordinates, np.asarray([0, 3]), context_size=2)
    second = exact_knn_indices(coordinates, np.asarray([0, 3]), context_size=2)
    np.testing.assert_array_equal(first, [[1, 2], [1, 0]])
    np.testing.assert_array_equal(first, second)
    assert 0 not in first[0]
    assert 3 not in first[1]


def _context_dataset(
    corpus: Path, *, condition: str, seed: int, context_size: int = 8
) -> SyntheticContextDataset:
    base = SyntheticExpressionDataset(corpus, "validation", cache_sections=1)
    return SyntheticContextDataset(
        base,
        condition=condition,
        context_size=context_size,
        shuffle_seed=seed,
    )


def test_true_context_contains_actual_same_section_neighbors(
    corpora: dict[str, Path],
) -> None:
    dataset = _context_dataset(corpora["positive"], condition="spatial", seed=5)
    targets = np.asarray([0, 9, 31])
    batch = dataset.context_batch(0, targets)
    _, coordinates, cell_ids = dataset.dataset.spatial_section(0)
    expected = exact_knn_indices(coordinates, targets, context_size=8)
    np.testing.assert_array_equal(batch.context_indices, expected)
    np.testing.assert_array_equal(batch.true_context_indices, expected)
    assert batch.section_id == dataset.dataset.section_ids[0]
    assert all(
        context_id in cell_ids
        for context_ids in batch.context_cell_ids
        for context_id in context_ids
    )
    assert batch.context_expression.shape == (3, 8, dataset.dataset.num_genes)


def test_shuffled_context_breaks_local_molecular_correspondence_and_is_seeded(
    corpora: dict[str, Path],
) -> None:
    targets = np.asarray([0, 9, 31, 47])
    true_batch = _context_dataset(
        corpora["positive"], condition="spatial", seed=5
    ).context_batch(0, targets)
    shuffled = _context_dataset(
        corpora["positive"], condition="shuffled", seed=5
    ).context_batch(0, targets)
    repeated = _context_dataset(
        corpora["positive"], condition="shuffled", seed=5
    ).context_batch(0, targets)
    changed = _context_dataset(
        corpora["positive"], condition="shuffled", seed=6
    ).context_batch(0, targets)

    assert true_batch.context_indices.shape == shuffled.context_indices.shape
    np.testing.assert_array_equal(
        true_batch.relative_coordinates_um, shuffled.relative_coordinates_um
    )
    np.testing.assert_array_equal(shuffled.context_indices, repeated.context_indices)
    assert not np.array_equal(shuffled.context_indices, changed.context_indices)
    assert shuffled.section_id == true_batch.section_id
    for target, true_indices, shuffled_indices in zip(
        targets,
        true_batch.context_indices,
        shuffled.context_indices,
        strict=True,
    ):
        assert target not in shuffled_indices
        assert set(true_indices).isdisjoint(shuffled_indices)


def test_spatial_transformer_shape_leakage_interface_and_matched_capacity() -> None:
    config = SpatialTransformerConfig(
        num_genes=24,
        context_size=8,
        embedding_dim=16,
        num_layers=1,
        num_heads=4,
        ffn_width=32,
    )
    spatial = SpatialTransformer(config)
    shuffled = SpatialTransformer(replace(config))
    focal = torch.randn(3, 24)
    masked = build_masked_batch(
        focal, ["cell-a", "cell-b", "cell-c"], MaskingConfig(fraction=0.50, seed=7)
    )
    context = torch.randn(3, 8, 24)
    offsets = torch.randn(3, 8, 2)
    output = spatial(masked.masked_expression, masked.visibility, context, offsets)
    assert output.shape == focal.shape
    assert spatial.parameter_count == shuffled.parameter_count
    assert torch.all(masked.masked_expression[masked.mask] == 0.0)
    assert "targets" not in inspect.signature(spatial.forward).parameters
    forbidden = {"intrinsic_latent", "spatial_latent", "gene_class", "section_id"}
    assert forbidden.isdisjoint(inspect.signature(spatial.forward).parameters)


def test_fixed_validation_masks_are_condition_independent(
    corpora: dict[str, Path],
) -> None:
    base = SyntheticExpressionDataset(corpora["positive"], "validation")
    indices = base.deterministic_indices(19, seed=991)
    masking = MaskingConfig(policy="high_random", fraction=0.85, seed=101)
    spatial_hash = _evaluation_mask_digest(base, indices, masking, batch_size=7)
    shuffled_hash = _evaluation_mask_digest(base, indices, masking, batch_size=5)
    assert spatial_hash == shuffled_hash


def _tiny_context_experiment(
    corpus: Path, output: Path, condition: str
) -> SSLExperimentConfig:
    return SSLExperimentConfig(
        experiment_name=f"tiny-{condition}",
        output_dir=str(output),
        seed=17,
        device="cpu",
        condition=condition,
        data=DataConfig(corpus_path=str(corpus), cache_sections=1),
        context=ContextConfig(context_size=8, shuffle_seed=17),
        model={
            "context_size": 8,
            "embedding_dim": 16,
            "num_layers": 1,
            "num_heads": 4,
            "ffn_width": 32,
            "dropout": 0.0,
            "coordinate_scale_um": 120.0,
        },
        masking=MaskingConfig(policy="high_random", fraction=0.85, seed=17),
        optimizer=OptimizerConfig(learning_rate=1e-3, weight_decay=0.0),
        training=TrainLoopConfig(
            batch_size=4, steps=2, evaluation_interval=1, mask_epoch_steps=1
        ),
        evaluation=EvaluationConfig(
            batch_size=4,
            max_cells=8,
            observation_seed=991,
            masking_seed=17,
        ),
    )


def test_matched_contextual_training_smoke_is_stable_and_mask_matched(
    corpora: dict[str, Path], tmp_path: Path
) -> None:
    spatial = run_experiment(
        _tiny_context_experiment(corpora["positive"], tmp_path / "spatial", "spatial")
    )
    shuffled = run_experiment(
        _tiny_context_experiment(corpora["positive"], tmp_path / "shuffled", "shuffled")
    )
    assert spatial["parameter_count"] == shuffled["parameter_count"]
    assert spatial["evaluation_observations"] == shuffled["evaluation_observations"]
    assert np.isfinite(spatial["final_training_interval_masked_mse"])
    assert np.isfinite(shuffled["final_training_interval_masked_mse"])


@pytest.mark.parametrize("context_size", [0, 64, 65])
def test_invalid_context_sizes_fail_clearly(
    corpora: dict[str, Path], context_size: int
) -> None:
    base = SyntheticExpressionDataset(corpora["positive"], "validation")
    with pytest.raises(ValueError, match="context_size|2 \\* K"):
        SyntheticContextDataset(
            base,
            condition="shuffled",
            context_size=context_size,
            shuffle_seed=1,
        )
