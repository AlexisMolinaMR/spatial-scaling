from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from spatial_scaling.data.synthetic.config import SyntheticConfig, load_config
from spatial_scaling.data.synthetic.generator import (
    SyntheticCorpusGenerator,
    SyntheticSection,
)
from spatial_scaling.data.synthetic.qc import (
    compute_corpus_qc,
    correlation_by_distance,
)
from spatial_scaling.data.synthetic.random import component_rng
from spatial_scaling.data.synthetic.storage import load_section, write_section

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def small_config() -> SyntheticConfig:
    return SyntheticConfig.from_dict(
        {
            "master_seed": 41,
            "num_sections": 4,
            "cells_per_section": 768,
            "num_genes": 24,
            "noise_std": 0.2,
            "geometry": {"width_um": 800.0, "height_um": 800.0},
            "latent": {"intrinsic_dim": 8, "spatial_dim": 12},
            "spatial_field": {
                "implementation": "rbf_rff",
                "correlation_length_um": 120.0,
                "num_fourier_features": 96,
            },
            "gene_classes": {"intrinsic": 8, "mixed": 8, "spatial": 8},
            "splits": {"train": 2, "validation": 1, "test": 1},
            "qc": {
                "num_distance_bins": 8,
                "max_distance_um": 480.0,
                "pairs_per_section": 40_000,
                "max_sections": 4,
            },
        }
    )


@pytest.fixture(scope="module")
def generator(small_config: SyntheticConfig) -> SyntheticCorpusGenerator:
    return SyntheticCorpusGenerator(small_config)


@pytest.fixture(scope="module")
def section(generator: SyntheticCorpusGenerator) -> SyntheticSection:
    return generator.generate_section(0)


def assert_sections_equal(left: SyntheticSection, right: SyntheticSection) -> None:
    assert left.section_id == right.section_id
    assert left.split == right.split
    for field in (
        "cell_id",
        "coordinates_um",
        "expression",
        "intrinsic_latent",
        "spatial_latent",
        "intrinsic_expression",
        "spatial_expression",
        "noise_expression",
    ):
        np.testing.assert_array_equal(getattr(left, field), getattr(right, field))


def test_same_seed_and_config_are_reproducible(small_config: SyntheticConfig) -> None:
    left = SyntheticCorpusGenerator(small_config).generate_section(2)
    right = SyntheticCorpusGenerator(small_config).generate_section(2)
    assert_sections_equal(left, right)


def test_different_seed_changes_generated_data(small_config: SyntheticConfig) -> None:
    left = SyntheticCorpusGenerator(small_config).generate_section(0)
    right_config = replace(small_config, master_seed=small_config.master_seed + 1)
    right = SyntheticCorpusGenerator(right_config).generate_section(0)
    assert not np.array_equal(left.coordinates_um, right.coordinates_um)
    assert not np.array_equal(left.expression, right.expression)


def test_section_generation_is_order_invariant(
    generator: SyntheticCorpusGenerator,
) -> None:
    forward = {item.section_id: item for item in generator.generate_sections([0, 2, 3])}
    reverse = {item.section_id: item for item in generator.generate_sections([3, 2, 0])}
    for section_id, expected in forward.items():
        assert_sections_equal(expected, reverse[section_id])


def test_standalone_section_equals_section_from_corpus(
    generator: SyntheticCorpusGenerator,
) -> None:
    standalone = generator.generate_section(2)
    from_corpus = next(
        item
        for item in generator.generate_sections()
        if item.section_id == "section_0002"
    )
    assert_sections_equal(standalone, from_corpus)


def test_section_splits_are_disjoint(generator: SyntheticCorpusGenerator) -> None:
    splits = generator.split_definition()
    split_sets = {name: set(ids) for name, ids in splits.items()}
    assert len(split_sets["train"]) == 2
    assert len(split_sets["validation"]) == 1
    assert len(split_sets["test"]) == 1
    assert split_sets["train"].isdisjoint(split_sets["validation"])
    assert split_sets["train"].isdisjoint(split_sets["test"])
    assert split_sets["validation"].isdisjoint(split_sets["test"])


def test_output_shapes_and_gene_classes(
    generator: SyntheticCorpusGenerator,
    section: SyntheticSection,
    small_config: SyntheticConfig,
) -> None:
    cells = small_config.cells_per_section
    genes = small_config.num_genes
    assert section.coordinates_um.shape == (cells, 2)
    assert section.expression.shape == (cells, genes)
    assert section.intrinsic_latent.shape == (cells, small_config.latent.intrinsic_dim)
    assert section.spatial_latent.shape == (cells, small_config.latent.spatial_dim)
    assert section.intrinsic_expression.shape == (cells, genes)
    assert section.spatial_expression.shape == (cells, genes)
    assert section.noise_expression.shape == (cells, genes)
    classes, counts = np.unique(
        generator.gene_loadings.gene_classes, return_counts=True
    )
    assert dict(zip(classes, counts, strict=True)) == {
        "intrinsic": 8,
        "mixed": 8,
        "spatial": 8,
    }


def test_clean_gene_class_constraints(generator: SyntheticCorpusGenerator) -> None:
    loadings = generator.gene_loadings
    intrinsic = loadings.gene_classes == "intrinsic"
    spatial = loadings.gene_classes == "spatial"
    assert np.count_nonzero(loadings.spatial[:, intrinsic]) == 0
    assert np.count_nonzero(loadings.intrinsic[:, spatial]) == 0


def test_spatial_latent_correlation_decays_and_intrinsic_does_not(
    section: SyntheticSection, small_config: SyntheticConfig
) -> None:
    edges = np.array([0.0, 60.0, 120.0, 240.0, 480.0])
    spatial, spatial_counts = correlation_by_distance(
        section.coordinates_um,
        section.spatial_latent,
        edges,
        100_000,
        component_rng(1, "test", "spatial"),
    )
    intrinsic, intrinsic_counts = correlation_by_distance(
        section.coordinates_um,
        section.intrinsic_latent,
        edges,
        100_000,
        component_rng(1, "test", "intrinsic"),
    )
    assert np.all(spatial_counts > 100)
    assert np.all(intrinsic_counts > 100)
    assert spatial[0] > 0.55
    assert spatial[0] > spatial[1] > spatial[2]
    assert spatial[0] - spatial[-1] > 0.5
    assert np.max(np.abs(intrinsic)) < 0.08


def test_lambda_zero_removes_spatial_expression(
    small_config: SyntheticConfig,
) -> None:
    positive = SyntheticCorpusGenerator(small_config).generate_section(1)
    null_config = replace(small_config, lambda_s=0.0)
    null = SyntheticCorpusGenerator(null_config).generate_section(1)
    np.testing.assert_array_equal(positive.coordinates_um, null.coordinates_um)
    np.testing.assert_array_equal(positive.intrinsic_latent, null.intrinsic_latent)
    np.testing.assert_array_equal(positive.spatial_latent, null.spatial_latent)
    np.testing.assert_array_equal(positive.noise_expression, null.noise_expression)
    assert np.count_nonzero(null.spatial_expression) == 0
    np.testing.assert_allclose(
        null.expression, null.intrinsic_expression + null.noise_expression, rtol=1e-6
    )


def test_positive_qc_orders_gene_classes_and_recovers_length(
    generator: SyntheticCorpusGenerator, small_config: SyntheticConfig
) -> None:
    metrics = compute_corpus_qc(
        generator.generate_sections(), small_config, generator.gene_loadings
    )
    class_metrics = metrics["expression_by_gene_class"]
    intrinsic = class_metrics["intrinsic"]["short_range_correlation"]
    mixed = class_metrics["mixed"]["short_range_correlation"]
    spatial = class_metrics["spatial"]["short_range_correlation"]
    assert spatial > mixed > intrinsic
    assert abs(intrinsic) < 0.08
    estimated = metrics["spatial_latent"]["estimated_correlation_length_um"]
    assert estimated == pytest.approx(120.0, rel=0.35)
    assert (
        metrics["section_independence"]["mean_absolute_off_diagonal_correlation"] < 0.08
    )


def test_null_qc_expression_spatial_signal_collapses(
    small_config: SyntheticConfig,
) -> None:
    null_config = replace(small_config, lambda_s=0.0)
    generator = SyntheticCorpusGenerator(null_config)
    metrics = compute_corpus_qc(
        generator.generate_sections(), null_config, generator.gene_loadings
    )
    for gene_class in ("intrinsic", "mixed", "spatial"):
        short_range = metrics["expression_by_gene_class"][gene_class][
            "short_range_correlation"
        ]
        assert abs(short_range) < 0.08


def test_positive_and_null_configs_differ_only_in_lambda() -> None:
    positive = load_config(ROOT / "configs/pilot/synthetic_spatial.yaml")
    null = load_config(ROOT / "configs/pilot/synthetic_null.yaml")
    positive_values = positive.to_dict()
    null_values = null.to_dict()
    assert positive_values.pop("lambda_s") == 1.0
    assert null_values.pop("lambda_s") == 0.0
    assert positive_values == null_values
    assert type(SyntheticCorpusGenerator(positive)) is type(
        SyntheticCorpusGenerator(null)
    )


def test_section_storage_round_trip(tmp_path: Path, section: SyntheticSection) -> None:
    (tmp_path / "sections").mkdir()
    path = write_section(tmp_path, section)
    assert_sections_equal(section, load_section(path))
    with np.load(path) as stored:
        assert {
            "cell_id",
            "section_id",
            "split",
            "x_um",
            "y_um",
            "expression",
            "intrinsic_latent",
            "spatial_latent",
            "intrinsic_expression",
            "spatial_expression",
            "noise_expression",
        } <= set(stored.files)


@pytest.mark.parametrize(
    "values, message",
    [
        ({"cells_per_section": 0}, "cells_per_section must be positive"),
        (
            {
                "num_genes": 10,
                "gene_classes": {"intrinsic": 3, "mixed": 3, "spatial": 3},
            },
            "gene class counts must sum",
        ),
        (
            {"num_sections": 3, "splits": {"train": 1, "validation": 1, "test": 0}},
            "split section counts must sum",
        ),
        (
            {"spatial_field": {"correlation_length_um": -1.0}},
            "correlation_length_um must be positive",
        ),
        (
            {"gene_classes": {"intrinsic_beta": 0.1}},
            "clean Pilot v0 intrinsic_beta must be zero",
        ),
    ],
)
def test_invalid_configurations_fail_clearly(
    values: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        SyntheticConfig.from_dict(values)
