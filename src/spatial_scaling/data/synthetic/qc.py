"""Quantitative, scalable QC for claimed synthetic spatial structure."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")

from spatial_scaling.data.synthetic.config import SyntheticConfig
from spatial_scaling.data.synthetic.expression import GeneLoadings
from spatial_scaling.data.synthetic.generator import SyntheticSection
from spatial_scaling.data.synthetic.random import component_rng

GENE_CLASSES = ("intrinsic", "mixed", "spatial")


@dataclass
class _CurveAccumulator:
    sums: np.ndarray
    counts: np.ndarray

    @classmethod
    def empty(cls, num_bins: int) -> _CurveAccumulator:
        return cls(np.zeros(num_bins), np.zeros(num_bins, dtype=np.int64))

    def add(self, values: np.ndarray, counts: np.ndarray) -> None:
        valid = np.isfinite(values)
        self.sums[valid] += values[valid] * counts[valid]
        self.counts += counts

    def means(self) -> np.ndarray:
        return np.divide(
            self.sums,
            self.counts,
            out=np.full_like(self.sums, np.nan),
            where=self.counts > 0,
        )


@dataclass
class _VarianceAccumulator:
    count: int = 0
    total: float = 0.0
    total_square: float = 0.0

    def add(self, values: np.ndarray) -> None:
        values64 = np.asarray(values, dtype=np.float64)
        self.count += values64.size
        self.total += float(values64.sum())
        self.total_square += float(np.square(values64).sum())

    def variance(self) -> float:
        if self.count == 0:
            return float("nan")
        mean = self.total / self.count
        return self.total_square / self.count - mean * mean


def correlation_by_distance(
    coordinates_um: np.ndarray,
    values: np.ndarray,
    bin_edges_um: np.ndarray,
    num_pairs: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate standardized covariance using randomly sampled cell pairs."""
    if coordinates_um.ndim != 2 or coordinates_um.shape[1] != 2:
        raise ValueError("coordinates_um must have shape (cells, 2)")
    if values.ndim == 1:
        values = values[:, None]
    if values.shape[0] != coordinates_um.shape[0]:
        raise ValueError("values and coordinates must have the same cell count")
    if coordinates_um.shape[0] < 2:
        raise ValueError("at least two cells are required for pairwise QC")

    standardized = values.astype(np.float64, copy=True)
    standardized -= standardized.mean(axis=0, keepdims=True)
    standard_deviations = standardized.std(axis=0, keepdims=True)
    valid_features = standard_deviations[0] > 1e-12
    if not np.any(valid_features):
        return (
            np.full(len(bin_edges_um) - 1, np.nan),
            np.zeros(len(bin_edges_um) - 1, dtype=np.int64),
        )
    standardized = (
        standardized[:, valid_features] / standard_deviations[:, valid_features]
    )

    first = rng.integers(0, len(coordinates_um), size=num_pairs)
    second = rng.integers(0, len(coordinates_um) - 1, size=num_pairs)
    second += second >= first
    distances = np.linalg.norm(coordinates_um[first] - coordinates_um[second], axis=1)
    products = np.mean(standardized[first] * standardized[second], axis=1)
    bin_index = np.searchsorted(bin_edges_um, distances, side="right") - 1
    valid = (bin_index >= 0) & (bin_index < len(bin_edges_um) - 1)
    counts = np.bincount(bin_index[valid], minlength=len(bin_edges_um) - 1)
    sums = np.bincount(
        bin_index[valid], weights=products[valid], minlength=len(bin_edges_um) - 1
    )
    means = np.divide(
        sums,
        counts,
        out=np.full(len(counts), np.nan),
        where=counts > 0,
    )
    return means, counts


def _section_pair_curves(
    section: SyntheticSection,
    config: SyntheticConfig,
    loadings: GeneLoadings,
    bin_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]:
    rng = component_rng(config.master_seed, section.section_id, "qc", "pairs")
    latent_curve, latent_count = correlation_by_distance(
        section.coordinates_um,
        section.spatial_latent,
        bin_edges,
        config.qc.pairs_per_section,
        rng,
    )
    class_curves = {}
    for gene_class in GENE_CLASSES:
        mask = loadings.gene_classes == gene_class
        class_curves[gene_class] = correlation_by_distance(
            section.coordinates_um,
            section.expression[:, mask],
            bin_edges,
            config.qc.pairs_per_section,
            component_rng(
                config.master_seed, section.section_id, "qc", "expression_pairs"
            ),
        )
    return latent_curve, latent_count, class_curves


def _cross_section_correlation(spatial_states: list[np.ndarray]) -> np.ndarray:
    count = len(spatial_states)
    matrix = np.eye(count)
    for first in range(count):
        for second in range(first + 1, count):
            cells = min(len(spatial_states[first]), len(spatial_states[second]))
            dimensions = min(
                spatial_states[first].shape[1], spatial_states[second].shape[1]
            )
            left = spatial_states[first][:cells, :dimensions].astype(np.float64)
            right = spatial_states[second][:cells, :dimensions].astype(np.float64)
            left -= left.mean(axis=0, keepdims=True)
            right -= right.mean(axis=0, keepdims=True)
            denominator = np.linalg.norm(left) * np.linalg.norm(right)
            value = float(np.sum(left * right) / denominator) if denominator else np.nan
            matrix[first, second] = value
            matrix[second, first] = value
    return matrix


def compute_corpus_qc(
    sections: Iterable[SyntheticSection],
    config: SyntheticConfig,
    loadings: GeneLoadings,
) -> dict[str, object]:
    """Compute corpus QC without retaining expression for all sections."""
    bin_edges = np.linspace(
        0.0, config.qc.max_distance_um, config.qc.num_distance_bins + 1
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    latent = _CurveAccumulator.empty(config.qc.num_distance_bins)
    expression = {
        gene_class: _CurveAccumulator.empty(config.qc.num_distance_bins)
        for gene_class in GENE_CLASSES
    }
    variance = {
        group: {
            component: _VarianceAccumulator()
            for component in ("intrinsic", "spatial", "noise", "expression")
        }
        for group in ("overall", *GENE_CLASSES)
    }
    spatial_states: list[np.ndarray] = []
    section_ids: list[str] = []
    sections_evaluated = 0

    for section in sections:
        sections_evaluated += 1
        components = {
            "intrinsic": section.intrinsic_expression,
            "spatial": section.spatial_expression,
            "noise": section.noise_expression,
            "expression": section.expression,
        }
        for component, values in components.items():
            variance["overall"][component].add(values)
            for gene_class in GENE_CLASSES:
                mask = loadings.gene_classes == gene_class
                variance[gene_class][component].add(values[:, mask])

        if len(spatial_states) < config.qc.max_sections:
            latent_curve, latent_counts, class_curves = _section_pair_curves(
                section, config, loadings, bin_edges
            )
            latent.add(latent_curve, latent_counts)
            for gene_class, (curve, counts) in class_curves.items():
                expression[gene_class].add(curve, counts)
            spatial_states.append(section.spatial_latent.copy())
            section_ids.append(section.section_id)

    if sections_evaluated == 0:
        raise ValueError("QC requires at least one section")
    cross_section = _cross_section_correlation(spatial_states)
    off_diagonal = cross_section[~np.eye(len(cross_section), dtype=bool)]
    variance_metrics: dict[str, dict[str, float]] = {}
    for group, accumulators in variance.items():
        component_variances = {
            name: accumulator.variance() for name, accumulator in accumulators.items()
        }
        additive_total = sum(
            component_variances[name] for name in ("intrinsic", "spatial", "noise")
        )
        component_variances.update(
            {
                f"{name}_fraction_of_additive_variance": (
                    component_variances[name] / additive_total
                    if additive_total
                    else 0.0
                )
                for name in ("intrinsic", "spatial", "noise")
            }
        )
        variance_metrics[group] = component_variances

    latent_means = latent.means()
    fit_mask = (
        np.isfinite(latent_means)
        & (latent_means > 0.05)
        & (bin_centers <= config.qc.max_distance_um)
    )
    estimated_length = None
    if np.count_nonzero(fit_mask) >= 3:
        slope, _intercept = np.polyfit(
            np.square(bin_centers[fit_mask]), np.log(latent_means[fit_mask]), 1
        )
        if slope < 0:
            estimated_length = float(np.sqrt(-1.0 / (2.0 * slope)))
    short_range_mask = bin_centers <= config.spatial_field.correlation_length_um / 2.0

    return {
        "sections_evaluated": sections_evaluated,
        "pairwise_qc_sections": section_ids,
        "distance_bin_edges_um": bin_edges.tolist(),
        "distance_bin_centers_um": bin_centers.tolist(),
        "spatial_latent": {
            "empirical_correlation": latent_means.tolist(),
            "pair_counts": latent.counts.tolist(),
            "expected_rbf_correlation": np.exp(
                -np.square(bin_centers)
                / (2.0 * config.spatial_field.correlation_length_um**2)
            ).tolist(),
            "estimated_correlation_length_um": estimated_length,
            "configured_correlation_length_um": (
                config.spatial_field.correlation_length_um
            ),
            "relative_length_error": (
                abs(estimated_length - config.spatial_field.correlation_length_um)
                / config.spatial_field.correlation_length_um
                if estimated_length is not None
                else None
            ),
            "short_range_correlation": float(
                np.nanmean(latent_means[short_range_mask])
            ),
        },
        "expression_by_gene_class": {
            gene_class: {
                "empirical_correlation": accumulator.means().tolist(),
                "pair_counts": accumulator.counts.tolist(),
                "short_range_correlation": float(
                    np.nanmean(accumulator.means()[short_range_mask])
                ),
            }
            for gene_class, accumulator in expression.items()
        },
        "variance_decomposition": variance_metrics,
        "section_independence": {
            "section_ids": section_ids,
            "matched_cell_spatial_correlation_matrix": cross_section.tolist(),
            "mean_absolute_off_diagonal_correlation": (
                float(np.mean(np.abs(off_diagonal))) if off_diagonal.size else None
            ),
            "max_absolute_off_diagonal_correlation": (
                float(np.max(np.abs(off_diagonal))) if off_diagonal.size else None
            ),
        },
    }


def write_qc_artifacts(
    metrics: dict[str, object], output_dir: str | Path, config: SyntheticConfig
) -> None:
    """Write machine-readable QC and four auditable scientific plots."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    centers = np.asarray(metrics["distance_bin_centers_um"])

    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    latent = metrics["spatial_latent"]
    axis.plot(centers, latent["empirical_correlation"], marker="o", label="Empirical")
    axis.plot(centers, latent["expected_rbf_correlation"], label="Expected RBF")
    axis.axvline(
        config.spatial_field.correlation_length_um,
        color="black",
        linestyle="--",
        label="Configured length",
    )
    axis.set(xlabel="Cell-pair distance (µm)", ylabel="Standardized covariance")
    axis.set_title("Spatial latent correlation by distance")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "spatial_latent_correlation.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    for gene_class in GENE_CLASSES:
        curve = metrics["expression_by_gene_class"][gene_class]
        axis.plot(centers, curve["empirical_correlation"], marker="o", label=gene_class)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(xlabel="Cell-pair distance (µm)", ylabel="Standardized covariance")
    axis.set_title("Expression spatial structure by gene class")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "expression_correlation_by_gene_class.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    groups = ["overall", *GENE_CLASSES]
    components = ["intrinsic", "spatial", "noise"]
    positions = np.arange(len(groups))
    bottom = np.zeros(len(groups))
    for component in components:
        values = np.array(
            [
                metrics["variance_decomposition"][group][
                    f"{component}_fraction_of_additive_variance"
                ]
                for group in groups
            ]
        )
        axis.bar(positions, values, bottom=bottom, label=component)
        bottom += values
    axis.set_xticks(positions, groups)
    axis.set(ylabel="Fraction of additive component variance")
    axis.set_title("Expression variance decomposition")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "variance_decomposition.png", dpi=160)
    plt.close(figure)

    matrix = np.asarray(
        metrics["section_independence"]["matched_cell_spatial_correlation_matrix"]
    )
    figure, axis = plt.subplots(figsize=(5.2, 4.6))
    image = axis.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    axis.set(xlabel="Section index", ylabel="Section index")
    axis.set_title("Cross-section spatial realization correlation")
    figure.colorbar(image, ax=axis, label="Matched-cell correlation")
    figure.tight_layout()
    figure.savefig(output / "section_independence.png", dpi=160)
    plt.close(figure)
