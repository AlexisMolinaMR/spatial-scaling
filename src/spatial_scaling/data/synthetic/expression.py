"""Gene-class loadings and decomposed expression generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spatial_scaling.data.synthetic.config import SyntheticConfig


@dataclass(frozen=True)
class GeneLoadings:
    """Shared vocabulary and intrinsic/spatial projection matrices."""

    gene_ids: np.ndarray
    gene_classes: np.ndarray
    intrinsic_alpha: np.ndarray
    spatial_beta: np.ndarray
    intrinsic: np.ndarray
    spatial: np.ndarray


def _unit_columns(rows: int, columns: int, rng: np.random.Generator) -> np.ndarray:
    values = rng.normal(size=(rows, columns))
    norms = np.linalg.norm(values, axis=0, keepdims=True)
    return values / norms


def generate_gene_loadings(
    config: SyntheticConfig,
    intrinsic_rng: np.random.Generator,
    spatial_rng: np.random.Generator,
) -> GeneLoadings:
    """Generate globally shared clean-class projections."""
    classes = config.gene_classes
    gene_classes = np.array(
        ["intrinsic"] * classes.intrinsic
        + ["mixed"] * classes.mixed
        + ["spatial"] * classes.spatial,
        dtype="U9",
    )
    alpha_by_class = {
        "intrinsic": classes.intrinsic_alpha,
        "mixed": classes.mixed_alpha,
        "spatial": classes.spatial_alpha,
    }
    beta_by_class = {
        "intrinsic": classes.intrinsic_beta,
        "mixed": classes.mixed_beta,
        "spatial": classes.spatial_beta,
    }
    alpha = np.array([alpha_by_class[item] for item in gene_classes])
    beta = np.array([beta_by_class[item] for item in gene_classes])
    intrinsic = (
        _unit_columns(config.latent.intrinsic_dim, config.num_genes, intrinsic_rng)
        * alpha
    )
    spatial = (
        _unit_columns(config.latent.spatial_dim, config.num_genes, spatial_rng) * beta
    )
    gene_ids = np.array(
        [f"gene_{index:04d}" for index in range(config.num_genes)], dtype="U9"
    )
    return GeneLoadings(
        gene_ids=gene_ids,
        gene_classes=gene_classes,
        intrinsic_alpha=alpha.astype(np.float32),
        spatial_beta=beta.astype(np.float32),
        intrinsic=intrinsic.astype(np.float32),
        spatial=spatial.astype(np.float32),
    )


def generate_expression(
    intrinsic_latent: np.ndarray,
    spatial_latent: np.ndarray,
    loadings: GeneLoadings,
    lambda_s: float,
    noise_std: float,
    noise_rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return expression and its inspectable additive components."""
    intrinsic = intrinsic_latent @ loadings.intrinsic
    spatial = lambda_s * (spatial_latent @ loadings.spatial)
    noise = noise_rng.normal(scale=noise_std, size=intrinsic.shape).astype(np.float32)
    expression = intrinsic + spatial + noise
    return (
        expression.astype(np.float32),
        intrinsic.astype(np.float32),
        spatial.astype(np.float32),
        noise,
    )
