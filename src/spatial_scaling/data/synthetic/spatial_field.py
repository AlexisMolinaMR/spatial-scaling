"""Modular spatial latent fields."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from spatial_scaling.data.synthetic.config import SpatialFieldConfig


class SpatialField(Protocol):
    """Interface implemented by replaceable spatial-field generators."""

    def sample(
        self,
        coordinates_um: np.ndarray,
        latent_dim: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample a spatial latent state at the supplied coordinates."""


class RBFRandomFourierField:
    """RBF-correlated Gaussian field approximation using Fourier features.

    For an RBF kernel with length scale ``ell``, its spectral density is
    Gaussian with frequencies ``omega ~ N(0, ell^-2 I)``. Each latent
    dimension receives an independent field realization.
    """

    def __init__(self, config: SpatialFieldConfig) -> None:
        self.config = config

    def sample(
        self,
        coordinates_um: np.ndarray,
        latent_dim: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        num_cells = coordinates_um.shape[0]
        num_features = self.config.num_fourier_features
        frequency_scale = 1.0 / self.config.correlation_length_um
        normalizer = np.sqrt(2.0 / num_features)
        output = np.empty((num_cells, latent_dim), dtype=np.float32)

        for dimension in range(latent_dim):
            frequencies = rng.normal(scale=frequency_scale, size=(num_features, 2))
            phases = rng.uniform(0.0, 2.0 * np.pi, size=num_features)
            coefficients = rng.normal(size=num_features)
            features = np.cos(coordinates_um @ frequencies.T + phases)
            output[:, dimension] = normalizer * (features @ coefficients)
        return output


def build_spatial_field(config: SpatialFieldConfig) -> SpatialField:
    """Construct the configured field implementation."""
    if config.implementation == "rbf_rff":
        return RBFRandomFourierField(config)
    raise ValueError(
        f"unsupported spatial field implementation: {config.implementation}"
    )
