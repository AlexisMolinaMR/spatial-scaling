"""Replaceable tissue geometry implementations."""

from __future__ import annotations

import numpy as np

from spatial_scaling.data.synthetic.config import GeometryConfig


def sample_uniform_rectangle(
    num_cells: int,
    config: GeometryConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample local section coordinates in micrometres."""
    coordinates = rng.random((num_cells, 2), dtype=np.float64)
    coordinates[:, 0] *= config.width_um
    coordinates[:, 1] *= config.height_um
    return coordinates.astype(np.float32)
