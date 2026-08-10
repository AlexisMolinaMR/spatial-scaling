"""Deterministic local-context construction for spatial experiments."""

from spatial_scaling.spatial.neighborhoods import exact_knn_indices
from spatial_scaling.spatial.shuffling import (
    shuffled_context_indices,
    target_anchor_permutation,
)

__all__ = [
    "exact_knn_indices",
    "shuffled_context_indices",
    "target_anchor_permutation",
]
