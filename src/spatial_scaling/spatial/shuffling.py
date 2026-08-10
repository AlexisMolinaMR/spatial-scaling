"""Matched same-section controls that break target-to-context correspondence."""

from __future__ import annotations

import hashlib

import numpy as np

from spatial_scaling.spatial.neighborhoods import (
    exact_knn_indices,
    validate_context_size,
)


def _stable_seed(*components: object) -> int:
    digest = hashlib.blake2b(digest_size=8)
    for component in components:
        encoded = str(component).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    return int.from_bytes(digest.digest(), "little")


def target_anchor_permutation(
    num_cells: int, *, section_id: str, seed: int
) -> np.ndarray:
    """Map every target bijectively to a different same-section anchor."""
    if not isinstance(num_cells, int) or isinstance(num_cells, bool) or num_cells < 2:
        raise ValueError("num_cells must be an integer of at least 2")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("shuffle seed must be an integer")
    rng = np.random.default_rng(
        _stable_seed("pilot-v0-context-anchors", section_id, seed)
    )
    ordering = rng.permutation(num_cells)
    shift = int(rng.integers(1, num_cells))
    anchors = np.empty(num_cells, dtype=np.int64)
    anchors[ordering] = np.roll(ordering, -shift)
    if np.any(anchors == np.arange(num_cells)):
        raise RuntimeError("internal error: shuffled anchors are not a derangement")
    return anchors


def shuffled_context_indices(
    coordinates: np.ndarray,
    target_indices: np.ndarray,
    true_context_indices: np.ndarray,
    *,
    section_id: str,
    context_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select molecular context around wrong, seeded same-section anchors.

    Relative-coordinate slots are supplied separately by the caller from the
    target's true KNN geometry. This function only assigns molecular identities.
    Every returned identity belongs to the same section but is neither the focal
    cell nor one of its true K nearest neighbors. Candidate cells retain their
    local ordering around the incorrect anchor, preserving contextual molecular
    covariance and broad expression marginals more closely than iid sampling.
    """
    num_cells = len(coordinates)
    validate_context_size(num_cells, context_size)
    if num_cells < 2 * context_size + 1:
        raise ValueError(
            "shuffled control requires cells_per_section >= 2 * K + 1 so true "
            "neighbors can be excluded"
        )
    targets = np.asarray(target_indices)
    true_indices = np.asarray(true_context_indices)
    if targets.ndim != 1 or not np.issubdtype(targets.dtype, np.integer):
        raise TypeError("target_indices must be a one-dimensional integer array")
    if true_indices.shape != (len(targets), context_size):
        raise ValueError("true_context_indices must have shape [targets, K]")

    anchors_by_target = target_anchor_permutation(
        num_cells, section_id=section_id, seed=seed
    )
    anchors = anchors_by_target[targets]
    candidate_size = min(num_cells - 1, 2 * context_size + 2)
    candidates = exact_knn_indices(coordinates, anchors, candidate_size)
    result = np.empty_like(true_indices, dtype=np.int64)
    for row, target in enumerate(targets.astype(np.int64, copy=False)):
        forbidden = np.zeros(num_cells, dtype=bool)
        forbidden[true_indices[row]] = True
        forbidden[target] = True
        allowed = candidates[row][~forbidden[candidates[row]]]
        if len(allowed) < context_size:
            raise RuntimeError("insufficient incorrect-context candidates")
        result[row] = allowed[:context_size]
    return result, anchors
