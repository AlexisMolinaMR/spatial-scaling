"""Matched same-section controls that break target-to-context correspondence."""

from __future__ import annotations

import hashlib

import numpy as np

from spatial_scaling.spatial.neighborhoods import (
    exact_distance_excluded_knn_indices,
    validate_context_size,
    validate_minimum_distance_um,
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
    minimum_distance_um: float = 0.0,
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
    threshold = validate_minimum_distance_um(minimum_distance_um)
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
    result = np.empty_like(true_indices, dtype=np.int64)
    positions = np.asarray(coordinates, dtype=np.float64)
    for row, target in enumerate(targets.astype(np.int64, copy=False)):
        anchor = int(anchors[row])
        delta = positions - positions[anchor]
        squared_distance = np.einsum("ij,ij->i", delta, delta)
        forbidden = np.zeros(num_cells, dtype=bool)
        forbidden[true_indices[row]] = True
        forbidden[target] = True
        forbidden[anchor] = True
        allowed = np.flatnonzero(
            (~forbidden) & (squared_distance >= threshold * threshold)
        )
        if len(allowed) < context_size:
            raise ValueError(
                f"target {int(target)} has {len(allowed)} eligible shuffled "
                f"context cells at d_min={threshold:g} um; requires K={context_size}"
            )
        allowed_distances = squared_distance[allowed]
        candidates = np.argpartition(allowed_distances, context_size - 1)[:context_size]
        cutoff = allowed_distances[candidates].max()
        closest = allowed[allowed_distances <= cutoff]
        order = np.lexsort((closest, squared_distance[closest]))
        result[row] = closest[order[:context_size]]
    return result, anchors


def _extreme_witness_indices(
    coordinates: np.ndarray, *, minimum_count: int
) -> np.ndarray:
    """Return a deterministic subset used only to certify lower bounds."""
    positions = np.asarray(coordinates, dtype=np.float64)
    center = np.median(positions, axis=0)
    squared_radius = np.einsum("ij,ij->i", positions - center, positions - center)
    indices = np.arange(len(positions), dtype=np.int64)
    order = np.lexsort((indices, -squared_radius))
    return order[: min(len(order), minimum_count)]


def matched_eligible_target_indices(
    coordinates: np.ndarray,
    *,
    section_id: str,
    context_size: int,
    seed: int,
    minimum_distance_um: float,
) -> np.ndarray:
    """Return targets valid for both true and shuffled distance controls.

    Extreme cells provide an exact lower-bound certificate for the common case.
    Any target not certified by that subset is checked by the full exact
    selectors, so eligibility is never approximate.
    """
    positions = np.asarray(coordinates, dtype=np.float64)
    num_cells = len(positions)
    validate_context_size(num_cells, context_size)
    threshold = validate_minimum_distance_um(minimum_distance_um)
    if num_cells < 2 * context_size + 1:
        raise ValueError("shuffled control requires cells_per_section >= 2 * K + 1")
    if threshold == 0.0:
        return np.arange(num_cells, dtype=np.int64)

    witness_count = min(num_cells, max(6 * context_size + 16, 128))
    witnesses = _extreme_witness_indices(positions, minimum_count=witness_count)
    witness_delta = positions[:, None, :] - positions[witnesses][None, :, :]
    witness_distance_sq = np.einsum("ijk,ijk->ij", witness_delta, witness_delta)
    witness_eligible = witness_distance_sq >= threshold * threshold
    witness_columns = {int(index): column for column, index in enumerate(witnesses)}
    for index, column in witness_columns.items():
        witness_eligible[index, column] = False
    lower_bounds = witness_eligible.sum(axis=1)
    anchors = target_anchor_permutation(num_cells, section_id=section_id, seed=seed)
    certified = (lower_bounds >= context_size) & (
        lower_bounds[anchors] >= 2 * context_size + 1
    )
    valid = certified.copy()
    for target in np.flatnonzero(~certified):
        try:
            true_indices = exact_distance_excluded_knn_indices(
                positions,
                np.asarray([target]),
                context_size,
                minimum_distance_um=threshold,
            )
            shuffled_context_indices(
                positions,
                np.asarray([target]),
                true_indices,
                section_id=section_id,
                context_size=context_size,
                seed=seed,
                minimum_distance_um=threshold,
            )
        except ValueError:
            continue
        valid[target] = True
    return np.flatnonzero(valid).astype(np.int64, copy=False)
