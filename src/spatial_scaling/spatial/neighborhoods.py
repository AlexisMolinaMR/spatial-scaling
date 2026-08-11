"""Exact deterministic Euclidean neighborhoods within one tissue section."""

from __future__ import annotations

import numpy as np


def validate_context_size(num_cells: int, context_size: int) -> None:
    """Validate K for a section containing ``num_cells`` observations."""
    if not isinstance(context_size, int) or isinstance(context_size, bool):
        raise TypeError("context_size must be an integer")
    if not 1 <= context_size < num_cells:
        raise ValueError(
            f"context_size must satisfy 1 <= K < cells_per_section ({num_cells})"
        )


def _validate_coordinates(coordinates: np.ndarray) -> np.ndarray:
    values = np.asarray(coordinates)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("coordinates must have shape [cells, 2]")
    if values.shape[0] < 2:
        raise ValueError("at least two cells are required for a neighborhood")
    if not np.issubdtype(values.dtype, np.number) or not np.all(np.isfinite(values)):
        raise ValueError("coordinates must be finite numeric values")
    return values.astype(np.float64, copy=False)


def exact_knn_indices(
    coordinates: np.ndarray,
    query_indices: np.ndarray,
    context_size: int,
) -> np.ndarray:
    """Return exact Euclidean KNN indices with focal cells excluded.

    Neighborhoods are computed independently inside the supplied section.
    Squared Euclidean distance determines order; global cell offset breaks
    exact-distance ties. The chunk-free per-query implementation bounds peak
    temporary memory by the section size and supports later changes to K
    without changing the returned ``[targets, K]`` interface.
    """
    positions = _validate_coordinates(coordinates)
    validate_context_size(len(positions), context_size)
    queries = np.asarray(query_indices)
    if queries.ndim != 1 or not np.issubdtype(queries.dtype, np.integer):
        raise TypeError("query_indices must be a one-dimensional integer array")
    if np.any(queries < 0) or np.any(queries >= len(positions)):
        raise IndexError("query index is outside its section")

    result = np.empty((len(queries), context_size), dtype=np.int64)
    all_indices = np.arange(len(positions), dtype=np.int64)
    for row, query in enumerate(queries.astype(np.int64, copy=False)):
        delta = positions - positions[query]
        squared_distance = np.einsum("ij,ij->i", delta, delta)
        squared_distance[query] = np.inf
        candidates = np.argpartition(squared_distance, context_size - 1)[:context_size]
        threshold = squared_distance[candidates].max()
        eligible = all_indices[squared_distance <= threshold]
        order = np.lexsort((eligible, squared_distance[eligible]))
        result[row] = eligible[order[:context_size]]
    return result


def validate_minimum_distance_um(minimum_distance_um: float) -> float:
    """Validate a physical-distance exclusion threshold in micrometres."""
    if isinstance(minimum_distance_um, bool) or not isinstance(
        minimum_distance_um, (int, float, np.integer, np.floating)
    ):
        raise TypeError("minimum_distance_um must be numeric")
    value = float(minimum_distance_um)
    if not np.isfinite(value) or value < 0:
        raise ValueError("minimum_distance_um must be finite and non-negative")
    return value


def exact_distance_excluded_knn_indices(
    coordinates: np.ndarray,
    query_indices: np.ndarray,
    context_size: int,
    *,
    minimum_distance_um: float,
) -> np.ndarray:
    """Return the closest K cells at least ``minimum_distance_um`` away.

    Selection is exact within the supplied section. The focal cell is always
    excluded, including when the threshold is zero. Distance then global cell
    offset defines a deterministic total ordering.
    """
    positions = _validate_coordinates(coordinates)
    validate_context_size(len(positions), context_size)
    threshold = validate_minimum_distance_um(minimum_distance_um)
    queries = np.asarray(query_indices)
    if queries.ndim != 1 or not np.issubdtype(queries.dtype, np.integer):
        raise TypeError("query_indices must be a one-dimensional integer array")
    if np.any(queries < 0) or np.any(queries >= len(positions)):
        raise IndexError("query index is outside its section")

    result = np.empty((len(queries), context_size), dtype=np.int64)
    all_indices = np.arange(len(positions), dtype=np.int64)
    squared_threshold = threshold * threshold
    for row, query in enumerate(queries.astype(np.int64, copy=False)):
        delta = positions - positions[query]
        squared_distance = np.einsum("ij,ij->i", delta, delta)
        eligible_mask = squared_distance >= squared_threshold
        eligible_mask[query] = False
        eligible = all_indices[eligible_mask]
        if len(eligible) < context_size:
            raise ValueError(
                f"query {int(query)} has {len(eligible)} eligible context cells "
                f"at d_min={threshold:g} um; requires K={context_size}"
            )
        candidate_distances = squared_distance[eligible]
        candidates = np.argpartition(candidate_distances, context_size - 1)[
            :context_size
        ]
        cutoff = candidate_distances[candidates].max()
        closest = eligible[candidate_distances <= cutoff]
        order = np.lexsort((closest, squared_distance[closest]))
        result[row] = closest[order[:context_size]]
    return result


def selected_context_distances_um(
    coordinates: np.ndarray,
    query_indices: np.ndarray,
    context_indices: np.ndarray,
) -> np.ndarray:
    """Measure selected context distances with shape ``[targets, K]``."""
    positions = _validate_coordinates(coordinates)
    queries = np.asarray(query_indices)
    contexts = np.asarray(context_indices)
    if queries.ndim != 1 or not np.issubdtype(queries.dtype, np.integer):
        raise TypeError("query_indices must be a one-dimensional integer array")
    if contexts.ndim != 2 or contexts.shape[0] != len(queries):
        raise ValueError("context_indices must have shape [targets, K]")
    if not np.issubdtype(contexts.dtype, np.integer):
        raise TypeError("context_indices must be an integer array")
    if np.any(queries < 0) or np.any(queries >= len(positions)):
        raise IndexError("query index is outside its section")
    if np.any(contexts < 0) or np.any(contexts >= len(positions)):
        raise IndexError("context index is outside its section")
    delta = positions[contexts] - positions[queries, None, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", delta, delta))


def summarize_selected_context_distances_um(
    distances_um: np.ndarray,
) -> list[dict[str, float | str]]:
    """Summarize per-target nearest, median, and farthest context distance."""
    values = np.asarray(distances_um, dtype=np.float64)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError("distances_um must be a non-empty [targets, K] array")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("distances_um must contain finite non-negative distances")
    per_target = {
        "nearest": values.min(axis=1),
        "median": np.median(values, axis=1),
        "farthest": values.max(axis=1),
    }
    rows: list[dict[str, float | str]] = []
    for statistic, distribution in per_target.items():
        percentiles = np.percentile(distribution, [5, 25, 50, 75, 95])
        rows.append(
            {
                "selected_distance_statistic": statistic,
                "median_um": float(percentiles[2]),
                "p25_um": float(percentiles[1]),
                "p75_um": float(percentiles[3]),
                "p05_um": float(percentiles[0]),
                "p95_um": float(percentiles[4]),
            }
        )
    return rows


def kth_neighbor_radii_um(
    coordinates: np.ndarray,
    query_indices: np.ndarray,
    context_sizes: list[int] | tuple[int, ...],
) -> dict[int, np.ndarray]:
    """Return each target's true Euclidean distance to its Kth neighbor."""
    if not context_sizes:
        raise ValueError("context_sizes must not be empty")
    if len(set(context_sizes)) != len(context_sizes):
        raise ValueError("context_sizes must be unique")
    positions = _validate_coordinates(coordinates)
    for context_size in context_sizes:
        validate_context_size(len(positions), context_size)
    neighbors = exact_knn_indices(
        positions, query_indices, context_size=max(context_sizes)
    )
    queries = np.asarray(query_indices).astype(np.int64, copy=False)
    result: dict[int, np.ndarray] = {}
    for context_size in context_sizes:
        kth = neighbors[:, context_size - 1]
        delta = positions[kth] - positions[queries]
        result[context_size] = np.sqrt(np.einsum("ij,ij->i", delta, delta))
    return result


def summarize_radii_um(radii_um: np.ndarray) -> dict[str, float]:
    """Summarize a non-empty distribution of physical KNN radii."""
    values = np.asarray(radii_um, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("radii_um must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("radii_um must contain finite non-negative distances")
    percentiles = np.percentile(values, [5, 25, 50, 75, 95])
    return {
        "median_radius_um": float(percentiles[2]),
        "p25_radius_um": float(percentiles[1]),
        "p75_radius_um": float(percentiles[3]),
        "p05_radius_um": float(percentiles[0]),
        "p95_radius_um": float(percentiles[4]),
    }
