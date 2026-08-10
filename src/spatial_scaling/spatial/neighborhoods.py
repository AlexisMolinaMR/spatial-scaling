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
