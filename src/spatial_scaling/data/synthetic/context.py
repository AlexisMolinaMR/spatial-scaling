"""Contextual views over validated section-sharded synthetic corpora."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spatial_scaling.data.synthetic.dataset import SyntheticExpressionDataset
from spatial_scaling.spatial.neighborhoods import (
    exact_knn_indices,
    validate_context_size,
)
from spatial_scaling.spatial.shuffling import shuffled_context_indices

CONTEXT_CONDITIONS = ("spatial", "shuffled")


@dataclass(frozen=True)
class ContextBatch:
    """Model-safe focal and contextual arrays for one section."""

    focal_expression: np.ndarray
    context_expression: np.ndarray
    relative_coordinates_um: np.ndarray
    focal_cell_ids: tuple[str, ...]
    context_cell_ids: tuple[tuple[str, ...], ...]
    true_context_indices: np.ndarray
    context_indices: np.ndarray
    anchor_indices: np.ndarray | None
    section_id: str


class SyntheticContextDataset:
    """Construct deterministic K-cell contexts without crossing sections."""

    def __init__(
        self,
        dataset: SyntheticExpressionDataset,
        *,
        context_size: int,
        condition: str,
        shuffle_seed: int,
    ) -> None:
        if condition not in CONTEXT_CONDITIONS:
            raise ValueError(f"condition must be one of {CONTEXT_CONDITIONS}")
        validate_context_size(dataset.cells_per_section, context_size)
        if condition == "shuffled" and dataset.cells_per_section < 2 * context_size + 1:
            raise ValueError("shuffled control requires cells_per_section >= 2 * K + 1")
        self.dataset = dataset
        self.context_size = context_size
        self.condition = condition
        self.shuffle_seed = shuffle_seed

    def context_batch(
        self, section_offset: int, focal_offsets: np.ndarray
    ) -> ContextBatch:
        expression, coordinates, cell_ids = self.dataset.spatial_section(section_offset)
        targets = np.asarray(focal_offsets)
        if targets.ndim != 1 or not np.issubdtype(targets.dtype, np.integer):
            raise TypeError("focal_offsets must be a one-dimensional integer array")
        true_indices = exact_knn_indices(coordinates, targets, self.context_size)
        anchors: np.ndarray | None = None
        if self.condition == "spatial":
            context_indices = true_indices
        else:
            context_indices, anchors = shuffled_context_indices(
                coordinates,
                targets,
                true_indices,
                section_id=self.dataset.section_ids[section_offset],
                context_size=self.context_size,
                seed=self.shuffle_seed,
            )
        relative = coordinates[true_indices] - coordinates[targets, None, :]
        context_ids = tuple(
            tuple(cell_ids[index] for index in row) for row in context_indices
        )
        return ContextBatch(
            focal_expression=expression[targets].copy(),
            context_expression=expression[context_indices].copy(),
            relative_coordinates_um=relative.astype(np.float32, copy=False),
            focal_cell_ids=tuple(cell_ids[index] for index in targets),
            context_cell_ids=context_ids,
            true_context_indices=true_indices,
            context_indices=context_indices,
            anchor_indices=anchors,
            section_id=self.dataset.section_ids[section_offset],
        )
