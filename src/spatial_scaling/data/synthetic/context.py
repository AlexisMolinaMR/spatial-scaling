"""Contextual views over validated section-sharded synthetic corpora."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from spatial_scaling.data.synthetic.dataset import SyntheticExpressionDataset
from spatial_scaling.spatial.neighborhoods import (
    exact_distance_excluded_knn_indices,
    exact_knn_indices,
    selected_context_distances_um,
    validate_context_size,
    validate_minimum_distance_um,
)
from spatial_scaling.spatial.shuffling import (
    matched_eligible_target_indices,
    shuffled_context_indices,
)

CONTEXT_CONDITIONS = ("spatial", "shuffled")
CONTEXT_SELECTION_MODES = ("nearest", "distance_exclusion")


@dataclass(frozen=True)
class ContextSelectionConfig:
    """Scientific context-selection dimension kept outside model capacity."""

    mode: str = "nearest"
    minimum_distance_um: float = 0.0
    eligibility_minimum_distance_um: float = 0.0

    def validate(self) -> None:
        if self.mode not in CONTEXT_SELECTION_MODES:
            raise ValueError(f"mode must be one of {CONTEXT_SELECTION_MODES}")
        minimum = validate_minimum_distance_um(self.minimum_distance_um)
        eligibility = validate_minimum_distance_um(self.eligibility_minimum_distance_um)
        if self.mode == "nearest" and (minimum != 0.0 or eligibility != 0.0):
            raise ValueError("nearest selection requires zero distance thresholds")
        if eligibility < minimum:
            raise ValueError(
                "eligibility_minimum_distance_um must be at least minimum_distance_um"
            )

    def to_dict(self) -> dict[str, str | float]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object] | None) -> ContextSelectionConfig:
        if values is None:
            return cls()
        if not isinstance(values, dict):
            raise TypeError("context selection configuration must be a mapping")
        try:
            result = cls(**values)
        except TypeError as error:
            raise ValueError(
                f"invalid context selection configuration: {error}"
            ) from error
        result.validate()
        return result


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
    selected_distances_um: np.ndarray
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
        selection: ContextSelectionConfig | None = None,
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
        self.selection = selection or ContextSelectionConfig()
        self.selection.validate()
        self._eligibility_cache: dict[tuple[int, float], np.ndarray] = {}

    def eligible_offsets(
        self, section_offset: int, *, minimum_distance_um: float | None = None
    ) -> np.ndarray:
        """Return the exact focal subset valid for both matched conditions."""
        if not 0 <= section_offset < len(self.dataset.section_ids):
            raise IndexError(section_offset)
        threshold = validate_minimum_distance_um(
            self.selection.eligibility_minimum_distance_um
            if minimum_distance_um is None
            else minimum_distance_um
        )
        key = (section_offset, threshold)
        cached = self._eligibility_cache.get(key)
        if cached is not None:
            return cached
        _, coordinates, _ = self.dataset.spatial_section(section_offset)
        values = matched_eligible_target_indices(
            coordinates,
            section_id=self.dataset.section_ids[section_offset],
            context_size=self.context_size,
            seed=self.shuffle_seed,
            minimum_distance_um=threshold,
        )
        values.flags.writeable = False
        self._eligibility_cache[key] = values
        return values

    def context_batch(
        self, section_offset: int, focal_offsets: np.ndarray
    ) -> ContextBatch:
        expression, coordinates, cell_ids = self.dataset.spatial_section(section_offset)
        targets = np.asarray(focal_offsets)
        if targets.ndim != 1 or not np.issubdtype(targets.dtype, np.integer):
            raise TypeError("focal_offsets must be a one-dimensional integer array")
        if self.selection.mode == "nearest":
            true_indices = exact_knn_indices(coordinates, targets, self.context_size)
        else:
            true_indices = exact_distance_excluded_knn_indices(
                coordinates,
                targets,
                self.context_size,
                minimum_distance_um=self.selection.minimum_distance_um,
            )
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
                minimum_distance_um=self.selection.minimum_distance_um,
            )
        relative = coordinates[true_indices] - coordinates[targets, None, :]
        selected_distances = selected_context_distances_um(
            coordinates, targets, true_indices
        )
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
            selected_distances_um=selected_distances,
            anchor_indices=anchors,
            section_id=self.dataset.section_ids[section_offset],
        )
