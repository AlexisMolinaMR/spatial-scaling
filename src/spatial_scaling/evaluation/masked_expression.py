"""Auditable masked-expression MSE overall and by synthetic gene class."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from spatial_scaling.data.synthetic.dataset import GENE_CLASSES, GeneMetadata


@dataclass
class MaskedMetricAccumulator:
    gene_metadata: GeneMetadata
    squared_error_sums: dict[str, float] = field(init=False)
    counts: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        names = ("all", *GENE_CLASSES)
        self.squared_error_sums = dict.fromkeys(names, 0.0)
        self.counts = dict.fromkeys(names, 0)

    def update(
        self, predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
    ) -> None:
        if predictions.shape != targets.shape or predictions.shape != mask.shape:
            raise ValueError(
                "predictions, targets, and mask must have identical shapes"
            )
        if predictions.shape[1] != len(self.gene_metadata.gene_ids):
            raise ValueError("metric gene dimension does not match gene metadata")
        errors = torch.square(predictions.detach() - targets.detach())
        self._update_one("all", errors, mask)
        classes = np.asarray(self.gene_metadata.gene_classes)
        for name in GENE_CLASSES:
            gene_mask = torch.as_tensor(
                classes == name, dtype=torch.bool, device=mask.device
            ).unsqueeze(0)
            self._update_one(name, errors, mask & gene_mask)

    def _update_one(
        self, name: str, errors: torch.Tensor, selected: torch.Tensor
    ) -> None:
        count = int(selected.sum().item())
        if count:
            self.squared_error_sums[name] += float(errors[selected].sum().item())
            self.counts[name] += count

    def compute(self) -> dict[str, float | int]:
        result: dict[str, float | int] = {}
        for name in ("all", *GENE_CLASSES):
            count = self.counts[name]
            if count == 0:
                raise ValueError(f"no masked targets accumulated for {name}")
            result[f"masked_mse_{name}"] = self.squared_error_sums[name] / count
            result[f"masked_count_{name}"] = count
        return result
