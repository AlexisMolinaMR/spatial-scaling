"""Masked-only molecular reconstruction objective."""

from __future__ import annotations

import torch


def masked_mse(
    predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Mean squared error over masked targets and nowhere else."""
    if predictions.shape != targets.shape or predictions.shape != mask.shape:
        raise ValueError("predictions, targets, and mask must have identical shapes")
    if mask.dtype != torch.bool:
        raise TypeError("mask must be boolean")
    if not torch.any(mask):
        raise ValueError("masked MSE requires at least one masked target")
    return torch.square(predictions - targets)[mask].mean()
