"""Order-independent deterministic masks for focal-cell expression."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
import torch

MASKING_POLICIES = ("random", "high_random")


@dataclass(frozen=True)
class MaskingConfig:
    """Configuration for uniform focal-gene masking.

    ``high_random`` uses the same unbiased gene sampling as ``random`` but
    enforces at least 50% masking. This makes its additional difficulty come
    only from reduced focal information, not synthetic metadata or structure.
    """

    policy: str = "random"
    fraction: float = 0.30
    seed: int = 0
    mask_value: float = 0.0

    def validate(self) -> None:
        if self.policy not in MASKING_POLICIES:
            raise ValueError(
                f"masking policy must be one of {MASKING_POLICIES}; got {self.policy!r}"
            )
        if not isinstance(self.fraction, (int, float)) or not math.isfinite(
            self.fraction
        ):
            raise TypeError("masking fraction must be a finite number")
        if not 0.0 < self.fraction < 1.0:
            raise ValueError("masking fraction must be strictly between 0 and 1")
        if self.policy == "high_random" and self.fraction < 0.50:
            raise ValueError("high_random masking requires fraction >= 0.50")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("masking seed must be an integer")
        if not isinstance(self.mask_value, (int, float)) or not math.isfinite(
            self.mask_value
        ):
            raise TypeError("mask_value must be a finite number")


@dataclass(frozen=True)
class MaskedBatch:
    """Leakage-resistant model inputs and their masked reconstruction targets."""

    masked_expression: torch.Tensor
    visibility: torch.Tensor
    targets: torch.Tensor
    mask: torch.Tensor


def _cell_seed(seed: int, epoch: int, cell_id: str) -> int:
    digest = hashlib.blake2b(digest_size=8)
    for component in ("pilot-v0-mask", seed, epoch, cell_id):
        encoded = str(component).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    return int.from_bytes(digest.digest(), "little")


def mask_for_cell(
    num_genes: int,
    cell_id: str,
    config: MaskingConfig,
    *,
    epoch: int = 0,
) -> np.ndarray:
    """Return a fixed-size mask determined only by cell identity and seeds."""
    config.validate()
    if not isinstance(num_genes, int) or isinstance(num_genes, bool) or num_genes < 2:
        raise ValueError("num_genes must be an integer of at least 2")
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
        raise ValueError("epoch must be a non-negative integer")
    num_masked = min(num_genes - 1, max(1, round(config.fraction * num_genes)))
    rng = np.random.default_rng(_cell_seed(config.seed, epoch, cell_id))
    order = rng.permutation(num_genes)
    mask = np.zeros(num_genes, dtype=bool)
    mask[order[:num_masked]] = True
    return mask


def masks_for_cells(
    num_genes: int,
    cell_ids: tuple[str, ...] | list[str],
    config: MaskingConfig,
    *,
    epoch: int = 0,
) -> np.ndarray:
    """Build masks independently of sample or DataLoader ordering."""
    return np.stack(
        [mask_for_cell(num_genes, cell_id, config, epoch=epoch) for cell_id in cell_ids]
    )


def build_masked_batch(
    expression: torch.Tensor,
    cell_ids: tuple[str, ...] | list[str],
    config: MaskingConfig,
    *,
    epoch: int = 0,
) -> MaskedBatch:
    """Replace targets by a mask value and expose a separate visibility bit."""
    if expression.ndim != 2:
        raise ValueError("expression must have shape [batch, genes]")
    if expression.shape[0] != len(cell_ids):
        raise ValueError("cell_ids length must match expression batch size")
    mask_array = masks_for_cells(expression.shape[1], cell_ids, config, epoch=epoch)
    mask = torch.as_tensor(mask_array, dtype=torch.bool, device=expression.device)
    masked_expression = expression.clone()
    masked_expression.masked_fill_(mask, float(config.mask_value))
    return MaskedBatch(
        masked_expression=masked_expression,
        visibility=~mask,
        targets=expression,
        mask=mask,
    )
