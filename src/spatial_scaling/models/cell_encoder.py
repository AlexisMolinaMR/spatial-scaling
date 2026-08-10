"""Deliberately small focal-cell-only masked-expression baseline."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import torch
from torch import nn


@dataclass(frozen=True)
class CellMLPConfig:
    num_genes: int
    hidden_dims: tuple[int, ...] = (128, 128)
    activation: str = "gelu"
    dropout: float = 0.0

    def validate(self) -> None:
        if not isinstance(self.num_genes, int) or self.num_genes < 2:
            raise ValueError("num_genes must be an integer of at least 2")
        if not self.hidden_dims or any(
            not isinstance(width, int) or width <= 0 for width in self.hidden_dims
        ):
            raise ValueError("hidden_dims must contain positive integers")
        if self.activation not in {"gelu", "relu"}:
            raise ValueError("activation must be 'gelu' or 'relu'")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class CellMLP(nn.Module):
    """Predict masked genes from focal expression and its visibility mask only."""

    def __init__(self, config: CellMLPConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        activation: type[nn.Module] = (
            nn.GELU if config.activation == "gelu" else nn.ReLU
        )
        widths = (2 * config.num_genes, *config.hidden_dims)
        layers: list[nn.Module] = []
        for input_width, output_width in pairwise(widths):
            layers.extend((nn.Linear(input_width, output_width), activation()))
            if config.dropout:
                layers.append(nn.Dropout(config.dropout))
        self.encoder = nn.Sequential(*layers)
        self.prediction_head = nn.Linear(config.hidden_dims[-1], config.num_genes)

    def forward(
        self, masked_expression: torch.Tensor, visibility: torch.Tensor
    ) -> torch.Tensor:
        if masked_expression.ndim != 2:
            raise ValueError("masked_expression must have shape [batch, genes]")
        if masked_expression.shape != visibility.shape:
            raise ValueError("visibility must match masked_expression shape")
        if masked_expression.shape[1] != self.config.num_genes:
            raise ValueError("input gene dimension does not match model configuration")
        model_input = torch.cat(
            (masked_expression, visibility.to(masked_expression)), dim=1
        )
        return self.prediction_head(self.encoder(model_input))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
