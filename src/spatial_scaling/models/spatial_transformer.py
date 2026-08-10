"""Modest cell-token transformer for the first spatial falsification test."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class SpatialTransformerConfig:
    num_genes: int
    context_size: int = 256
    embedding_dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    ffn_width: int = 128
    dropout: float = 0.0
    coordinate_scale_um: float = 200.0

    def validate(self) -> None:
        for name, minimum in (("num_genes", 2), ("context_size", 1)):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}")
        for name in ("embedding_dim", "num_layers", "num_heads", "ffn_width"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.embedding_dim % self.num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.coordinate_scale_um <= 0:
            raise ValueError("coordinate_scale_um must be positive")


class SpatialTransformer(nn.Module):
    """Predict focal masked genes from cell tokens and relative 2D geometry.

    Focal and context cells pass through one shared encoder. Context cells are
    fully molecularly observed in Pilot v0 and therefore receive all-one
    visibility vectors. Only the focal expression is masked and predicted.
    The contextual block is isolated behind ``self.context_encoder`` so dense
    attention can later be replaced without changing the neighborhood or model
    input interfaces.
    """

    def __init__(self, config: SpatialTransformerConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.cell_encoder = nn.Sequential(
            nn.Linear(2 * config.num_genes, config.embedding_dim),
            nn.GELU(),
            nn.LayerNorm(config.embedding_dim),
        )
        self.relative_position_encoder = nn.Sequential(
            nn.Linear(2, config.embedding_dim),
            nn.GELU(),
            nn.Linear(config.embedding_dim, config.embedding_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ffn_width,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(
            layer, num_layers=config.num_layers, enable_nested_tensor=False
        )
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(config.embedding_dim),
            nn.Linear(config.embedding_dim, config.num_genes),
        )

    def forward(
        self,
        masked_focal_expression: torch.Tensor,
        focal_visibility: torch.Tensor,
        context_expression: torch.Tensor,
        relative_coordinates_um: torch.Tensor,
    ) -> torch.Tensor:
        if masked_focal_expression.ndim != 2:
            raise ValueError("masked focal expression must have shape [batch, genes]")
        if focal_visibility.shape != masked_focal_expression.shape:
            raise ValueError("focal visibility must match focal expression")
        batch_size, num_genes = masked_focal_expression.shape
        expected_context = (batch_size, self.config.context_size, num_genes)
        if context_expression.shape != expected_context:
            raise ValueError(
                f"context expression must have shape {expected_context}; got "
                f"{tuple(context_expression.shape)}"
            )
        expected_coordinates = (batch_size, self.config.context_size, 2)
        if relative_coordinates_um.shape != expected_coordinates:
            raise ValueError(
                f"relative coordinates must have shape {expected_coordinates}"
            )
        if num_genes != self.config.num_genes:
            raise ValueError("input gene dimension does not match model configuration")

        expression = torch.cat(
            (masked_focal_expression[:, None, :], context_expression), dim=1
        )
        context_visibility = torch.ones_like(context_expression, dtype=torch.bool)
        visibility = torch.cat(
            (focal_visibility[:, None, :], context_visibility), dim=1
        )
        cell_input = torch.cat((expression, visibility.to(expression)), dim=-1)
        tokens = self.cell_encoder(cell_input)

        focal_coordinates = torch.zeros(
            (batch_size, 1, 2),
            dtype=relative_coordinates_um.dtype,
            device=relative_coordinates_um.device,
        )
        relative_coordinates = torch.cat(
            (focal_coordinates, relative_coordinates_um), dim=1
        )
        tokens = tokens + self.relative_position_encoder(
            relative_coordinates / self.config.coordinate_scale_um
        )
        contextualized = self.context_encoder(tokens)
        return self.prediction_head(contextualized[:, 0])

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
