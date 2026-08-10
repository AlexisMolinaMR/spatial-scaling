"""Configuration schema for the Pilot v0 synthetic corpus."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class GeometryConfig:
    width_um: float = 2000.0
    height_um: float = 2000.0


@dataclass(frozen=True)
class LatentConfig:
    intrinsic_dim: int = 16
    spatial_dim: int = 16


@dataclass(frozen=True)
class SpatialFieldConfig:
    implementation: str = "rbf_rff"
    correlation_length_um: float = 200.0
    num_fourier_features: int = 256


@dataclass(frozen=True)
class GeneClassConfig:
    intrinsic: int = 96
    mixed: int = 64
    spatial: int = 96
    intrinsic_alpha: float = 1.0
    intrinsic_beta: float = 0.0
    mixed_alpha: float = 2**-0.5
    mixed_beta: float = 2**-0.5
    spatial_alpha: float = 0.0
    spatial_beta: float = 1.0


@dataclass(frozen=True)
class SplitConfig:
    train: int = 48
    validation: int = 8
    test: int = 8


@dataclass(frozen=True)
class QCConfig:
    num_distance_bins: int = 16
    max_distance_um: float = 800.0
    pairs_per_section: int = 50_000
    max_sections: int = 8


@dataclass(frozen=True)
class SyntheticConfig:
    generator_version: str = "pilot-v0-rff-1"
    master_seed: int = 17
    num_sections: int = 64
    cells_per_section: int = 8192
    num_genes: int = 256
    lambda_s: float = 1.0
    noise_std: float = 0.20
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    latent: LatentConfig = field(default_factory=LatentConfig)
    spatial_field: SpatialFieldConfig = field(default_factory=SpatialFieldConfig)
    gene_classes: GeneClassConfig = field(default_factory=GeneClassConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    qc: QCConfig = field(default_factory=QCConfig)

    def validate(self) -> None:
        """Reject invalid or scientifically ambiguous configurations."""
        if not isinstance(self.master_seed, int) or isinstance(self.master_seed, bool):
            raise TypeError("master_seed must be an integer")
        positive_ints = {
            "num_sections": self.num_sections,
            "cells_per_section": self.cells_per_section,
            "num_genes": self.num_genes,
            "latent.intrinsic_dim": self.latent.intrinsic_dim,
            "latent.spatial_dim": self.latent.spatial_dim,
            "spatial_field.num_fourier_features": (
                self.spatial_field.num_fourier_features
            ),
            "qc.num_distance_bins": self.qc.num_distance_bins,
            "qc.pairs_per_section": self.qc.pairs_per_section,
            "qc.max_sections": self.qc.max_sections,
        }
        for name, value in positive_ints.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(
                    f"{name} must be an integer; got {type(value).__name__}"
                )
            if value <= 0:
                raise ValueError(f"{name} must be positive; got {value}")

        remaining_ints = {
            "gene_classes.intrinsic": self.gene_classes.intrinsic,
            "gene_classes.mixed": self.gene_classes.mixed,
            "gene_classes.spatial": self.gene_classes.spatial,
            "splits.train": self.splits.train,
            "splits.validation": self.splits.validation,
            "splits.test": self.splits.test,
        }
        for name, value in remaining_ints.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(
                    f"{name} must be an integer; got {type(value).__name__}"
                )

        finite_values = {
            "geometry.width_um": self.geometry.width_um,
            "geometry.height_um": self.geometry.height_um,
            "correlation_length_um": self.spatial_field.correlation_length_um,
            "qc.max_distance_um": self.qc.max_distance_um,
            "noise_std": self.noise_std,
            "lambda_s": self.lambda_s,
            "gene_classes.intrinsic_alpha": self.gene_classes.intrinsic_alpha,
            "gene_classes.intrinsic_beta": self.gene_classes.intrinsic_beta,
            "gene_classes.mixed_alpha": self.gene_classes.mixed_alpha,
            "gene_classes.mixed_beta": self.gene_classes.mixed_beta,
            "gene_classes.spatial_alpha": self.gene_classes.spatial_alpha,
            "gene_classes.spatial_beta": self.gene_classes.spatial_beta,
        }
        for name, value in finite_values.items():
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise TypeError(f"{name} must be a finite number")

        if self.geometry.width_um <= 0 or self.geometry.height_um <= 0:
            raise ValueError("geometry width_um and height_um must be positive")
        if self.spatial_field.correlation_length_um <= 0:
            raise ValueError("correlation_length_um must be positive")
        if self.qc.max_distance_um <= 0:
            raise ValueError("qc.max_distance_um must be positive")
        if self.noise_std < 0:
            raise ValueError("noise_std must be non-negative")
        if self.lambda_s < 0:
            raise ValueError("lambda_s must be non-negative")
        if self.spatial_field.implementation != "rbf_rff":
            raise ValueError(
                "spatial_field.implementation must be 'rbf_rff' for Pilot v0"
            )

        class_total = (
            self.gene_classes.intrinsic
            + self.gene_classes.mixed
            + self.gene_classes.spatial
        )
        if class_total != self.num_genes:
            raise ValueError(
                "gene class counts must sum to num_genes; "
                f"got {class_total} != {self.num_genes}"
            )
        if (
            min(
                self.gene_classes.intrinsic,
                self.gene_classes.mixed,
                self.gene_classes.spatial,
            )
            <= 0
        ):
            raise ValueError("all three gene class counts must be positive")
        if self.gene_classes.intrinsic_beta != 0.0:
            raise ValueError("clean Pilot v0 intrinsic_beta must be zero")
        if self.gene_classes.spatial_alpha != 0.0:
            raise ValueError("clean Pilot v0 spatial_alpha must be zero")
        if self.gene_classes.intrinsic_alpha <= 0.0:
            raise ValueError("intrinsic_alpha must be positive")
        if self.gene_classes.mixed_alpha <= 0.0 or self.gene_classes.mixed_beta <= 0.0:
            raise ValueError("mixed_alpha and mixed_beta must be positive")
        if self.gene_classes.spatial_beta <= 0.0:
            raise ValueError("spatial_beta must be positive")

        split_total = self.splits.train + self.splits.validation + self.splits.test
        if min(self.splits.train, self.splits.validation, self.splits.test) < 0:
            raise ValueError("split section counts must be non-negative")
        if split_total != self.num_sections:
            raise ValueError(
                "split section counts must sum to num_sections; "
                f"got {split_total} != {self.num_sections}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the complete resolved configuration."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> SyntheticConfig:
        """Construct and validate a config from parsed YAML."""
        known = {field_.name for field_ in cls.__dataclass_fields__.values()}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown top-level configuration keys: {sorted(unknown)}")
        nested = dict(values)
        schemas = {
            "geometry": GeometryConfig,
            "latent": LatentConfig,
            "spatial_field": SpatialFieldConfig,
            "gene_classes": GeneClassConfig,
            "splits": SplitConfig,
            "qc": QCConfig,
        }
        for key, schema in schemas.items():
            if key in nested:
                try:
                    nested[key] = schema(**nested[key])
                except TypeError as error:
                    raise ValueError(f"invalid {key} configuration: {error}") from error
        try:
            config = cls(**nested)
        except TypeError as error:
            raise ValueError(f"invalid configuration: {error}") from error
        config.validate()
        return config


def load_config(path: str | Path) -> SyntheticConfig:
    """Load and validate YAML configuration from ``path``."""
    with Path(path).open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise TypeError("configuration root must be a mapping")
    return SyntheticConfig.from_dict(values)
