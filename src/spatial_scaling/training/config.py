"""Explicit configuration schema for Pilot v0 focal-cell SSL."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from spatial_scaling.models.cell_encoder import CellMLPConfig
from spatial_scaling.training.masking import MaskingConfig


@dataclass(frozen=True)
class DataConfig:
    corpus_path: str = "outputs/pilot_v0_synthetic_spatial"
    cache_sections: int = 2


@dataclass(frozen=True)
class OptimizerConfig:
    name: str = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4


@dataclass(frozen=True)
class TrainLoopConfig:
    batch_size: int = 512
    steps: int = 2000
    evaluation_interval: int = 250
    mask_epoch_steps: int = 250


@dataclass(frozen=True)
class EvaluationConfig:
    split: str = "validation"
    batch_size: int = 1024
    max_cells: int | None = 65536
    observation_seed: int = 991
    masking_seed: int = 992


@dataclass(frozen=True)
class SSLExperimentConfig:
    experiment_name: str = "pilot_ssl"
    output_dir: str = "outputs/pilot_ssl/run"
    seed: int = 123
    device: str = "auto"
    data: DataConfig = field(default_factory=DataConfig)
    model: dict[str, Any] = field(
        default_factory=lambda: {
            "hidden_dims": [128, 128],
            "activation": "gelu",
            "dropout": 0.0,
        }
    )
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainLoopConfig = field(default_factory=TrainLoopConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self) -> None:
        if not self.experiment_name:
            raise ValueError("experiment_name must not be empty")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an integer")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
        if self.data.cache_sections < 1:
            raise ValueError("data.cache_sections must be at least 1")
        if self.optimizer.name != "adamw":
            raise ValueError("optimizer.name must be 'adamw'")
        if self.optimizer.learning_rate <= 0:
            raise ValueError("optimizer.learning_rate must be positive")
        if self.optimizer.weight_decay < 0:
            raise ValueError("optimizer.weight_decay must be non-negative")
        for name in (
            "batch_size",
            "steps",
            "evaluation_interval",
            "mask_epoch_steps",
        ):
            if getattr(self.training, name) <= 0:
                raise ValueError(f"training.{name} must be positive")
        if self.training.evaluation_interval > self.training.steps:
            raise ValueError("evaluation_interval cannot exceed training steps")
        if self.evaluation.split not in {"validation", "test"}:
            raise ValueError("evaluation.split must be 'validation' or 'test'")
        if self.evaluation.batch_size <= 0:
            raise ValueError("evaluation.batch_size must be positive")
        if self.evaluation.max_cells is not None and self.evaluation.max_cells <= 0:
            raise ValueError("evaluation.max_cells must be positive or null")
        self.masking.validate()
        unknown_model = set(self.model) - {"hidden_dims", "activation", "dropout"}
        if unknown_model:
            raise ValueError(
                f"unknown model configuration keys: {sorted(unknown_model)}"
            )
        CellMLPConfig(num_genes=2, **self._normalized_model()).validate()

    def _normalized_model(self) -> dict[str, Any]:
        values = dict(self.model)
        if "hidden_dims" in values:
            values["hidden_dims"] = tuple(values["hidden_dims"])
        return values

    def model_config(self, num_genes: int) -> CellMLPConfig:
        return CellMLPConfig(num_genes=num_genes, **self._normalized_model())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> SSLExperimentConfig:
        known = {item.name for item in cls.__dataclass_fields__.values()}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
        nested = dict(values)
        schemas = {
            "data": DataConfig,
            "masking": MaskingConfig,
            "optimizer": OptimizerConfig,
            "training": TrainLoopConfig,
            "evaluation": EvaluationConfig,
        }
        for key, schema in schemas.items():
            if key in nested:
                if not isinstance(nested[key], dict):
                    raise TypeError(f"{key} configuration must be a mapping")
                try:
                    nested[key] = schema(**nested[key])
                except TypeError as error:
                    raise ValueError(f"invalid {key} configuration: {error}") from error
        try:
            config = cls(**nested)
        except TypeError as error:
            raise ValueError(f"invalid experiment configuration: {error}") from error
        config.validate()
        return config


def apply_override(values: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"override must have KEY=VALUE form: {override!r}")
    dotted_key, raw_value = override.split("=", maxsplit=1)
    keys = dotted_key.split(".")
    if any(not key for key in keys):
        raise ValueError(f"invalid override key: {dotted_key!r}")
    target = values
    for key in keys[:-1]:
        child = target.get(key)
        if not isinstance(child, dict):
            raise TypeError(f"override parent is not a mapping: {key!r}")
        target = child
    if keys[-1] not in target:
        raise ValueError(f"override key does not exist: {dotted_key}")
    target[keys[-1]] = yaml.safe_load(raw_value)


def load_experiment_config(
    path: str | Path, overrides: list[str] | tuple[str, ...] = ()
) -> SSLExperimentConfig:
    with Path(path).open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise TypeError("configuration root must be a mapping")
    for override in overrides:
        apply_override(values, override)
    return SSLExperimentConfig.from_dict(values)
