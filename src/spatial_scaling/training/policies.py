"""Explicit data selection and optimization policies for N-train scaling."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ValidationPlateauController:
    """Deterministic validation-check-driven learning-rate reducer."""

    reduction_factor: float
    reduction_patience: int
    minimum_learning_rate: float
    insufficient_checks: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.reduction_factor < 1.0:
            raise ValueError("reduction_factor must be between zero and one")
        if (
            not isinstance(self.reduction_patience, int)
            or isinstance(self.reduction_patience, bool)
            or self.reduction_patience <= 0
        ):
            raise ValueError("reduction_patience must be a positive integer")
        if self.minimum_learning_rate <= 0.0:
            raise ValueError("minimum_learning_rate must be positive")

    def update(
        self, *, improved: bool, current_learning_rate: float
    ) -> tuple[float, bool]:
        """Return the next LR and whether an actual reduction occurred."""
        if current_learning_rate <= 0.0:
            raise ValueError("current_learning_rate must be positive")
        if improved:
            self.insufficient_checks = 0
            return current_learning_rate, False
        self.insufficient_checks += 1
        if self.insufficient_checks < self.reduction_patience:
            return current_learning_rate, False
        next_learning_rate = max(
            current_learning_rate * self.reduction_factor,
            self.minimum_learning_rate,
        )
        if next_learning_rate >= current_learning_rate:
            return current_learning_rate, False
        self.insufficient_checks = 0
        return next_learning_rate, True


@dataclass(frozen=True)
class ValidationPolicyDecision:
    """One deterministic validation-policy decision."""

    improved: bool
    learning_rate: float
    lr_reduced: bool
    should_stop: bool
    insufficient_checks: int


@dataclass
class ValidationConvergenceController:
    """Coordinate plateau reduction and early stopping without shared events."""

    best_loss: float
    improvement_threshold: float
    early_stopping_patience: int
    plateau: ValidationPlateauController | None = None
    insufficient_checks: int = 0

    def __post_init__(self) -> None:
        if self.improvement_threshold < 0.0:
            raise ValueError("improvement_threshold must be non-negative")
        if (
            not isinstance(self.early_stopping_patience, int)
            or isinstance(self.early_stopping_patience, bool)
            or self.early_stopping_patience <= 0
        ):
            raise ValueError("early_stopping_patience must be a positive integer")

    def update(
        self, *, validation_loss: float, current_learning_rate: float
    ) -> ValidationPolicyDecision:
        """Update counters and return the LR/stop decision for one validation."""
        improved = self.best_loss - validation_loss > self.improvement_threshold
        if improved:
            self.best_loss = validation_loss
            self.insufficient_checks = 0
        else:
            self.insufficient_checks += 1
        next_lr = current_learning_rate
        lr_reduced = False
        if self.plateau is not None:
            next_lr, lr_reduced = self.plateau.update(
                improved=improved,
                current_learning_rate=current_learning_rate,
            )
            if lr_reduced:
                self.insufficient_checks = 0
        return ValidationPolicyDecision(
            improved=improved,
            learning_rate=next_lr,
            lr_reduced=lr_reduced,
            should_stop=(
                not lr_reduced
                and self.insufficient_checks >= self.early_stopping_patience
            ),
            insufficient_checks=self.insufficient_checks,
        )


@dataclass(frozen=True)
class TrainingDataSelection:
    """One nested subset of the corpus's fixed training split."""

    scaling_axis: str
    subset_selection_seed: int
    ordered_train_section_ids: tuple[str, ...]
    active_train_section_ids: tuple[str, ...]

    def validate(self) -> None:
        if self.scaling_axis != "N_train_sections":
            raise ValueError("scaling_axis must be 'N_train_sections'")
        if not isinstance(self.subset_selection_seed, int) or isinstance(
            self.subset_selection_seed, bool
        ):
            raise TypeError("subset_selection_seed must be an integer")
        if not self.ordered_train_section_ids:
            raise ValueError("ordered_train_section_ids must not be empty")
        if len(set(self.ordered_train_section_ids)) != len(
            self.ordered_train_section_ids
        ):
            raise ValueError("ordered_train_section_ids must be unique")
        if not self.active_train_section_ids:
            raise ValueError("active_train_section_ids must not be empty")
        expected = self.ordered_train_section_ids[: len(self.active_train_section_ids)]
        if self.active_train_section_ids != expected:
            raise ValueError("active training sections must be an ordered prefix")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> TrainingDataSelection:
        converted = dict(values)
        for key in ("ordered_train_section_ids", "active_train_section_ids"):
            value = converted.get(key)
            if isinstance(value, list):
                converted[key] = tuple(value)
        result = cls(**converted)
        result.validate()
        return result


@dataclass(frozen=True)
class TrainingExecutionPolicy:
    """Common optimization stopping and sampling policy."""

    name: str
    sampling: str
    lr_schedule: str = "constant"
    minimum_epochs: int | None = None
    maximum_epochs: int | None = None
    validation_interval_epochs: int | None = None
    early_stopping_patience: int | None = None
    improvement_threshold: float | None = None
    absolute_max_steps: int | None = None
    fixed_steps: int | None = None
    restore_best_validation: bool = True
    lr_reduction_factor: float | None = None
    lr_reduction_patience: int | None = None
    minimum_learning_rate: float | None = None

    def validate(self) -> None:
        if self.name not in {
            "converged_data_frontier",
            "fixed_compute_2000_steps",
            "smoke_minimum_exposure",
            "prolonged_constant_lr",
            "validation_plateau_lr_decay",
        }:
            raise ValueError(f"unknown training execution policy: {self.name!r}")
        if self.lr_schedule not in {"constant", "validation_plateau_decay"}:
            raise ValueError(f"unknown LR schedule: {self.lr_schedule!r}")
        if self.sampling == "deterministic_shuffled_epochs":
            required = {
                "minimum_epochs": self.minimum_epochs,
                "maximum_epochs": self.maximum_epochs,
                "validation_interval_epochs": self.validation_interval_epochs,
                "early_stopping_patience": self.early_stopping_patience,
                "improvement_threshold": self.improvement_threshold,
                "absolute_max_steps": self.absolute_max_steps,
            }
            if any(value is None for value in required.values()):
                raise ValueError(f"epoch policy requires {sorted(required)}")
            for name in (
                "minimum_epochs",
                "maximum_epochs",
                "validation_interval_epochs",
                "early_stopping_patience",
                "absolute_max_steps",
            ):
                value = getattr(self, name)
                if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                    raise ValueError(f"{name} must be a positive integer")
            if self.minimum_epochs > self.maximum_epochs:
                raise ValueError("minimum_epochs cannot exceed maximum_epochs")
            if self.validation_interval_epochs != 1:
                raise ValueError("Pilot v0 validates once after every complete pass")
            if self.improvement_threshold < 0:
                raise ValueError("improvement_threshold must be non-negative")
            if self.fixed_steps is not None:
                raise ValueError("epoch policy cannot also set fixed_steps")
            scheduler_fields = (
                self.lr_reduction_factor,
                self.lr_reduction_patience,
                self.minimum_learning_rate,
            )
            if self.lr_schedule == "constant":
                if any(value is not None for value in scheduler_fields):
                    raise ValueError(
                        "constant LR policy cannot set plateau scheduler fields"
                    )
            else:
                if any(value is None for value in scheduler_fields):
                    raise ValueError(
                        "plateau LR policy requires factor, patience, and minimum LR"
                    )
                ValidationPlateauController(
                    reduction_factor=self.lr_reduction_factor,
                    reduction_patience=self.lr_reduction_patience,
                    minimum_learning_rate=self.minimum_learning_rate,
                )
                if self.lr_reduction_patience >= self.early_stopping_patience:
                    raise ValueError(
                        "LR-reduction patience must be shorter than early stopping"
                    )
                if self.name != "validation_plateau_lr_decay":
                    raise ValueError(
                        "validation plateau schedule requires its named policy"
                    )
        elif self.sampling == "historical_random_with_replacement":
            if self.fixed_steps != 2000:
                raise ValueError("fixed-compute policy must use exactly 2,000 steps")
            epoch_fields = (
                self.minimum_epochs,
                self.maximum_epochs,
                self.validation_interval_epochs,
                self.early_stopping_patience,
                self.improvement_threshold,
                self.absolute_max_steps,
            )
            if any(value is not None for value in epoch_fields):
                raise ValueError("fixed-compute policy cannot set epoch fields")
            if self.lr_schedule != "constant":
                raise ValueError("fixed-compute policy requires constant LR")
            if any(
                value is not None
                for value in (
                    self.lr_reduction_factor,
                    self.lr_reduction_patience,
                    self.minimum_learning_rate,
                )
            ):
                raise ValueError("fixed-compute policy cannot set scheduler fields")
        else:
            raise ValueError(f"unknown sampling policy: {self.sampling!r}")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        values = asdict(self)
        for key in (
            "lr_reduction_factor",
            "lr_reduction_patience",
            "minimum_learning_rate",
        ):
            if values[key] is None:
                values.pop(key)
        return values

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> TrainingExecutionPolicy:
        result = cls(**values)
        result.validate()
        return result
