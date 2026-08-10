"""Deterministic single-section and corpus generation."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np

from spatial_scaling.data.synthetic.config import SyntheticConfig
from spatial_scaling.data.synthetic.expression import (
    generate_expression,
    generate_gene_loadings,
)
from spatial_scaling.data.synthetic.geometry import sample_uniform_rectangle
from spatial_scaling.data.synthetic.random import component_rng
from spatial_scaling.data.synthetic.spatial_field import build_spatial_field


@dataclass(frozen=True)
class SyntheticSection:
    """One independent tissue section with preserved additive ground truth."""

    cell_id: np.ndarray
    section_id: str
    split: str
    coordinates_um: np.ndarray
    expression: np.ndarray
    intrinsic_latent: np.ndarray
    spatial_latent: np.ndarray
    intrinsic_expression: np.ndarray
    spatial_expression: np.ndarray
    noise_expression: np.ndarray


class SyntheticCorpusGenerator:
    """Component-seeded Pilot v0 generator."""

    def __init__(self, config: SyntheticConfig) -> None:
        config.validate()
        self.config = config
        self.spatial_field = build_spatial_field(config.spatial_field)
        self.gene_loadings = generate_gene_loadings(
            config,
            component_rng(config.master_seed, "gene_loadings", "intrinsic"),
            component_rng(config.master_seed, "gene_loadings", "spatial"),
        )

    def section_id(self, section_index: int) -> str:
        self._validate_section_index(section_index)
        return f"section_{section_index:04d}"

    def split_for_section(self, section_index: int) -> str:
        self._validate_section_index(section_index)
        if section_index < self.config.splits.train:
            return "train"
        if section_index < self.config.splits.train + self.config.splits.validation:
            return "validation"
        return "test"

    def split_definition(self) -> dict[str, list[str]]:
        result = {"train": [], "validation": [], "test": []}
        for index in range(self.config.num_sections):
            result[self.split_for_section(index)].append(self.section_id(index))
        return result

    def generate_section(self, section_index: int) -> SyntheticSection:
        """Generate a section independently of corpus order or prior calls."""
        section_id = self.section_id(section_index)
        coordinates = sample_uniform_rectangle(
            self.config.cells_per_section,
            self.config.geometry,
            component_rng(self.config.master_seed, section_id, "coordinates"),
        )
        intrinsic_latent = (
            component_rng(self.config.master_seed, section_id, "intrinsic")
            .normal(
                size=(self.config.cells_per_section, self.config.latent.intrinsic_dim)
            )
            .astype(np.float32)
        )
        spatial_latent = self.spatial_field.sample(
            coordinates,
            self.config.latent.spatial_dim,
            component_rng(self.config.master_seed, section_id, "spatial"),
        )
        expression, intrinsic, spatial, noise = generate_expression(
            intrinsic_latent,
            spatial_latent,
            self.gene_loadings,
            self.config.lambda_s,
            self.config.noise_std,
            component_rng(self.config.master_seed, section_id, "noise"),
        )
        cell_ids = np.array(
            [
                f"{section_id}_cell_{cell_index:07d}"
                for cell_index in range(self.config.cells_per_section)
            ],
            dtype="U30",
        )
        return SyntheticSection(
            cell_id=cell_ids,
            section_id=section_id,
            split=self.split_for_section(section_index),
            coordinates_um=coordinates,
            expression=expression,
            intrinsic_latent=intrinsic_latent,
            spatial_latent=spatial_latent,
            intrinsic_expression=intrinsic,
            spatial_expression=spatial,
            noise_expression=noise,
        )

    def generate_sections(
        self, section_indices: Iterable[int] | None = None
    ) -> Iterator[SyntheticSection]:
        """Yield sections in any requested order without shared RNG state."""
        indices = (
            range(self.config.num_sections)
            if section_indices is None
            else section_indices
        )
        for index in indices:
            yield self.generate_section(index)

    def _validate_section_index(self, section_index: int) -> None:
        if not 0 <= section_index < self.config.num_sections:
            raise IndexError(
                f"section index must be in [0, {self.config.num_sections}); "
                f"got {section_index}"
            )
