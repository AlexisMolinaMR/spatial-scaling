"""Section-sharded corpus storage with explicit ground truth."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from spatial_scaling.data.synthetic.generator import (
    SyntheticCorpusGenerator,
    SyntheticSection,
)


def git_commit_sha() -> str | None:
    """Return the current commit when generation occurs in a Git worktree."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def prepare_output_directory(output_dir: str | Path) -> Path:
    """Create a new corpus directory without overwriting prior results."""
    path = Path(output_dir)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    (path / "sections").mkdir()
    (path / "qc").mkdir()
    return path


def write_corpus_metadata(
    output_dir: str | Path, generator: SyntheticCorpusGenerator
) -> None:
    """Write resolved configuration, split membership, and provenance."""
    config = generator.config
    metadata: dict[str, Any] = {
        "generator_version": config.generator_version,
        "git_commit_sha": git_commit_sha(),
        "master_seed": config.master_seed,
        "lambda_s": config.lambda_s,
        "correlation_length_um": config.spatial_field.correlation_length_um,
        "noise_std": config.noise_std,
        "latent_dimensions": {
            "intrinsic": config.latent.intrinsic_dim,
            "spatial": config.latent.spatial_dim,
        },
        "num_fourier_features": config.spatial_field.num_fourier_features,
        "section_geometry_um": {
            "width": config.geometry.width_um,
            "height": config.geometry.height_um,
        },
        "cells_per_section": config.cells_per_section,
        "num_sections": config.num_sections,
        "num_genes": config.num_genes,
        "gene_class_counts": {
            "intrinsic": config.gene_classes.intrinsic,
            "mixed": config.gene_classes.mixed,
            "spatial": config.gene_classes.spatial,
        },
        "split_unit": "section",
        "split_definition": generator.split_definition(),
        "resolved_config": config.to_dict(),
    }
    path = Path(output_dir) / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def write_gene_ground_truth(
    output_dir: str | Path, generator: SyntheticCorpusGenerator
) -> None:
    """Write gene classes, scalar coefficients, and full loading matrices."""
    output = Path(output_dir)
    loadings = generator.gene_loadings
    with (output / "genes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gene_id", "gene_class", "intrinsic_alpha", "spatial_beta"])
        writer.writerows(
            zip(
                loadings.gene_ids.tolist(),
                loadings.gene_classes.tolist(),
                loadings.intrinsic_alpha.tolist(),
                loadings.spatial_beta.tolist(),
                strict=True,
            )
        )
    np.savez_compressed(
        output / "gene_loadings.npz",
        gene_id=loadings.gene_ids,
        gene_class=loadings.gene_classes,
        intrinsic_alpha=loadings.intrinsic_alpha,
        spatial_beta=loadings.spatial_beta,
        intrinsic_loading=loadings.intrinsic,
        spatial_loading=loadings.spatial,
    )


def write_section(output_dir: str | Path, section: SyntheticSection) -> Path:
    """Write one independently loadable compressed NumPy shard."""
    path = Path(output_dir) / "sections" / f"{section.section_id}.npz"
    num_cells = len(section.cell_id)
    np.savez_compressed(
        path,
        cell_id=section.cell_id,
        section_id=np.full(num_cells, section.section_id, dtype="U12"),
        split=np.full(num_cells, section.split, dtype="U10"),
        x_um=section.coordinates_um[:, 0],
        y_um=section.coordinates_um[:, 1],
        expression=section.expression,
        intrinsic_latent=section.intrinsic_latent,
        spatial_latent=section.spatial_latent,
        intrinsic_expression=section.intrinsic_expression,
        spatial_expression=section.spatial_expression,
        noise_expression=section.noise_expression,
    )
    return path


def load_section(path: str | Path) -> SyntheticSection:
    """Load one section shard into the canonical in-memory representation."""
    with np.load(path) as data:
        section_ids = data["section_id"]
        splits = data["split"]
        if not np.all(section_ids == section_ids[0]):
            raise ValueError(f"section_id is not constant within shard: {path}")
        if not np.all(splits == splits[0]):
            raise ValueError(f"split is not constant within shard: {path}")
        return SyntheticSection(
            cell_id=data["cell_id"].copy(),
            section_id=str(section_ids[0]),
            split=str(splits[0]),
            coordinates_um=np.column_stack((data["x_um"], data["y_um"])),
            expression=data["expression"].copy(),
            intrinsic_latent=data["intrinsic_latent"].copy(),
            spatial_latent=data["spatial_latent"].copy(),
            intrinsic_expression=data["intrinsic_expression"].copy(),
            spatial_expression=data["spatial_expression"].copy(),
            noise_expression=data["noise_expression"].copy(),
        )
