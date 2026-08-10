from __future__ import annotations

import json
from pathlib import Path

from spatial_scaling.data.synthetic.cli import main

ROOT = Path(__file__).parents[1]


def test_cli_generates_sharded_corpus_and_qc(tmp_path: Path) -> None:
    output = tmp_path / "corpus"
    result = main(
        [
            "--config",
            str(ROOT / "configs/pilot/synthetic_spatial.yaml"),
            "--output",
            str(output),
            "--override",
            "num_sections=3",
            "--override",
            "cells_per_section=96",
            "--override",
            "splits.train=1",
            "--override",
            "splits.validation=1",
            "--override",
            "splits.test=1",
            "--override",
            "spatial_field.num_fourier_features=24",
            "--override",
            "qc.num_distance_bins=4",
            "--override",
            "qc.pairs_per_section=10000",
            "--override",
            "qc.max_sections=3",
        ]
    )
    assert result == 0
    assert len(list((output / "sections").glob("*.npz"))) == 3
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["split_unit"] == "section"
    assert metadata["resolved_config"]["cells_per_section"] == 96
    assert (output / "genes.csv").is_file()
    assert (output / "gene_loadings.npz").is_file()
    assert (output / "qc/metrics.json").is_file()
    assert (output / "qc/spatial_latent_correlation.png").is_file()
    assert (output / "qc/expression_correlation_by_gene_class.png").is_file()
    assert (output / "qc/variance_decomposition.png").is_file()
    assert (output / "qc/section_independence.png").is_file()
