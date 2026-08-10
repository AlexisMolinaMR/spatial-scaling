"""Command-line entry point for corpus generation and QC."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import yaml

from spatial_scaling.data.synthetic.config import SyntheticConfig
from spatial_scaling.data.synthetic.generator import SyntheticCorpusGenerator
from spatial_scaling.data.synthetic.qc import compute_corpus_qc, write_qc_artifacts
from spatial_scaling.data.synthetic.storage import (
    load_section,
    prepare_output_directory,
    write_corpus_metadata,
    write_gene_ground_truth,
    write_section,
)


def _apply_override(values: dict[str, object], override: str) -> None:
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
        raise ValueError(f"override key does not exist in configuration: {dotted_key}")
    target[keys[-1]] = yaml.safe_load(raw_value)


def _load_with_overrides(path: Path, overrides: Sequence[str]) -> SyntheticConfig:
    with path.open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise TypeError("configuration root must be a mapping")
    for override in overrides:
        _apply_override(values, override)
    return SyntheticConfig.from_dict(values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a section-sharded Pilot v0 synthetic corpus and QC."
    )
    parser.add_argument("--config", type=Path, required=True, help="YAML configuration")
    parser.add_argument(
        "--output", type=Path, required=True, help="new output directory"
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override an existing dotted configuration key; repeat as needed",
    )
    parser.add_argument(
        "--skip-qc", action="store_true", help="generate data without QC artifacts"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _load_with_overrides(args.config, args.override)
    generator = SyntheticCorpusGenerator(config)
    output = prepare_output_directory(args.output)
    write_corpus_metadata(output, generator)
    write_gene_ground_truth(output, generator)
    section_paths = [
        write_section(output, section) for section in generator.generate_sections()
    ]
    if not args.skip_qc:
        metrics = compute_corpus_qc(
            (load_section(path) for path in section_paths),
            config,
            generator.gene_loadings,
        )
        write_qc_artifacts(metrics, output / "qc", config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
