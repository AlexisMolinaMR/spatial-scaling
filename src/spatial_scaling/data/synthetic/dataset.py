"""Lazy, split-safe loading for the Pilot v0 synthetic corpus."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset

SPLITS = ("train", "validation", "test")
GENE_CLASSES = ("intrinsic", "mixed", "spatial")


@dataclass(frozen=True)
class GeneMetadata:
    """Gene identifiers and evaluation-only synthetic classes."""

    gene_ids: tuple[str, ...]
    gene_classes: tuple[str, ...]

    def class_mask(self, gene_class: str) -> np.ndarray:
        if gene_class not in GENE_CLASSES:
            raise ValueError(f"unknown gene class: {gene_class}")
        return np.asarray(self.gene_classes) == gene_class


def _stable_seed(*components: object) -> int:
    digest = hashlib.blake2b(digest_size=8)
    for component in components:
        encoded = str(component).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    return int.from_bytes(digest.digest(), "little")


class SyntheticExpressionDataset(Dataset[dict[str, Any]]):
    """Map-style expression dataset backed by section-level NPZ shards.

    Only shards from the requested pre-existing section split are addressable.
    A small per-process LRU cache avoids loading the full corpus while keeping
    repeated access within a section efficient. Coordinates and generator
    ground truth are deliberately absent from returned model examples.
    """

    def __init__(
        self,
        corpus_path: str | Path,
        split: str,
        *,
        cache_sections: int = 2,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}; got {split!r}")
        if cache_sections < 1:
            raise ValueError("cache_sections must be at least 1")
        self.corpus_path = Path(corpus_path)
        self.split = split
        self.cache_sections = cache_sections
        self.metadata = self._load_metadata()
        self.gene_metadata = self._load_genes()
        self.section_ids_by_split = self._validate_split_definition()
        self.section_ids = self.section_ids_by_split[split]
        if not self.section_ids:
            raise ValueError(f"corpus split {split!r} contains no sections")
        self.cells_per_section = self._positive_metadata_int("cells_per_section")
        self.num_genes = self._positive_metadata_int("num_genes")
        if self.num_genes != len(self.gene_metadata.gene_ids):
            raise ValueError("metadata num_genes does not match genes.csv")
        self._shard_paths = {
            section_id: self.corpus_path / "sections" / f"{section_id}.npz"
            for section_id in self.section_ids
        }
        missing = [
            str(path) for path in self._shard_paths.values() if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(f"missing section shards: {missing[:3]}")
        self._cache: OrderedDict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = (
            OrderedDict()
        )
        self._validate_shard(self.section_ids[0])

    def _load_metadata(self) -> dict[str, Any]:
        path = self.corpus_path / "metadata.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing corpus metadata: {path}")
        values = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise TypeError("corpus metadata must be a JSON object")
        if values.get("split_unit") != "section":
            raise ValueError("corpus split_unit must be 'section'")
        return values

    def _load_genes(self) -> GeneMetadata:
        path = self.corpus_path / "genes.csv"
        if not path.is_file():
            raise FileNotFoundError(f"missing gene metadata: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        expected = {"gene_id", "gene_class"}
        if not rows or not expected.issubset(rows[0]):
            raise ValueError("genes.csv must contain gene_id and gene_class columns")
        gene_ids = tuple(row["gene_id"] for row in rows)
        gene_classes = tuple(row["gene_class"] for row in rows)
        if len(set(gene_ids)) != len(gene_ids):
            raise ValueError("gene identifiers must be unique")
        invalid = sorted(set(gene_classes) - set(GENE_CLASSES))
        if invalid:
            raise ValueError(f"unknown gene classes in genes.csv: {invalid}")
        return GeneMetadata(gene_ids=gene_ids, gene_classes=gene_classes)

    def _validate_split_definition(self) -> dict[str, tuple[str, ...]]:
        definition = self.metadata.get("split_definition")
        if not isinstance(definition, dict) or set(definition) != set(SPLITS):
            raise ValueError(f"split_definition must contain exactly {SPLITS}")
        result: dict[str, tuple[str, ...]] = {}
        for name in SPLITS:
            values = definition[name]
            if not isinstance(values, list) or not all(
                isinstance(value, str) and value for value in values
            ):
                raise ValueError(f"split_definition.{name} must be a list of IDs")
            result[name] = tuple(values)
        split_sets = {name: set(values) for name, values in result.items()}
        for index, left in enumerate(SPLITS):
            for right in SPLITS[index + 1 :]:
                if not split_sets[left].isdisjoint(split_sets[right]):
                    raise ValueError(f"section leakage between {left} and {right}")
        all_ids = tuple(section for name in SPLITS for section in result[name])
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("section IDs must be globally unique")
        if len(all_ids) != self._positive_metadata_int("num_sections"):
            raise ValueError("split section count does not match num_sections")
        return result

    def _positive_metadata_int(self, name: str) -> int:
        value = self.metadata.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"metadata {name} must be a positive integer")
        return value

    def _validate_shard(self, section_id: str) -> None:
        expression, cell_ids, coordinates = self._load_shard(section_id)
        if expression.shape != (self.cells_per_section, self.num_genes):
            raise ValueError(
                f"invalid expression shape for {section_id}: {expression.shape}"
            )
        if cell_ids.shape != (self.cells_per_section,):
            raise ValueError(
                f"invalid cell_id shape for {section_id}: {cell_ids.shape}"
            )
        if coordinates.shape != (self.cells_per_section, 2):
            raise ValueError(
                f"invalid coordinate shape for {section_id}: {coordinates.shape}"
            )

    def _load_shard(self, section_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cached = self._cache.get(section_id)
        if cached is not None:
            self._cache.move_to_end(section_id)
            return cached
        path = self._shard_paths[section_id]
        with np.load(path, allow_pickle=False) as data:
            required = {
                "expression",
                "cell_id",
                "section_id",
                "split",
                "x_um",
                "y_um",
            }
            if not required.issubset(data.files):
                raise ValueError(f"section shard lacks required arrays: {path}")
            stored_sections = data["section_id"]
            stored_splits = data["split"]
            if not np.all(stored_sections == section_id):
                raise ValueError(f"section_id mismatch in shard: {path}")
            if not np.all(stored_splits == self.split):
                raise ValueError(f"split mismatch in shard: {path}")
            expression = data["expression"].astype(np.float32, copy=True)
            cell_ids = data["cell_id"].astype(str, copy=True)
            coordinates = np.column_stack((data["x_um"], data["y_um"])).astype(
                np.float32, copy=False
            )
        expression.flags.writeable = False
        cell_ids.flags.writeable = False
        coordinates.flags.writeable = False
        value = (expression, cell_ids, coordinates)
        self._cache[section_id] = value
        while len(self._cache) > self.cache_sections:
            self._cache.popitem(last=False)
        return value

    def __len__(self) -> int:
        return len(self.section_ids) * self.cells_per_section

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not isinstance(index, (int, np.integer)):
            raise TypeError("dataset index must be an integer")
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        section_offset, cell_offset = divmod(index, self.cells_per_section)
        section_id = self.section_ids[section_offset]
        expression, cell_ids, _ = self._load_shard(section_id)
        return {
            "expression": expression[cell_offset],
            "cell_id": str(cell_ids[cell_offset]),
            "section_id": section_id,
            "split": self.split,
        }

    def section_batch(
        self, section_offset: int, cell_offsets: np.ndarray
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        """Read selected cells from one split section without global shuffling."""
        if not 0 <= section_offset < len(self.section_ids):
            raise IndexError(section_offset)
        offsets = np.asarray(cell_offsets)
        if offsets.ndim != 1 or not np.issubdtype(offsets.dtype, np.integer):
            raise TypeError("cell_offsets must be a one-dimensional integer array")
        if np.any(offsets < 0) or np.any(offsets >= self.cells_per_section):
            raise IndexError("cell offset is outside its section")
        expression, cell_ids, _ = self._load_shard(self.section_ids[section_offset])
        return expression[offsets].copy(), tuple(cell_ids[offsets].tolist())

    def spatial_section(
        self, section_offset: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read model-safe expression, coordinates, and IDs for one section.

        Generator latent states, expression decompositions, gene classes, and
        split labels are deliberately not returned through this interface.
        """
        if not 0 <= section_offset < len(self.section_ids):
            raise IndexError(section_offset)
        expression, cell_ids, coordinates = self._load_shard(
            self.section_ids[section_offset]
        )
        return expression, coordinates, cell_ids

    def deterministic_indices(self, count: int | None, seed: int) -> np.ndarray:
        """Choose a fixed observation subset without changing split membership."""
        if count is None or count >= len(self):
            return np.arange(len(self), dtype=np.int64)
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError("count must be a positive integer or None")
        rng = np.random.default_rng(
            _stable_seed("synthetic-expression-subset", self.split, seed)
        )
        return np.sort(rng.choice(len(self), size=count, replace=False))
