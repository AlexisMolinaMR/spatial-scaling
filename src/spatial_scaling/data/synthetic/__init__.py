"""Pilot v0 synthetic spatial-expression generator."""

from spatial_scaling.data.synthetic.config import SyntheticConfig, load_config
from spatial_scaling.data.synthetic.context import ContextBatch, SyntheticContextDataset
from spatial_scaling.data.synthetic.dataset import (
    GeneMetadata,
    SyntheticExpressionDataset,
)
from spatial_scaling.data.synthetic.generator import (
    SyntheticCorpusGenerator,
    SyntheticSection,
)

__all__ = [
    "ContextBatch",
    "GeneMetadata",
    "SyntheticConfig",
    "SyntheticContextDataset",
    "SyntheticCorpusGenerator",
    "SyntheticExpressionDataset",
    "SyntheticSection",
    "load_config",
]
