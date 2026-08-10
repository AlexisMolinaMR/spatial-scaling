"""Pilot v0 synthetic spatial-expression generator."""

from spatial_scaling.data.synthetic.config import SyntheticConfig, load_config
from spatial_scaling.data.synthetic.generator import (
    SyntheticCorpusGenerator,
    SyntheticSection,
)

__all__ = [
    "SyntheticConfig",
    "SyntheticCorpusGenerator",
    "SyntheticSection",
    "load_config",
]
