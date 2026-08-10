"""Stable, component-addressed random streams."""

from __future__ import annotations

import hashlib

import numpy as np


def component_seed(master_seed: int, *components: object) -> int:
    """Derive a stable 128-bit seed without Python's randomized ``hash``."""
    payload = "\x1f".join([str(master_seed), *(str(item) for item in components)])
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=16).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def component_rng(master_seed: int, *components: object) -> np.random.Generator:
    """Return an RNG uniquely addressed by conceptual components."""
    return np.random.default_rng(component_seed(master_seed, *components))
