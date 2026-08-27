"""Deterministic proxy subset construction without dataset or filesystem I/O."""
from __future__ import annotations

import hashlib

import numpy as np


def nested_stratified_order(labels: np.ndarray, seed: int, namespace: str) -> np.ndarray:
    """Return one deterministic, nested, round-robin stratified permutation.

    Every budget ``N`` uses ``order[:N]``.  The namespace participates in a
    stable SHA-256-derived seed, avoiding Python's process-randomized ``hash``.
    """
    y = np.asarray(labels)
    if y.ndim != 1 or len(y) == 0 or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("labels must be a non-empty integer vector")
    classes = np.unique(y)
    if not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError("labels must contain contiguous classes starting at zero")
    digest = hashlib.sha256(f"proxy_nested_v1|{int(seed)}|{namespace}".encode()).digest()
    rng_seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(rng_seed)
    queues = [list(rng.permutation(np.flatnonzero(y == cls))) for cls in classes]
    order: list[int] = []
    while any(queues):
        for cls in rng.permutation(classes):
            if queues[int(cls)]:
                order.append(int(queues[int(cls)].pop()))
    result = np.asarray(order, dtype=np.int64)
    if not np.array_equal(np.sort(result), np.arange(len(y))):
        raise AssertionError("nested split is not a permutation")
    return result
