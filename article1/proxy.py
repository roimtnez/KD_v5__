"""Nested public-proxy subsets; never change the private partitions."""

from __future__ import annotations

import numpy as np


def proxy_positions(indices, labels, size: int | None, seed: int) -> np.ndarray:
    """Return cache row positions, stratified and nested across requested sizes.

    Shuffle each class in canonical dataset-index order, then interleave classes.
    Selection uses public labels only. Return rows in the original cache order
    so full-proxy runs retain their historical batch ordering exactly.
    """
    indices, labels = np.asarray(indices), np.asarray(labels)
    if (
        indices.ndim != 1
        or not len(indices)
        or labels.shape != indices.shape
        or indices.dtype.kind not in "iu"
        or labels.dtype.kind not in "iu"
        or (indices < 0).any()
        or (labels < 0).any()
        or len(np.unique(indices)) != len(indices)
    ):
        raise ValueError(
            "proxy requires unique nonnegative integer indices and aligned labels"
        )
    size = len(indices) if size is None else size
    if not isinstance(size, (int, np.integer)) or not 1 <= size <= len(indices):
        raise ValueError("proxy size must be between 1 and the reserved proxy size")
    if size == len(indices):
        return np.arange(len(indices), dtype=np.int64)
    rng = np.random.default_rng(seed)
    groups = []
    for label in np.unique(labels):
        rows = np.flatnonzero(labels == label)
        rows = rows[np.argsort(indices[rows])]
        groups.append(rng.permutation(rows))
    selected = []
    for depth in range(max(map(len, groups))):
        for group in groups:
            if depth < len(group):
                selected.append(group[depth])
                if len(selected) == size:
                    return np.sort(np.asarray(selected, dtype=np.int64))
    raise AssertionError("unreachable proxy selection")
