"""Proxy-first, disjoint Article-1 partitions."""
from __future__ import annotations

from pathlib import Path
import json
import os
import tempfile

import numpy as np

from article1 import REGIMES


def balanced_order(labels: np.ndarray, seed: int) -> np.ndarray:
    """A deterministic class-balanced order: every prefix is nested."""
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or not len(labels):
        raise ValueError("labels must be non-empty [N]")
    rng = np.random.default_rng(seed)
    queues = [list(rng.permutation(np.flatnonzero(labels == c))) for c in range(int(labels.max()) + 1)]
    if any(not q for q in queues):
        raise ValueError("each class must occur in the training data")
    out: list[int] = []
    while any(queues):
        for c in rng.permutation(len(queues)):
            if queues[c]:
                out.append(queues[c].pop())
    return np.asarray(out, dtype=np.int64)


def _assign(pool: np.ndarray, labels: np.ndarray, regime: str, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    if regime not in REGIMES:
        raise ValueError(f"unsupported regime {regime!r}")
    if regime == "iid":
        groups = [[] for _ in range(k)]
        for c in range(int(labels.max()) + 1):
            rows = pool[labels[pool] == c].copy(); rng.shuffle(rows)
            for client, part in enumerate(np.array_split(rows, k)):
                groups[client].append(part)
        return [np.concatenate(parts) for parts in groups]
    if regime.startswith("alpha"):
        alpha = {"alpha1p0": 1.0, "alpha0p5": 0.5, "alpha0p1": 0.1}[regime]
        groups = [[] for _ in range(k)]
        for c in range(int(labels.max()) + 1):
            rows = pool[labels[pool] == c].copy(); rng.shuffle(rows)
            cuts = (rng.dirichlet(np.full(k, alpha)) * len(rows)).astype(int)
            cuts[np.argmax(cuts)] += len(rows) - cuts.sum()
            start = 0
            for client, count in enumerate(cuts):
                groups[client].append(rows[start:start + count]); start += count
        return [np.concatenate(parts) for parts in groups]
    classes = int(labels.max()) + 1
    per_client = 1 if regime == "single" else 2
    buckets = {c: rng.permutation(pool[labels[pool] == c]) for c in range(classes)}
    groups = [[] for _ in range(k)]
    ownership = [[(client * per_client + j) % classes for j in range(per_client)] for client in range(k)]
    for c, values in buckets.items():
        owners = [client for client, owned in enumerate(ownership) if c in owned]
        for client, part in zip(owners, np.array_split(values, len(owners))):
            groups[client].append(part)
    # ``group`` holds one array per class owned by the client.  Converting the
    # list directly with ``asarray`` produces a 2-D array for ``single`` and a
    # ragged-array error for ``multi``; both are index vectors, not matrices.
    return [
        np.concatenate(group).astype(np.int64, copy=False)
        if group else np.empty(0, dtype=np.int64)
        for group in groups
    ]


def _split_client(indices: np.ndarray, seed: int, holdout_fraction: float, test_fraction: float) -> dict[str, np.ndarray]:
    values = np.asarray(indices, dtype=np.int64).copy()
    rng = np.random.default_rng(seed); rng.shuffle(values)
    n_holdout = int(round(len(values) * holdout_fraction))
    n_test = int(round(len(values) * test_fraction))
    if n_holdout + n_test >= len(values):
        raise ValueError("holdout and local-test fractions leave no train examples")
    return {"train_idx": values[n_holdout + n_test:], "holdout_idx": values[:n_holdout], "test_idx": values[n_holdout:n_holdout + n_test]}


def validate_splits(proxy_idx: np.ndarray, clients: list[dict[str, np.ndarray]]) -> None:
    """Fail closed on overlap across proxy, roles, or clients."""
    all_values = [np.asarray(proxy_idx, dtype=np.int64)]
    for client in clients:
        required = {"train_idx", "holdout_idx", "test_idx"}
        if set(client) != required:
            raise ValueError("each client requires train_idx, holdout_idx and test_idx")
        roles = [np.asarray(client[key], dtype=np.int64) for key in sorted(required)]
        if any(len(x) != len(np.unique(x)) for x in roles):
            raise ValueError("duplicate example within a split")
        if any(np.intersect1d(roles[i], roles[j]).size for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("overlap between client roles")
        all_values.extend(roles)
    merged = np.concatenate(all_values)
    if len(merged) != len(np.unique(merged)):
        raise ValueError("proxy/client or cross-client split overlap")


def make_partitions(
    labels: np.ndarray, *, regime: str, seed: int, clients: int = 10,
    proxy_size: int = 10_000, holdout_fraction: float = 0.2, test_fraction: float = 0.15,
) -> tuple[np.ndarray, list[dict[str, np.ndarray]]]:
    """Reserve public proxy examples before non-IID assignment and role splitting."""
    labels = np.asarray(labels, dtype=np.int64)
    order = balanced_order(labels, seed)
    if proxy_size <= 0 or proxy_size >= len(labels):
        raise ValueError("proxy_size must be in (0,N)")
    proxy_idx, pool = order[:proxy_size], order[proxy_size:]
    assigned = _assign(pool, labels, regime, clients, seed)
    rows = [_split_client(values, seed + cid + 1, holdout_fraction, test_fraction) for cid, values in enumerate(assigned)]
    validate_splits(proxy_idx, rows)
    return proxy_idx, rows


def save_partitions(path: Path, *, proxy_idx: np.ndarray, clients: list[dict[str, np.ndarray]], metadata: dict) -> None:
    """Write immutable index files once; existing files must match the request."""
    path = Path(path)
    validate_splits(proxy_idx, clients)
    if (path / "metadata.json").exists():
        raise FileExistsError(f"partitions already exist: {path}")
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / "proxy.npz", proxy_idx=np.asarray(proxy_idx, dtype=np.int64))
    for cid, split in enumerate(clients):
        np.savez_compressed(path / f"client_{cid:03d}.npz", **split)
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
