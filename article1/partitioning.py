"""Proxy-first, disjoint Article-1 partitions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from article1 import PROTOCOL_VERSION, REGIMES
from article1.hashes import array_sha256, file_sha256, git_commit

ROLES = ("train", "validation", "expertise")
SPLIT_FRACTIONS = {"train": 0.7, "validation": 0.1, "expertise": 0.2}
SPLIT_RULE = "stratified_largest_remainder_70_10_20_v1"


def balanced_order(labels: np.ndarray, seed: int) -> np.ndarray:
    """A deterministic class-balanced order: every prefix is nested."""
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or not len(labels):
        raise ValueError("labels must be non-empty [N]")
    rng = np.random.default_rng(seed)
    queues = [
        list(rng.permutation(np.flatnonzero(labels == c)))
        for c in range(int(labels.max()) + 1)
    ]
    if any(not q for q in queues):
        raise ValueError("each class must occur in the training data")
    out: list[int] = []
    while any(queues):
        for c in rng.permutation(len(queues)):
            if queues[c]:
                out.append(queues[c].pop())
    return np.asarray(out, dtype=np.int64)


def _assign(
    pool: np.ndarray, labels: np.ndarray, regime: str, k: int, seed: int
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    if regime not in REGIMES:
        raise ValueError(f"unsupported regime {regime!r}")
    if regime == "iid":
        groups = [[] for _ in range(k)]
        for c in range(int(labels.max()) + 1):
            rows = pool[labels[pool] == c].copy()
            rng.shuffle(rows)
            for client, part in enumerate(np.array_split(rows, k)):
                groups[client].append(part)
        return [np.concatenate(parts) for parts in groups]
    if regime.startswith("alpha"):
        alpha = {"alpha1p0": 1.0, "alpha0p5": 0.5, "alpha0p1": 0.1}[regime]
        groups = [[] for _ in range(k)]
        for c in range(int(labels.max()) + 1):
            rows = pool[labels[pool] == c].copy()
            rng.shuffle(rows)
            cuts = (rng.dirichlet(np.full(k, alpha)) * len(rows)).astype(int)
            cuts[np.argmax(cuts)] += len(rows) - cuts.sum()
            start = 0
            for client, count in enumerate(cuts):
                groups[client].append(rows[start : start + count])
                start += count
        return [np.concatenate(parts) for parts in groups]
    classes = int(labels.max()) + 1
    per_client = 1 if regime == "single" else 2
    buckets = {c: rng.permutation(pool[labels[pool] == c]) for c in range(classes)}
    groups = [[] for _ in range(k)]
    ownership = [
        [(client * per_client + j) % classes for j in range(per_client)]
        for client in range(k)
    ]
    for c, values in buckets.items():
        owners = [client for client, owned in enumerate(ownership) if c in owned]
        for client, part in zip(owners, np.array_split(values, len(owners))):
            groups[client].append(part)
    # ``group`` holds one array per class owned by the client.  Converting the
    # list directly with ``asarray`` produces a 2-D array for ``single`` and a
    # ragged-array error for ``multi``; both are index vectors, not matrices.
    return [
        np.concatenate(group).astype(np.int64, copy=False)
        if group
        else np.empty(0, dtype=np.int64)
        for group in groups
    ]


def _split_client(indices: np.ndarray, labels: np.ndarray, seed: int) -> dict:
    """Split each local class 70/10/20 using integer largest remainders.

    Ties prefer train, then expertise, then validation. No oversampling, no
    forced class coverage, no silent reassignment from another split/client.
    """
    rng = np.random.default_rng(seed)
    groups = {role: [] for role in ROLES}
    for c in np.unique(labels[indices]):
        values = rng.permutation(indices[labels[indices] == c])
        quotas = len(values) * np.array([7, 1, 2])
        counts, remainders = quotas // 10, quotas % 10
        priority = sorted(range(3), key=lambda i: (-remainders[i], (0, 2, 1)[i]))
        for i in priority[: len(values) - int(counts.sum())]:
            counts[i] += 1
        for role, part in zip(ROLES, np.split(values, np.cumsum(counts)[:-1])):
            groups[role].append(part)
    return {
        role + "_idx": rng.permutation(np.concatenate(parts)).astype(np.int64)
        if parts
        else np.empty(0, dtype=np.int64)
        for role, parts in groups.items()
    }


def validate_splits(
    proxy_idx: np.ndarray, clients: list[dict], total_examples: int | None = None
) -> None:
    """Fail on invalid indices, empty roles, overlap or incomplete coverage."""
    required = {role + "_idx" for role in ROLES}
    all_values = [np.asarray(proxy_idx)]
    for cid, client in enumerate(clients):
        if set(client) != required:
            raise ValueError(
                "each client requires train_idx, validation_idx and expertise_idx"
            )
        for key in sorted(required):
            values = np.asarray(client[key])
            if not len(values):
                raise ValueError(
                    f"client {cid}: empty {key}; inspect allocation, do not resample silently"
                )
            all_values.append(values)
    for values in all_values:
        if values.ndim != 1 or values.dtype.kind not in "iu" or (values < 0).any():
            raise ValueError("indices must be nonnegative integer vectors")
    if not len(proxy_idx):
        raise ValueError("empty proxy")
    merged = np.concatenate(all_values)
    if len(merged) != len(np.unique(merged)):
        raise ValueError("duplicate indices or proxy/client/role overlap")
    if total_examples is not None and not np.array_equal(
        np.sort(merged), np.arange(total_examples)
    ):
        raise ValueError("splits do not cover the official training set exactly")


def make_partitions(
    labels: np.ndarray,
    *,
    regime: str,
    seed: int,
    clients: int = 10,
    proxy_size: int = 10_000,
) -> tuple[np.ndarray, list[dict[str, np.ndarray]]]:
    """Reserve public proxy examples before non-IID assignment and role splitting."""
    labels = np.asarray(labels, dtype=np.int64)
    if clients != 10 or set(np.unique(labels)) != set(range(10)):
        raise ValueError("Article 1 fixes 10 clients and classes 0..9")
    order = balanced_order(labels, seed)
    if proxy_size <= 0 or proxy_size >= len(labels):
        raise ValueError("proxy_size must be in (0,N)")
    proxy_idx, pool = order[:proxy_size], order[proxy_size:]
    assigned = _assign(pool, labels, regime, clients, seed)
    rows = [
        _split_client(values, labels, seed + cid + 1)
        for cid, values in enumerate(assigned)
    ]
    validate_splits(proxy_idx, rows, total_examples=len(labels))
    return proxy_idx, rows


def save_partitions(
    path: Path,
    *,
    proxy_idx: np.ndarray,
    clients: list[dict],
    labels: np.ndarray,
    metadata: dict,
) -> None:
    """Persist a new partition with a manifest of immutable index files."""
    path = Path(path)
    labels = np.asarray(labels, dtype=np.int64)
    validate_splits(proxy_idx, clients, total_examples=len(labels))
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"partition directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path / "proxy.npz", proxy_idx=np.asarray(proxy_idx, dtype=np.int64)
    )
    for cid, split in enumerate(clients):
        np.savez_compressed(path / f"client_{cid:03d}.npz", **split)
    provenance = {
        **metadata,
        "protocol_version": PROTOCOL_VERSION,
        "creation_commit": git_commit(),
        "split_rule": SPLIT_RULE,
        "split_fractions": SPLIT_FRACTIONS,
        "training_size": len(labels),
        "training_labels_sha256": array_sha256(labels),
        "index_files_sha256": {
            p.name: file_sha256(p) for p in sorted(path.glob("*.npz"))
        },
    }
    (path / "metadata.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )


def load_partitions(path: Path, labels: np.ndarray | None = None):
    """Read current-protocol partitions and validate manifest, coverage and roles."""
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text())
    if metadata.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("incompatible partitions: create new article1-v3 partitions")
    if (
        metadata.get("split_rule") != SPLIT_RULE
        or metadata.get("split_fractions") != SPLIT_FRACTIONS
    ):
        raise ValueError("incompatible local split policy")
    files = {p.name: file_sha256(p) for p in path.glob("*.npz")}
    if files != metadata.get("index_files_sha256"):
        raise ValueError("partition index files differ from their manifest")
    with np.load(path / "proxy.npz", allow_pickle=False) as data:
        proxy = data["proxy_idx"]
    clients = []
    for p in sorted(path.glob("client_*.npz")):
        with np.load(p, allow_pickle=False) as data:
            clients.append(dict(data))
    if len(clients) != 10:
        raise ValueError("Article 1 requires exactly 10 clients")
    validate_splits(proxy, clients, total_examples=metadata["training_size"])
    if labels is not None:
        labels = np.asarray(labels, dtype=np.int64)
        if (
            len(labels) != metadata["training_size"]
            or array_sha256(labels) != metadata["training_labels_sha256"]
        ):
            raise ValueError(
                "partition labels differ from the official training labels"
            )
    return proxy, clients, metadata
