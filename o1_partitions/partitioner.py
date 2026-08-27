"""Unified, self-contained partitioning (REFACTOR_PLAN phase 2, full implementation).

This module *is* the partitioning implementation. It absorbs the flat
Dirichlet/iid/single/multi logic that used to live in ``partitioning.py`` +
``sweep_partitions.py`` (minus dead options like ``min_size`` resampling and
``quantity_mode``/``lognormal_sigma``), and
dispatches CIFAR-100 K=20 to the superclass partitioner. The genuinely-shared
naming helpers (``build_partition_name``/``partition_exists``) live here too.

Distribution reports are a derived, deterministic part of the partition
contract. Missing reports are backfilled from ``clients/c*.npz`` without
changing any saved index.

Behaviour parity: the flat path reproduces the legacy partitions index-for-index
(same dual ``default_rng(seed)`` usage — one for class assignment, a separate one
for the 3-way split — and the same non-stratified split). Locked by
tests/test_partition_parity (RUN_PARTITION_REGEN=1) and tests/test_partitioner.

Data contract (unchanged): each ``clients/cXXX.npz`` carries ``train_idx``,
``holdout_idx`` and (when ``local_test_frac>0``) ``local_test_idx`` — teacher
training early-stops on ``holdout_idx``.
"""
from __future__ import annotations

import json
import hashlib
import os
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import yaml

from paths import DATA_DIR
from data.dataset_config import get_dataset_config
from oracle_distillation.utils import client_npz_name

DEFAULT_PROXY_SIZE = 10_000

# ---------------------------------------------------------------------------
# Canonical partition modes + naming helpers (single source of truth)
# ---------------------------------------------------------------------------

CANONICAL_MODES = ("dirichlet", "iid", "single", "multi")
_MODE_ALIASES = {
    "dirichlet": "dirichlet",
    "iid": "iid", "iid_balanced": "iid",
    "single": "single", "single_class": "single",
    "multi": "multi", "multi_class": "multi",
}


def normalize_partition_mode(mode: str) -> str:
    """Map any accepted alias to its canonical name. Raises if unknown."""
    if mode not in _MODE_ALIASES:
        raise ValueError(
            f"Unknown partition mode '{mode}'. Accepted: {sorted(_MODE_ALIASES)}"
        )
    return _MODE_ALIASES[mode]


def fmt_float(x: float) -> str:
    s = f"{x:.10g}"
    if "e" in s or "E" in s:
        s = f"{x:.8f}".rstrip("0").rstrip(".")
    if "." not in s:
        s = s + ".0"
    return s.replace(".", "p")


def fmt_k(n: int) -> str:
    return f"{n // 1000}k" if n % 1000 == 0 else str(n)


def alpha_tag(mode: str, alpha: float) -> str:
    """folder tag: dirichlet -> 'alpha0p1'; iid/single/multi -> the mode name."""
    canonical = normalize_partition_mode(mode)
    if canonical in ("iid", "single", "multi"):
        return canonical
    return f"alpha{fmt_float(alpha)}"


def build_partition_name(
    *,
    dataset: str,            # dataset *label* (e.g. "cifar10"), not the registry key
    num_clients: int,
    pool_size: int,
    proxy_size: int,
    partition_mode: str,
    alpha: float,
    holdout_frac: float,
) -> str:
    """Canonical relative path: ``<dataset>/K<N>/<tag>__<pool>-<proxy>-<holdout>``."""
    tag = alpha_tag(partition_mode, alpha)
    return (
        f"{dataset}/K{num_clients}/"
        f"{tag}__{fmt_k(pool_size)}-{fmt_k(proxy_size)}-{fmt_float(holdout_frac)}"
    )


def partition_exists(out_dir: Path, num_clients: int) -> bool:
    """Ready iff all per-client NPZ files are present (config.yaml optional)."""
    clients_dir = Path(out_dir) / "clients"
    if not clients_dir.is_dir():
        return False
    return len(sorted(clients_dir.glob("c*.npz"))) == num_clients


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class Partitioner(ABC):
    """Common interface for all partitioning strategies."""

    @abstractmethod
    def partition_dir(self, out_root: Path) -> Path: ...

    @abstractmethod
    def num_clients(self) -> int: ...

    @abstractmethod
    def _create(self, out_root: Path) -> None:
        """Write clients/*.npz (and metadata) under ``partition_dir(out_root)``."""

    def exists(self, out_root: Path) -> bool:
        return partition_exists(self.partition_dir(out_root), self.num_clients())

    def _write_reports(self, out_dir: Path) -> None:
        """Create deterministic derived reports; subclasses opt in."""

    def _validate_existing(self, out_dir: Path) -> None:
        """Fail closed when persisted inputs do not match the requested replica."""

    def ensure(self, out_root: Path, *, force: bool = False, raise_on_error: bool = False) -> Path:
        """Create the partition if missing (or if ``force``); return its directory."""
        out_root = Path(out_root).resolve()
        out_root.mkdir(parents=True, exist_ok=True)
        out_dir = self.partition_dir(out_root)

        if not force and self.exists(out_root):
            try:
                self._validate_existing(out_dir)
                self._write_reports(out_dir)
            except Exception as e:  # noqa: BLE001
                if raise_on_error:
                    raise
                print(f"[WARN] Partition exists but reports could not be backfilled: {e}")
            print(f"[SKIP] Partition already exists: {out_dir}")
            return out_dir

        out_dir.mkdir(parents=True, exist_ok=True)
        run_log = out_dir / "run.log"
        try:
            self._create(out_root)
            self._write_reports(out_dir)
            run_log.write_text("OK\n", encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            run_log.write_text(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
                               encoding="utf-8")
            if raise_on_error:
                raise
            print(f"[ERROR] Partition generation failed for {out_dir}: {e}")
        return out_dir


# ---------------------------------------------------------------------------
# Flat-label partitioner (dirichlet / iid / single / multi)
# ---------------------------------------------------------------------------

@dataclass
class DirichletPartitioner(Partitioner):
    """Flat-label partitioner for cifar / mnist / fmnist.

    Reproduces the legacy ``make_one_partition`` exactly: class assignment and
    the 3-way split each use their own ``np.random.default_rng(seed)``.
    """
    dataset: str                       # registry key: "cifar" | "mnist" | "fmnist" | "cinic"
    n_clients: int
    partition_mode: str                # canonical or alias
    alpha: float = 0.0
    seed: int = 42
    data_dir: Path = Path("data")
    proxy_size: int = DEFAULT_PROXY_SIZE
    holdout_frac: float = 0.2
    local_test_frac: float = 0.15
    classes_per_client: Optional[Union[int, Sequence[int]]] = None
    proxy_split_npz: Optional[Path] = None

    def num_clients(self) -> int:
        return self.n_clients

    @property
    def _label(self) -> str:
        return get_dataset_config(self.dataset).label

    def partition_dir(self, out_root: Path) -> Path:
        rel = build_partition_name(
            dataset=self._label,
            num_clients=self.n_clients,
            pool_size=get_dataset_config(self.dataset).train_size - self.proxy_size,
            proxy_size=self.proxy_size,
            partition_mode=self.partition_mode,
            alpha=self.alpha,
            holdout_frac=self.holdout_frac,
        )
        return Path(out_root) / rel

    # ---- class-assignment algorithms (each seeds its own RNG, as legacy) ----

    def _assign(self, pool_labels: np.ndarray) -> List[np.ndarray]:
        mode = normalize_partition_mode(self.partition_mode)
        if mode == "iid":
            return self._iid(pool_labels)
        if mode == "single":
            return self._single(pool_labels)
        if mode == "multi":
            return self._multi(pool_labels)
        if mode == "dirichlet":
            return self._dirichlet(pool_labels)
        raise ValueError(f"Unhandled mode: {mode}")

    def _dirichlet(self, labels: np.ndarray) -> List[np.ndarray]:
        rng = np.random.default_rng(self.seed)
        labels = labels.astype(int)
        num_classes = len(np.unique(labels))
        idx_by_class = [np.where(labels == c)[0] for c in range(num_classes)]
        client_bins: List[List[np.ndarray]] = [[] for _ in range(self.n_clients)]
        for c in range(num_classes):
            idx_c = idx_by_class[c].copy()
            rng.shuffle(idx_c)
            props = rng.dirichlet(self.alpha * np.ones(self.n_clients))
            counts = (props * len(idx_c)).astype(int)
            diff = len(idx_c) - counts.sum()
            if diff > 0:
                counts[np.argmax(props)] += diff
            elif diff < 0:
                counts[np.argmax(counts)] += diff
            start = 0
            for k in range(self.n_clients):
                end = start + counts[k]
                if end > start:
                    client_bins[k].append(idx_c[start:end])
                start = end
        return [np.concatenate(b) if b else np.array([], dtype=int) for b in client_bins]

    def _iid(self, labels: np.ndarray) -> List[np.ndarray]:
        rng = np.random.default_rng(self.seed)
        labels = labels.astype(int)
        client_bins: List[List[np.ndarray]] = [[] for _ in range(self.n_clients)]
        for c in np.unique(labels):
            idx_c = np.where(labels == c)[0].copy()
            rng.shuffle(idx_c)
            chunks = np.array_split(idx_c, self.n_clients)
            order = rng.permutation(self.n_clients)
            for chunk, k in zip(chunks, order):
                if len(chunk):
                    client_bins[int(k)].append(chunk)
        return [np.concatenate(b) if b else np.array([], dtype=int) for b in client_bins]

    def _single(self, labels: np.ndarray) -> List[np.ndarray]:
        rng = np.random.default_rng(self.seed)
        labels = labels.astype(int)
        classes = np.unique(labels)
        num_classes = len(classes)
        if self.n_clients != num_classes:
            raise ValueError(
                f"single requires num_clients == #classes; got {self.n_clients} vs {num_classes}"
            )
        idx_by_class = [np.where(labels == c)[0] for c in classes]
        order = rng.permutation(self.n_clients)
        client_bins: List[List[np.ndarray]] = [[] for _ in range(self.n_clients)]
        for class_idx, k in zip(range(num_classes), order):
            idx_c = idx_by_class[class_idx]
            if len(idx_c):
                client_bins[int(k)].append(idx_c)
        return [np.concatenate(b) if b else np.array([], dtype=int) for b in client_bins]

    def _multi(self, labels: np.ndarray) -> List[np.ndarray]:
        rng = np.random.default_rng(self.seed)
        labels = labels.astype(int)
        classes = np.unique(labels)
        C = len(classes)
        if C == 0:
            raise ValueError("No classes found in labels pool.")
        cpc = self.classes_per_client if self.classes_per_client is not None else (2, 3)
        if isinstance(cpc, int):
            per_client = np.array([int(cpc)] * self.n_clients, dtype=int)
        else:
            options = list(cpc)
            if not options:
                raise ValueError("classes_per_client must be int or non-empty sequence")
            per_client = rng.choice(options, size=self.n_clients)
        S = int(per_client.sum())
        assign_pool = list(classes.tolist())
        remaining = S - C
        if remaining < 0:
            raise ValueError("Not enough total class slots to cover all classes once")
        if remaining > 0:
            extra = rng.choice(classes, size=remaining, replace=True)
            assign_pool.extend(int(x) for x in extra.tolist())
        for _ in range(100):
            rng.shuffle(assign_pool)
            it = iter(assign_pool)
            client_classes: List[List[int]] = []
            ok = True
            for k in range(self.n_clients):
                m = int(per_client[k])
                cls = [int(next(it)) for _ in range(m)] if m > 0 else []
                if len(set(cls)) != len(cls):
                    ok = False
                    break
                client_classes.append(cls)
            if ok:
                break
        else:
            raise RuntimeError("Failed to generate unique per-client class assignments")
        class_to_clients = {int(c): [] for c in classes}
        for k, cls_list in enumerate(client_classes):
            for c in cls_list:
                class_to_clients[int(c)].append(k)
        idx_by_class = {int(c): np.where(labels == c)[0].copy() for c in classes}
        client_bins: List[List[np.ndarray]] = [[] for _ in range(self.n_clients)]
        for c in classes:
            c = int(c)
            clients_with_c = class_to_clients.get(c, [])
            if not clients_with_c:
                continue
            idx_c = idx_by_class[c]
            rng.shuffle(idx_c)
            chunks = np.array_split(idx_c, len(clients_with_c))
            for k, chunk in zip(clients_with_c, chunks):
                if len(chunk):
                    client_bins[int(k)].append(chunk)
        return [np.concatenate(b) if b else np.array([], dtype=int) for b in client_bins]

    # ---- 3-way split (separate fresh RNG, as legacy) -----------------------

    def _three_way_split(self, client_indices: List[np.ndarray]):
        rng = np.random.default_rng(self.seed)
        use_local_test = self.local_test_frac > 0.0
        train_rel, hold_rel, test_rel = [], [], []
        for idx in client_indices:
            idx = np.asarray(idx, dtype=int).copy()
            rng.shuffle(idx)
            n = len(idx)
            if use_local_test:
                n_test = max(1, int(n * self.local_test_frac))
                local_test, rest = idx[:n_test], idx[n_test:]
            else:
                local_test, rest = np.array([], dtype=int), idx
            n_hold = int(len(rest) * self.holdout_frac)
            hold, tr = rest[:n_hold], rest[n_hold:]
            train_rel.append(tr)
            hold_rel.append(hold)
            test_rel.append(local_test)
        return train_rel, hold_rel, test_rel

    # ---- data loading + creation ------------------------------------------

    def _load_labels(self) -> np.ndarray:
        from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
        root = str(self.data_dir) if self.data_dir else DATA_DIR
        if self.dataset == "mnist":
            ds = MNIST(root=root, train=True, download=True)
        elif self.dataset == "fmnist":
            ds = FashionMNIST(root=root, train=True, download=True)
        else:
            ds = CIFAR10(root=root, train=True, download=True)
        return np.array(ds.targets, dtype=int)

    def _proxy_path(self) -> Path:
        if self.proxy_split_npz is not None:
            return Path(self.proxy_split_npz)
        return get_dataset_config(self.dataset).proxy_path_for_seed(self.seed, self.data_dir)

    @staticmethod
    def _proxy_idx_sha256(path: Path) -> str:
        with np.load(path, allow_pickle=True) as split:
            proxy_idx = np.ascontiguousarray(split["proxy_idx"].astype(np.int64))
        digest = hashlib.sha256()
        digest.update(proxy_idx.tobytes())
        return digest.hexdigest()

    def _validate_existing(self, out_dir: Path) -> None:
        config_path = Path(out_dir) / "config.yaml"
        if not config_path.is_file():
            raise ValueError(f"existing partition lacks provenance config: {config_path}")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        observed_seed = int(config.get("seed", -1))
        if observed_seed != int(self.seed):
            raise ValueError(
                f"existing partition seed mismatch {observed_seed}!={self.seed}: {out_dir}"
            )
        recorded_partition_name = str(config.get("partition_name", ""))
        if recorded_partition_name and not Path(out_dir).as_posix().endswith(recorded_partition_name):
            raise ValueError(
                "existing partition directory/config name mismatch; refusing ambiguous layout: "
                f"directory={out_dir}, config.partition_name={recorded_partition_name}"
            )
        expected_proxy = self._proxy_path()
        if not expected_proxy.is_file():
            raise FileNotFoundError(f"seed-specific proxy split is missing: {expected_proxy}")
        recorded_proxy = Path(str(config.get("proxy_split_npz", "")))
        if recorded_proxy.name != expected_proxy.name:
            raise ValueError(
                "existing partition uses an incompatible proxy split; refusing silent reuse: "
                f"recorded={recorded_proxy}, expected={expected_proxy}"
            )
        expected_hash = self._proxy_idx_sha256(expected_proxy)
        recorded_hash = str(config.get("proxy_idx_sha256", ""))
        if int(self.seed) != 42 and not recorded_hash:
            raise ValueError(
                f"existing seed-{self.seed} partition lacks proxy_idx_sha256: {out_dir}"
            )
        if recorded_hash and recorded_hash != expected_hash:
            raise ValueError(
                f"existing partition proxy_idx hash mismatch: {recorded_hash}!={expected_hash}"
            )

    def _create(self, out_root: Path) -> None:
        out_dir = self.partition_dir(out_root)
        out_dir.mkdir(parents=True, exist_ok=True)

        labels_full = self._load_labels()
        proxy_npz = self._proxy_path()
        split = np.load(proxy_npz)
        pool_idx = split["train_idx"].astype(int)
        proxy_idx = split["proxy_idx"].astype(int)
        pool_labels = labels_full[pool_idx]

        client_indices = self._assign(pool_labels)
        train_rel, hold_rel, test_rel = self._three_way_split(client_indices)

        clients_dir = out_dir / "clients"
        clients_dir.mkdir(parents=True, exist_ok=True)
        use_local_test = self.local_test_frac > 0.0
        client_files: List[str] = []
        for k in range(self.n_clients):
            arrays = dict(
                train_idx=pool_idx[train_rel[k]],
                holdout_idx=pool_idx[hold_rel[k]],
            )
            if use_local_test:
                arrays["local_test_idx"] = pool_idx[test_rel[k]]
            rel = f"clients/{client_npz_name(k)}"
            np.savez_compressed(out_dir / rel, **arrays)
            client_files.append(rel)

        self._write_config(out_dir, proxy_npz, len(pool_idx), len(proxy_idx), client_files)
        print(f"[OK] Partitions saved to: {out_dir}")

    def _write_reports(self, out_dir: Path) -> None:
        from o1_partitions.partition_reports import report_complete, write_partition_reports

        if report_complete(out_dir):
            return
        write_partition_reports(
            out_dir,
            self._load_labels(),
            num_classes=get_dataset_config(self.dataset).num_classes,
        )

    def _write_config(self, out_dir, proxy_npz, pool_size, proxy_size, client_files):
        mode = normalize_partition_mode(self.partition_mode)
        cpc = None
        if mode == "multi":
            c = self.classes_per_client if self.classes_per_client is not None else (2, 3)
            cpc = [int(c)] if isinstance(c, int) else [int(x) for x in c]
        portable_proxy = os.path.relpath(Path(proxy_npz).resolve(), Path.cwd().resolve())
        proxy_idx_sha256 = self._proxy_idx_sha256(Path(proxy_npz))
        cfg = {
            "dataset": self._label,
            "partition_name": build_partition_name(
                dataset=self._label, num_clients=self.n_clients,
                pool_size=pool_size, proxy_size=proxy_size,
                partition_mode=mode, alpha=self.alpha, holdout_frac=self.holdout_frac,
            ),
            "seed": int(self.seed),
            "num_clients": int(self.n_clients),
            "partition_mode": mode,
            "alpha": float(self.alpha) if mode == "dirichlet" else None,
            "classes_per_client": cpc,
            "proxy_split_npz": portable_proxy,
            "proxy_idx_sha256": proxy_idx_sha256,
            "pool_size": int(pool_size),
            "proxy_size": int(proxy_size),
            "holdout_frac": float(self.holdout_frac),
            "local_test_frac": float(self.local_test_frac),
        }
        (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        (out_dir / "metadata.json").write_text(json.dumps({
            "pool_idx_path": portable_proxy,
            "proxy_idx_sha256": proxy_idx_sha256,
            "pool_size": int(pool_size),
            "proxy_size": int(proxy_size),
            "client_files": client_files,
        }, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# CIFAR-100 K=20 superclass-tessellated partitioner
# ---------------------------------------------------------------------------

def _cifar100_regime(group_label: str) -> str:
    """Normalize a CLI group label to a k10 regime label.

    A bare float becomes the coarse Dirichlet regime (``dir_<a>``, sibling-coherent);
    every other label is passed through to ``partition_cifar100_k10.parse_regime``
    for validation (single/multi/iid, single_super/multi_super, dir_/coarse_/fine_).
    """
    g = group_label.strip()
    try:
        float(g)
        return f"dir_{g}"
    except ValueError:
        return g


@dataclass
class SuperclassPartitioner(Partitioner):
    """CIFAR-100 fine+superclass (dual-label) partitioner.

    Emits clients/*.npz via ``data.partition_cifar100_k10`` (the hierarchy-aware
    successor to the deprecated k20 module). The k10 naming scheme
    ``cifar100/K<k>/<regime_tag>`` is the single source shared with ``Paths``.
    """
    regime: str
    seed: int = 42
    k: int = 10
    multi_supers: tuple = (3, 4)
    data_dir: Path = Path("data")
    proxy_path: Optional[Path] = None
    holdout_frac: float = 0.15
    local_test_frac: float = 0.2

    def num_clients(self) -> int:
        return self.k

    def _proxy(self) -> Path:
        return Path(self.proxy_path) if self.proxy_path else get_dataset_config("cifar100").proxy_path

    def partition_dir(self, out_root: Path) -> Path:
        from data.partition_cifar100_k10 import partition_relname
        return Path(out_root) / partition_relname(self.k, self.regime)

    def _create(self, out_root: Path) -> None:
        from data.partition_cifar100_k10 import run as run_k10
        run_k10(
            proxy_path=self._proxy(), out_root=Path(out_root), data_dir=Path(self.data_dir),
            regime=self.regime, seed=self.seed, K=self.k, multi_supers=tuple(self.multi_supers),
            holdout_frac=self.holdout_frac, local_test_frac=self.local_test_frac, force=True,
        )

    def _write_reports(self, out_dir: Path) -> None:
        from o1_partitions.partition_reports import (
            load_dataset_labels, report_complete, write_partition_reports,
        )

        if report_complete(out_dir):
            return
        write_partition_reports(
            out_dir,
            load_dataset_labels("cifar100", self.data_dir),
            num_classes=get_dataset_config("cifar100").num_classes,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_partitioner(
    *,
    dataset: str,
    group_label: str,
    partition_mode: str,
    alpha: float = 0.0,
    num_clients: int,
    seed: int = 42,
    data_dir: Path = Path("data"),
    proxy_size: int = DEFAULT_PROXY_SIZE,
    holdout_frac: float = 0.2,
    local_test_frac: float = 0.15,
    classes_per_client: Optional[Sequence[int]] = None,
    proxy_split_npz: Optional[Path] = None,
) -> Partitioner:
    """Pick the partitioner for ``dataset`` (the single dispatch point)."""
    if dataset == "cifar100":
        return SuperclassPartitioner(
            regime=_cifar100_regime(group_label), seed=seed, k=num_clients,
            data_dir=Path(data_dir), proxy_path=proxy_split_npz,
            holdout_frac=holdout_frac, local_test_frac=local_test_frac,
        )
    return DirichletPartitioner(
        dataset=dataset, n_clients=num_clients, partition_mode=partition_mode, alpha=alpha,
        seed=seed, data_dir=Path(data_dir), proxy_size=proxy_size, holdout_frac=holdout_frac,
        local_test_frac=local_test_frac, classes_per_client=classes_per_client,
        proxy_split_npz=proxy_split_npz,
    )
