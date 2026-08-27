"""Shared run-layout helpers for the KD CLIs (global + personal).

Single naming scheme for ``partitions/`` and ``runs/``, built by
``build_partition_name``:

    partitions/<dataset>/K<N>/<tag>__<pool>-<proxy>-<holdout>/
    runs/<dataset>/K<N>/<tag>__<pool>-<proxy>-<holdout>/
        teachers/
        <proxy_dataset>/
            proxy_analysis/
            distillation/<method>/

``Paths`` exposes both a ``RunConfig``-driven API (used by the global CLI,
which builds configs declaratively from alpha/group labels) and a plain
``rel: str``-driven API (used by the personal CLI, which discovers existing
``rel`` strings by scanning the filesystem rather than declaring them).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np

from o1_partitions.partitioner import build_partition_name
from data.dataset_config import get_dataset_config, dataset_from_label

DEFAULT_PROXY_SIZE = 10_000
DEFAULT_HOLDOUT_FRAC = 0.15
DEFAULT_LOCAL_TEST_FRAC = 0.2


@dataclass(frozen=True)
class RunConfig:
    group_label: str                       # CLI label: "0.1", "iid", "single", ...
    num_clients: int
    partition_mode: str                    # canonical: "dirichlet" | "iid" | "single" | "multi"
    alpha: float                           # 0.0 for non-dirichlet
    classes_per_client: Optional[Tuple[int, ...]] = None


def build_run_configs(group_labels: Iterable[str], num_clients: int) -> list[RunConfig]:
    configs: list[RunConfig] = []
    for g in group_labels:
        if g == "iid":
            configs.append(RunConfig(g, num_clients, "iid", 1.0))
        elif g == "single":
            configs.append(RunConfig(g, num_clients, "single", 0.0))
        elif g == "multi":
            configs.append(RunConfig(g, num_clients, "multi", 0.0, (2, 3)))
        else:
            # Dirichlet labels: bare "0.1", or the CIFAR-100 k10 prefixes
            # dir_/coarse_/fine_/dir_coarse_/dir_fine_ (the prefix is preserved in
            # group_label; only the trailing alpha is parsed). group_label is what
            # the CIFAR-100 path keys on, so coarse vs fine stay distinct.
            label_clean = g
            for pfx in ("dir_coarse_", "dir_fine_", "coarse_", "fine_", "dir_"):
                if g.startswith(pfx):
                    label_clean = g[len(pfx):]
                    break
            try:
                alpha_val = float(label_clean)
            except ValueError:
                print(f"[WARN] group label '{g}' ignored (not float or keyword)")
                continue
            configs.append(RunConfig(g, num_clients, "dirichlet", alpha_val))
    return configs


@dataclass(frozen=True)
class Paths:
    work_root: Path
    data_dir: Path
    dataset: str  # proxy dataset: "cifar" | "cinic" | "cifar100" | ...
    proxy_size: int = DEFAULT_PROXY_SIZE
    holdout_frac: float = DEFAULT_HOLDOUT_FRAC
    local_test_frac: float = DEFAULT_LOCAL_TEST_FRAC

    @property
    def partitions_root(self) -> Path:
        return self.work_root / "partitions"

    def partitions_root_for_seed(self, seed: int) -> Path:
        """Return the partition root inside an already seed-scoped work root.

        ``work_root`` is ``.../study_i/seed_<seed>/raw_work``. Adding another
        ``seed_<seed>`` below ``partitions`` duplicated the replicate namespace
        and made path discovery unnecessarily fragile.
        """
        return self.partitions_root

    @property
    def runs_root(self) -> Path:
        return self.work_root / "runs"

    @property
    def proxy_split_npz(self) -> Path:
        """Legacy seed-42 compatibility path; new code must pass a seed."""
        return get_dataset_config(self.dataset).proxy_path

    def proxy_split_npz_for_seed(self, seed: int) -> Path:
        return get_dataset_config(self.dataset).proxy_path_for_seed(seed, self.data_dir)

    # -------------------------------------------------------- RunConfig API
    # Used by the global CLI, which declares configs from alpha/group labels.

    def rel(self, cfg: RunConfig) -> str:
        # CIFAR-100 uses the hierarchy-explicit k10 naming scheme
        # (cifar100/K<k>/<regime_tag>), not the flat <tag>__<sizes> convention.
        if self.dataset == "cifar100":
            from data.partition_cifar100_k10 import partition_relname
            from o1_partitions.partitioner import _cifar100_regime
            return partition_relname(cfg.num_clients, _cifar100_regime(cfg.group_label))
        ds_cfg = get_dataset_config(self.dataset)
        return build_partition_name(
            dataset=ds_cfg.label,
            num_clients=cfg.num_clients,
            pool_size=ds_cfg.train_size - self.proxy_size,
            proxy_size=self.proxy_size,
            partition_mode=cfg.partition_mode,
            alpha=cfg.alpha,
            holdout_frac=self.holdout_frac,
        )

    def partition_dir(self, cfg: RunConfig, seed: int = 42) -> Path:
        return self.partitions_root_for_seed(seed) / self.rel(cfg)

    def run_dir(self, cfg: RunConfig) -> Path:
        return self.run_dir_for_rel(self.rel(cfg))

    def teachers_dir(self, cfg: RunConfig, seed: int) -> Path:
        return self.teachers_dir_for_rel(self.rel(cfg), seed)

    def proxy_dir(self, cfg: RunConfig, seed: int) -> Path:
        return self.proxy_dir_for_rel(self.rel(cfg), seed)

    def method_dir(self, cfg: RunConfig, method: str, seed: int) -> Path:
        return self.method_dir_for_rel(self.rel(cfg), method, seed)

    # ------------------------------------------------------------- rel API
    # Used by the personal CLI, which discovers existing ``rel`` strings by
    # scanning the filesystem rather than declaring RunConfigs.

    def partition_dir_for_rel(self, rel: str, seed: int = 42) -> Path:
        """Resolve the partition paired with a run seed.

        Keeping ``seed=42`` as the default retains compatibility with legacy
        callers, while all new callers should pass their explicit run seed.
        """
        return self.partitions_root_for_seed(seed) / rel

    def run_dir_for_rel(self, rel: str) -> Path:
        return self.runs_root / rel

    def teachers_dir_for_rel(self, rel: str, seed: int) -> Path:
        return self.run_dir_for_rel(rel) / "teachers"

    def proxy_dir_for_rel(self, rel: str, seed: int) -> Path:
        return self.run_dir_for_rel(rel) / "proxy_analysis"

    def method_dir_for_rel(self, rel: str, method: str, seed: int) -> Path:
        return self.run_dir_for_rel(rel) / "distillation" / method

    def personal_class_mask_dir_for_rel(self, rel: str, global_source: str, seed: int) -> Path:
        """personal_class_mask output dir, keyed by --global_source so sweeping
        ``g`` (feddf/energy/consensus/confidence/expert/oracle) never clobbers a
        sibling run on disk — unlike the other personal methods (one config each,
        no clobbering risk), personal_class_mask is the one method swept over g."""
        return self.run_dir_for_rel(rel) / "personal_class_mask" / global_source

    # ----------------------------------------------------------------- misc

    def global_csv(self) -> Path:
        return self.work_root / "methods_results.csv"


def discover_proxy_jobs(
    runs_root: Path,
    datasets: Iterable[str],
    seed: int,
    rel_filter=None,
) -> list[Tuple[str, str, Path]]:
    """Scan a ``runs/`` tree for ``(rel, dataset, proxy_analysis.npz)``.

    Inverse of ``Paths.proxy_dir_for_rel(rel, seed)``; the layout (rooted at
    ``<work_root>/runs``, e.g. ``v2_<dataset>/runs``) is::

        <dataset>/K<N>/<alpha_dir>/proxy_analysis/proxy_analysis.npz

    so ``rel = <dataset>/K<N>/<alpha_dir>``. The leading ``<dataset>`` is the
    path *label* (e.g. ``cifar10``); the returned ``dataset`` is the registry
    *key* (``cifar``), which is what ``datasets`` filters on and what the CLIs
    use for loader/proxy lookups. Only the requested ``seed`` is matched.
    """
    datasets = set(datasets)
    results: list[Tuple[str, str, Path]] = []
    runs_root = Path(runs_root)
    if not runs_root.is_dir():
        return results
    for proxy_dir in sorted(runs_root.rglob("proxy_analysis")):
        npz = proxy_dir / "proxy_analysis.npz"
        if not npz.is_file():
            continue
        with np.load(npz, allow_pickle=True) as payload:
            if "seed" not in payload.files:
                if int(seed) != 42:
                    raise ValueError(f"proxy analysis lacks seed metadata: {npz}")
            elif int(np.asarray(payload["seed"]).item()) != int(seed):
                raise ValueError(f"proxy analysis seed mismatch for seed={seed}: {npz}")
        # proxy_dir = runs/<rel>/proxy_analysis  ->  rel = parent
        try:
            rel = str(proxy_dir.parent.relative_to(runs_root)).replace("\\", "/")
        except ValueError:
            continue
        try:
            dataset = dataset_from_label(rel.split("/")[0])
        except ValueError:
            continue  # leading dir is not a known dataset label
        if datasets and dataset not in datasets:
            continue
        if rel_filter and not any(tok in rel for tok in rel_filter):
            continue
        results.append((rel, dataset, npz))
    return results
