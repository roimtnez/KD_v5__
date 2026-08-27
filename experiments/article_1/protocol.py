"""Small, explicit protocol for the canonical Article 1 experiments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DATASETS = ("cifar", "mnist", "fmnist")
SEEDS = (42, 43, 44)
REGIMES = ("iid", "alpha1p0", "alpha0p5", "alpha0p1", "multi", "single")
PROXY_SIZES = (100, 250, 500, 1000, 2500, 5000, 10000)
THRESHOLD_OFFSETS = (-0.05, 0.0, 0.05)

DATASET_LABEL = {"cifar": "cifar10", "mnist": "mnist", "fmnist": "fmnist"}
CANONICAL_THRESHOLD = {"cifar": 0.7, "mnist": 0.9, "fmnist": 0.8}

CENTRAL_METHODS = ("feddf", "support_only", "expert_full", "expert_v2")
ORACLE_METHODS = ("oracle_full", "oracle_v2", "supervised_matched")
# A mechanistic control, not a new selector: it retains EXPERT-full's class
# support while matching EXPERT-support's entropy one proxy example at a time.
MECHANISM_CONTROL_METHODS = (
    "expert_full",
    "expert_v2",
    "expert_full_entropy_matched",
)
# The deliberately small, paper-opening comparison.  These methods all use the
# same teacher sources, proxy prefix and paired KD recipe at N=10000.
FINAL_BASELINE_METHODS = ("feddf", "energy", "expert_full", "oracle_full")
CURVE_METHODS = (
    "feddf", "expert_v2", "oracle_v2", "supervised_matched", "supervised_standard",
)
# The historical curve is a seed-42 EXPERT-support diagnostic.  This separate
# set is the only method set used for the Article-1 proxy sample-efficiency
# claim.  In particular, EXPERT-full—not EXPERT-support—is the expertise-aware
# method under test.
PROXY_EFFICIENCY_BASELINE_METHODS = (
    "feddf", "expert_full", "supervised_matched", "supervised_standard",
)
PROXY_EFFICIENCY_ENERGY_METHODS = ("energy",)
PROXY_EFFICIENCY_ORACLE_REFERENCE = ("oracle_full",)
ALL_METHODS = CENTRAL_METHODS + tuple(x for x in ORACLE_METHODS if x not in CENTRAL_METHODS) + (
    "expert_full_entropy_matched",
    "energy",
    "supervised_standard",
)


@dataclass(frozen=True)
class StudentRecipe:
    arch: str
    temperature: float = 8.0
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    kd_epochs: int = 30
    supervised_epochs: int = 100


def threshold_for(dataset: str, offset: float = 0.0) -> float:
    if dataset not in CANONICAL_THRESHOLD:
        raise ValueError(f"unsupported Article-1 dataset {dataset!r}")
    value = CANONICAL_THRESHOLD[dataset] + float(offset)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"expertise threshold outside [0,1]: {value}")
    return value


def threshold_tag(value: float) -> str:
    return f"tau_{value:.2f}".replace(".", "p")


def source_dir(root: Path, dataset: str, seed: int, regime: str) -> Path:
    return Path(root) / "sources" / dataset / f"seed_{seed}" / regime


def run_dir(root: Path, dataset: str, seed: int, regime: str, threshold: float, size: int) -> Path:
    return (
        Path(root) / "runs" / dataset / f"seed_{seed}" / regime
        / threshold_tag(threshold) / f"N_{int(size)}"
    )


def regime_metadata(regime: str) -> dict:
    if regime == "iid":
        return {"regime_family": "iid", "heterogeneity_rank": 0, "specialization_rank": 0}
    if regime.startswith("alpha"):
        rank = {"alpha1p0": 1, "alpha0p5": 2, "alpha0p1": 3}[regime]
        return {"regime_family": "dirichlet", "heterogeneity_rank": rank, "specialization_rank": None}
    rank = {"multi": 1, "single": 2}[regime]
    return {"regime_family": "specialization", "heterogeneity_rank": None, "specialization_rank": rank}
