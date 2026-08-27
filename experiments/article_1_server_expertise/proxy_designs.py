"""Deterministic proxy-size and proxy-composition designs for Article 1."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .protocols import ProxyDesignSpec


@dataclass(frozen=True)
class ProxyDesign:
    positions: np.ndarray
    diagnostics: dict


def _rng(spec: ProxyDesignSpec, name: str) -> np.random.Generator:
    raw = f"{spec.namespace}|{spec.seed}|{spec.size}|{spec.composition}|{name}".encode()
    seed = int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")
    return np.random.default_rng(seed)


def _allocate(total: int, weights: np.ndarray, capacities: np.ndarray) -> np.ndarray:
    """Largest-remainder allocation with hard per-class capacities."""
    if total > int(capacities.sum()):
        raise ValueError("requested proxy size exceeds eligible examples")
    expected = total * weights / weights.sum()
    counts = np.minimum(np.floor(expected).astype(int), capacities)
    remaining = total - int(counts.sum())
    fractions = expected - np.floor(expected)
    # Stable tie-breaking is important for reproducible nested experimental arms.
    for cls in np.argsort(-fractions, kind="stable"):
        take = min(int(capacities[cls] - counts[cls]), remaining)
        counts[cls] += take
        remaining -= take
        if remaining == 0:
            break
    if remaining:
        for cls in np.flatnonzero(capacities > counts):
            take = min(int(capacities[cls] - counts[cls]), remaining)
            counts[cls] += take
            remaining -= take
            if remaining == 0:
                break
    if remaining:
        raise AssertionError("unable to allocate requested proxy examples")
    return counts


def build_proxy_design(labels: np.ndarray, spec: ProxyDesignSpec) -> ProxyDesign:
    """Choose a total-size-controlled proxy subset and persistable diagnostics."""
    y = np.asarray(labels, dtype=np.int64)
    if y.ndim != 1 or len(y) == 0 or (y < 0).any():
        raise ValueError("labels must be a non-empty non-negative vector")
    classes = int(y.max()) + 1
    spec.validate(classes, len(y))
    by_class = [np.flatnonzero(y == cls) for cls in range(classes)]
    capacities = np.array([len(values) for values in by_class], dtype=int)
    if (capacities == 0).any():
        raise ValueError("the source proxy must contain every class")

    if spec.composition == "uniform":
        positions = _rng(spec, "uniform").permutation(len(y))[: spec.size]
        allocation = np.bincount(y[positions], minlength=classes)
    else:
        eligible = np.ones(classes, dtype=bool)
        if spec.composition == "reduced_coverage":
            eligible[list(spec.dropped_classes)] = False
        eligible_capacity = capacities * eligible
        if spec.composition == "long_tail":
            # Every class remains eligible; ratio is max/min desired count.
            weights = np.geomspace(spec.long_tail_ratio, 1.0, classes)
        else:
            weights = eligible.astype(float)
        allocation = _allocate(spec.size, weights, eligible_capacity)
        picked = []
        for cls, count in enumerate(allocation):
            if count:
                picked.append(_rng(spec, f"class-{cls}").permutation(by_class[cls])[:count])
        positions = np.concatenate(picked) if picked else np.empty(0, dtype=int)
        positions = _rng(spec, "final-order").permutation(positions)

    actual = np.bincount(y[positions], minlength=classes).astype(int)
    if len(positions) != spec.size or len(np.unique(positions)) != spec.size:
        raise AssertionError("proxy design is not a valid subset")
    return ProxyDesign(
        positions=positions.astype(np.int64),
        diagnostics={
            "proxy_size": int(spec.size),
            "proxy_composition": spec.composition,
            "proxy_design": spec.to_dict(),
            "class_counts": actual.tolist(),
            "covered_classes": np.flatnonzero(actual > 0).astype(int).tolist(),
            "coverage_fraction": float((actual > 0).mean()),
            "class_count_ratio": float(actual.max() / actual[actual > 0].min()),
        },
    )
