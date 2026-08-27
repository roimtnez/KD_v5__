"""Fail-closed integrity checks for seed-scoped artifact chains."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


_SEED_PATTERNS = (re.compile(r"^seed_(\d+)$"), re.compile(r"^seed(\d+)$"))


def seeds_in_path(path: Path | str) -> set[int]:
    found: set[int] = set()
    for part in Path(path).parts:
        for pattern in _SEED_PATTERNS:
            match = pattern.match(part)
            if match:
                found.add(int(match.group(1)))
    return found


def assert_artifact_seed(path: Path | str, seed: int, *, allow_legacy_42: bool = False) -> None:
    found = seeds_in_path(path)
    if not found and int(seed) == 42 and allow_legacy_42:
        return
    if found != {int(seed)}:
        raise ValueError(f"artifact path {path} carries seeds {sorted(found)}, expected only {seed}")


def assert_single_seed_chain(seed: int, artifacts: Mapping[str, Path | str], *, allow_legacy_42: bool = False) -> None:
    for name, path in artifacts.items():
        try:
            assert_artifact_seed(path, seed, allow_legacy_42=allow_legacy_42)
        except ValueError as exc:
            raise ValueError(f"{name}: {exc}") from exc


def assert_aligned_proxy_idx(reference: np.ndarray, **arrays: np.ndarray) -> None:
    ref = np.asarray(reference, dtype=np.int64)
    if ref.ndim != 1 or len(np.unique(ref)) != len(ref):
        raise ValueError("reference proxy_idx must be one-dimensional and unique")
    for name, candidate in arrays.items():
        value = np.asarray(candidate, dtype=np.int64)
        if not np.array_equal(ref, value):
            raise ValueError(f"proxy_idx mismatch for {name}")


def assert_ln_split(order: np.ndarray, N: int, *, n_total: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.asarray(order, dtype=np.int64)
    if len(order) != n_total or not np.array_equal(np.sort(order), np.arange(n_total)):
        raise ValueError("L_N order must be a permutation of the proxy positions")
    L = order[:int(N)]
    U = order[int(N):]
    if np.intersect1d(L, U).size:
        raise ValueError("leakage: L_N and U_N overlap")
    return L, U


def validate_csv_grid(
    frame: pd.DataFrame,
    *,
    key_columns: Sequence[str],
    expected_rows: Iterable[tuple],
) -> None:
    if "seed" not in frame.columns:
        raise ValueError("CSV is missing required seed column")
    if frame.duplicated(list(key_columns)).any():
        raise ValueError(f"CSV contains duplicate keys {list(key_columns)}")
    actual = {tuple(row) for row in frame[list(key_columns)].itertuples(index=False, name=None)}
    expected = set(expected_rows)
    if actual != expected:
        raise ValueError(f"grid mismatch: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
