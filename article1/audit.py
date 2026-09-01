"""Read-only integrity audit for a completed Article-1 grid.

It intentionally checks provenance and paired-comparison invariants before any
scientific interpretation.  It does not alter results, partitions, or caches.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from article1 import DATASETS, REGIMES, SEEDS, THRESHOLDS
from article1.distillation import METHODS, authority_from_holdout, build_target


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _issue(issues: list[dict], kind: str, **details) -> None:
    issues.append({"kind": kind, **details})


def audit(
    results_path: Path, *, source_root: Path,
    datasets: Iterable[str] = DATASETS, seeds: Iterable[int] = SEEDS,
    regimes: Iterable[str] = REGIMES, methods: Iterable[str] = METHODS,
    temperatures: Iterable[float] = (8.0,),
) -> dict:
    """Return a JSON-serializable audit. ``ok`` is false on any invariant error."""
    results_path, source_root = Path(results_path), Path(source_root)
    expected_datasets, expected_seeds = tuple(datasets), tuple(int(x) for x in seeds)
    expected_regimes, expected_methods = tuple(regimes), tuple(methods)
    issues: list[dict] = []
    with results_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"dataset", "regime", "seed", "method", "run_id", "cache_sha256", "M_sha256", "proxy_sha256", "student_init_sha256", "batch_order_sha256", "updates", "temperature"}
    missing_fields = required - set(rows[0] if rows else ())
    if missing_fields:
        _issue(issues, "missing_result_columns", columns=sorted(missing_fields))
        return {"ok": False, "rows": len(rows), "issues": issues}

    key_rows: dict[tuple[str, int, str, str], list[dict]] = defaultdict(list)
    run_ids = Counter()
    for row in rows:
        try:
            key = (row["dataset"], int(row["seed"]), row["regime"], row["method"])
        except (KeyError, ValueError):
            _issue(issues, "invalid_result_identity", row=row)
            continue
        key_rows[key].append(row); run_ids[row["run_id"]] += 1
    for key, matching in key_rows.items():
        if len(matching) != 1: _issue(issues, "duplicate_condition_method", condition=key, rows=len(matching))
    for run_id, count in run_ids.items():
        if count != 1: _issue(issues, "duplicate_run_id", run_id=run_id, rows=count)

    conditions = [(d, s, r) for d in expected_datasets for s in expected_seeds for r in expected_regimes]
    for condition in conditions:
        observed = {method for (*prefix, method) in key_rows if tuple(prefix) == condition}
        absent, unexpected = sorted(set(expected_methods) - observed), sorted(observed - set(expected_methods))
        if absent: _issue(issues, "missing_methods", condition=condition, methods=absent)
        if unexpected: _issue(issues, "unexpected_methods", condition=condition, methods=unexpected)
        arm_rows = [key_rows[(*condition, method)][0] for method in observed if len(key_rows[(*condition, method)]) == 1]
        if arm_rows:
            paired = ("cache_sha256", "M_sha256", "proxy_sha256", "student_init_sha256", "batch_order_sha256", "updates", "temperature")
            for field in paired:
                values = {row[field] for row in arm_rows}
                if len(values) != 1: _issue(issues, "unpaired_arms", condition=condition, field=field, values=sorted(values))
        _audit_source(condition, source_root, arm_rows, issues)

    expected_rows = len(conditions) * len(expected_methods)
    sanity = _target_sanity(conditions, source_root, temperatures=tuple(temperatures), issues=issues)
    return {"ok": not issues, "rows": len(rows), "expected_rows": expected_rows,
            "target_sanity": sanity,
            "conditions": len(conditions), "methods": len(expected_methods), "issues": issues}


def _target_sanity(
    conditions: list[tuple[str, int, str]], source_root: Path, *,
    temperatures: tuple[float, ...], issues: list[dict],
) -> dict:
    """Rebuild targets from caches and enforce algebraic protocol invariants."""
    checked = 0
    for dataset, seed, regime in conditions:
        cache_path = source_root / f"{dataset}-seed{seed}-{regime}" / "teacher_cache.npz"
        if not cache_path.is_file():
            continue
        with np.load(cache_path, allow_pickle=False) as cache:
            logits, labels, mask = cache["logits"], cache["labels"], cache["M"]
        for temperature in temperatures:
            targets = {method: build_target(logits, labels, mask, method=method, temperature=temperature) for method in METHODS}
            checked += len(targets)
            feddf = targets["feddf_logit"]
            for method, target in targets.items():
                if not np.isfinite(target.probabilities).all() or not np.allclose(target.probabilities.sum(axis=1), 1.0, atol=1e-6):
                    _issue(issues, "invalid_target_distribution", condition=(dataset, seed, regime), method=method, temperature=temperature)
                if not np.allclose(target.weights.sum(axis=1), 1.0, atol=1e-7) or (target.weights < 0).any():
                    _issue(issues, "invalid_teacher_weights", condition=(dataset, seed, regime), method=method, temperature=temperature)
                if target.fallback.any() and not np.array_equal(target.probabilities[target.fallback], feddf.probabilities[target.fallback]):
                    _issue(issues, "fallback_differs_from_feddf", condition=(dataset, seed, regime), method=method, temperature=temperature)
            expert = [targets[name].selected for name in ("expert_logit", "expert_prob", "expert_prob_sr")]
            if not (np.array_equal(expert[0], expert[1]) and np.array_equal(expert[1], expert[2])):
                _issue(issues, "expert_selection_differs", condition=(dataset, seed, regime), temperature=temperature)
            if not np.array_equal(targets["oracle_logit"].selected, targets["oracle_prob"].selected):
                _issue(issues, "oracle_selection_differs", condition=(dataset, seed, regime), temperature=temperature)
            # Pure construction must be bitwise repeatable with immutable inputs.
            repeated = build_target(logits, labels, mask, method="expert_prob_sr", temperature=temperature)
            if not (np.array_equal(repeated.probabilities, targets["expert_prob_sr"].probabilities) and np.array_equal(repeated.weights, targets["expert_prob_sr"].weights)):
                _issue(issues, "target_construction_not_exactly_reproducible", condition=(dataset, seed, regime), temperature=temperature)
    return {"temperatures": list(temperatures), "targets_checked": checked}


def _audit_source(condition: tuple[str, int, str], source_root: Path, arm_rows: list[dict], issues: list[dict]) -> None:
    dataset, seed, regime = condition
    source = source_root / f"{dataset}-seed{seed}-{regime}"
    metadata_path, cache_path = source / "metadata.json", source / "teacher_cache.npz"
    if not metadata_path.is_file() or not cache_path.is_file():
        _issue(issues, "missing_source", condition=condition, source=str(source)); return
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (metadata.get("dataset"), metadata.get("seed"), metadata.get("regime")) != condition:
        _issue(issues, "source_identity_mismatch", condition=condition)
    digest = _sha256(cache_path)
    if metadata.get("cache_sha256") != digest:
        _issue(issues, "cache_metadata_hash_mismatch", condition=condition)
    for row in arm_rows:
        if row["cache_sha256"] != digest:
            _issue(issues, "result_cache_hash_mismatch", condition=condition, method=row["method"])
    with np.load(cache_path, allow_pickle=False) as cache:
        needed = {"proxy_idx", "labels", "logits", "M", "holdout_accuracy", "holdout_counts"}
        absent = needed - set(cache.files)
        if absent:
            _issue(issues, "cache_missing_arrays", condition=condition, arrays=sorted(absent)); return
        logits, mask = cache["logits"], cache["M"]
        if logits.ndim != 3 or logits.shape[1:] != mask.shape or not np.isfinite(logits).all():
            _issue(issues, "invalid_proxy_logits_or_M_shape", condition=condition)
        expected_mask = authority_from_holdout(cache["holdout_accuracy"], cache["holdout_counts"], THRESHOLDS[dataset])
        if not np.array_equal(mask, expected_mask):
            _issue(issues, "M_not_reproducible_from_holdout", condition=condition)
        m_hash = hashlib.sha256(mask.tobytes()).hexdigest()
        p_hash = hashlib.sha256(cache["proxy_idx"].tobytes()).hexdigest()
        for row in arm_rows:
            if row["M_sha256"] != m_hash: _issue(issues, "result_M_hash_mismatch", condition=condition, method=row["method"])
            if row["proxy_sha256"] != p_hash: _issue(issues, "result_proxy_hash_mismatch", condition=condition, method=row["method"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--source-root", type=Path, default=Path("OUTPUTS/article1/sources"))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--regimes", nargs="+", choices=REGIMES, default=list(REGIMES))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--temperatures", nargs="+", type=float, default=[8.0], help="target-sanity temperatures")
    args = parser.parse_args()
    report = audit(args.results, source_root=args.source_root, datasets=args.datasets, seeds=args.seeds,
                   regimes=args.regimes, methods=args.methods, temperatures=args.temperatures)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ok"]: raise SystemExit(1)


if __name__ == "__main__": main()
