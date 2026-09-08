"""Read-only validation and reuse planning for Article-1 RQ2 temperature cells."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from article1.distillation import build_target
from article1.hashes import array_sha256, file_sha256

RQ2_DATASET = "cifar"
RQ2_REGIMES = ("iid", "alpha0p1", "single")
RQ2_SEEDS = (42, 43, 44)
RQ2_METHODS = ("expert_logit", "expert_prob")
PAIR_FIELDS = (
    "cache_sha256",
    "M_sha256",
    "proxy_sha256",
    "student_init_sha256",
    "batch_order_sha256",
    "updates",
    "fallback_rate",
    "mean_selected_teachers",
)
TARGET_FIELDS = (
    "target_accuracy",
    "target_nll",
    "target_entropy",
    "fallback_count",
    "fallback_rate",
    "mean_selected_teachers",
    "effective_teachers",
    "pre_restriction_outside_support_mass",
)


def rq2_cells(temperatures: Iterable[float]) -> set[tuple[str, str, int, str, float]]:
    return {
        (RQ2_DATASET, regime, seed, method, float(temperature))
        for regime in RQ2_REGIMES
        for seed in RQ2_SEEDS
        for method in RQ2_METHODS
        for temperature in temperatures
    }


def _key(row: dict) -> tuple[str, str, int, str, float]:
    return (
        row["dataset"],
        row["regime"],
        int(row["seed"]),
        row["method"],
        float(row["temperature"]),
    )


def _empty_or_equal(value: str | None, expected: float | None) -> bool:
    if expected is None:
        return value in (None, "", "nan", "NaN")
    try:
        return bool(np.isclose(float(value), float(expected), rtol=0.0, atol=1e-12))
    except (TypeError, ValueError):
        return False


def validate_cells(
    results_paths: Iterable[Path],
    *,
    source_root: Path,
    temperatures: Iterable[float] = (1.0, 4.0),
    reject_foreign: Path | None = None,
) -> dict:
    """Classify the fixed RQ2 cells without modifying CSVs or caches.

    A present row is reusable only when it matches its immutable cache and a
    freshly rebuilt target.  Rows from legacy result CSVs are accepted when
    their hashes verify; provenance fields absent from old rows are never
    fabricated.
    """
    expected = rq2_cells(temperatures)
    grouped: dict[tuple[str, str, int, str, float], list[dict]] = defaultdict(list)
    foreign: list[dict] = []
    for path in map(Path, results_paths):
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    key = _key(row)
                except (KeyError, ValueError):
                    continue
                if key in expected:
                    grouped[key].append(row)
                elif reject_foreign is not None and path == reject_foreign:
                    foreign.append({"path": str(path), "identity": key})

    cache_data: dict[tuple[str, int, str], tuple[dict, dict]] = {}
    target_data: dict[tuple[str, int, str, float], dict] = {}

    def validate_row(key: tuple[str, str, int, str, float], row: dict) -> list[str]:
        dataset, regime, seed, method, temperature = key
        condition = dataset, seed, regime
        errors: list[str] = []
        if condition not in cache_data:
            source = Path(source_root) / f"{dataset}-seed{seed}-{regime}"
            cache_path, metadata_path = (
                source / "teacher_cache.npz",
                source / "metadata.json",
            )
            if not cache_path.is_file() or not metadata_path.is_file():
                return ["missing_source"]
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            with np.load(cache_path, allow_pickle=False) as cache:
                cache_data[condition] = (
                    metadata,
                    {
                        name: cache[name]
                        for name in ("logits", "labels", "M", "proxy_idx")
                    },
                )
            cache_data[condition][1]["cache_sha256"] = file_sha256(cache_path)
        metadata, cache = cache_data[condition]
        if metadata.get("protocol") != "article1-v3":
            errors.append("source_protocol")
        if row.get("cache_sha256") != cache["cache_sha256"]:
            errors.append("cache_sha256")
        if row.get("M_sha256") != array_sha256(cache["M"]):
            errors.append("M_sha256")
        if row.get("proxy_sha256") != array_sha256(cache["proxy_idx"]):
            errors.append("proxy_sha256")
        target_key = condition + (temperature,)
        if target_key not in target_data:
            target_data[target_key] = {
                name: build_target(
                    cache["logits"],
                    cache["labels"],
                    cache["M"],
                    method=name,
                    temperature=temperature,
                )
                for name in RQ2_METHODS
            }
        target = target_data[target_key][method]
        if (
            not np.isfinite(target.probabilities).all()
            or (target.probabilities < 0).any()
            or not np.allclose(target.probabilities.sum(axis=1), 1.0, atol=1e-6)
        ):
            errors.append("invalid_target")
        for field in TARGET_FIELDS:
            if field in row and not _empty_or_equal(row[field], target.metrics[field]):
                errors.append(field)
        return errors

    status: dict[tuple[str, str, int, str, float], dict] = {}
    for key in sorted(expected):
        rows = grouped[key]
        if not rows:
            status[key] = {"status": "absent", "errors": []}
        elif len(rows) > 1:
            status[key] = {"status": "duplicated", "errors": ["multiple_rows"]}
        else:
            errors = validate_row(key, rows[0])
            status[key] = {
                "status": "valid_reusable" if not errors else "incompatible",
                "errors": errors,
            }

    pair_issues: list[dict] = []
    for regime in RQ2_REGIMES:
        for seed in RQ2_SEEDS:
            for temperature in map(float, temperatures):
                left_key = RQ2_DATASET, regime, seed, "expert_logit", temperature
                right_key = RQ2_DATASET, regime, seed, "expert_prob", temperature
                left, right = grouped[left_key], grouped[right_key]
                if (
                    len(left) == len(right) == 1
                    and status[left_key]["status"]
                    == status[right_key]["status"]
                    == "valid_reusable"
                ):
                    differing = [
                        field
                        for field in PAIR_FIELDS
                        if left[0].get(field) != right[0].get(field)
                    ]
                    condition = RQ2_DATASET, seed, regime
                    targets = target_data[condition + (temperature,)]
                    if not np.array_equal(
                        targets["expert_logit"].selected,
                        targets["expert_prob"].selected,
                    ):
                        differing.append("expert_routing")
                    if differing:
                        pair_issues.append(
                            {
                                "condition": left_key[:3] + (temperature,),
                                "fields": differing,
                            }
                        )

    counts: dict[str, int] = defaultdict(int)
    for value in status.values():
        counts[value["status"]] += 1
    return {
        "ok": not pair_issues
        and not foreign
        and not counts["duplicated"]
        and not counts["incompatible"],
        "expected_cells": len(expected),
        "counts": dict(counts),
        "pair_issues": pair_issues,
        "foreign_rows": foreign,
        "cells": [
            {
                "dataset": key[0],
                "regime": key[1],
                "seed": key[2],
                "method": key[3],
                "temperature": key[4],
                **value,
            }
            for key, value in sorted(status.items())
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--source-root", type=Path, default=Path("OUTPUTS/article1_v3/sources")
    )
    parser.add_argument("--temperatures", type=float, nargs="+", default=[1.0, 4.0])
    parser.add_argument(
        "--isolated-results",
        type=Path,
        help="require this CSV to contain no rows outside RQ2",
    )
    args = parser.parse_args()
    report = validate_cells(
        args.results,
        source_root=args.source_root,
        temperatures=args.temperatures,
        reject_foreign=args.isolated_results,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
