#!/usr/bin/env python3
"""Build the authoritative Article-1 fully-labelled proxy-size outputs.

This collector intentionally does not read the old paper tables.  It reads raw
student metrics, keeps the three Article-1 seeds as repetitions within a fixed
dataset/regime cell, and writes explicit missingness rather than silently
letting seed-42 rows stand in for a completed sample-efficiency experiment.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from experiments.article_1 import OUTPUT_ROOT, PROTOCOL_VERSION
from experiments.article_1.io import write_csv, write_json
from experiments.article_1.protocol import (
    CANONICAL_THRESHOLD, DATASETS, PROXY_EFFICIENCY_BASELINE_METHODS,
    PROXY_EFFICIENCY_ENERGY_METHODS, PROXY_EFFICIENCY_ORACLE_REFERENCE,
    PROXY_SIZES, REGIMES, SEEDS,
)


STUDY_ID = "article_1_proxy_sample_efficiency_v1"
# Direct supervised-standard intentionally has a different CE/augmentation
# budget.  The KD-budget control is supervised_matched, which is paired with
# FedDF and EXPERT-full.
MATCHED_METHODS = ("feddf", "expert_full", "supervised_matched")
PAIR_FIELDS = ("student_init_sha256", "train_order_sha256", "total_updates", "paired_run_id")


def _canonical(row: dict) -> bool:
    try:
        return math.isclose(
            float(row["threshold"]), CANONICAL_THRESHOLD[str(row["dataset"])], abs_tol=1e-9,
        )
    except (KeyError, TypeError, ValueError):
        return False


def _key(row: dict) -> tuple[str, int, str, int, str]:
    return (
        str(row["dataset"]), int(row["seed"]), str(row["regime"]),
        int(row["proxy_size"]), str(row["method"]),
    )


def _expected(methods: tuple[str, ...]) -> list[tuple[str, int, str, int, str]]:
    return [
        (dataset, seed, regime, size, method)
        for dataset in DATASETS for seed in SEEDS for regime in REGIMES
        for size in PROXY_SIZES for method in methods
    ]


def _historical_reuse(key: tuple[str, int, str, int, str]) -> bool:
    """Coordinates known to be compatible before this final study was opened."""
    _, seed, _, size, method = key
    return (
        (method == "feddf" and (seed == 42 or size == 10000))
        or (method == "expert_full" and size == 10000)
        or (method in {"supervised_matched", "supervised_standard"} and seed == 42)
    )


def _read_students(root: Path) -> tuple[dict[tuple, dict], list[dict]]:
    found: dict[tuple, dict] = {}
    duplicates: list[dict] = []
    for path in sorted((root / "runs").glob("**/students/*/metrics.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if not _canonical(row):
            continue
        key = _key(row)
        row["artifact"] = str(path)
        if key in found:
            duplicates.append({"key": list(key), "first": found[key]["artifact"], "second": str(path)})
        else:
            found[key] = row
    return found, duplicates


def _summary(rows: list[dict], value: str) -> dict:
    values = [float(row[value]) for row in rows]
    n = len(values)
    return {
        "n_seed": n,
        "mean": mean(values) if n else None,
        "sd": stdev(values) if n > 1 else None,
        "sem": stdev(values) / math.sqrt(n) if n > 1 else None,
        "min": min(values) if n else None,
        "max": max(values) if n else None,
    }


def _complete_target_label_check(rows: list[dict]) -> list[dict]:
    """Verify completed rows really point to a fully-labelled saved prefix."""
    failures: list[dict] = []
    for row in rows:
        target_path = Path(row["artifact"]).parents[2] / "targets.npz"
        try:
            with np.load(target_path, allow_pickle=False) as target:
                labels = np.asarray(target["labels"])
                indices = np.asarray(target["proxy_idx"])
            if len(labels) != int(row["proxy_size"]) or len(indices) != len(labels):
                failures.append({"artifact": row["artifact"], "reason": "proxy_labels_not_complete"})
        except (OSError, KeyError, ValueError) as error:
            failures.append({"artifact": row["artifact"], "reason": f"cannot_read_targets:{error}"})
    return failures


def _target_coverage(root: Path, methods: tuple[str, ...]) -> tuple[list[dict], list[dict]]:
    """Audit all planned cells, including ones whose student is still missing."""
    coverage: list[dict] = []
    failures: list[dict] = []
    for dataset in DATASETS:
        threshold = CANONICAL_THRESHOLD[dataset]
        tag = f"tau_{threshold:.2f}".replace(".", "p")
        for seed in SEEDS:
            for regime in REGIMES:
                for size in PROXY_SIZES:
                    path = root / "runs" / dataset / f"seed_{seed}" / regime / tag / f"N_{size}" / "targets.npz"
                    record = {
                        "dataset": dataset, "seed": seed, "regime": regime, "proxy_size": size,
                        "target_artifact": str(path), "exists": path.is_file(),
                    }
                    try:
                        with np.load(path, allow_pickle=False) as target:
                            labels = np.asarray(target["labels"])
                            indices = np.asarray(target["proxy_idx"])
                            missing_methods = [method for method in methods if f"target__{method}" not in target.files]
                        record.update({
                            "label_count": len(labels), "index_count": len(indices),
                            "fully_labeled": len(labels) == size and len(indices) == size,
                            "missing_target_methods": missing_methods,
                        })
                        if not record["fully_labeled"] or missing_methods:
                            failures.append({**record, "reason": "incomplete_labels_or_target_method"})
                    except (OSError, KeyError, ValueError) as error:
                        record.update({"fully_labeled": False, "missing_target_methods": list(methods)})
                        failures.append({**record, "reason": f"cannot_read_target:{error}"})
                    coverage.append(record)
    return coverage, failures


def _paired_effect_rows(available: dict[tuple, dict], methods: tuple[str, ...]) -> list[dict]:
    comparisons = (("expert_full", "feddf"), ("expert_full", "supervised_matched"),
                   ("expert_full", "supervised_standard"))
    output: list[dict] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            for regime in REGIMES:
                for size in PROXY_SIZES:
                    for left, right in comparisons:
                        if left not in methods or right not in methods:
                            continue
                        left_row = available.get((dataset, seed, regime, size, left))
                        right_row = available.get((dataset, seed, regime, size, right))
                        if not left_row or not right_row:
                            continue
                        output.append({
                            "dataset": dataset, "seed": seed, "regime": regime, "proxy_size": size,
                            "comparison": f"{left}_minus_{right}",
                            "left_method": left, "right_method": right,
                            "accuracy_delta": float(left_row["student_test_accuracy"])
                            - float(right_row["student_test_accuracy"]),
                        })
    return output


def _paired_summaries(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["regime"], row["proxy_size"], row["comparison"])].append(row)
    return [
        {"dataset": key[0], "regime": key[1], "proxy_size": key[2], "comparison": key[3],
         **_summary(value, "accuracy_delta")}
        for key, value in sorted(grouped.items())
    ]


def _curve_summaries(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["regime"], int(row["proxy_size"]), row["method"])].append(row)
    return [
        {"dataset": key[0], "regime": key[1], "proxy_size": key[2], "method": key[3],
         **_summary(value, "student_test_accuracy")}
        for key, value in sorted(grouped.items())
    ]


def _catchup_rows(available: dict[tuple, dict], methods: tuple[str, ...]) -> list[dict]:
    """Same-size supervised catch-up, with an explicit sustained variant."""
    output: list[dict] = []
    for supervised in ("supervised_standard", "supervised_matched"):
        if supervised not in methods:
            continue
        for dataset in DATASETS:
            for seed in SEEDS:
                for regime in REGIMES:
                    pairs = []
                    for size in PROXY_SIZES:
                        expert = available.get((dataset, seed, regime, size, "expert_full"))
                        direct = available.get((dataset, seed, regime, size, supervised))
                        if expert is not None and direct is not None:
                            pairs.append((size, float(direct["student_test_accuracy"])
                                          - float(expert["student_test_accuracy"])))
                    for tolerance_pp in (0, 1, 2):
                        threshold = -tolerance_pp / 100.0
                        first = next((size for size, delta in pairs if delta >= threshold), None)
                        sustained = next(
                            (size for index, (size, _) in enumerate(pairs)
                             if all(delta >= threshold for _, delta in pairs[index:])), None,
                        )
                        output.append({
                            "dataset": dataset, "seed": seed, "regime": regime,
                            "supervised_method": supervised, "tolerance_pp": tolerance_pp,
                            "observed_size_count": len(pairs), "first_same_size_catchup": first,
                            "first_sustained_same_size_catchup": sustained,
                            "complete_curve": len(pairs) == len(PROXY_SIZES),
                        })
    return output


def _attainment_rows(available: dict[tuple, dict], methods: tuple[str, ...]) -> list[dict]:
    """How much comparator proxy is needed to attain each EXPERT-full level."""
    output: list[dict] = []
    for comparator in ("feddf", "supervised_standard", "supervised_matched"):
        if comparator not in methods:
            continue
        for dataset in DATASETS:
            for seed in SEEDS:
                for regime in REGIMES:
                    comparator_rows = [
                        available[(dataset, seed, regime, size, comparator)]
                        for size in PROXY_SIZES if (dataset, seed, regime, size, comparator) in available
                    ]
                    for reference_size in PROXY_SIZES:
                        reference = available.get((dataset, seed, regime, reference_size, "expert_full"))
                        if reference is None:
                            continue
                        for tolerance_pp in (0, 1, 2):
                            required = float(reference["student_test_accuracy"]) - tolerance_pp / 100.0
                            hit = next(
                                (row for row in comparator_rows
                                 if float(row["student_test_accuracy"]) >= required), None,
                            )
                            output.append({
                                "dataset": dataset, "seed": seed, "regime": regime,
                                "reference_method": "expert_full", "reference_proxy_size": reference_size,
                                "reference_accuracy": float(reference["student_test_accuracy"]),
                                "comparator": comparator, "tolerance_pp": tolerance_pp,
                                "first_comparator_proxy_size": int(hit["proxy_size"]) if hit else None,
                                "comparator_accuracy_at_crossing": (
                                    float(hit["student_test_accuracy"]) if hit else None
                                ),
                            })
    return output


def build_outputs(root: Path, *, include_energy: bool = False) -> dict:
    methods = PROXY_EFFICIENCY_BASELINE_METHODS + (
        PROXY_EFFICIENCY_ENERGY_METHODS if include_energy else ()
    )
    expected = _expected(methods)
    found, duplicates = _read_students(root)
    available = {key: found[key] for key in expected if key in found}
    completed = []
    missing = []
    for key in expected:
        if key not in available:
            missing.append({
                "dataset": key[0], "seed": key[1], "regime": key[2], "proxy_size": key[3],
                "method": key[4], "required_action": "train_student_after_target_exists",
            })
            continue
        row = dict(available[key])
        row["study_id"] = STUDY_ID
        row["fully_labeled_proxy"] = True
        row["artifact_role"] = "main" if key[4] in PROXY_EFFICIENCY_BASELINE_METHODS else "energy_extension"
        row["artifact_origin"] = "reused_compatible" if _historical_reuse(key) else "newly_trained"
        completed.append(row)

    references = []
    for key, row in found.items():
        dataset, seed, regime, size, method = key
        if method in PROXY_EFFICIENCY_ORACLE_REFERENCE and seed == 42 and size == 10000:
            references.append({**row, "study_id": STUDY_ID, "artifact_role": "reference_only",
                               "fully_labeled_proxy": True,
                               "reference_scope": "N10000_seed42_only_not_main_effect"})

    target_failures = _complete_target_label_check(completed)
    target_coverage, planned_target_failures = _target_coverage(root, methods)
    paired_failures = []
    for dataset in DATASETS:
        for seed in SEEDS:
            for regime in REGIMES:
                for size in PROXY_SIZES:
                    rows = [available.get((dataset, seed, regime, size, method)) for method in MATCHED_METHODS]
                    rows = [row for row in rows if row is not None]
                    if len(rows) > 1:
                        for field in PAIR_FIELDS:
                            if len({row.get(field) for row in rows}) != 1:
                                paired_failures.append({
                                    "dataset": dataset, "seed": seed, "regime": regime,
                                    "proxy_size": size, "field": field,
                                })

    effects = _paired_effect_rows(available, methods)
    output_dir = root / "summary" / "proxy_sample_efficiency"
    write_csv(output_dir / "results.csv", completed)
    write_csv(output_dir / "missing.csv", missing)
    write_csv(output_dir / "oracle_reference.csv", references)
    write_csv(output_dir / "curve_summary_by_dataset_regime.csv", _curve_summaries(completed))
    write_csv(output_dir / "paired_effects_by_seed.csv", effects)
    write_csv(output_dir / "paired_effect_summary.csv", _paired_summaries(effects))
    write_csv(output_dir / "supervised_catchup.csv", _catchup_rows(available, methods))
    write_csv(output_dir / "expert_attainment_crossings.csv", _attainment_rows(available, methods))
    write_csv(output_dir / "target_label_failures.csv", target_failures)
    write_csv(output_dir / "target_coverage.csv", target_coverage)
    write_csv(output_dir / "planned_target_failures.csv", planned_target_failures)
    write_csv(output_dir / "pairing_failures.csv", paired_failures)

    expected_by_method = {method: sum(key[4] == method for key in expected) for method in methods}
    observed_by_method = {method: sum(row["method"] == method for row in completed) for method in methods}
    manifest = {
        "study_id": STUDY_ID,
        "protocol_version": PROTOCOL_VERSION,
        "scientific_scope": "fully_labelled_total_proxy_size_only",
        "proxy_sizes": list(PROXY_SIZES),
        "main_methods": list(PROXY_EFFICIENCY_BASELINE_METHODS),
        "energy": {
            "included": include_energy,
            "role": "post_baseline_extension" if include_energy else "not_in_baseline",
            "historical_results_compatible": False,
        },
        "oracle_full": {
            "role": "reference_only", "allowed_coordinates": "seed=42,N=10000",
            "included_rows": len(references),
        },
        "seeds": list(SEEDS),
        "regimes": list(REGIMES),
        "seed_interpretation": "experimental repetitions; regimes are conditions, not independent replications",
        "fully_labeled_proxy": True,
        "target_conditions": {
            "available": sum(bool(row["exists"]) for row in target_coverage),
            "expected": len(target_coverage),
            "fully_labeled_and_method_complete": len(planned_target_failures) == 0,
        },
        "expected_students": len(expected),
        "completed_students": len(completed),
        "missing_students": len(missing),
        "complete": (
            len(missing) == 0 and not duplicates and not target_failures
            and not planned_target_failures and not paired_failures
        ),
        "coverage_by_method": {
            method: {"observed": observed_by_method[method], "expected": expected_by_method[method],
                     "complete": observed_by_method[method] == expected_by_method[method]}
            for method in methods
        },
        "historical_reuse_rule": {
            "feddf": "seed=42 all sizes plus N=10000 for seeds 43/44",
            "expert_full": "N=10000 all seeds/regimes",
            "supervised": "seed=42 all sizes/regimes",
        },
        "quality_checks": {
            "duplicate_raw_cells": len(duplicates),
            "target_label_failures": len(target_failures),
            "planned_target_failures": len(planned_target_failures),
            "paired_control_failures": len(paired_failures),
        },
        "outputs": [
            "results.csv", "missing.csv", "oracle_reference.csv", "curve_summary_by_dataset_regime.csv",
            "paired_effects_by_seed.csv", "paired_effect_summary.csv", "supervised_catchup.csv",
            "expert_attainment_crossings.csv", "target_label_failures.csv", "pairing_failures.csv",
            "target_coverage.csv", "planned_target_failures.csv",
        ],
        "execution_commands": {
            "baseline_targets": "PYTHONPATH=. conda run -n FLWR python -m experiments.article_1.run --stage targets --preset proxy-efficiency",
            "baseline_students": "PYTHONPATH=. conda run -n FLWR python -m experiments.article_1.run --stage students --preset proxy-efficiency --device cuda",
            "collect": "PYTHONPATH=. conda run -n FLWR python -m experiments.article_1.collect_proxy_efficiency",
            "energy_after_baseline": "PYTHONPATH=. conda run -n FLWR python -m experiments.article_1.run --stage targets --preset proxy-efficiency-energy --force && PYTHONPATH=. conda run -n FLWR python -m experiments.article_1.run --stage students --preset proxy-efficiency-energy --device cuda && PYTHONPATH=. conda run -n FLWR python -m experiments.article_1.collect_proxy_efficiency --include-energy",
        },
    }
    write_json(output_dir / "manifest.json", manifest, force=True)
    write_json(output_dir / "audit.json", {
        "status": "verified" if manifest["complete"] else "incomplete",
        "duplicate_raw_cells": duplicates,
        "target_label_failures": target_failures,
        "planned_target_failures": planned_target_failures,
        "paired_control_failures": paired_failures,
        "missing_students": len(missing),
    }, force=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path(OUTPUT_ROOT))
    parser.add_argument("--include-energy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_outputs(args.output_root, include_energy=args.include_energy)
    print(
        f"[PROXY EFFICIENCY] complete={manifest['complete']} "
        f"students={manifest['completed_students']}/{manifest['expected_students']} "
        f"summary={args.output_root / 'summary' / 'proxy_sample_efficiency'}"
    )


if __name__ == "__main__":
    main()
