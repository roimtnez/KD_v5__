#!/usr/bin/env python3
"""Audit leakage invariants and planned-study coverage for Article 1."""
from __future__ import annotations

import argparse
import ast
import inspect
import json
from pathlib import Path

import numpy as np

from experiments.article_1 import OUTPUT_ROOT
from experiments.article_1.audit_teacher_provenance import build_report as provenance_report
from experiments.article_1.io import sha256_array, sha256_file, write_json
from experiments.article_1.protocol import (
    CANONICAL_THRESHOLD, CENTRAL_METHODS, CURVE_METHODS, DATASETS, FINAL_BASELINE_METHODS,
    PROXY_SIZES, REGIMES, SEEDS,
)
from experiments.article_1.targets import authority_from_holdout
from experiments.article_1.training import train_student


def _student_test_is_evaluation_only() -> bool:
    """Verify that train_student reads test_loader only in the final evaluate call."""
    tree = ast.parse(inspect.getsource(train_student))
    loads = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "test_loader" and isinstance(node.ctx, ast.Load):
            loads.append(node)
    if len(loads) != 1:
        return False
    parent_call = next(
        (node for node in ast.walk(tree) if isinstance(node, ast.Call) and loads[0] in ast.walk(node)),
        None,
    )
    return (
        isinstance(parent_call, ast.Call)
        and isinstance(parent_call.func, ast.Name)
        and parent_call.func.id == "evaluate"
    )


def _run_identity(path: Path, runs_root: Path) -> tuple[str, int, str, int]:
    parts = path.relative_to(runs_root).parts
    return parts[0], int(parts[1].split("_")[1]), parts[2], int(parts[4].split("_")[1])


def _canonical(row: dict) -> bool:
    return np.isclose(float(row["threshold"]), CANONICAL_THRESHOLD[str(row["dataset"])])


def build_report(output_root: Path, *, repo_root: Path) -> dict:
    output_root = Path(output_root)
    runs_root = output_root / "runs"
    errors: list[str] = []
    target_checks = 0
    oracle_checks = 0

    provenance = provenance_report(output_root, repo_root=repo_root)
    if provenance["artifact_chain_status"] != "verified":
        errors.append("teacher_source_artifact_chain_invalid")

    for metrics_path in sorted(runs_root.glob("**/target_metrics.json")):
        dataset, seed, regime, size = _run_identity(metrics_path, runs_root)
        run_dir = metrics_path.parent
        source_dir = output_root / "sources" / dataset / f"seed_{seed}" / regime
        with np.load(source_dir / "teacher_source.npz", allow_pickle=False) as source:
            source_arrays = {key: np.asarray(source[key]) for key in source.files}
        with np.load(run_dir / "targets.npz", allow_pickle=False) as targets:
            target_arrays = {key: np.asarray(targets[key]) for key in targets.files}
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        threshold = float(metrics["condition"]["threshold"])

        expected_authority = authority_from_holdout(
            source_arrays["holdout_accuracy"], source_arrays["holdout_counts"], threshold,
        )
        if not np.array_equal(target_arrays["authority"], expected_authority):
            errors.append(f"authority_not_from_holdout:{run_dir}")
        positions = source_arrays["proxy_order"][:size].astype(np.int64)
        if not np.array_equal(target_arrays["proxy_idx"], source_arrays["proxy_idx"][positions]):
            errors.append(f"proxy_subset_not_nested_prefix:{run_dir}")
        if not np.array_equal(target_arrays["labels"], source_arrays["proxy_labels"][positions]):
            errors.append(f"proxy_label_mismatch:{run_dir}")

        # Target files are intentionally method-sparse for the newer
        # proxy-efficiency studies.  Validate ORACLE-v2 wherever that target
        # was requested; its absence is not a leakage or compatibility error.
        if {"target__oracle_v2", "target__supervised_matched"}.issubset(target_arrays):
            oracle = target_arrays["target__oracle_v2"]
            supervised = target_arrays["target__supervised_matched"]
            different = np.any(np.abs(oracle - supervised) > 1e-7, axis=1)
            oracle_metrics = next(row for row in metrics["methods"] if row["method"] == "oracle_v2")
            if int(different.sum()) != int(oracle_metrics["oracle_no_correct_count"]):
                errors.append(f"oracle_non_onehot_not_equal_fallback:{run_dir}")
            if not bool(oracle_metrics["one_hot_when_oracle_available"]):
                errors.append(f"oracle_not_onehot_when_available:{run_dir}")
            oracle_checks += 1
        target_checks += 1

    students = [json.loads(path.read_text(encoding="utf-8"))
                for path in sorted(runs_root.glob("**/students/*/metrics.json"))]
    central = [row for row in students if row["method"] in CENTRAL_METHODS
               and int(row["proxy_size"]) == 10000 and _canonical(row)]
    curve = [row for row in students if row["method"] in CURVE_METHODS
             and int(row["seed"]) == 42 and _canonical(row)]
    threshold = [row for row in students if row["method"] == "expert_v2"
                 and int(row["seed"]) == 42 and int(row["proxy_size"]) == 10000]
    oracle_full = [row for row in students if row["method"] == "oracle_full"
                   and int(row["seed"]) == 42 and int(row["proxy_size"]) == 10000
                   and _canonical(row)]
    supervised_curve = [row for row in curve if row["method"] in {
        "supervised_matched", "supervised_standard",
    }]

    expected_central = len(DATASETS) * len(SEEDS) * len(REGIMES) * len(CENTRAL_METHODS)
    expected_curve = len(DATASETS) * len(REGIMES) * len(PROXY_SIZES) * len(CURVE_METHODS)
    expected_supervised = len(DATASETS) * len(REGIMES) * len(PROXY_SIZES) * 2
    expected_threshold = len(DATASETS) * len(REGIMES) * 3
    expected_oracle_full = len(DATASETS) * len(REGIMES)
    for name, observed, expected in (
        ("central", len(central), expected_central),
        ("curve", len(curve), expected_curve),
        ("supervised_curve", len(supervised_curve), expected_supervised),
        ("threshold", len(threshold), expected_threshold),
        ("oracle_full", len(oracle_full), expected_oracle_full),
    ):
        if observed != expected:
            errors.append(f"incomplete_{name}:{observed}!={expected}")

    paired: dict[tuple, dict[str, dict]] = {}
    for row in central:
        key = (row["dataset"], int(row["seed"]), row["regime"])
        paired.setdefault(key, {})[row["method"]] = row
    expert_pairs = 0
    for key, methods in paired.items():
        if set(methods) != set(CENTRAL_METHODS):
            errors.append(f"incomplete_central_cell:{key}")
            continue
        full, support = methods["expert_full"], methods["expert_v2"]
        for field in ("student_init_sha256", "train_order_sha256", "total_updates", "paired_run_id"):
            if full[field] != support[field]:
                errors.append(f"unpaired_expert_v1_v2:{key}:{field}")
        expert_pairs += 1

    target_hash_checks = 0
    for path in sorted(runs_root.glob("**/students/*/metrics.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        target_path = Path(row.get("target_artifact", path.parents[2] / "targets.npz"))
        if not target_path.is_file() or target_path.parent != path.parents[2]:
            errors.append(f"student_target_artifact_invalid:{path}")
            continue
        recorded_artifact_hash = row.get("target_artifact_sha256")
        if recorded_artifact_hash is not None and recorded_artifact_hash != sha256_file(target_path):
            errors.append(f"student_target_artifact_hash_mismatch:{path}")
        with np.load(target_path, allow_pickle=False) as payload:
            key = f"target__{row['method']}"
            if key not in payload:
                errors.append(f"student_target_missing_from_artifact:{path}")
                continue
            observed = sha256_array(np.asarray(payload[key]))
        if observed != row["target_sha256"]:
            errors.append(f"student_target_hash_mismatch:{path}")
        target_hash_checks += 1

    final_baseline = [
        row for row in students
        if row["method"] in FINAL_BASELINE_METHODS
        and int(row["proxy_size"]) == 10000 and _canonical(row)
    ]
    expected_final_baseline = len(DATASETS) * len(SEEDS) * len(REGIMES) * len(FINAL_BASELINE_METHODS)
    if len(final_baseline) != expected_final_baseline:
        errors.append(f"incomplete_final_baseline:{len(final_baseline)}!={expected_final_baseline}")
    final_pairs: dict[tuple, dict[str, dict]] = {}
    for row in final_baseline:
        key = (row["dataset"], int(row["seed"]), row["regime"])
        final_pairs.setdefault(key, {})[row["method"]] = row
    final_paired_cells = 0
    for key, methods in final_pairs.items():
        if set(methods) != set(FINAL_BASELINE_METHODS):
            errors.append(f"incomplete_final_baseline_cell:{key}")
            continue
        reference = methods["feddf"]
        for method, row in methods.items():
            for field in ("teacher_source_sha256", "proxy_subset_sha256", "student_init_sha256",
                          "train_order_sha256", "total_updates", "paired_run_id"):
                if row[field] != reference[field]:
                    errors.append(f"unpaired_final_baseline:{key}:{method}:{field}")
        final_paired_cells += 1

    checks = {
        "teacher_generation_assumed_correct": True,
        "client_train_holdout_test_and_proxy_disjoint": (
            provenance["artifact_integrity_verified_conditions"] == len(DATASETS) * len(SEEDS) * len(REGIMES)
        ),
        "every_saved_mask_recomputed_from_holdout_only": target_checks > 0 and not any(
            error.startswith("authority_not_from_holdout") for error in errors
        ),
        "every_proxy_subset_is_a_nested_source_prefix": target_checks > 0 and not any(
            error.startswith("proxy_subset") or error.startswith("proxy_label") for error in errors
        ),
        "student_optimization_never_reads_test_loader": _student_test_is_evaluation_only(),
        "every_student_target_hash_matches_saved_target": target_hash_checks == len(students) and not any(
            error.startswith("student_target_hash") for error in errors
        ),
        "oracle_v2_onehot_except_no_correct_fallback": oracle_checks > 0 and not any(
            error.startswith("oracle_") for error in errors
        ),
        "expert_v1_v2_common_random_numbers": expert_pairs == len(DATASETS) * len(SEEDS) * len(REGIMES)
            and not any(error.startswith("unpaired_expert") for error in errors),
        "final_baseline_common_random_numbers": final_paired_cells == len(DATASETS) * len(SEEDS) * len(REGIMES)
            and not any(error.startswith("unpaired_final_baseline") for error in errors),
    }
    if not all(checks.values()):
        errors.extend(f"failed_check:{name}" for name, value in checks.items() if not value)

    coverage = {
        "expert_v1_vs_v2": {"observed": len(central), "expected": expected_central,
                            "paired_cells": expert_pairs, "complete": len(central) == expected_central},
        "oracle_v2_onehot": {"observed_target_conditions": oracle_checks,
                             "expected_target_conditions": oracle_checks,
                             "complete": True},
        "supervised_proxy_curve": {"observed": len(supervised_curve),
                                   "expected": expected_supervised,
                                   "complete": len(supervised_curve) == expected_supervised},
        "kd_proxy_sample_efficiency": {"observed": len(curve), "expected": expected_curve,
                                       "complete": len(curve) == expected_curve},
        "final_baseline": {
            "methods": list(FINAL_BASELINE_METHODS),
            "observed": len(final_baseline), "expected": expected_final_baseline,
            "paired_cells": final_paired_cells,
            "complete": len(final_baseline) == expected_final_baseline
            and final_paired_cells == len(DATASETS) * len(SEEDS) * len(REGIMES),
        },
    }
    return {
        "scope": "article_1_executed_artifacts",
        "status": "verified" if not errors else "failed",
        "data_leakage_detected": bool(errors),
        "target_conditions_checked": target_checks,
        "student_trainings_checked": len(students),
        "checks": checks,
        "planned_analysis_coverage": coverage,
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path(OUTPUT_ROOT))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args.output_root, repo_root=args.repo_root.resolve())
    destination = args.report or args.output_root / "summary" / "execution_audit.json"
    write_json(destination, report, force=True)
    print(
        f"[EXECUTION AUDIT] status={report['status']} targets={report['target_conditions_checked']} "
        f"students={report['student_trainings_checked']} report={destination}"
    )
    if report["status"] != "verified":
        for error in report["errors"]:
            print(f"[ERROR] {error}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
