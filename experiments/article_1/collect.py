#!/usr/bin/env python3
"""Collect compact Article-1 result tables and proxy-size crossing points."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from experiments.article_1 import OUTPUT_ROOT
from experiments.article_1.io import write_csv
from experiments.article_1.protocol import CANONICAL_THRESHOLD, DATASETS, FINAL_BASELINE_METHODS, REGIMES, SEEDS


def collect_rows(root: Path) -> tuple[list[dict], list[dict]]:
    targets: list[dict] = []
    students: list[dict] = []
    # Controls added after the frozen primary corpus are immutable sidecars.
    for path in sorted((Path(root) / "runs").glob("**/target_metrics*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["methods"]:
            targets.append({**row, "artifact": str(path)})
    for path in sorted((Path(root) / "runs").glob("**/students/*/metrics.json")):
        students.append({**json.loads(path.read_text(encoding="utf-8")), "artifact": str(path)})
    return targets, students


def crossing_rows(students: list[dict], reference_method: str) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in students:
        key = (row["dataset"], row["seed"], row["regime"], row["threshold"])
        grouped[key].append(row)
    result = []
    for key, rows in grouped.items():
        references = [row for row in rows if row["method"] == reference_method]
        if not references:
            continue
        reference = max(references, key=lambda row: int(row["proxy_size"]))
        reference_accuracy = float(reference["student_test_accuracy"])
        methods = sorted({row["method"] for row in rows})
        for method in methods:
            candidates = sorted(
                (row for row in rows if row["method"] == method),
                key=lambda row: int(row["proxy_size"]),
            )
            for tolerance in (0.01, 0.02):
                crossing = next(
                    (row for row in candidates
                     if float(row["student_test_accuracy"]) >= reference_accuracy - tolerance),
                    None,
                )
                result.append({
                    "dataset": key[0], "seed": key[1], "regime": key[2], "threshold": key[3],
                    "method": method, "reference_method": reference_method,
                    "reference_proxy_size": int(reference["proxy_size"]),
                    "reference_accuracy": reference_accuracy,
                    "tolerance_pp": int(tolerance * 100),
                    "first_proxy_size": int(crossing["proxy_size"]) if crossing else None,
                    "crossing_accuracy": float(crossing["student_test_accuracy"]) if crossing else None,
                })
    return result


def final_baseline_rows(students: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return the four-method opening comparison and its paired effects."""
    selected = [
        row for row in students
        if row.get("method") in FINAL_BASELINE_METHODS
        and int(row["proxy_size"]) == 10000
        and abs(float(row["threshold"]) - CANONICAL_THRESHOLD[row["dataset"]]) < 1e-12
    ]
    selected.sort(key=lambda row: (row["dataset"], int(row["seed"]), row["regime"], row["method"]))
    grouped: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in selected:
        grouped[(row["dataset"], int(row["seed"]), row["regime"])][row["method"]] = row
    effects: list[dict] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            for regime in REGIMES:
                rows = grouped.get((dataset, seed, regime), {})
                record = {"dataset": dataset, "seed": seed, "regime": regime,
                          "complete": set(rows) == set(FINAL_BASELINE_METHODS)}
                for method in FINAL_BASELINE_METHODS:
                    record[f"accuracy__{method}"] = (
                        float(rows[method]["student_test_accuracy"]) if method in rows else None
                    )
                if record["complete"]:
                    record.update({
                        "expert_full_minus_feddf": record["accuracy__expert_full"] - record["accuracy__feddf"],
                        "expert_full_minus_energy": record["accuracy__expert_full"] - record["accuracy__energy"],
                        "oracle_full_minus_expert_full": record["accuracy__oracle_full"] - record["accuracy__expert_full"],
                        "paired_run_id": rows["feddf"]["paired_run_id"],
                        "teacher_source_sha256": rows["feddf"]["teacher_source_sha256"],
                        "proxy_subset_sha256": rows["feddf"]["proxy_subset_sha256"],
                    })
                effects.append(record)
    return selected, effects


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path(OUTPUT_ROOT))
    parser.add_argument("--reference-method", default="expert_v2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets, students = collect_rows(args.output_root)
    summary = args.output_root / "summary"
    baseline, baseline_effects = final_baseline_rows(students)
    write_csv(summary / "target_results.csv", targets)
    write_csv(summary / "student_results.csv", students)
    write_csv(summary / "crossings.csv", crossing_rows(students, args.reference_method))
    write_csv(summary / "final_baseline_results.csv", baseline)
    write_csv(summary / "final_baseline_paired_effects.csv", baseline_effects)
    print(f"[OK] targets={len(targets)} students={len(students)} baseline={len(baseline)} summary={summary}")


if __name__ == "__main__":
    main()
