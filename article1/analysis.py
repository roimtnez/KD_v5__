"""The sole results consumer: paired effects by dataset and regime."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def paired_effects(results_csv: Path, reference: str = "feddf_logit", temperature: float = 8.0) -> list[dict]:
    """Return seed-paired accuracy differences; regimes are not pseudo-replicates."""
    with Path(results_csv).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected_temperature = float(temperature)
    by_condition: dict[tuple[str, str, str, float], dict[str, float]] = defaultdict(dict)
    for row in rows:
        if float(row["temperature"]) != selected_temperature:
            continue
        key = (row["dataset"], row.get("regime", ""), row["seed"], selected_temperature)
        if row["method"] in by_condition[key]:
            raise ValueError(f"duplicate result identity in analysis: {key + (row['method'],)}")
        by_condition[key][row["method"]] = float(row["student_test_accuracy"])
    grouped: dict[tuple[str, str, str, float], list[float]] = defaultdict(list)
    for (dataset, regime, _, row_temperature), methods in by_condition.items():
        if reference not in methods: continue
        for method, value in methods.items():
            if method != reference: grouped[(dataset, regime, method, row_temperature)].append(value - methods[reference])
    return [{"dataset": dataset, "regime": regime, "method": method, "temperature": row_temperature, "paired_seeds": len(values),
             "mean_accuracy_delta": float(np.mean(values)), "sd_accuracy_delta": float(np.std(values, ddof=1)) if len(values) > 1 else None}
            for (dataset, regime, method, row_temperature), values in sorted(grouped.items())]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--reference", default="feddf_logit")
    parser.add_argument("--temperature", type=float, default=8.0)
    args = parser.parse_args()
    print(json.dumps(paired_effects(args.results, args.reference, args.temperature), indent=2, allow_nan=False))
