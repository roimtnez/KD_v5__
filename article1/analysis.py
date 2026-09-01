"""The sole results consumer: paired effects by dataset and regime."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def paired_effects(results_csv: Path, reference: str = "feddf_logit") -> list[dict]:
    """Return seed-paired accuracy differences; regimes are not pseudo-replicates."""
    with Path(results_csv).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_condition: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        key = (row["dataset"], row.get("regime", ""), row["seed"])
        by_condition[key][row["method"]] = float(row["student_test_accuracy"])
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for (dataset, regime, _), methods in by_condition.items():
        if reference not in methods: continue
        for method, value in methods.items():
            if method != reference: grouped[(dataset, regime, method)].append(value - methods[reference])
    return [{"dataset": dataset, "regime": regime, "method": method, "paired_seeds": len(values),
             "mean_accuracy_delta": float(np.mean(values)), "sd_accuracy_delta": float(np.std(values, ddof=1)) if len(values) > 1 else None}
            for (dataset, regime, method), values in sorted(grouped.items())]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--reference", default="feddf_logit")
    args = parser.parse_args()
    print(json.dumps(paired_effects(args.results, args.reference), indent=2, allow_nan=False))
