"""Update derived target diagnostics from immutable teacher caches.

This is a schema migration, not a KD rerun: student metrics and run identities
remain unchanged.  It replaces the ambiguous support-mass column and adds the
effective-teacher diagnostic for already completed arms.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from article1.distillation import build_target


def backfill(results_path: Path, source_root: Path) -> int:
    path = Path(results_path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    caches: dict[tuple[str, int, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for row in rows:
        key = row["dataset"], int(row["seed"]), row["regime"]
        if key not in caches:
            cache_path = Path(source_root) / f"{key[0]}-seed{key[1]}-{key[2]}" / "teacher_cache.npz"
            with np.load(cache_path, allow_pickle=False) as cache:
                caches[key] = cache["logits"], cache["labels"], cache["M"]
        target = build_target(*caches[key], method=row["method"], temperature=float(row["temperature"]))
        row.pop("mean_outside_support_mass", None)
        for name in ("effective_teachers", "pre_restriction_outside_support_mass"):
            value = target.metrics[name]
            row[name] = "" if value is None else str(value)
    fields = sorted(set().union(*(row.keys() for row in rows)))
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--source-root", type=Path, default=Path("OUTPUTS/article1/sources"))
    args = parser.parse_args()
    print(f"updated {backfill(args.results, args.source_root)} rows")


if __name__ == "__main__": main()
