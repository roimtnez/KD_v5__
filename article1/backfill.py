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


def backfill(results_path: Path, source_root: Path, *, dry_run: bool = False) -> dict:
    path = Path(results_path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    caches: dict[tuple[str, int, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    changed_rows = 0
    for row in rows:
        key = row["dataset"], int(row["seed"]), row["regime"]
        if key not in caches:
            cache_path = Path(source_root) / f"{key[0]}-seed{key[1]}-{key[2]}" / "teacher_cache.npz"
            with np.load(cache_path, allow_pickle=False) as cache:
                caches[key] = cache["logits"], cache["labels"], cache["M"]
        target = build_target(*caches[key], method=row["method"], temperature=float(row["temperature"]))
        previous = (row.get("mean_outside_support_mass"), row.get("effective_teachers"), row.get("pre_restriction_outside_support_mass"))
        row.pop("mean_outside_support_mass", None)
        for name in ("effective_teachers", "pre_restriction_outside_support_mass"):
            value = target.metrics[name]
            row[name] = "" if value is None else str(value)
        if previous != (None, row.get("effective_teachers"), row.get("pre_restriction_outside_support_mass")):
            changed_rows += 1
    fields = sorted(set().union(*(row.keys() for row in rows)))
    report = {"rows": len(rows), "changed_rows": changed_rows, "removed_columns": ["mean_outside_support_mass"],
              "derived_columns": ["pre_restriction_outside_support_mass", "effective_teachers"], "dry_run": dry_run}
    if dry_run:
        return report
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--source-root", type=Path, default=Path("OUTPUTS/article1/sources"))
    parser.add_argument("--dry-run", action="store_true", help="report derived-metric changes without writing")
    args = parser.parse_args()
    print(json.dumps(backfill(args.results, args.source_root, dry_run=args.dry_run), sort_keys=True))


if __name__ == "__main__": main()
