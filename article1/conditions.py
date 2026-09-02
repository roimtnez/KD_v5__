"""Create the one-row-per-condition provenance/coverage table."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from article1 import DATASETS, REGIMES, SEEDS, THRESHOLDS
from article1 import PROTOCOL_VERSION
from article1.hashes import array_sha256, artifact_commit, file_sha256


def _array_hash(*arrays: np.ndarray) -> str:
    # Compatibility shim for the condition-table's composite holdout hash.
    import hashlib
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(array_sha256(array).encode())
    return digest.hexdigest()


def _directory_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(path.glob("*.npz")):
        digest.update(child.name.encode()); digest.update(file_sha256(child).encode())
    metadata = path / "metadata.json"
    if metadata.is_file(): digest.update(file_sha256(metadata).encode())
    return digest.hexdigest()


def condition_rows(source_root: Path, *, git_root: Path = Path(".")) -> list[dict]:
    rows: list[dict] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            for regime in REGIMES:
                source = Path(source_root) / f"{dataset}-seed{seed}-{regime}"
                metadata_path, cache_path = source / "metadata.json", source / "teacher_cache.npz"
                if not metadata_path.is_file() or not cache_path.is_file():
                    raise FileNotFoundError(f"missing source for {dataset}/seed={seed}/{regime}")
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                partition_dir = Path(metadata["proxy_source"]).parent
                with np.load(cache_path, allow_pickle=False) as cache:
                    M = cache["M"].astype(np.uint8)
                    counts, accuracy = cache["holdout_counts"], cache["holdout_accuracy"]
                    proxy = cache["proxy_idx"]
                per_class = M.sum(axis=0)
                rows.append({
                    "dataset": dataset, "regime": regime, "seed": seed,
                    "expertise_threshold": THRESHOLDS[dataset],
                    "M_density": float(M.mean()),
                    "experts_per_class_min": int(per_class.min()),
                    "experts_per_class_mean": float(per_class.mean()),
                    "experts_per_class_max": int(per_class.max()),
                    "classes_without_expert": json.dumps(np.flatnonzero(per_class == 0).astype(int).tolist()),
                    "partition_sha256": _directory_hash(partition_dir),
                    "holdout_sha256": _array_hash(counts, accuracy),
                    "proxy_sha256": array_sha256(proxy),
                    "teachers_sha256": metadata["teacher_fingerprint"],
                    "protocol_version": metadata.get("protocol_version", metadata.get("protocol", PROTOCOL_VERSION)),
                    "artifact_creation_commit": artifact_commit(metadata),
                })
    return rows


def write_table(path: Path, rows: list[dict]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("OUTPUTS/article1/sources"))
    parser.add_argument("--output", type=Path, default=Path("OUTPUTS/article1/conditions.csv"))
    args = parser.parse_args()
    rows = condition_rows(args.source_root)
    write_table(args.output, rows)
    print(f"wrote {len(rows)} conditions to {args.output}")


if __name__ == "__main__": main()
