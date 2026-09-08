"""Create the one-row-per-condition provenance/coverage table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from article1 import DATASETS, PROTOCOL_VERSION, REGIMES, SEEDS, THRESHOLDS
from article1.hashes import array_sha256, artifact_commit, file_sha256
from article1.partitioning import load_partitions


def _array_hash(*arrays: np.ndarray) -> str:
    # Keep the composite hash compatible with existing conditions.csv files.
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(array_sha256(array).encode())
    return digest.hexdigest()


def _directory_hash(path: Path) -> str:
    if not path.is_dir() or not (path / "proxy.npz").is_file():
        raise FileNotFoundError(f"partition files missing: {path}")
    digest = hashlib.sha256()
    for child in sorted(path.glob("*.npz")):
        digest.update(child.name.encode())
        digest.update(file_sha256(child).encode())
    metadata = path / "metadata.json"
    if metadata.is_file():
        digest.update(file_sha256(metadata).encode())
    return digest.hexdigest()


def condition_rows(
    source_root: Path, *, partition_root: Path | None = None
) -> list[dict]:
    partition_root = (
        Path(partition_root)
        if partition_root is not None
        else Path(source_root).parent / "partitions"
    )
    rows: list[dict] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            for regime in REGIMES:
                source = Path(source_root) / f"{dataset}-seed{seed}-{regime}"
                metadata_path, cache_path = (
                    source / "metadata.json",
                    source / "teacher_cache.npz",
                )
                if not metadata_path.is_file() or not cache_path.is_file():
                    raise FileNotFoundError(
                        f"missing source for {dataset}/seed={seed}/{regime}"
                    )
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata.get("protocol") != PROTOCOL_VERSION:
                    raise ValueError(f"incompatible source protocol: {source}")
                if metadata.get("cache_sha256") != file_sha256(cache_path):
                    raise ValueError(f"cache hash mismatch: {source}")
                partition_dir = partition_root / f"{dataset}-seed{seed}-{regime}"
                load_partitions(partition_dir)
                if metadata.get("partition_metadata_sha256") != file_sha256(
                    partition_dir / "metadata.json"
                ):
                    raise ValueError(f"cache/partition provenance mismatch: {source}")
                with np.load(cache_path, allow_pickle=False) as cache:
                    M = cache["M"].astype(np.uint8)
                    counts, accuracy = (
                        cache["expertise_counts"],
                        cache["expertise_accuracy"],
                    )
                    proxy = cache["proxy_idx"]
                per_class = M.sum(axis=0)
                rows.append(
                    {
                        "dataset": dataset,
                        "regime": regime,
                        "seed": seed,
                        "expertise_threshold": THRESHOLDS[dataset],
                        "M_density": float(M.mean()),
                        "experts_per_class_min": int(per_class.min()),
                        "experts_per_class_mean": float(per_class.mean()),
                        "experts_per_class_max": int(per_class.max()),
                        "classes_without_expert": json.dumps(
                            np.flatnonzero(per_class == 0).astype(int).tolist()
                        ),
                        "partition_sha256": _directory_hash(partition_dir),
                        "expertise_sha256": _array_hash(counts, accuracy),
                        "proxy_sha256": array_sha256(proxy),
                        "teachers_sha256": metadata["teacher_fingerprint"],
                        "protocol_version": metadata.get(
                            "protocol_version",
                            metadata.get("protocol", PROTOCOL_VERSION),
                        ),
                        "artifact_creation_commit": artifact_commit(metadata),
                    }
                )
    return rows


def write_table(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path, default=Path("OUTPUTS/article1_v3/sources")
    )
    parser.add_argument(
        "--partition-root", type=Path, help="default: partitions/ next to sources/"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("OUTPUTS/article1_v3/conditions.csv")
    )
    args = parser.parse_args()
    rows = condition_rows(args.source_root, partition_root=args.partition_root)
    write_table(args.output, rows)
    print(f"wrote {len(rows)} conditions to {args.output}")


if __name__ == "__main__":
    main()
