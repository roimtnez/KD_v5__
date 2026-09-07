#!/usr/bin/env python3
"""Explicit partition/teacher/KD stages; reuse only identical cached KD recipes."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from article1 import DATASETS, REGIMES, SEEDS
from article1.distillation import METHODS, kd_config, metadata_identity
from article1.hashes import array_sha256, file_sha256


def command(*arguments: object) -> list[str]:
    return [sys.executable, "-m", "article1.runner", *map(str, arguments)]


def run(arguments: list[str], *, dry_run: bool) -> None:
    print("+", " ".join(arguments), flush=True)
    if not dry_run:
        subprocess.run(arguments, check=True)


def completed_run_ids(paths: list[Path]) -> set[str]:
    """Collect recorded executions; ambiguity is an error, not a silent skip."""
    found: set[str] = set()
    for path in dict.fromkeys(p.resolve() for p in paths):
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                run_id = row.get("run_id")
                if not run_id or run_id in found:
                    raise ValueError(f"missing or duplicate run_id in {path}: {run_id}")
                found.add(run_id)
    return found


def cache_identity(cache: Path) -> dict:
    """Fingerprint the actual cache, not just its dataset/regime filename."""
    metadata = json.loads(cache.with_name("metadata.json").read_text())
    digest = file_sha256(cache)
    if metadata.get("cache_sha256") != digest:
        raise ValueError(f"cache hash differs from metadata: {cache}")
    with np.load(cache, allow_pickle=False) as data:
        return dict(
            source_hash=digest,
            proxy_hash=array_sha256(data["proxy_idx"]),
            mask_hash=array_sha256(data["M"]),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("OUTPUTS/article1"))
    parser.add_argument(
        "--results",
        type=Path,
        help="default: <output-root>/results.csv; T!=8 requires another file",
    )
    parser.add_argument(
        "--reuse-results",
        type=Path,
        nargs="*",
        default=[],
        help="additional CSVs with completed run IDs",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--stage", choices=("all", "partition", "teachers", "distill"), default="all"
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=list(DATASETS)
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS)
    )
    parser.add_argument("--regimes", nargs="+", choices=REGIMES, default=list(REGIMES))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--temperatures", nargs="+", type=float, default=[8.0])
    parser.add_argument(
        "--proxy-size",
        type=int,
        default=10_000,
        help="partition stage only; NOT a proxy-size KD ablation",
    )
    parser.add_argument("--teacher-epochs", type=int, default=50)
    parser.add_argument("--student-epochs", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if any(not np.isfinite(t) or t <= 0 for t in args.temperatures):
        parser.error("temperatures must be positive and finite")
    if min(args.teacher_epochs, args.student_epochs) <= 0:
        parser.error("epoch budgets must be positive")
    main_results = args.output_root / "results.csv"
    results = args.results or main_results
    if (
        any(t != 8 for t in args.temperatures)
        and results.resolve() == main_results.resolve()
    ):
        parser.error("T!=8 requires --results pointing to a separate CSV")
    done = completed_run_ids([results, *args.reuse_results])
    for dataset in args.datasets:
        for seed in args.seeds:
            for regime in args.regimes:
                key = f"{dataset}-seed{seed}-{regime}"
                partitions = args.output_root / "partitions" / key
                source = args.output_root / "sources" / key
                cache = source / "teacher_cache.npz"
                if (
                    args.stage in {"all", "partition"}
                    and not (partitions / "metadata.json").is_file()
                ):
                    run(
                        command(
                            "partition",
                            "--dataset",
                            dataset,
                            "--seed",
                            seed,
                            "--regime",
                            regime,
                            "--data-dir",
                            args.data_dir,
                            "--output",
                            partitions,
                            "--proxy-size",
                            args.proxy_size,
                        ),
                        dry_run=args.dry_run,
                    )
                if args.stage in {"all", "teachers"} and not (
                    cache.is_file() and cache.with_name("metadata.json").is_file()
                ):
                    run(
                        command(
                            "teachers",
                            "--dataset",
                            dataset,
                            "--seed",
                            seed,
                            "--regime",
                            regime,
                            "--data-dir",
                            args.data_dir,
                            "--partitions",
                            partitions,
                            "--output",
                            source,
                            "--epochs",
                            args.teacher_epochs,
                            "--device",
                            args.device,
                        ),
                        dry_run=args.dry_run,
                    )
                if args.stage not in {"all", "distill"}:
                    continue
                fingerprint = cache_identity(cache) if cache.is_file() else None
                for temperature in args.temperatures:
                    for method in args.methods:
                        run_id = (
                            metadata_identity(
                                method=method,
                                temperature=temperature,
                                config=kd_config(args.student_epochs),
                                **fingerprint,
                            )
                            if fingerprint
                            else None
                        )
                        if run_id in done:
                            print(
                                f"# skip identical recipe: {key}/{method}/T={temperature:g}",
                                flush=True,
                            )
                            continue
                        run(
                            command(
                                "distill",
                                "--dataset",
                                dataset,
                                "--seed",
                                seed,
                                "--method",
                                method,
                                "--cache",
                                cache,
                                "--data-dir",
                                args.data_dir,
                                "--results",
                                results,
                                "--temperature",
                                temperature,
                                "--epochs",
                                args.student_epochs,
                                "--device",
                                args.device,
                            ),
                            dry_run=args.dry_run,
                        )


if __name__ == "__main__":
    main()
