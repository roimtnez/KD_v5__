#!/usr/bin/env python3
"""Launch the Article-1 grid without maintaining a shell script.

Examples:
    python run_article1_grid.py --device cuda
    python run_article1_grid.py --stage distill --methods expert_logit expert_prob expert_prob_sr
    python run_article1_grid.py --stage distill --temperatures 1 4 --methods expert_prob expert_prob_sr
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from article1 import DATASETS, REGIMES, SEEDS
from article1.distillation import METHODS


def command(*arguments: object) -> list[str]:
    return [sys.executable, "-m", "article1.runner", *map(str, arguments)]


def run(arguments: list[str], *, dry_run: bool) -> None:
    print("+", " ".join(arguments), flush=True)
    if not dry_run:
        subprocess.run(arguments, check=True)


def completed_methods(results: Path) -> set[tuple[str, int, str, str, float]]:
    """Read completed default-grid arms without trusting historical outputs."""
    if not results.is_file():
        return set()
    with results.open(newline="", encoding="utf-8") as handle:
        return {
            (row["dataset"], int(row["seed"]), row["regime"], row["method"], float(row["temperature"]))
            for row in csv.DictReader(handle)
        }


def skip(message: str) -> None:
    print(f"# skip: {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("OUTPUTS/article1"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--device", default="cuda", help="e.g. cuda, cuda:0, or cpu")
    parser.add_argument("--stage", choices=("all", "partition", "teachers", "distill"), default="all")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--regimes", nargs="+", choices=REGIMES, default=list(REGIMES))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--temperatures", nargs="+", type=float, default=[8.0])
    parser.add_argument("--proxy-size", type=int, default=10_000)
    parser.add_argument("--teacher-epochs", type=int, default=50)
    parser.add_argument("--student-epochs", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true", help="print commands without running them")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = args.output_root / "results.csv"
    done = completed_methods(results)
    for dataset in args.datasets:
        for seed in args.seeds:
            for regime in args.regimes:
                key = f"{dataset}-seed{seed}-{regime}"
                partitions = args.output_root / "partitions" / key
                source = args.output_root / "sources" / key
                cache = source / "teacher_cache.npz"

                if args.stage in {"all", "partition"}:
                    if (partitions / "metadata.json").is_file():
                        skip(f"partition {key} already exists")
                    else:
                        run(command(
                            "partition", "--dataset", dataset, "--seed", seed, "--regime", regime,
                            "--data-dir", args.data_dir, "--output", partitions,
                            "--proxy-size", args.proxy_size,
                        ), dry_run=args.dry_run)
                if args.stage in {"all", "teachers"}:
                    if cache.is_file() and (source / "metadata.json").is_file():
                        skip(f"teacher cache {key} already exists")
                    else:
                        run(command(
                            "teachers", "--dataset", dataset, "--seed", seed, "--regime", regime,
                            "--data-dir", args.data_dir, "--partitions", partitions, "--output", source,
                            "--epochs", args.teacher_epochs, "--device", args.device,
                        ), dry_run=args.dry_run)
                if args.stage in {"all", "distill"}:
                    for temperature in args.temperatures:
                        for method in args.methods:
                            identity = (dataset, seed, regime, method, temperature)
                            if identity in done:
                                skip(f"distill {dataset}-seed{seed}-{regime}-{method}-T{temperature:g} already recorded")
                            else:
                                run(command(
                                    "distill", "--dataset", dataset, "--seed", seed, "--method", method,
                                    "--cache", cache, "--data-dir", args.data_dir, "--results", results,
                                    "--temperature", temperature, "--epochs", args.student_epochs, "--device", args.device,
                                ), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
