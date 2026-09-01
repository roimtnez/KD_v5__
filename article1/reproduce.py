"""Repeat one complete KD cell and require bitwise-identical outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

from article1 import DATASETS, SEEDS
from article1.distillation import METHODS
from article1.runner import distill


def reproduce(**kwargs) -> dict:
    """Run the same KD cell twice; results upsert to the same immutable run id."""
    first = distill(**kwargs)
    second = distill(**kwargs)
    fields = ("run_id", "student_final_sha256", "student_init_sha256", "batch_order_sha256",
              "student_test_accuracy", "student_test_nll", "target_accuracy", "target_nll", "target_entropy")
    differences = {field: (first[field], second[field]) for field in fields if first[field] != second[field]}
    if differences:
        raise RuntimeError(f"KD cell is not exactly reproducible: {differences}")
    return {field: first[field] for field in fields}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--results", type=Path, default=Path("OUTPUTS/article1/results.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--temperature", type=float, default=8.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = reproduce(cache=args.cache, data_dir=args.data_dir, results=args.results, method=args.method,
                       dataset=args.dataset, seed=args.seed, temperature=args.temperature, epochs=args.epochs,
                       device=args.device)
    print(result)


if __name__ == "__main__": main()
