"""Run the fixed Article-1 v3 workflow. Default: print a plan, do not execute.

Use --execute to run; --partitions-only stops before all model training.
The optional proxy curve is enabled only with --with-proxy-curve.
Run with the repository's Python environment (requirements + requirements-dev).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

from article1 import DATASETS, PROTOCOL_VERSION, REGIMES, SEEDS
from article1.partitioning import ROLES, load_partitions, make_partitions

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "OUTPUTS" / "article1_v3"


def check_partition(path, labels, *, dataset, seed, regime):
    """Check stored indices against a fresh deterministic construction."""
    proxy, clients, meta = load_partitions(path, labels)
    if (meta.get("dataset"), meta.get("seed"), meta.get("regime")) != (
        dataset,
        seed,
        regime,
    ):
        raise ValueError(f"partition identity mismatch: {path}")
    expected_proxy, expected_clients = make_partitions(labels, regime=regime, seed=seed)
    np.testing.assert_array_equal(proxy, expected_proxy)
    np.testing.assert_array_equal(
        np.bincount(labels[proxy], minlength=10), np.full(10, 1000)
    )
    for actual, expected in zip(clients, expected_clients):
        for role in ROLES:
            np.testing.assert_array_equal(
                actual[role + "_idx"], expected[role + "_idx"]
            )
    return {
        "dataset": dataset,
        "seed": seed,
        "regime": regime,
        **{
            role + "_min": min(len(c[role + "_idx"]) for c in clients) for role in ROLES
        },
    }


def check_all_partitions():
    from torchvision import datasets

    constructors = dict(
        zip(DATASETS, (datasets.MNIST, datasets.FashionMNIST, datasets.CIFAR10))
    )
    rows = []
    for dataset in DATASETS:
        labels = np.asarray(
            constructors[dataset](
                str(ROOT / "data"), train=True, download=True
            ).targets,
            dtype=np.int64,
        )
        for seed in SEEDS:
            for regime in REGIMES:
                path = OUT / "partitions" / f"{dataset}-seed{seed}-{regime}"
                rows.append(
                    check_partition(
                        path, labels, dataset=dataset, seed=seed, regime=regime
                    )
                )
    (OUT / "partition_check.json").write_text(
        json.dumps(
            {"protocol": PROTOCOL_VERSION, "ok": True, "conditions": rows}, indent=2
        )
        + "\n"
    )
    print(
        "Verified 54 partitions: exact reconstruction, balance, coverage and disjoint splits.",
        flush=True,
    )


def execute_notebook(name, *, dataset=None, seed=None):
    import nbformat
    from nbclient import NotebookClient

    notebook = nbformat.read(ROOT / "notebooks" / name, as_version=4)
    if dataset is not None:
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.source = re.sub(
                    r"^DATASET = .*$",
                    f"DATASET = {dataset!r}",
                    cell.source,
                    flags=re.MULTILINE,
                )
                cell.source = re.sub(
                    r"^SEED = .*$", f"SEED = {seed}", cell.source, flags=re.MULTILINE
                )
    # A real kernel is required; no silent fallback that could hide execution failures.
    NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()
    folder = OUT / "notebooks"
    folder.mkdir(parents=True, exist_ok=True)
    filename = f"{dataset}-seed{seed}-{name}" if dataset is not None else name
    nbformat.write(notebook, folder / filename)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute", action="store_true", help="actually run the printed workflow"
    )
    parser.add_argument(
        "--partitions-only",
        action="store_true",
        help="create/check partitions and their plots, no training",
    )
    parser.add_argument(
        "--with-proxy-curve",
        action="store_true",
        help="also run 48 smaller-proxy CIFAR cells",
    )
    parser.add_argument(
        "--skip-notebooks",
        action="store_true",
        help="leave notebook execution to the user",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    os.chdir(ROOT)
    if PROTOCOL_VERSION != "article1-v3":
        raise ValueError("this workflow is frozen for article1-v3")

    def run(*arguments):
        cmd = [sys.executable, *map(str, arguments)]
        print("+ " + shlex.join(cmd), flush=True)
        if args.execute:
            subprocess.run(cmd, check=True, cwd=ROOT)

    def grid(*arguments):
        run(
            "run_article1_grid.py",
            "--output-root",
            OUT,
            "--device",
            args.device,
            *arguments,
        )

    def runner(*arguments):
        run("-m", "article1.runner", *arguments, "--device", args.device)

    def note(text):
        print("\n" + text, flush=True)

    note("1. Quick checks, then create all 54 partitions.")
    run(
        "-m",
        "compileall",
        "-q",
        "article1",
        "run_article1_grid.py",
        "run_article1_pipeline.py",
    )
    run("-m", "pytest", "-q")
    grid("--stage", "partition")
    note("2. Verify every partition before training.")
    if args.execute:
        check_all_partitions()
        if args.device.startswith("cuda") and not args.partitions_only:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA requested but unavailable; no training started"
                )
    if not args.skip_notebooks:
        note(
            "3. Execute partition notebook for all three datasets and seeds; export PNG/PDF."
        )
        if args.execute:
            for dataset in DATASETS:
                for seed in SEEDS:
                    execute_notebook(
                        "article1_partition_diagnostics.ipynb",
                        dataset=dataset,
                        seed=seed,
                    )
    if args.partitions_only:
        return

    pilot = OUT / "sources" / "mnist-seed42-iid" / "teacher_cache.npz"
    note(
        "4. Pilot: ten MNIST-IID teachers, three main methods, then exact KD repetition."
    )
    grid(
        "--stage",
        "all",
        "--datasets",
        "mnist",
        "--seeds",
        42,
        "--regimes",
        "iid",
        "--methods",
        "feddf_logit",
        "expert_logit",
        "oracle_logit",
    )
    run(
        "-m",
        "article1.reproduce",
        "--dataset",
        "mnist",
        "--seed",
        42,
        "--method",
        "expert_logit",
        "--cache",
        pilot,
        "--device",
        args.device,
        "--results",
        OUT / "reproducibility_check.csv",
    )
    note(
        "5. Main v3 grid: reuse pilot teachers and main-grid cells; 486 T=8 results in total."
    )
    grid("--stage", "teachers")
    grid("--stage", "distill")
    run("-m", "article1.conditions")
    run("-m", "article1.audit", OUT / "results.csv")
    note("6. Focused temperature contrast: 36 new T=1/4 cells; reuse T=8 for analysis.")
    grid(
        "--stage",
        "distill",
        "--datasets",
        "cifar",
        "--regimes",
        "iid",
        "alpha0p1",
        "single",
        "--methods",
        "expert_logit",
        "expert_prob",
        "--temperatures",
        1,
        4,
        "--results",
        OUT / "results_rq2_temperature.csv",
    )
    run(
        "-m",
        "article1.rq2",
        "--results",
        OUT / "results.csv",
        OUT / "results_rq2_temperature.csv",
        "--source-root",
        OUT / "sources",
        "--temperatures",
        1,
        4,
        8,
        "--isolated-results",
        OUT / "results_rq2_temperature.csv",
    )
    note("7. Nine supervised full-proxy cells, reusable across regimes.")
    for dataset in DATASETS:
        for seed in SEEDS:
            runner(
                "supervised",
                "--dataset",
                dataset,
                "--seed",
                seed,
                "--cache",
                OUT / "sources" / f"{dataset}-seed{seed}-iid" / "teacher_cache.npz",
                "--results",
                OUT / "results_supervised_proxy.csv",
                "--updates",
                1200,
                "--skip-existing",
            )
    if args.with_proxy_curve:
        note(
            "8. Optional curve: CIFAR N=100/500/1000/5000, three seeds, 12 CE + 36 EXPERT cells."
        )
        for seed in SEEDS:
            for size in (100, 500, 1000, 5000):
                common = [
                    "--dataset",
                    "cifar",
                    "--seed",
                    seed,
                    "--proxy-size",
                    size,
                    "--updates",
                    1200,
                    "--results",
                    OUT / "results_proxy_size.csv",
                    "--skip-existing",
                ]
                runner(
                    "supervised",
                    "--cache",
                    OUT / "sources" / f"cifar-seed{seed}-iid" / "teacher_cache.npz",
                    *common,
                )
                for regime in ("iid", "alpha0p1", "single"):
                    runner(
                        "distill",
                        "--method",
                        "expert_logit",
                        "--temperature",
                        8,
                        "--cache",
                        OUT
                        / "sources"
                        / f"cifar-seed{seed}-{regime}"
                        / "teacher_cache.npz",
                        *common,
                    )
    if not args.skip_notebooks:
        note(
            "9. Execute definitive results notebook; supervised/size-curve analysis remains a separate research checkpoint."
        )
        if args.execute:
            execute_notebook("article1_definitive_analysis.ipynb")
    note(
        "Workflow complete."
        if args.execute
        else "PLAN ONLY. Add --execute to run; --partitions-only stops before training."
    )


if __name__ == "__main__":
    main()
