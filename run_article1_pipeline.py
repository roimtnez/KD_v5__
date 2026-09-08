"""Sequential Article-1 v3 checkpoints. Plan by default; execute one explicit phase.

Teachers and RQ1 precede separate aggregation, support and supervised studies.
The pipeline never advances to the next scientific checkpoint automatically.
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
from article1.distillation import kd_config, metadata_identity
from article1.experiments import (
    BASELINE_METHODS,
    FOCAL_REGIMES,
    OPTIONAL_PHASES,
    PHASES,
    T8_BLOCKS,
)
from article1.hashes import file_sha256
from article1.partitioning import ROLES, load_partitions, make_partitions
from run_article1_grid import cache_identity

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


def execute_notebook(name, *, dataset=None, seed=None, stage=None):
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
    if stage is not None:
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.source = re.sub(
                    r"^STAGE = .*$",
                    f"STAGE = {stage!r}",
                    cell.source,
                    flags=re.MULTILINE,
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
    filename = (
        f"{dataset}-seed{seed}-{name}" if dataset is not None else f"{stage}-{name}"
    )
    nbformat.write(notebook, folder / filename)


PILOT = ("mnist", 42, "alpha0p1")
PILOT_METHODS = ("feddf_logit", "oracle_logit")


def check_pilot():
    """A reproduction report is a technical prerequisite, not scientific approval."""
    cache = OUT / "sources" / "mnist-seed42-alpha0p1" / "teacher_cache.npz"
    report = json.loads(
        (OUT / "pilot_selection_reproducibility.report.json").read_text()
    )
    expected = metadata_identity(
        method="oracle_logit",
        temperature=8.0,
        config=kd_config(),
        **cache_identity(cache),
    )
    if (
        report.get("reproducible") is not True
        or report.get("result", {}).get("run_id") != expected
    ):
        raise ValueError(
            "pilot report does not match the current cache/recipe; run --phase pilot"
        )


def check_sources():
    """Verify all teacher conditions and their link to the current partitions."""
    for dataset in DATASETS:
        for seed in SEEDS:
            for regime in REGIMES:
                key = f"{dataset}-seed{seed}-{regime}"
                partitions = OUT / "partitions" / key
                load_partitions(partitions)
                source = OUT / "sources" / key
                metadata = json.loads((source / "metadata.json").read_text())
                if (
                    metadata.get("dataset"),
                    metadata.get("seed"),
                    metadata.get("regime"),
                ) != (dataset, seed, regime):
                    raise ValueError(f"source identity mismatch: {source}")
                if metadata.get("partition_metadata_sha256") != file_sha256(
                    partitions / "metadata.json"
                ):
                    raise ValueError(f"source/partition mismatch: {source}")
                cache_identity(source / "teacher_cache.npz")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=PHASES + OPTIONAL_PHASES,
        help="one checkpoint only; omission prints the sequence",
    )
    parser.add_argument(
        "--execute", action="store_true", help="execute the selected phase, then stop"
    )
    parser.add_argument("--skip-notebooks", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    os.chdir(ROOT)
    if not args.phase:
        if args.execute:
            parser.error(
                "--execute requires an explicit --phase; there is no automatic all-phase run"
            )
        for number, phase in enumerate(PHASES, 1):
            print(
                f"{number}. python run_article1_pipeline.py --phase {phase} --execute"
            )
        print("Optional phases: " + ", ".join(OPTIONAL_PHASES))
        print("Review each checkpoint before choosing the next. No commands executed.")
        return
    if PROTOCOL_VERSION != "article1-v3":
        raise ValueError("this workflow is frozen for article1-v3")
    phase = args.phase

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

    def audit_block(block, *, pilot=False):
        filename, methods = T8_BLOCKS[block]
        extra = (
            ["--datasets", "mnist", "--seeds", 42, "--regimes", "alpha0p1"]
            if pilot
            else []
        )
        run(
            "-m",
            "article1.audit",
            OUT / filename,
            "--subset",
            "--methods",
            *(PILOT_METHODS if pilot else methods),
            *extra,
        )

    def notebook(stage):
        if not args.skip_notebooks:
            print(f"Notebook analysis checkpoint: {stage}", flush=True)
            if args.execute:
                execute_notebook("article1_definitive_analysis.ipynb", stage=stage)

    print(
        f"Checkpoint: {phase}. No subsequent phase will run automatically.", flush=True
    )
    if args.execute and phase != "partitions" and args.device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
    if phase == "partitions":
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
        if args.execute:
            check_all_partitions()
            if not args.skip_notebooks:
                for dataset in DATASETS:
                    for seed in SEEDS:
                        execute_notebook(
                            "article1_partition_diagnostics.ipynb",
                            dataset=dataset,
                            seed=seed,
                        )
    elif phase == "pilot":
        if args.execute:
            check_all_partitions()
        grid(
            "--stage",
            "teachers",
            "--datasets",
            "mnist",
            "--seeds",
            42,
            "--regimes",
            "alpha0p1",
        )
        # These two runs are part of RQ1. No probability/SR student runs here.
        grid(
            "--stage",
            "distill",
            "--datasets",
            "mnist",
            "--seeds",
            42,
            "--regimes",
            "alpha0p1",
            "--methods",
            *PILOT_METHODS,
            "--results",
            OUT / T8_BLOCKS["rq1"][0],
        )
        # The audit reconstructs every target variant, without training those students.
        audit_block("rq1", pilot=True)
        run(
            "-m",
            "article1.reproduce",
            "--dataset",
            "mnist",
            "--seed",
            42,
            "--method",
            "oracle_logit",
            "--cache",
            OUT / "sources" / "mnist-seed42-alpha0p1" / "teacher_cache.npz",
            "--device",
            args.device,
            "--results",
            OUT / "pilot_selection_reproducibility.csv",
            "--report",
            OUT / "pilot_selection_reproducibility.report.json",
        )
    elif phase == "teachers":
        if args.execute:
            check_pilot()
            check_all_partitions()
        grid("--stage", "teachers")
        if args.execute:
            check_sources()
        run("-m", "article1.conditions")
    else:
        if args.execute:
            check_sources()
        if phase == "baseline":
            # Keep this ordering explicit: full baseline runs are long, and
            # probability-space methods must be executed before logit-space ones.
            grid(
                "--stage",
                "distill",
                "--methods",
                *BASELINE_METHODS,
                "--results",
                OUT / "results_baseline.csv",
            )
            run(
                "-m",
                "article1.audit",
                OUT / "results_baseline.csv",
                "--methods",
                *BASELINE_METHODS,
            )
            notebook(phase)
        elif phase in T8_BLOCKS:
            if phase == "rq1":
                if args.execute:
                    check_pilot()
                run("-m", "article1.conditions")
            else:
                audit_block("rq1")
            if phase in {"expertise", "support"}:
                audit_block("aggregation")
            if phase == "support":
                audit_block("expertise")
            filename, methods = T8_BLOCKS[phase]
            grid(
                "--stage", "distill", "--methods", *methods, "--results", OUT / filename
            )
            audit_block(phase)
            notebook(phase)
        elif phase in {"expert-logit", "temperature"}:
            audit_block("expertise")
            focal = ["--datasets", "cifar", "--regimes", *FOCAL_REGIMES]
            diagnostic = OUT / "results_expert_logit_focal.csv"
            grid(
                "--stage",
                "distill",
                *focal,
                "--methods",
                "expert_logit",
                "--results",
                diagnostic,
            )
            run("-m", "article1.audit", diagnostic, "--methods", "expert_logit", *focal)
            temperatures = [8]
            files = [OUT / T8_BLOCKS["expertise"][0], diagnostic]
            extra = OUT / "results_expert_temperature.csv"
            if phase == "temperature":
                grid(
                    "--stage",
                    "distill",
                    *focal,
                    "--methods",
                    "expert_logit",
                    "expert_prob",
                    "--temperatures",
                    1,
                    4,
                    "--results",
                    extra,
                )
                files.append(extra)
                temperatures = [1, 4, 8]
            arguments = [
                "-m",
                "article1.rq2",
                "--results",
                *files,
                "--source-root",
                OUT / "sources",
                "--temperatures",
                *temperatures,
            ]
            if phase == "temperature":
                arguments += ["--isolated-results", extra]
            run(*arguments)
            if args.execute:
                from article1.analysis import (
                    export,
                    focal_comparisons,
                    plot_temperature,
                    summarize,
                )

                effect = focal_comparisons(OUT, temperatures=temperatures)
                export(
                    OUT / phase,
                    {
                        "paired": effect,
                        "summary": summarize(effect, groups=["regime", "temperature"]),
                    },
                    {"operator_effect": plot_temperature(effect)},
                )
        elif phase == "supervised":
            audit_block("rq1")
            for dataset in DATASETS:
                for seed in SEEDS:
                    runner(
                        "supervised",
                        "--dataset",
                        dataset,
                        "--seed",
                        seed,
                        "--cache",
                        OUT
                        / "sources"
                        / f"{dataset}-seed{seed}-iid"
                        / "teacher_cache.npz",
                        "--results",
                        OUT / "results_supervised_proxy.csv",
                        "--updates",
                        1200,
                        "--skip-existing",
                    )
        elif phase == "proxy-curve":
            audit_block("expertise")
            # Complete/reuse the three CIFAR full-proxy references before the smaller subsets.
            for seed in SEEDS:
                runner(
                    "supervised",
                    "--dataset",
                    "cifar",
                    "--seed",
                    seed,
                    "--cache",
                    OUT / "sources" / f"cifar-seed{seed}-iid" / "teacher_cache.npz",
                    "--results",
                    OUT / "results_supervised_proxy.csv",
                    "--updates",
                    1200,
                    "--skip-existing",
                )
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
                        OUT / "results_proxy_size_expert_prob.csv",
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
                            "expert_prob",
                            "--temperature",
                            8,
                            "--cache",
                            OUT
                            / "sources"
                            / f"cifar-seed{seed}-{regime}"
                            / "teacher_cache.npz",
                            *common,
                        )
    print(
        "Checkpoint finished. Review its evidence before choosing the next phase."
        if args.execute
        else "PLAN ONLY. Add --execute to run this phase."
    )


if __name__ == "__main__":
    main()
