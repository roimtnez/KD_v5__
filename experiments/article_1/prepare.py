#!/usr/bin/env python3
"""Build clean Article-1 teacher sources from retained historical checkpoints.

Only teacher weights and their original client split indices are reused. All
holdout/test measurements and proxy logits are recomputed into the new output
tree. Historical masks, targets, students and proxy-analysis arrays are never
read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data.dataset_config import get_dataset_config
from experiments.article_1 import OUTPUT_ROOT, PROTOCOL_VERSION
from experiments.article_1.io import sha256_array, sha256_file, write_json, write_npz
from experiments.article_1.protocol import DATASET_LABEL, DATASETS, REGIMES, SEEDS, source_dir
from experiments.article_1.targets import nested_balanced_order
from oracle_distillation.checkpoints import load_checkpoint_state
from oracle_distillation.models import build_model
from oracle_distillation.utils import collect_logits, resolve_device


def _legacy_condition(
    legacy_root: Path, dataset: str, seed: int, regime: str,
) -> tuple[Path, Path]:
    label = DATASET_LABEL[dataset]
    work = Path(legacy_root) / f"seed_{seed}" / "raw_work"
    runs = sorted((work / "runs" / label / "K10").glob(f"{regime}__*"))
    if len(runs) != 1:
        raise FileNotFoundError(
            f"expected one retained teacher run for {dataset}/seed_{seed}/{regime}, found {runs}"
        )
    run = runs[0]
    partition = work / "partitions" / label / "K10" / run.name
    if not partition.is_dir():
        raise FileNotFoundError(f"missing original split indices: {partition}")
    return run, partition


def _client_splits(partition: Path) -> list[dict[str, np.ndarray]]:
    files = sorted((partition / "clients").glob("c*.npz"))
    if not files:
        raise FileNotFoundError(f"no client splits below {partition}")
    clients: list[dict[str, np.ndarray]] = []
    all_indices: list[np.ndarray] = []
    for expected_cid, path in enumerate(files):
        if path.stem != f"c{expected_cid:03d}":
            raise ValueError(f"client files are not contiguous at {path}")
        with np.load(path, allow_pickle=False) as payload:
            required = {"train_idx", "holdout_idx", "local_test_idx"}
            if not required.issubset(payload.files):
                raise ValueError(f"{path} does not contain train/holdout/test")
            split = {
                "train_idx": np.asarray(payload["train_idx"], dtype=np.int64),
                "holdout_idx": np.asarray(payload["holdout_idx"], dtype=np.int64),
                "test_idx": np.asarray(payload["local_test_idx"], dtype=np.int64),
            }
        values = list(split.values())
        for value in values:
            if len(np.unique(value)) != len(value):
                raise ValueError(f"duplicated indices in {path}")
        if any(np.intersect1d(values[i], values[j]).size for i in range(3) for j in range(i + 1, 3)):
            raise ValueError(f"train/holdout/test overlap in {path}")
        clients.append(split)
        all_indices.extend(values)
    concatenated = np.concatenate(all_indices)
    if len(np.unique(concatenated)) != len(concatenated):
        raise ValueError(f"examples overlap between clients below {partition}")
    return clients


def _per_class(logits: np.ndarray, labels: np.ndarray, classes: int) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(logits).argmax(axis=1)
    counts = np.bincount(labels, minlength=classes).astype(np.int64)
    accuracy = np.full(classes, np.nan, dtype=np.float32)
    for cls in np.flatnonzero(counts):
        rows = labels == cls
        accuracy[cls] = float((predictions[rows] == cls).mean())
    return accuracy, counts


def _fingerprint(hashes: list[str]) -> str:
    digest = hashlib.sha256()
    for value in hashes:
        digest.update(value.encode("ascii"))
    return digest.hexdigest()


def audit_condition(
    *, dataset: str, seed: int, regime: str, legacy_root: Path, data_dir: Path,
) -> dict:
    """Read-only preflight: splits, checkpoints and proxy separation."""
    run, partition = _legacy_condition(legacy_root, dataset, seed, regime)
    clients = _client_splits(partition)
    checkpoints = [run / "teachers" / f"cid_{cid:03d}.pt" for cid in range(len(clients))]
    missing = [str(path) for path in checkpoints if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing retained teacher checkpoints: {missing}")
    proxy_path = get_dataset_config(dataset).proxy_path_for_seed(seed, data_dir)
    with np.load(proxy_path, allow_pickle=False) as payload:
        proxy_idx = np.asarray(payload["proxy_idx"], dtype=np.int64)
    allocated = np.concatenate([value for split in clients for value in split.values()])
    overlap = int(np.intersect1d(proxy_idx, allocated).size)
    if overlap:
        raise ValueError(f"proxy/client overlap in {dataset}/seed_{seed}/{regime}: {overlap}")
    return {
        "dataset": dataset, "seed": int(seed), "regime": regime,
        "clients": len(clients), "checkpoints": len(checkpoints),
        "proxy_examples": len(proxy_idx), "proxy_client_overlap": overlap,
        "run": str(run), "partition": str(partition),
    }


def prepare_condition(
    *,
    dataset: str,
    seed: int,
    regime: str,
    legacy_root: Path,
    output_root: Path,
    data_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    force: bool,
) -> Path:
    destination = source_dir(output_root, dataset, seed, regime)
    source_npz = destination / "teacher_source.npz"
    source_json = destination / "source.json"
    if source_npz.is_file() and source_json.is_file() and not force:
        print(f"[SKIP] {dataset} seed={seed} regime={regime}: {source_npz}")
        return source_npz

    run, partition = _legacy_condition(legacy_root, dataset, seed, regime)
    clients = _client_splits(partition)
    checkpoints = [run / "teachers" / f"cid_{cid:03d}.pt" for cid in range(len(clients))]
    missing = [str(path) for path in checkpoints if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing retained teacher checkpoints: {missing}")

    config = get_dataset_config(dataset)
    eval_dataset = config.load_train_eval_dataset(data_dir)
    proxy_path = config.proxy_path_for_seed(seed, data_dir)
    if not proxy_path.is_file():
        raise FileNotFoundError(f"missing seed-specific proxy split: {proxy_path}")
    with np.load(proxy_path, allow_pickle=False) as proxy:
        proxy_idx = np.asarray(proxy["proxy_idx"], dtype=np.int64)
    allocated = np.concatenate([value for split in clients for value in split.values()])
    if np.intersect1d(proxy_idx, allocated).size:
        raise ValueError("public proxy overlaps a client train/holdout/test split")

    proxy_loader = DataLoader(
        Subset(eval_dataset, proxy_idx.tolist()), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=device.type == "cuda",
    )
    k, c = len(clients), config.num_classes
    proxy_logits = np.empty((len(proxy_idx), k, c), dtype=np.float16)
    proxy_labels: np.ndarray | None = None
    holdout_accuracy = np.full((k, c), np.nan, dtype=np.float32)
    test_accuracy = np.full((k, c), np.nan, dtype=np.float32)
    holdout_counts = np.zeros((k, c), dtype=np.int64)
    test_counts = np.zeros((k, c), dtype=np.int64)
    checkpoint_hashes: list[str] = []

    for cid, (checkpoint, split) in enumerate(zip(checkpoints, clients)):
        model = build_model(config.arch, num_classes=c).to(device)
        model.load_state_dict(load_checkpoint_state(checkpoint, map_location=device), strict=True)
        model.eval()
        for role, accuracy, counts in (
            ("holdout_idx", holdout_accuracy, holdout_counts),
            ("test_idx", test_accuracy, test_counts),
        ):
            indices = split[role]
            loader = DataLoader(
                Subset(eval_dataset, indices.tolist()), batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=device.type == "cuda",
            )
            logits, labels = collect_logits(model, loader, device)
            accuracy[cid], counts[cid] = _per_class(logits, labels, c)
        logits, labels = collect_logits(model, proxy_loader, device)
        proxy_logits[:, cid] = logits.astype(np.float16)
        labels = labels.astype(np.int64)
        if proxy_labels is None:
            proxy_labels = labels
        elif not np.array_equal(proxy_labels, labels):
            raise AssertionError("teacher proxy labels disagree")
        checkpoint_hashes.append(sha256_file(checkpoint))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert proxy_labels is not None
    proxy_order = nested_balanced_order(proxy_labels, seed)
    for cid, split in enumerate(clients):
        write_npz(destination / "splits" / f"client_{cid:03d}.npz", force=force, **split)
    write_npz(
        source_npz, force=force,
        proxy_idx=proxy_idx,
        proxy_labels=proxy_labels,
        proxy_logits=proxy_logits,
        proxy_order=proxy_order,
        holdout_accuracy=holdout_accuracy,
        holdout_counts=holdout_counts,
        test_accuracy=test_accuracy,
        test_counts=test_counts,
        train_sizes=np.asarray([len(split["train_idx"]) for split in clients], dtype=np.int64),
        holdout_sizes=np.asarray([len(split["holdout_idx"]) for split in clients], dtype=np.int64),
        test_sizes=np.asarray([len(split["test_idx"]) for split in clients], dtype=np.int64),
    )
    write_json(source_json, {
        "protocol_version": PROTOCOL_VERSION,
        "scientific_role": "clean_teacher_source",
        "dataset": dataset,
        "seed": int(seed),
        "regime": regime,
        "teacher_architecture": config.arch,
        "num_teachers": k,
        "num_classes": c,
        "split_roles": {
            "train_idx": "teacher_parameter_fitting_only",
            "holdout_idx": "teacher_early_stopping_and_expertise_estimation",
            "test_idx": "teacher_evaluation_only",
        },
        "reused_artifacts": "teacher_checkpoints_and_original_split_indices_only",
        "legacy_run": str(run),
        "legacy_partition": str(partition),
        "checkpoint_paths": [str(path) for path in checkpoints],
        "checkpoint_sha256": checkpoint_hashes,
        "teacher_fingerprint": _fingerprint(checkpoint_hashes),
        "proxy_split": str(proxy_path),
        "proxy_idx_sha256": sha256_array(proxy_idx),
        "teacher_source_sha256": sha256_file(source_npz),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }, force=force)
    print(f"[OK] {dataset} seed={seed} regime={regime}: {source_npz}")
    return source_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, default=Path("OUTPUTS/experiments/study_i"))
    parser.add_argument("--output-root", type=Path, default=Path(OUTPUT_ROOT))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--regimes", nargs="+", choices=REGIMES, default=list(REGIMES))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        for dataset in args.datasets:
            for seed in args.seeds:
                for regime in args.regimes:
                    run, partition = _legacy_condition(args.legacy_root, dataset, seed, regime)
                    print(f"{dataset} seed={seed} regime={regime}: {run} | {partition}")
        return
    if args.audit_only:
        records = []
        for dataset in args.datasets:
            for seed in args.seeds:
                for regime in args.regimes:
                    record = audit_condition(
                        dataset=dataset, seed=seed, regime=regime,
                        legacy_root=args.legacy_root, data_dir=args.data_dir,
                    )
                    records.append(record)
                    print(f"[AUDIT OK] {dataset} seed={seed} regime={regime}")
        print(
            f"[AUDIT COMPLETE] conditions={len(records)} "
            f"checkpoints={sum(row['checkpoints'] for row in records)}"
        )
        return
    device = resolve_device(args.device)
    for dataset in args.datasets:
        for seed in args.seeds:
            for regime in args.regimes:
                prepare_condition(
                    dataset=dataset, seed=seed, regime=regime,
                    legacy_root=args.legacy_root, output_root=args.output_root,
                    data_dir=args.data_dir, device=device, batch_size=args.batch_size,
                    num_workers=args.num_workers, force=args.force,
                )


if __name__ == "__main__":
    main()
