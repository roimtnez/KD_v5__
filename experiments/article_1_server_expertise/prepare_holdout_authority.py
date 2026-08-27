#!/usr/bin/env python3
"""Regenerate Article-1 expertise masks from existing teachers' local holdouts.

This performs inference only: it does not retrain teachers and it leaves the
historical proxy-analysis cache untouched.  It writes a content-addressed
authority sidecar accepted by :mod:`experiment_1`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.dataset_config import get_dataset_config
from oracle_distillation.checkpoints import load_checkpoint_state
from oracle_distillation.models import build_model
from oracle_distillation.utils import collect_logits, resolve_device
from experiments.article_1_server_expertise.artifacts import sha256_file
from experiments.article_1_server_expertise.experiment_1 import (
    THRESHOLD_BY_DATASET_LABEL, SourceArtifact, _load_cache, _select, discover_sources,
)
from experiments.article_1_server_expertise.storage import save_npz_once, write_json_once


DATASET_NAME = {"cifar10": "cifar", "mnist": "mnist", "fmnist": "fmnist", "cinic": "cinic", "cifar100": "cifar100"}


def _checkpoint_for(run_dir: Path, cid: int) -> Path:
    path = Path(run_dir) / "teachers" / f"cid_{cid:03d}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"missing retained teacher checkpoint: {path}")
    return path


def _per_class_accuracy(logits: np.ndarray, labels: np.ndarray, classes: int) -> tuple[np.ndarray, np.ndarray]:
    prediction = logits.argmax(axis=1)
    counts = np.bincount(labels, minlength=classes).astype(np.int64)
    accuracy = np.full(classes, np.nan, dtype=np.float32)
    for cls in np.flatnonzero(counts):
        rows = labels == cls
        accuracy[cls] = float((prediction[rows] == cls).mean())
    return accuracy, counts


def prepare_one(source: SourceArtifact, *, output_root: Path, data_dir: Path,
                device: torch.device, probe_examples: int) -> Path:
    cache = _load_cache(source.path)
    logits_cache = cache["teacher_logits_cache"].astype(np.float32)
    k, classes = logits_cache.shape[1:]
    dataset_name = DATASET_NAME[source.dataset]
    config = get_dataset_config(dataset_name)
    eval_dataset = config.load_train_eval_dataset(data_dir)
    threshold = THRESHOLD_BY_DATASET_LABEL[source.dataset]
    per_class = np.full((k, classes), np.nan, dtype=np.float32)
    class_counts = np.zeros((k, classes), dtype=np.int64)
    checkpoint_hashes: list[str] = []
    probe_positions = np.arange(min(probe_examples, len(cache["proxy_idx"])))
    probe_loader = DataLoader(Subset(eval_dataset, cache["proxy_idx"][probe_positions].astype(int).tolist()),
                              batch_size=256, shuffle=False, num_workers=0)
    for cid in range(k):
        checkpoint = _checkpoint_for(source.run_dir, cid)
        model = build_model(config.arch, num_classes=classes).to(device)
        model.load_state_dict(load_checkpoint_state(checkpoint, map_location=device), strict=True)
        model.eval()
        probe_logits, _ = collect_logits(model, probe_loader, device)
        # A deterministic probe makes reuse of the cached full proxy logits auditable
        # without repeating the expensive 10k-example forward pass.
        if not np.allclose(probe_logits, logits_cache[probe_positions, cid], rtol=2e-3, atol=2e-2):
            raise ValueError(f"teacher checkpoint does not match cached proxy logits: {checkpoint}")
        client_path = source.partition_dir / "clients" / f"c{cid:03d}.npz"
        with np.load(client_path, allow_pickle=False) as split:
            if "holdout_idx" not in split:
                raise ValueError(f"{client_path} lacks holdout_idx")
            holdout_idx = np.asarray(split["holdout_idx"], dtype=np.int64)
        holdout_loader = DataLoader(Subset(eval_dataset, holdout_idx.tolist()), batch_size=256,
                                    shuffle=False, num_workers=0)
        holdout_logits, labels = collect_logits(model, holdout_loader, device)
        per_class[cid], class_counts[cid] = _per_class_accuracy(holdout_logits, labels.astype(np.int64), classes)
        checkpoint_hashes.append(sha256_file(checkpoint))
        del model
    authority = ((per_class >= threshold) & (class_counts > 0)).astype(np.uint8)
    source_hash = sha256_file(source.path)
    out = Path(output_root) / "holdout_authority" / source_hash
    save_npz_once(out / "authority.npz", authority=authority, holdout_acc_per_class=per_class,
                  holdout_class_counts=class_counts, source_proxy_sha256=np.array(source_hash))
    write_json_once(out / "provenance.json", {
        "source_proxy_analysis": str(source.path), "source_proxy_sha256": source_hash,
        "dataset": source.dataset, "seed": source.seed, "regime": source.regime,
        "partition_dir": str(source.partition_dir), "mask_source": f"holdout_acc_per_class>={threshold}",
        "threshold": threshold, "checkpoint_sha256": checkpoint_hashes,
        "proxy_logit_probe_examples": int(len(probe_positions)), "authority_support_sizes": authority.sum(axis=1).astype(int).tolist(),
        "zero_support_teachers": np.flatnonzero(authority.sum(axis=1) == 0).astype(int).tolist(),
        "scientific_status": "valid_holdout_mask_pending_downstream_analysis",
    })
    return out / "authority.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=Path("OUTPUTS/experiments/study_i"))
    parser.add_argument("--output-root", type=Path, default=Path("OUTPUTS/experiments/article_1_server_expertise/experiment_1"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--datasets", nargs="*", choices=tuple(DATASET_NAME))
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--regimes", nargs="*", choices=("iid", "alpha1p0", "alpha0p5", "alpha0p1", "multi", "single"))
    parser.add_argument("--probe-examples", type=int, default=32)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = _select(discover_sources(args.study_root), args)
    if not sources:
        raise FileNotFoundError(f"no sources found below {args.study_root}")
    device = resolve_device(args.device)
    for source in sources:
        output = prepare_one(source, output_root=args.output_root, data_dir=args.data_dir,
                             device=device, probe_examples=args.probe_examples)
        print(f"[OK] {source.dataset} seed={source.seed} regime={source.regime}: {output}")


if __name__ == "__main__":
    main()
