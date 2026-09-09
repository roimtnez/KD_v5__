"""Train local teachers once, select by validation, estimate expertise independently, then cache proxy logits."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from article1 import PROTOCOL_VERSION, THRESHOLDS
from article1.datasets import datasets_for, labels_of
from article1.distillation import authority_from_expertise
from article1.hashes import file_sha256, git_commit
from article1.models import build_model
from article1.partitioning import load_partitions


def configure_determinism() -> None:
    """Make the fixed Article-1 recipe bitwise reproducible on CUDA or CPU."""
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _seed(seed: int) -> None:
    configure_determinism()
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _hash_state(state: dict) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(state[key].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def logits_for(model, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    values: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for x, y in loader:
        values.append(
            model(x.to(device, non_blocking=True)).cpu().numpy().astype(np.float32)
        )
        labels.append(y.numpy())
    return np.concatenate(values), np.concatenate(labels).astype(np.int64)


def _per_class(
    logits: np.ndarray, labels: np.ndarray, classes: int
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(labels, minlength=classes).astype(np.int64)
    accuracy = np.zeros(classes, dtype=np.float32)
    pred = logits.argmax(axis=1)
    for c in np.flatnonzero(counts):
        rows = labels == c
        accuracy[c] = (pred[rows] == c).mean()
    return accuracy, counts


def _loader(
    dataset,
    indices: np.ndarray,
    batch: int,
    shuffle: bool,
    device: str | torch.device = "cpu",
) -> DataLoader:
    return DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=batch,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.device(device).type == "cuda",
    )


def train_and_cache(
    *,
    dataset: str,
    data_dir: Path,
    partition_dir: Path,
    output_dir: Path,
    seed: int,
    regime: str,
    epochs: int = 50,
    patience: int = 5,
    batch_size: int = 64,
    device: str = "cpu",
) -> Path:
    """Create one selected checkpoint per teacher plus the shared proxy cache.

    Proxy labels are allowed for server routing, never for checkpoint selection
    or M estimation. Validation selects the checkpoint; expertise is evaluated
    only after the model is frozen. Full training labels validate partition identity.
    """
    dev = torch.device(device)
    train_ds, eval_ds, _ = datasets_for(dataset, data_dir)
    if epochs <= 0 or patience <= 0 or batch_size <= 0:
        raise ValueError("training budgets must be positive")
    proxy_idx, clients, partition_metadata = load_partitions(
        partition_dir, np.asarray(labels_of(eval_ds), dtype=np.int64)
    )
    if (
        partition_metadata.get("dataset"),
        partition_metadata.get("regime"),
        partition_metadata.get("seed"),
    ) != (dataset, regime, seed):
        raise ValueError("partition identity does not match teacher condition")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"teacher output is not empty: {output_dir}")
    checkpoints = output_dir / "teachers"
    checkpoints.mkdir(parents=True)
    all_logits: list[np.ndarray] = []
    expertise_acc = []
    expertise_counts = []
    selection_records = []
    hashes = []
    proxy_loader = _loader(eval_ds, proxy_idx, 256, False, dev)
    proxy_labels: np.ndarray | None = None
    for cid, split in enumerate(clients):
        _seed(seed + cid)
        model = build_model(dataset).to(dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        best_state = None
        best_accuracy = -1.0
        remaining = patience
        train_loader, validation_loader = (
            _loader(train_ds, split["train_idx"], batch_size, True, dev),
            _loader(eval_ds, split["validation_idx"], 256, False, dev),
        )
        best_epoch = 0
        for epoch in range(epochs):
            model.train()
            for x, y in train_loader:
                loss = F.cross_entropy(
                    model(x.to(dev, non_blocking=True)),
                    y.to(dev, non_blocking=True),
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            v_logits, v_labels = logits_for(model, validation_loader, dev)
            accuracy = float((v_logits.argmax(1) == v_labels).mean())
            if accuracy > best_accuracy:
                best_epoch = epoch + 1
                best_accuracy, best_state, remaining = (
                    accuracy,
                    copy.deepcopy(model.state_dict()),
                    patience,
                )
            else:
                remaining -= 1
                if remaining <= 0:
                    break
        if best_state is None:
            raise RuntimeError("teacher has no selectable validation checkpoint")
        model.load_state_dict(best_state)
        state_hash = _hash_state(best_state)
        path = checkpoints / f"teacher_{cid:03d}.pt"
        torch.save(
            {
                "state_dict": best_state,
                "state_sha256": state_hash,
                "selected_epoch": best_epoch,
                "validation_accuracy": best_accuracy,
                "protocol_version": PROTOCOL_VERSION,
            },
            path,
        )
        hashes.append(state_hash)
        # No optimizer steps after selection. This split never selects the checkpoint.
        e_logits, e_labels = logits_for(
            model, _loader(eval_ds, split["expertise_idx"], 256, False, dev), dev
        )
        selection_records.append(
            {
                "client": cid,
                "selected_epoch": best_epoch,
                "epochs_run": epoch + 1,
                "validation_accuracy": best_accuracy,
            }
        )
        p_logits, p_labels = logits_for(model, proxy_loader, dev)
        if proxy_labels is None:
            proxy_labels = p_labels
        elif not np.array_equal(proxy_labels, p_labels):
            raise AssertionError("proxy labels changed between teachers")
        e_a, e_c = _per_class(e_logits, e_labels, 10)
        all_logits.append(p_logits)
        expertise_acc.append(e_a)
        expertise_counts.append(e_c)
    assert proxy_labels is not None
    mask = authority_from_expertise(
        np.asarray(expertise_acc), np.asarray(expertise_counts), THRESHOLDS[dataset]
    )
    cache = output_dir / "teacher_cache.npz"
    np.savez_compressed(
        cache,
        proxy_idx=proxy_idx,
        labels=proxy_labels,
        logits=np.stack(all_logits, axis=1).astype(np.float32),
        M=mask,
        expertise_accuracy=np.asarray(expertise_acc),
        expertise_counts=np.asarray(expertise_counts),
    )
    source_hash = file_sha256(cache)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "protocol": PROTOCOL_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "dataset": dataset,
                "seed": seed,
                "regime": regime,
                "K": 10,
                "threshold": THRESHOLDS[dataset],
                "M_source": "expertise_accuracy_and_counts_only",
                "checkpoint_selection": "validation_accuracy_strict_improvement",
                "selection_records": selection_records,
                "partition_metadata_sha256": file_sha256(
                    Path(partition_dir) / "metadata.json"
                ),
                "split_rule": partition_metadata["split_rule"],
                "split_fractions": partition_metadata["split_fractions"],
                "proxy_source": str(Path(partition_dir) / "proxy.npz"),
                "cache_sha256": source_hash,
                "cache_creation_commit": git_commit(),
                "teacher_state_sha256": hashes,
                "teacher_fingerprint": hashlib.sha256(
                    "".join(hashes).encode()
                ).hexdigest(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return cache
