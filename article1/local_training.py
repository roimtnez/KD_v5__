"""Train local teachers once, select by holdout, then cache proxy logits."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from article1 import THRESHOLDS
from article1.datasets import datasets_for, labels_of
from article1.distillation import authority_from_holdout
from article1.models import build_model
from article1.partitioning import validate_splits


def _seed(seed: int) -> None:
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _hash_state(state: dict) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode()); digest.update(state[key].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def logits_for(model, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval(); values: list[np.ndarray] = []; labels: list[np.ndarray] = []
    for x, y in loader:
        values.append(model(x.to(device)).cpu().numpy().astype(np.float32)); labels.append(y.numpy())
    return np.concatenate(values), np.concatenate(labels).astype(np.int64)


def _per_class(logits: np.ndarray, labels: np.ndarray, classes: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(labels, minlength=classes).astype(np.int64)
    accuracy = np.zeros(classes, dtype=np.float32)
    pred = logits.argmax(axis=1)
    for c in np.flatnonzero(counts):
        rows = labels == c; accuracy[c] = (pred[rows] == c).mean()
    return accuracy, counts


def _loader(dataset, indices: np.ndarray, batch: int, shuffle: bool) -> DataLoader:
    return DataLoader(Subset(dataset, indices.tolist()), batch_size=batch, shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available())


def train_and_cache(
    *, dataset: str, data_dir: Path, partition_dir: Path, output_dir: Path,
    seed: int, regime: str, epochs: int = 50, patience: int = 5,
    batch_size: int = 64, device: str = "cpu",
) -> Path:
    """Create one selected checkpoint per teacher plus the shared proxy cache.

    The proxy labels are read only after training; they are allowed for target
    routing but never for checkpoint selection or M estimation.
    """
    dev = torch.device(device)
    train_ds, eval_ds, _ = datasets_for(dataset, data_dir)
    proxy_file = Path(partition_dir) / "proxy.npz"
    with np.load(proxy_file, allow_pickle=False) as data: proxy_idx = data["proxy_idx"].astype(np.int64)
    client_files = sorted(Path(partition_dir).glob("client_*.npz"))
    clients = []
    for path in client_files:
        with np.load(path, allow_pickle=False) as data:
            clients.append({key: data[key].astype(np.int64) for key in ("train_idx", "holdout_idx", "test_idx")})
    if len(clients) != 10: raise ValueError("Article 1 fixes K=10")
    validate_splits(proxy_idx, clients)
    output_dir = Path(output_dir); checkpoints = output_dir / "teachers"; checkpoints.mkdir(parents=True, exist_ok=True)
    all_logits: list[np.ndarray] = []; hold_acc = []; hold_counts = []; test_acc = []; test_counts = []; hashes = []
    proxy_loader = _loader(eval_ds, proxy_idx, 256, False)
    proxy_labels: np.ndarray | None = None
    for cid, split in enumerate(clients):
        _seed(seed + cid)
        model = build_model(dataset).to(dev); optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        best_state = None; best_accuracy = -1.0; remaining = patience
        train_loader, hold_loader = _loader(train_ds, split["train_idx"], batch_size, True), _loader(eval_ds, split["holdout_idx"], 256, False)
        for _ in range(epochs):
            model.train()
            for x, y in train_loader:
                loss = F.cross_entropy(model(x.to(dev)), y.to(dev)); optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            h_logits, h_labels = logits_for(model, hold_loader, dev)
            accuracy = float((h_logits.argmax(1) == h_labels).mean()) if len(h_labels) else 0.0
            if accuracy > best_accuracy:
                best_accuracy, best_state, remaining = accuracy, copy.deepcopy(model.state_dict()), patience
            else:
                remaining -= 1
                if remaining <= 0: break
        if best_state is None: raise RuntimeError("teacher has no selectable holdout checkpoint")
        model.load_state_dict(best_state); state_hash = _hash_state(best_state); path = checkpoints / f"teacher_{cid:03d}.pt"
        torch.save({"state_dict": best_state, "state_sha256": state_hash}, path); hashes.append(state_hash)
        h_logits, h_labels = logits_for(model, hold_loader, dev)
        t_logits, t_labels = logits_for(model, _loader(eval_ds, split["test_idx"], 256, False), dev)
        p_logits, p_labels = logits_for(model, proxy_loader, dev)
        if proxy_labels is None: proxy_labels = p_labels
        elif not np.array_equal(proxy_labels, p_labels): raise AssertionError("proxy labels changed between teachers")
        h_a, h_c = _per_class(h_logits, h_labels, 10); t_a, t_c = _per_class(t_logits, t_labels, 10)
        all_logits.append(p_logits); hold_acc.append(h_a); hold_counts.append(h_c); test_acc.append(t_a); test_counts.append(t_c)
    assert proxy_labels is not None
    mask = authority_from_holdout(np.asarray(hold_acc), np.asarray(hold_counts), THRESHOLDS[dataset])
    cache = output_dir / "teacher_cache.npz"
    np.savez_compressed(cache, proxy_idx=proxy_idx, labels=proxy_labels, logits=np.stack(all_logits, axis=1).astype(np.float32), M=mask,
                        holdout_accuracy=np.asarray(hold_acc), holdout_counts=np.asarray(hold_counts), test_accuracy=np.asarray(test_acc), test_counts=np.asarray(test_counts))
    source_hash = hashlib.sha256(cache.read_bytes()).hexdigest()
    (output_dir / "metadata.json").write_text(json.dumps({"protocol": "article1-v2", "dataset": dataset, "seed": seed, "regime": regime, "K": 10,
        "threshold": THRESHOLDS[dataset], "M_source": "holdout_accuracy_and_counts_only", "proxy_source": str(proxy_file), "cache_sha256": source_hash,
        "teacher_state_sha256": hashes, "teacher_fingerprint": hashlib.sha256("".join(hashes).encode()).hexdigest()}, indent=2, sort_keys=True) + "\n")
    return cache
