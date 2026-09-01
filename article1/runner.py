"""Three small CLI stages: partition, teachers, distill."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from article1 import DATASETS, REGIMES, SEEDS, THRESHOLDS
from article1.datasets import datasets_for, labels_of, test_loader
from article1.distillation import METHODS, build_target, kd_loss, metadata_identity
from article1.local_training import _hash_state, _seed, train_and_cache
from article1.models import build_model
from article1.partitioning import make_partitions, save_partitions


class _Proxy(Dataset):
    def __init__(self, dataset, indices): self.dataset, self.indices = dataset, np.asarray(indices, dtype=np.int64)
    def __len__(self): return len(self.indices)
    def __getitem__(self, position):
        x, _ = self.dataset[int(self.indices[position])]; return x, int(position)


class _PairedOrder(Sampler[int]):
    def __init__(self, n: int, seed: int): self.n, self.seed, self.epoch = n, seed, 0; self.digest = hashlib.sha256()
    def __len__(self): return self.n
    def __iter__(self):
        order = torch.randperm(self.n, generator=torch.Generator().manual_seed(self.seed + self.epoch))
        self.digest.update(order.numpy().tobytes()); return iter(order.tolist())


def _evaluate(model, loader, device):
    model.eval(); correct = total = 0; nll = 0.0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device)); y = y.to(device)
            correct += int((logits.argmax(1) == y).sum()); total += len(y); nll += float(F.cross_entropy(logits, y, reduction="sum"))
    return correct / max(total, 1), nll / max(total, 1)


def _update_table(path: Path, row: dict) -> None:
    """Atomically upsert by immutable run identity; never append duplicates."""
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(".lock")
    with lock.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        existing = []
        if path.exists():
            with path.open(newline="", encoding="utf-8") as source: existing = list(csv.DictReader(source))
        key = row["run_id"]
        existing = [old for old in existing if old.get("run_id") != key] + [row]
        fields = sorted(set().union(*(record.keys() for record in existing)))
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=fields); writer.writeheader(); writer.writerows(existing)
        os.replace(tmp, path); fcntl.flock(handle, fcntl.LOCK_UN)


def distill(*, cache: Path, data_dir: Path, results: Path, method: str, dataset: str, seed: int, epochs: int = 30, batch_size: int = 256, temperature: float = 8.0, device: str = "cpu") -> dict:
    """Consume q directly. Every arm shares teacher cache, initial state and order."""
    dev = torch.device(device)
    cache = Path(cache)
    metadata_path = cache.parent / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"cache metadata missing: {metadata_path}")
    source_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if source_metadata.get("protocol") != "article1-v2" or source_metadata.get("dataset") != dataset or int(source_metadata.get("seed", -1)) != seed:
        raise ValueError("teacher cache metadata is incompatible with this distillation condition")
    with np.load(cache, allow_pickle=False) as data:
        logits, labels, indices, mask = data["logits"], data["labels"], data["proxy_idx"], data["M"]
    config = {"epochs": epochs, "batch_size": batch_size, "optimizer": "AdamW", "lr": 1e-3, "weight_decay": 1e-4, "K": 10}
    cache_hash = hashlib.sha256(cache.read_bytes()).hexdigest()
    if source_metadata.get("cache_sha256") != cache_hash:
        raise ValueError("teacher cache hash does not match metadata")
    mask_hash = hashlib.sha256(mask.tobytes()).hexdigest(); proxy_hash = hashlib.sha256(indices.tobytes()).hexdigest()
    run_id = metadata_identity(method=method, temperature=temperature, config=config, source_hash=cache_hash, proxy_hash=proxy_hash, mask_hash=mask_hash)
    target = build_target(logits, labels, mask, method=method, temperature=temperature)
    train_ds, eval_ds, test_ds = datasets_for(dataset, data_dir)
    _seed(seed); initial = build_model(dataset).state_dict(); initial_hash = _hash_state(initial)
    model = build_model(dataset).to(dev); model.load_state_dict(copy.deepcopy(initial)); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    order = _PairedOrder(len(indices), seed)
    loader = DataLoader(_Proxy(eval_ds, indices), batch_size=min(batch_size, len(indices)), sampler=order, num_workers=0)
    q = torch.from_numpy(target.probabilities).to(dev)
    for epoch in range(epochs):
        order.epoch = epoch; model.train()
        for x, positions in loader:
            loss = kd_loss(model(x.to(dev)), q.index_select(0, positions.to(dev)), temperature)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    accuracy, nll = _evaluate(model, test_loader(test_ds), dev)
    row = {"run_id": run_id, "dataset": dataset, "regime": source_metadata["regime"], "seed": seed, "method": method, "temperature": temperature, "cache_sha256": cache_hash,
           "M_sha256": mask_hash, "proxy_sha256": proxy_hash, "student_init_sha256": initial_hash, "batch_order_sha256": order.digest.hexdigest(),
           "student_final_sha256": _hash_state(model.state_dict()), "updates": epochs * len(loader), "student_test_accuracy": accuracy, "student_test_nll": nll, **target.metrics}
    _update_table(Path(results), row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); subs = parser.add_subparsers(dest="stage", required=True)
    p = subs.add_parser("partition"); p.add_argument("--dataset", choices=DATASETS, required=True); p.add_argument("--regime", choices=REGIMES, required=True); p.add_argument("--seed", choices=SEEDS, type=int, required=True); p.add_argument("--data-dir", type=Path, default=Path("data")); p.add_argument("--output", type=Path, required=True); p.add_argument("--proxy-size", type=int, default=10_000)
    t = subs.add_parser("teachers"); t.add_argument("--dataset", choices=DATASETS, required=True); t.add_argument("--regime", choices=REGIMES, required=True); t.add_argument("--seed", choices=SEEDS, type=int, required=True); t.add_argument("--data-dir", type=Path, default=Path("data")); t.add_argument("--partitions", type=Path, required=True); t.add_argument("--output", type=Path, required=True); t.add_argument("--epochs", type=int, default=50); t.add_argument("--device", default="cpu")
    d = subs.add_parser("distill"); d.add_argument("--dataset", choices=DATASETS, required=True); d.add_argument("--seed", choices=SEEDS, type=int, required=True); d.add_argument("--cache", type=Path, required=True); d.add_argument("--data-dir", type=Path, default=Path("data")); d.add_argument("--results", type=Path, required=True); d.add_argument("--method", choices=METHODS, required=True); d.add_argument("--temperature", type=float, default=8.0); d.add_argument("--epochs", type=int, default=30); d.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.stage == "partition":
        _, eval_ds, _ = datasets_for(args.dataset, args.data_dir); proxy, clients = make_partitions(np.asarray(labels_of(eval_ds)), regime=args.regime, seed=args.seed, proxy_size=args.proxy_size)
        save_partitions(args.output, proxy_idx=proxy, clients=clients, metadata={"dataset": args.dataset, "regime": args.regime, "seed": args.seed, "K": 10, "proxy_size": len(proxy)})
    elif args.stage == "teachers": train_and_cache(dataset=args.dataset, data_dir=args.data_dir, partition_dir=args.partitions, output_dir=args.output, seed=args.seed, regime=args.regime, epochs=args.epochs, device=args.device)
    else: print(json.dumps(distill(cache=args.cache, data_dir=args.data_dir, results=args.results, method=args.method, dataset=args.dataset, seed=args.seed, temperature=args.temperature, epochs=args.epochs, device=args.device), indent=2))


if __name__ == "__main__": main()
