"""Small CLI stages: partition, teachers, KD distillation, and proxy supervision."""

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

from article1 import DATASETS, PROTOCOL_VERSION, REGIMES, SEEDS
from article1.datasets import datasets_for, labels_of, test_loader
from article1.distillation import (
    METHODS,
    build_target,
    kd_config,
    kd_loss,
    metadata_identity,
)
from article1.hashes import array_sha256, artifact_commit, file_sha256, git_commit
from article1.local_training import _hash_state, _seed, train_and_cache
from article1.models import build_model
from article1.partitioning import make_partitions, save_partitions
from article1.proxy import proxy_positions


class _Proxy(Dataset):
    def __init__(self, dataset, indices):
        self.dataset, self.indices = dataset, np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        x, _ = self.dataset[int(self.indices[position])]
        return x, int(position)


class _PairedOrder(Sampler[int]):
    def __init__(self, n: int, seed: int):
        self.n, self.seed, self.epoch = n, seed, 0
        self.digest = hashlib.sha256()

    def __len__(self):
        return self.n

    def __iter__(self):
        order = torch.randperm(
            self.n, generator=torch.Generator().manual_seed(self.seed + self.epoch)
        )
        self.digest.update(order.numpy().tobytes())
        return iter(order.tolist())


def _evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    nll = 0.0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            y = y.to(device)
            correct += int((logits.argmax(1) == y).sum())
            total += len(y)
            nll += float(F.cross_entropy(logits, y, reduction="sum"))
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
            with path.open(newline="", encoding="utf-8") as source:
                existing = list(csv.DictReader(source))
        if any(old.get("protocol_version") != PROTOCOL_VERSION for old in existing):
            raise ValueError(
                "results CSV belongs to another/unknown protocol; use OUTPUTS/article1_v3"
            )
        key = row["run_id"]
        existing = [old for old in existing if old.get("run_id") != key] + [row]
        fields = sorted(set().union(*(record.keys() for record in existing)))
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=fields)
            writer.writeheader()
            writer.writerows(existing)
        os.replace(tmp, path)
        fcntl.flock(handle, fcntl.LOCK_UN)


def supervised_proxy_identity(
    *, dataset: str, seed: int, proxy_hash: str, config: dict
) -> str:
    """Identity for a regime-independent, labelled-proxy baseline.

    Deliberately excludes the teacher cache and routing mask: neither is an input to
    supervised training.  This makes reuse across regimes explicit and auditable.
    """
    payload = {
        "dataset": dataset,
        "seed": int(seed),
        "method": "supervised_proxy_ce",
        "proxy_sha256": proxy_hash,
        "recipe": config,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_proxy(cache, dataset, seed, proxy_size):
    cache = Path(cache)
    metadata_path = cache.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("protocol") != PROTOCOL_VERSION
        or metadata.get("dataset") != dataset
        or int(metadata.get("seed", -1)) != seed
    ):
        raise ValueError("teacher cache metadata is incompatible with this condition")
    cache_hash = file_sha256(cache)
    if metadata.get("cache_sha256") != cache_hash:
        raise ValueError("teacher cache hash does not match metadata")
    with np.load(cache, allow_pickle=False) as data:
        indices, labels = data["proxy_idx"], data["labels"]
    positions = proxy_positions(indices, labels, proxy_size, seed)
    indices, labels = indices.astype(np.int64), labels.astype(np.int64)
    info = {
        "proxy_master_sha256": array_sha256(indices),
        "proxy_master_size": len(indices),
        "proxy_sha256": array_sha256(indices[positions]),
        "proxy_labels_sha256": array_sha256(labels[positions]),
        "proxy_size": len(positions),
        "proxy_selection": "full_original_order"
        if len(positions) == len(indices)
        else "nested_stratified_v1",
        "proxy_selection_seed": seed,
    }
    return metadata, cache_hash, indices[positions], labels[positions], positions, info


def _train_proxy(
    *,
    dataset,
    data_dir,
    indices,
    labels,
    seed,
    updates,
    batch_size,
    device,
    q=None,
    temperature=8.0,
):
    """Identical runtime for CE and KD, including partial final epochs."""
    if updates <= 0 or batch_size <= 0:
        raise ValueError("updates and batch_size must be positive")
    dev = torch.device(device)
    train_ds, eval_ds, test_ds = datasets_for(dataset, data_dir)
    del train_ds
    if (indices >= len(eval_ds)).any():
        raise ValueError("proxy index outside official training set")
    actual_labels = np.asarray(labels_of(eval_ds), dtype=np.int64)[indices]
    if not np.array_equal(labels, actual_labels):
        raise ValueError("cache labels do not match the public proxy labels")
    _seed(seed)
    initial = build_model(dataset).state_dict()
    initial_hash = _hash_state(initial)
    model = build_model(dataset).to(dev)
    model.load_state_dict(copy.deepcopy(initial))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    order = _PairedOrder(len(indices), seed)
    loader = DataLoader(
        _Proxy(eval_ds, indices),
        batch_size=min(batch_size, len(indices)),
        sampler=order,
        num_workers=0,
    )
    targets = torch.from_numpy(labels if q is None else q).to(dev)
    consumed = hashlib.sha256()
    completed = epoch = examples = 0
    last_loss = float("nan")
    while completed < updates:
        order.epoch = epoch
        model.train()
        for x, positions in loader:
            # Include batch boundaries and actual dataset indices, not unused suffixes.
            consumed.update(np.asarray([len(positions)], dtype=np.int64).tobytes())
            consumed.update(indices[positions.numpy()].tobytes())
            outputs = model(x.to(dev))
            batch_targets = targets.index_select(0, positions.to(dev))
            loss = (
                F.cross_entropy(outputs, batch_targets)
                if q is None
                else kd_loss(outputs, batch_targets, temperature)
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last_loss = float(loss.detach())
            completed += 1
            examples += len(positions)
            if completed == updates:
                break
        epoch += 1
    accuracy, nll = _evaluate(model, test_loader(test_ds), dev)
    return {
        "student_init_sha256": initial_hash,
        "batch_order_sha256": order.digest.hexdigest(),  # Historical epoch-permutation hash.
        "consumed_batches_sha256": consumed.hexdigest(),
        "student_final_sha256": _hash_state(model.state_dict()),
        "updates": completed,
        "epochs_completed": completed // len(loader),
        "epochs_started": epoch,
        "examples_seen": examples,
        "batch_size": min(batch_size, len(indices)),
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "scheduler": "none",
        "proxy_view": "deterministic_evaluation",
        "student_final_train_loss": last_loss,
        "student_test_accuracy": accuracy,
        "student_test_nll": nll,
        "protocol_version": PROTOCOL_VERSION,
    }


def _protect_main_results(results, *, study):
    if study and Path(results).name == "results.csv":
        raise ValueError(
            "use a separate CSV for supervised/proxy-size/update-budget runs"
        )


def distill(
    *,
    cache: Path,
    data_dir: Path,
    results: Path,
    method: str,
    dataset: str,
    seed: int,
    epochs: int = 30,
    batch_size: int = 256,
    temperature: float = 8.0,
    device: str = "cpu",
    updates: int | None = None,
    proxy_size: int | None = None,
) -> dict:
    """Consume q directly; optional subsets require an explicit update budget."""
    if batch_size <= 0 or epochs <= 0 or (updates is not None and updates <= 0):
        raise ValueError("budget and batch_size must be positive")
    if proxy_size is not None and updates is None:
        raise ValueError("--proxy-size requires --updates to keep the budget fixed")
    _protect_main_results(results, study=proxy_size is not None or updates is not None)
    metadata, cache_hash, indices, labels, positions, info = _load_proxy(
        cache, dataset, seed, proxy_size
    )
    with np.load(cache, allow_pickle=False) as data:
        logits, mask = data["logits"][positions], data["M"]
    config = kd_config(epochs, batch_size)
    total_updates = epochs * ((len(indices) + batch_size - 1) // batch_size)
    if updates is not None:
        total_updates = updates
        config.pop("epochs")
        config.update(
            updates=updates,
            proxy_size=len(indices),
            proxy_labels_sha256=info["proxy_labels_sha256"],
        )
    mask_hash = array_sha256(mask)
    run_id = metadata_identity(
        method=method,
        temperature=temperature,
        config=config,
        source_hash=cache_hash,
        proxy_hash=info["proxy_sha256"],
        mask_hash=mask_hash,
    )
    target = build_target(logits, labels, mask, method=method, temperature=temperature)
    trained = _train_proxy(
        dataset=dataset,
        data_dir=data_dir,
        indices=indices,
        labels=labels,
        seed=seed,
        updates=total_updates,
        batch_size=batch_size,
        device=device,
        q=target.probabilities,
        temperature=temperature,
    )
    row = {
        "run_id": run_id,
        "dataset": dataset,
        "regime": metadata["regime"],
        "seed": seed,
        "method": method,
        "temperature": temperature,
        "cache_sha256": cache_hash,
        "M_sha256": mask_hash,
        "cache_creation_commit": artifact_commit(metadata),
        "kd_execution_commit": git_commit(),
        "training_recipe_json": json.dumps(config, sort_keys=True),
        **info,
        **trained,
        **target.metrics,
    }
    _update_table(Path(results), row)
    return row


def supervised_proxy(
    *,
    cache: Path,
    data_dir: Path,
    results: Path,
    dataset: str,
    seed: int,
    updates: int = 1200,
    batch_size: int = 256,
    device: str = "cpu",
    proxy_size: int | None = None,
) -> dict:
    """CE on the same reserved public examples; independent of teacher knowledge."""
    if updates <= 0 or batch_size <= 0:
        raise ValueError("updates and batch_size must be positive")
    _protect_main_results(results, study=True)
    metadata, cache_hash, indices, labels, _, info = _load_proxy(
        cache, dataset, seed, proxy_size
    )
    config = {
        "loss": "cross_entropy",
        "optimizer": "AdamW",
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": min(batch_size, len(indices)),
        "updates": updates,
        "scheduler": None,
        "proxy_view": "deterministic_evaluation",
        "proxy_labels_sha256": info["proxy_labels_sha256"],
        "protocol_version": PROTOCOL_VERSION,
    }
    run_id = supervised_proxy_identity(
        dataset=dataset, seed=seed, proxy_hash=info["proxy_sha256"], config=config
    )
    trained = _train_proxy(
        dataset=dataset,
        data_dir=data_dir,
        indices=indices,
        labels=labels,
        seed=seed,
        updates=updates,
        batch_size=batch_size,
        device=device,
    )
    row = {
        "run_id": run_id,
        "dataset": dataset,
        "regime": "shared_proxy",
        "source_regime": metadata["regime"],
        "source_cache_sha256": cache_hash,
        "seed": seed,
        "method": "supervised_proxy_ce",
        "loss": "cross_entropy",
        "temperature": None,
        "execution_commit": git_commit(),
        "training_recipe_json": json.dumps(config, sort_keys=True),
        **info,
        **trained,
    }
    _update_table(Path(results), row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="stage", required=True)
    p = subs.add_parser("partition")
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--regime", choices=REGIMES, required=True)
    p.add_argument("--seed", choices=SEEDS, type=int, required=True)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--proxy-size", type=int, default=10_000)
    t = subs.add_parser("teachers")
    t.add_argument("--dataset", choices=DATASETS, required=True)
    t.add_argument("--regime", choices=REGIMES, required=True)
    t.add_argument("--seed", choices=SEEDS, type=int, required=True)
    t.add_argument("--data-dir", type=Path, default=Path("data"))
    t.add_argument("--partitions", type=Path, required=True)
    t.add_argument("--output", type=Path, required=True)
    t.add_argument("--epochs", type=int, default=50)
    t.add_argument("--device", default="cpu")
    d = subs.add_parser("distill")
    d.add_argument("--dataset", choices=DATASETS, required=True)
    d.add_argument("--seed", choices=SEEDS, type=int, required=True)
    d.add_argument("--cache", type=Path, required=True)
    d.add_argument("--data-dir", type=Path, default=Path("data"))
    d.add_argument("--results", type=Path, required=True)
    d.add_argument("--method", choices=METHODS, required=True)
    d.add_argument("--temperature", type=float, default=8.0)
    budget = d.add_mutually_exclusive_group()
    budget.add_argument("--epochs", type=int, default=30)
    budget.add_argument("--updates", type=int)
    d.add_argument("--proxy-size", type=int)
    d.add_argument("--batch-size", type=int, default=256)
    d.add_argument("--device", default="cpu")
    s = subs.add_parser("supervised")
    s.add_argument("--dataset", choices=DATASETS, required=True)
    s.add_argument("--seed", choices=SEEDS, type=int, required=True)
    s.add_argument("--cache", type=Path, required=True)
    s.add_argument("--data-dir", type=Path, default=Path("data"))
    s.add_argument("--results", type=Path, required=True)
    s.add_argument("--updates", type=int, default=1200)
    s.add_argument("--proxy-size", type=int)
    s.add_argument("--batch-size", type=int, default=256)
    s.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.stage == "partition":
        _, eval_ds, _ = datasets_for(args.dataset, args.data_dir)
        proxy, clients = make_partitions(
            np.asarray(labels_of(eval_ds)),
            regime=args.regime,
            seed=args.seed,
            proxy_size=args.proxy_size,
        )
        save_partitions(
            args.output,
            proxy_idx=proxy,
            clients=clients,
            labels=np.asarray(labels_of(eval_ds), dtype=np.int64),
            metadata={
                "dataset": args.dataset,
                "regime": args.regime,
                "seed": args.seed,
                "K": 10,
                "proxy_size": len(proxy),
            },
        )
    elif args.stage == "teachers":
        train_and_cache(
            dataset=args.dataset,
            data_dir=args.data_dir,
            partition_dir=args.partitions,
            output_dir=args.output,
            seed=args.seed,
            regime=args.regime,
            epochs=args.epochs,
            device=args.device,
        )
    elif args.stage == "distill":
        print(
            json.dumps(
                distill(
                    cache=args.cache,
                    data_dir=args.data_dir,
                    results=args.results,
                    method=args.method,
                    dataset=args.dataset,
                    seed=args.seed,
                    temperature=args.temperature,
                    epochs=args.epochs,
                    updates=args.updates,
                    proxy_size=args.proxy_size,
                    batch_size=args.batch_size,
                    device=args.device,
                ),
                indent=2,
            )
        )
    else:
        print(
            json.dumps(
                supervised_proxy(
                    cache=args.cache,
                    data_dir=args.data_dir,
                    results=args.results,
                    dataset=args.dataset,
                    seed=args.seed,
                    updates=args.updates,
                    proxy_size=args.proxy_size,
                    batch_size=args.batch_size,
                    device=args.device,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
