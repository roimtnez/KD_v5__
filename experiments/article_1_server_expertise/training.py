"""Shared student training for KD and supervised Article-1 controls."""
from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from oracle_distillation.distill.losses import loss_soft_kd
from oracle_distillation.models import build_model
from oracle_distillation.utils import set_seed


class PositionedDataset(Dataset):
    def __init__(self, base, dataset_indices: np.ndarray, labels: np.ndarray | None = None):
        self.base = base
        self.indices = np.asarray(dataset_indices, dtype=np.int64)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)
        if self.labels is not None and self.labels.shape != self.indices.shape:
            raise ValueError("labels must align with dataset_indices")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        image, dataset_label = self.base[int(self.indices[position])]
        label = int(dataset_label) if self.labels is None else int(self.labels[position])
        return image, int(position), label


class EpochPermutationSampler(Sampler[int]):
    """Deterministic epoch-wise permutations with an observable order hash."""

    def __init__(self, size: int, seed: int):
        self.size = int(size)
        self.seed = int(seed)
        self.epoch = 0
        self._digest = hashlib.sha256()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.size, generator=generator)
        self._digest.update(self.epoch.to_bytes(8, "little"))
        self._digest.update(order.numpy().tobytes())
        return iter(order.tolist())

    def __len__(self) -> int:
        return self.size

    @property
    def order_sha256(self) -> str:
        return self._digest.hexdigest()


def initial_state(arch: str, num_classes: int, seed: int) -> tuple[dict, str]:
    set_seed(seed)
    model = build_model(arch, num_classes=num_classes)
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(state[name].numpy()).tobytes())
    return state, digest.hexdigest()


@dataclass(frozen=True)
class TrainResult:
    metrics: dict
    state_dict: dict
    test_logits: np.ndarray
    test_labels: np.ndarray


@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray, float, float]:
    model.eval()
    logits_parts, label_parts = [], []
    for images, labels in loader:
        logits_parts.append(model(images.to(device)).detach().cpu().numpy())
        label_parts.append(labels.numpy())
    logits = np.concatenate(logits_parts)
    labels = np.concatenate(label_parts).astype(np.int64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_prob = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    accuracy = float((logits.argmax(axis=1) == labels).mean())
    nll = float(-log_prob[np.arange(len(labels)), labels].mean())
    return logits, labels, accuracy, nll


def train_student(
    *,
    arch: str,
    num_classes: int,
    init_state: dict,
    init_hash: str,
    train_dataset,
    test_loader,
    dataset_indices: np.ndarray,
    device: torch.device,
    seed: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    temperature: float,
    target_probabilities: np.ndarray | None = None,
    hard_labels: np.ndarray | None = None,
    max_updates: int | None = None,
) -> TrainResult:
    """Train from a supplied shared initialization with exact update control."""
    if (target_probabilities is None) == (hard_labels is None):
        raise ValueError("provide exactly one of target_probabilities or hard_labels")
    set_seed(seed)
    model = build_model(arch, num_classes=num_classes).to(device)
    model.load_state_dict(copy.deepcopy(init_state), strict=True)
    indices = np.asarray(dataset_indices, dtype=np.int64)
    labels = None if hard_labels is None else np.asarray(hard_labels, dtype=np.int64)
    positioned = PositionedDataset(train_dataset, indices, labels)
    sampler = EpochPermutationSampler(len(positioned), seed)
    loader = DataLoader(
        positioned,
        batch_size=min(int(batch_size), len(positioned)),
        sampler=sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )
    pseudo_logits = None
    if target_probabilities is not None:
        probs = np.asarray(target_probabilities, dtype=np.float32)
        if probs.shape != (len(positioned), num_classes):
            raise ValueError("target probabilities must align with dataset_indices")
        pseudo_logits = torch.from_numpy(
            float(temperature) * np.log(np.clip(probs, 1e-12, 1.0))
        ).to(device)

    planned = int(max_updates) if max_updates is not None else int(epochs * len(loader))
    if planned <= 0:
        raise ValueError("training requires at least one update")
    steps = epoch = 0
    started = perf_counter()
    last_loss = float("nan")
    while steps < planned:
        sampler.set_epoch(epoch)
        model.train()
        for images, positions, batch_labels in loader:
            output = model(images.to(device))
            if pseudo_logits is not None:
                target = pseudo_logits.index_select(0, positions.to(device))
                loss = loss_soft_kd(output, target, temperature)
            else:
                loss = F.cross_entropy(output, batch_labels.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().item())
            steps += 1
            if steps >= planned:
                break
        epoch += 1

    test_logits, test_labels, accuracy, nll = evaluate(model, test_loader, device)
    result = {
        "test_accuracy": accuracy,
        "test_nll": nll,
        "total_updates": steps,
        "epochs_completed_or_entered": epoch,
        "batch_size": min(int(batch_size), len(positioned)),
        "n_train": len(positioned),
        "final_train_loss": last_loss,
        "elapsed_seconds": perf_counter() - started,
        "student_init_sha256": init_hash,
        "student_init_seed": int(seed),
        "batch_order_seed": int(seed),
        "train_order_sha256": sampler.order_sha256,
        "optimizer": "adamw",
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "temperature": float(temperature) if pseudo_logits is not None else None,
        "planned_updates": planned,
        "updates_per_full_epoch": math.ceil(len(positioned) / min(int(batch_size), len(positioned))),
    }
    return TrainResult(
        metrics=result,
        state_dict={name: value.detach().cpu() for name, value in model.state_dict().items()},
        test_logits=test_logits,
        test_labels=test_labels,
    )
