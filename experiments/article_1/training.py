"""One small paired student-training loop for Article 1."""
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

from oracle_distillation.checkpoints import state_dict_sha256
from oracle_distillation.models import build_model
from oracle_distillation.utils import set_seed


class IndexedProxy(Dataset):
    def __init__(self, dataset, indices: np.ndarray, labels: np.ndarray):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        if self.indices.shape != self.labels.shape:
            raise ValueError("proxy indices and labels must align")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        image, _ = self.dataset[int(self.indices[position])]
        return image, int(position), int(self.labels[position])


class PairedSampler(Sampler[int]):
    def __init__(self, size: int, seed: int):
        self.size = int(size)
        self.seed = int(seed)
        self.epoch = 0
        self.digest = hashlib.sha256()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        order = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed + self.epoch))
        self.digest.update(self.epoch.to_bytes(8, "little"))
        self.digest.update(order.numpy().tobytes())
        return iter(order.tolist())

    def __len__(self) -> int:
        return self.size


def initial_state(arch: str, classes: int, seed: int) -> tuple[dict, str]:
    set_seed(seed)
    model = build_model(arch, num_classes=classes)
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    return state, state_dict_sha256(state)


@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total = correct = 0
    nll = 0.0
    for images, labels in loader:
        labels = labels.to(device)
        logits = model(images.to(device))
        nll += float(F.cross_entropy(logits, labels, reduction="sum").item())
        correct += int((logits.argmax(1) == labels).sum().item())
        total += len(labels)
    return correct / max(total, 1), nll / max(total, 1)


def train_student(
    *,
    arch: str,
    classes: int,
    init_state: dict,
    init_hash: str,
    train_dataset,
    test_loader,
    proxy_indices: np.ndarray,
    labels: np.ndarray,
    target_probabilities: np.ndarray,
    hard_labels: bool,
    temperature: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    init_seed: int,
    order_seed: int,
    device: torch.device,
    cosine: bool = False,
) -> dict:
    """Train once and return only result/provenance metrics, not a checkpoint."""
    indices = np.asarray(proxy_indices, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    targets = np.asarray(target_probabilities, dtype=np.float32)
    if targets.shape != (len(indices), classes):
        raise ValueError("target probabilities do not align with the proxy subset")
    set_seed(init_seed)
    model = build_model(arch, num_classes=classes).to(device)
    model.load_state_dict(copy.deepcopy(init_state), strict=True)
    sampler = PairedSampler(len(indices), order_seed)
    loader = DataLoader(
        IndexedProxy(train_dataset, indices, labels),
        batch_size=min(batch_size, len(indices)), sampler=sampler, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    updates = int(epochs) * len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates) if cosine else None
    soft_targets = torch.from_numpy(targets).to(device)
    started = perf_counter()
    last_loss = float("nan")
    steps = 0
    for epoch in range(int(epochs)):
        sampler.set_epoch(epoch)
        model.train()
        for images, positions, batch_labels in loader:
            logits = model(images.to(device))
            if hard_labels:
                loss = F.cross_entropy(logits, batch_labels.to(device))
            else:
                batch_targets = soft_targets.index_select(0, positions.to(device))
                loss = F.kl_div(
                    F.log_softmax(logits / temperature, dim=1),
                    batch_targets,
                    reduction="batchmean",
                ) * (temperature * temperature)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            steps += 1
            last_loss = float(loss.detach().item())
    test_accuracy, test_nll = evaluate(model, test_loader, device)
    return {
        "student_test_accuracy": float(test_accuracy),
        "student_test_nll": float(test_nll),
        "student_final_train_loss": last_loss,
        "student_elapsed_seconds": perf_counter() - started,
        "student_final_sha256": state_dict_sha256(model.state_dict()),
        "student_init_sha256": init_hash,
        "student_init_seed": int(init_seed),
        "batch_order_seed": int(order_seed),
        "train_order_sha256": sampler.digest.hexdigest(),
        "epochs": int(epochs),
        "total_updates": int(steps),
        "updates_per_epoch": int(math.ceil(len(indices) / min(batch_size, len(indices)))),
        "batch_size": int(min(batch_size, len(indices))),
        "optimizer": "adamw",
        "scheduler": "cosine_per_update" if cosine else "none",
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "loss": "cross_entropy" if hard_labels else "kd_kl",
        "temperature": None if hard_labels else float(temperature),
    }

