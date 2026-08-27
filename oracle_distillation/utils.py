from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Collection

import numpy as np
import torch
from torch.utils.data import DataLoader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Reproducibility flags (slower but deterministic)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device(device)


def seed_dir_name(seed: int) -> str:
    """Canonical directory name for a given seed, e.g. 42 → 'seed_42'."""
    return f"seed_{seed}"


def client_label(cid: int) -> str:
    """Canonical zero-padded client label, e.g. 3 → 'c003'."""
    return f"c{cid:03d}"


def client_npz_name(cid: int) -> str:
    """Canonical per-client partition filename, e.g. 3 → 'c003.npz'."""
    return f"{client_label(cid)}.npz"


@torch.no_grad()
def eval_ce_acc(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    ce = torch.nn.CrossEntropyLoss().to(device)
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += float(ce(logits, y).item())
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.size(0))
    return loss_sum / max(1, len(loader)), correct / max(1, total)


@torch.no_grad()
def eval_both_models_ce_acc(
    student: torch.nn.Module,
    finetuned: torch.nn.Module,
    known_classes: Collection[int],
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 10,
) -> Dict[str, Dict[str, Any]]:
    """
    Evalúa tres escenarios sobre el mismo DataLoader:

    1. solo student
    2. solo finetuned
    3. selección dinámica:
       - si y pertenece a known_classes -> finetuned
       - si no -> student

    Devuelve, para cada escenario:
    - loss media por muestra
    - accuracy global
    - accuracy por clase

    Args:
        student: modelo distilled / base
        finetuned: modelo finetuned
        known_classes: clases conocidas por el cliente
        loader: DataLoader con batches (x, y)
        device: dispositivo ('cpu', 'cuda', etc.)
        num_classes: número total de clases

    Returns:
        {
            "student": {
                "loss": float,
                "acc": float,
                "per_class_acc": np.ndarray,
            },
            "finetuned": {
                "loss": float,
                "acc": float,
                "per_class_acc": np.ndarray,
            },
            "dynamic": {
                "loss": float,
                "acc": float,
                "per_class_acc": np.ndarray,
            },
        }
    """

    device = torch.device(device) if isinstance(device, str) else device

    student = student.to(device)
    finetuned = finetuned.to(device)

    student.eval()
    finetuned.eval()

    criterion = torch.nn.CrossEntropyLoss(reduction="sum").to(device)
    known_classes = set(int(c) for c in known_classes)

    student_loss_sum = 0.0
    finetuned_loss_sum = 0.0
    dynamic_loss_sum = 0.0

    student_correct = 0
    finetuned_correct = 0
    dynamic_correct = 0

    total = 0

    student_correct_per_class = torch.zeros(num_classes, dtype=torch.long, device=device)
    student_total_per_class = torch.zeros(num_classes, dtype=torch.long, device=device)

    finetuned_correct_per_class = torch.zeros(num_classes, dtype=torch.long, device=device)
    finetuned_total_per_class = torch.zeros(num_classes, dtype=torch.long, device=device)

    dynamic_correct_per_class = torch.zeros(num_classes, dtype=torch.long, device=device)
    dynamic_total_per_class = torch.zeros(num_classes, dtype=torch.long, device=device)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits_student = student(x)
        logits_finetuned = finetuned(x)

        if logits_student.size(1) != logits_finetuned.size(1):
            raise ValueError(
                "student y finetuned deben producir logits con el mismo número de clases. "
                f"Recibido: {logits_student.size(1)} y {logits_finetuned.size(1)}."
            )

        batch_size = y.size(0)
        total += batch_size

        # --------------------------------------------------
        # SOLO STUDENT
        # --------------------------------------------------
        student_loss_sum += criterion(logits_student, y).item()
        pred_student = logits_student.argmax(dim=1)
        student_correct += (pred_student == y).sum().item()

        student_total_per_class += torch.bincount(y, minlength=num_classes)
        student_correct_labels = y[pred_student == y]
        if student_correct_labels.numel() > 0:
            student_correct_per_class += torch.bincount(
                student_correct_labels, minlength=num_classes
            )

        # --------------------------------------------------
        # SOLO FINETUNED
        # --------------------------------------------------
        finetuned_loss_sum += criterion(logits_finetuned, y).item()
        pred_finetuned = logits_finetuned.argmax(dim=1)
        finetuned_correct += (pred_finetuned == y).sum().item()

        finetuned_total_per_class += torch.bincount(y, minlength=num_classes)
        finetuned_correct_labels = y[pred_finetuned == y]
        if finetuned_correct_labels.numel() > 0:
            finetuned_correct_per_class += torch.bincount(
                finetuned_correct_labels, minlength=num_classes
            )

        # --------------------------------------------------
        # SELECCIÓN DINÁMICA
        # --------------------------------------------------
        use_finetuned = torch.tensor(
            [int(label) in known_classes for label in y.tolist()],
            device=device,
            dtype=torch.bool,
        )

        logits_dynamic = logits_student.clone()
        if use_finetuned.any():
            logits_dynamic[use_finetuned] = logits_finetuned[use_finetuned]

        dynamic_loss_sum += criterion(logits_dynamic, y).item()
        pred_dynamic = logits_dynamic.argmax(dim=1)
        dynamic_correct += (pred_dynamic == y).sum().item()

        dynamic_total_per_class += torch.bincount(y, minlength=num_classes)
        dynamic_correct_labels = y[pred_dynamic == y]
        if dynamic_correct_labels.numel() > 0:
            dynamic_correct_per_class += torch.bincount(
                dynamic_correct_labels, minlength=num_classes
            )

    total = max(1, total)

    def build_per_class_acc(correct_pc: torch.Tensor, total_pc: torch.Tensor) -> np.ndarray:
        correct_np = correct_pc.detach().cpu().numpy()
        total_np = total_pc.detach().cpu().numpy()

        per_class_acc = np.full(num_classes, np.nan, dtype=float)
        valid = total_np > 0
        per_class_acc[valid] = correct_np[valid] / total_np[valid]
        return per_class_acc

    return {
        "student": {
            "loss": float(student_loss_sum / total),
            "acc": float(student_correct / total),
            "per_class_acc": build_per_class_acc(
                student_correct_per_class,
                student_total_per_class,
            ),
        },
        "finetuned": {
            "loss": float(finetuned_loss_sum / total),
            "acc": float(finetuned_correct / total),
            "per_class_acc": build_per_class_acc(
                finetuned_correct_per_class,
                finetuned_total_per_class,
            ),
        },
        "dynamic": {
            "loss": float(dynamic_loss_sum / total),
            "acc": float(dynamic_correct / total),
            "per_class_acc": build_per_class_acc(
                dynamic_correct_per_class,
                dynamic_total_per_class,
            ),
        },
    }


def write_csv(rows: Iterable[Dict[str, Any]], path: Path, fieldnames: Optional[List[str]] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_list = list(rows)
    if not rows_list:
        return
    if fieldnames is None:
        fieldnames = sorted({k for row in rows_list for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_list)


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Shared numeric utilities (softmax, ECE, entropy)
# ---------------------------------------------------------------------------

def softmax_np(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    """Numerically stable (temperature-scaled) softmax over the last axis.

    With T=1.0 this is the plain softmax. T>1 softens the distribution; this is
    the single implementation shared by the KD target builders (which previously
    used a local ``_stable_softmax``). The exp argument always has a maximum of
    0 after the max-subtraction, so the denominator is >= 1 and no epsilon is
    needed.
    """
    x = logits.astype(np.float32) / float(T)
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def ece_np(logits: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """15-bin equal-width Expected Calibration Error."""
    probs = softmax_np(logits)
    conf  = probs.max(axis=1)
    pred  = probs.argmax(axis=1)
    acc   = (pred == labels).astype(np.float32)
    bins  = np.linspace(0.0, 1.0, n_bins + 1)
    ece   = 0.0
    n     = len(labels)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi)
        if not mask.any():
            continue
        ece += mask.sum() / n * abs(acc[mask].mean() - conf[mask].mean())
    return float(ece)


def entropy_mean_np(logits: np.ndarray) -> float:
    """Mean Shannon entropy H = -Σ p·log(p+ε) over samples."""
    p = softmax_np(logits)
    return float(-(p * np.log(p + 1e-12)).sum(axis=1).mean())


@torch.inference_mode()
def collect_logits(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run ``model`` over ``loader`` and return (logits[N,C] fp32, labels[N] int64).

    The loader is expected to yield (x, y, ...) tuples; only the first two
    elements are used. Single implementation shared by the KD runner and the
    teacher diagnostics (previously ``_collect_logits`` / ``_eval_logits``).
    """
    model.eval()
    logits_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    for batch in loader:
        x, y = batch[0], batch[1]
        out = model(x.to(device)).cpu().numpy()
        logits_list.append(out)
        labels_list.append(np.asarray(y))
    return (
        np.concatenate(logits_list, axis=0).astype(np.float32),
        np.concatenate(labels_list, axis=0).astype(np.int64),
    )


@torch.inference_mode()
def forward_logits_for_indices(
    model: torch.nn.Module,
    ds,
    idx: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
    num_workers: int = 2,
) -> np.ndarray:
    """[len(idx), C] logits of ``model`` over ``Subset(ds, idx)``, in ``idx`` order
    (``shuffle=False``) — the convention every cached-logits array in this codebase
    relies on (``teacher_logits_cache``, ``student_logits.npz``, ...).
    """
    from torch.utils.data import DataLoader, Subset
    loader = DataLoader(
        Subset(ds, [int(i) for i in idx]), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    model.eval()
    out: List[np.ndarray] = []
    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        out.append(model(x).float().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def per_class_accuracy(
    preds: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    absent_value: float = float("nan"),
) -> np.ndarray:
    """Per-class accuracy array of length ``num_classes``.

    Classes absent from ``labels`` get ``absent_value`` (NaN by default; pass
    0.0 to reproduce the runner's previous convention).
    """
    acc = np.full(num_classes, absent_value, dtype=np.float32)
    for c in range(num_classes):
        mask = labels == c
        if mask.any():
            acc[c] = float((preds[mask] == c).mean())
    return acc
