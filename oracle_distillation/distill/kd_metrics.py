"""KD evaluation / metrics (extracted from the KD engine).

Pure, composable evaluators used after a KD run to characterise the student and
the distillation quality. No training, no persistence, no config objects — each
function takes plain tensors/loaders and returns a metric dict, so global and
personal paths can pick the blocks they need.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from oracle_distillation.distill import losses
from oracle_distillation.utils import (
    ece_np, entropy_mean_np, collect_logits, per_class_accuracy,
)


@torch.no_grad()
def evaluate_distillation_quality(
    *,
    student: torch.nn.Module,
    target_tensor: torch.Tensor,       # [N_proxy, C] on device — FULL proxy
    y_true_proxy: torch.Tensor,        # [N_proxy] int64 on device
    pos_of_global: torch.Tensor,       # [N_dataset] on device
    train_loader: DataLoader,          # IndexedSubset — yields (x, y, gidx)
    val_loader: Optional[DataLoader],  # IndexedSubset or None
    temperature: float,
    device: torch.device,
) -> Dict[str, float]:
    """Distillation diagnostics after training (student in best state).

    The distill_* metrics that characterise target quality and how faithfully
    the student imitated it.
    """
    _nan = float("nan")

    # A) Global target statistics (on the full proxy tensor directly)
    T = temperature
    p_target = torch.softmax(target_tensor / T, dim=1)                  # [N, C]
    log_p    = torch.log(p_target + 1e-12)
    entropy  = -(p_target * log_p).sum(dim=1)                           # [N]
    distill_target_entropy_mean = float(entropy.mean().item())

    gt_mass = p_target[torch.arange(len(p_target), device=device), y_true_proxy]  # [N]
    distill_target_gt_mass_mean = float(gt_mass.mean().item())

    # B) Per-fold statistics: target_acc, student-target agreement, KL final
    def _fold_stats(loader: Optional[DataLoader], suffix: str) -> Dict[str, float]:
        if loader is None:
            return {
                f"distill_target_{suffix}_acc":                _nan,
                f"distill_student_target_agreement_{suffix}":  _nan,
                f"distill_kl_final_{suffix}":                  _nan,
            }
        student.eval()
        n_correct_target = n_agree = 0
        kl_sum = 0.0
        n = 0
        for batch in loader:
            x, y, gidx = batch[0], batch[1], batch[2]
            x    = x.to(device, non_blocking=True)
            y    = y.to(device, non_blocking=True)
            gidx = gidx.to(device, non_blocking=True)

            pos      = pos_of_global[gidx]          # [B] — same lookup as training
            target_b = target_tensor[pos]          # [B, C]

            student_logits = student(x)
            pred_target  = target_b.argmax(dim=1)
            pred_student = student_logits.argmax(dim=1)

            n_correct_target += int((pred_target == y).sum().item())
            n_agree          += int((pred_student == pred_target).sum().item())

            kl = losses.loss_soft_kd(student_logits, target_b, temperature, reduction="mean")
            kl_sum += float(kl.item()) * y.size(0)
            n += y.size(0)

        return {
            f"distill_target_{suffix}_acc":                n_correct_target / max(n, 1),
            f"distill_student_target_agreement_{suffix}":  n_agree          / max(n, 1),
            f"distill_kl_final_{suffix}":                  kl_sum           / max(n, 1),
        }

    train_stats = _fold_stats(train_loader, "train")
    val_stats   = _fold_stats(val_loader,   "val")

    return {
        "distill_target_entropy_mean":  distill_target_entropy_mean,
        "distill_target_gt_mass_mean":  distill_target_gt_mass_mean,
        **train_stats,
        **val_stats,
    }


def eval_per_class_acc(
    model: torch.nn.Module, loader: DataLoader, device: torch.device, num_classes: int,
) -> List[float]:
    """Per-class accuracy on loader. Absent classes get 0.0 (legacy convention)."""
    logits, labels = collect_logits(model, loader, device)
    preds = logits.argmax(axis=1)
    return per_class_accuracy(preds, labels, num_classes, absent_value=0.0).tolist()


def eval_student_extended(
    student: torch.nn.Module,
    test_loader: DataLoader,
    proxy_all_loader: DataLoader,          # full proxy (no train/val split)
    target_tensor: torch.Tensor,           # [N_proxy, C] — precomputed KD targets
    device: torch.device,
    num_classes: int,
    fine_to_coarse: Optional[np.ndarray],  # int64[C] or None
) -> Dict[str, Any]:
    """Extended student metrics for metrics.json (per-class, ECE, superclass, …)."""
    per_class_acc = eval_per_class_acc(student, test_loader, device, num_classes)

    test_logits, test_labels = collect_logits(student, test_loader, device)
    test_ece = ece_np(test_logits, test_labels)
    test_entropy_mean = entropy_mean_np(test_logits)

    proxy_logits, proxy_labels = collect_logits(student, proxy_all_loader, device)
    proxy_pred = proxy_logits.argmax(axis=1)
    proxy_acc  = float((proxy_pred == proxy_labels).mean())

    target_np = target_tensor.cpu().numpy()         # [N_proxy, C]
    target_pred = target_np.argmax(axis=1)
    student_target_agree = float((proxy_pred == target_pred).mean())

    out: Dict[str, Any] = {
        "model_test_acc_per_class":    per_class_acc,
        "model_test_ece":              float(test_ece),
        "model_test_entropy_mean":     float(test_entropy_mean),
        "model_proxy_acc":             float(proxy_acc),
        "distill_student_target_agree": float(student_target_agree),
    }

    # Per-superclass accuracy (CIFAR-100 only)
    if fine_to_coarse is not None:
        n_super = int(fine_to_coarse.max()) + 1
        super_correct = np.zeros(n_super, dtype=np.int64)
        super_total   = np.zeros(n_super, dtype=np.int64)
        fine_intra_correct = np.zeros(n_super, dtype=np.int64)
        fine_intra_total   = np.zeros(n_super, dtype=np.int64)
        test_pred_np  = test_logits.argmax(axis=1)
        test_super_true = fine_to_coarse[test_labels]
        test_super_pred = fine_to_coarse[np.clip(test_pred_np, 0, num_classes - 1)]
        for s in range(n_super):
            mask = test_super_true == s
            super_total[s]   = mask.sum()
            super_correct[s] = (test_super_pred[mask] == s).sum()
            intra_mask = mask & (test_super_pred == s)
            fine_intra_total[s]   = intra_mask.sum()
            fine_intra_correct[s] = (test_pred_np[intra_mask] == test_labels[intra_mask]).sum()
        super_denom = np.maximum(super_total, 1)
        intra_denom = np.maximum(fine_intra_total, 1)
        out["model_super_acc"]            = (super_correct / super_denom).tolist()
        out["model_fine_intra_super_acc"] = (fine_intra_correct / intra_denom).tolist()

    return out
