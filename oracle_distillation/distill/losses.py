from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

def loss_soft_kd(
    student_logits: torch.Tensor,
    teacher_avg_logits: torch.Tensor,
    temperature: float,
    reduction: str = "mean",
) -> torch.Tensor:
    log_pred = F.log_softmax(student_logits / temperature, dim=1)
    soft_targets = F.softmax(teacher_avg_logits / temperature, dim=1)
    kl = F.kl_div(log_pred, soft_targets, reduction="none").sum(dim=1)
    scale = temperature * temperature
    if reduction == "none":
        return kl * scale
    if reduction == "mean":
        return kl.mean() * scale
    raise ValueError(f"Unsupported reduction: {reduction}")


def loss_hard_labels(student_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(student_logits, targets)
