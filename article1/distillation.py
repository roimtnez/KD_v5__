"""Target construction and KD loss for Article 1.

All methods return ``q[N,C]``.  The common logit operator is
``softmax(sum_k w_k z_k / T)``; weights are scalar per teacher and sample.
EXPERT and ORACLE are supervised proxy routing rules, not label-free methods.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

METHODS = (
    "feddf_logit", "confidence_logit", "consensus_logit", "energy_logit",
    "expert_logit", "oracle_logit", "expert_prob", "expert_prob_sr", "oracle_prob",
)
EPS = 1e-12


@dataclass(frozen=True)
class Target:
    probabilities: np.ndarray
    weights: np.ndarray
    selected: np.ndarray
    fallback: np.ndarray
    metrics: dict[str, float | int | str | None]


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable last-axis softmax at a positive temperature."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim < 1 or not np.isfinite(z).all():
        raise ValueError("logits must be finite")
    z = z / float(temperature)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def authority_from_holdout(
    accuracy: np.ndarray, counts: np.ndarray, threshold: float,
) -> np.ndarray:
    """Return M[k,c]; zero observations can never establish expertise."""
    accuracy = np.asarray(accuracy, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.int64)
    if accuracy.ndim != 2 or accuracy.shape != counts.shape or (counts < 0).any():
        raise ValueError("holdout accuracy/counts must be aligned [K,C]")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0,1]")
    return ((counts > 0) & (accuracy >= threshold)).astype(np.uint8)


def _validate(logits: np.ndarray, labels: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if z.ndim != 3 or min(z.shape) <= 0 or not np.isfinite(z).all():
        raise ValueError("logits must be finite [N,K,C]")
    if y.shape != (z.shape[0],) or (y < 0).any() or (y >= z.shape[2]).any():
        raise ValueError("labels must be [N] and in the class range")
    if mask is None:
        return z, y, None
    m = np.asarray(mask, dtype=np.uint8)
    if m.shape != z.shape[1:] or not np.isin(m, (0, 1)).all():
        raise ValueError("M must be binary [K,C]")
    return z, y, m


def _teacher_softmax(values: np.ndarray) -> np.ndarray:
    values = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(values)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def _routing(method: str, z: np.ndarray, y: np.ndarray, m: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, str]:
    """Return scalar teacher weights and a selection indicator before fallback."""
    n, k, _ = z.shape
    all_selected = np.ones((n, k), dtype=bool)
    if method == "feddf_logit":
        return np.full((n, k), 1.0 / k), all_selected, "uniform"
    if method == "confidence_logit":
        # T_weight=tau=1; this is deliberately not MSP/sum(MSP).
        msp = softmax(z, 1.0).max(axis=2)
        return _teacher_softmax(msp), all_selected, "softmax_teachers(MSP)"
    if method == "energy_logit":
        # negative energy = logsumexp(z), T_weight=tau=1.
        max_z = z.max(axis=2, keepdims=True)
        score = (max_z[..., 0] + np.log(np.exp(z - max_z).sum(axis=2)))
        return _teacher_softmax(score), all_selected, "softmax_teachers(logsumexp(logits))"
    if method == "consensus_logit":
        vote = softmax(z, 1.0).mean(axis=1).argmax(axis=1)
        selected = z.argmax(axis=2) == vote[:, None]
    elif method.startswith("expert"):
        if m is None:
            raise ValueError(f"{method} requires M estimated from holdout")
        selected = m[:, y].T.astype(bool)
    elif method.startswith("oracle"):
        selected = z.argmax(axis=2) == y[:, None]
    else:  # pragma: no cover - public input is checked by build_target
        raise ValueError(f"unknown method {method}")
    weights = selected.astype(np.float64)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1.0)
    return weights, selected, "selected_uniform"


def _fallback(z: np.ndarray, temperature: float) -> np.ndarray:
    return softmax(z.mean(axis=1), temperature)


def build_target(
    logits: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray | None,
    *, method: str,
    temperature: float = 8.0,
) -> Target:
    """Build one target with the protocol's common empty-selection fallback.

    ``expert_prob_sr`` masks each selected teacher's *probability* distribution,
    then renormalizes it.  A zero-M row is safe when that teacher is unselected;
    selecting it is impossible because it lacks M[k,y].
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    z, y, m = _validate(logits, labels, mask)
    weights, selected, weight_rule = _routing(method, z, y, m)
    fallback = selected.sum(axis=1) == 0
    fallback_q = _fallback(z, temperature)
    if method.endswith("_logit"):
        q = softmax((weights[..., None] * z).sum(axis=1), temperature)
        aggregation = "weighted_mean_logits"
    else:
        teacher_p = softmax(z, temperature)
        outside_mass: np.ndarray | None = None
        if method == "expert_prob_sr":
            assert m is not None
            masked = teacher_p * m[None, :, :]
            retained = masked.sum(axis=2)
            # Only selected values are used; selected teachers necessarily have
            # a true-label support bit and thus strictly positive retained mass.
            valid = retained > EPS
            teacher_p = np.zeros_like(teacher_p)
            teacher_p[valid] = masked[valid] / retained[valid, None]
            outside_mass = 1.0 - retained
        q = (weights[..., None] * teacher_p).sum(axis=1)
        aggregation = "mean_probabilities" if method != "expert_prob_sr" else "mean_support_restricted_probabilities"
    q[fallback] = fallback_q[fallback]
    # Report a normalized effective weight vector even on fallback rows.  The
    # fallback itself remains an explicit logit-mean exception for probability
    # arms, but its teacher weights are the FedDF-uniform weights.
    weights[fallback] = 1.0 / z.shape[1]
    q /= q.sum(axis=1, keepdims=True)
    if not np.isfinite(q).all() or (q < 0).any():
        raise AssertionError("target is not a finite normalized distribution")
    counts = selected.sum(axis=1)
    metrics: dict[str, float | int | str | None] = {
        "method": method, "aggregation": aggregation, "weight_rule": weight_rule,
        "target_accuracy": float((q.argmax(axis=1) == y).mean()),
        "target_nll": float(-np.log(np.clip(q[np.arange(len(y)), y], EPS, 1.0)).mean()),
        "target_entropy": float(-(q * np.log(np.clip(q, EPS, 1.0))).sum(axis=1).mean()),
        "mean_selected_teachers": float(counts.mean()),
        "fallback_count": int(fallback.sum()), "fallback_rate": float(fallback.mean()),
        "pre_restriction_outside_support_mass": None,
        "effective_teachers": float((1.0 / np.square(weights).sum(axis=1)).mean()),
    }
    if method in {"expert_prob", "expert_prob_sr"}:
        assert m is not None
        # This is computed from each selected teacher's full distribution before
        # restriction, excluding fallback. It is a property of the condition,
        # not an effect created by the SR output.
        full = softmax(z, temperature)
        outside_mass = 1.0 - (full * m[None, :, :]).sum(axis=2)
        event = selected & ~fallback[:, None]
        metrics["pre_restriction_outside_support_mass"] = float(outside_mass[event].mean()) if event.any() else None
    return Target(q.astype(np.float32), weights.astype(np.float32), selected, fallback, metrics)


def metadata_identity(*, method: str, temperature: float, config: dict, source_hash: str, proxy_hash: str, mask_hash: str) -> str:
    """Compact cache/result identity; callers persist this alongside outputs."""
    import hashlib
    import json
    payload = {"method": method, "temperature": temperature, "config": config,
               "source_hash": source_hash, "proxy_hash": proxy_hash, "mask_hash": mask_hash}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def kd_loss(student_logits, q, temperature: float):
    """``T² KL(q || softmax(student_logits/T))``; q is already probabilities."""
    import torch.nn.functional as F
    return F.kl_div(F.log_softmax(student_logits / temperature, dim=1), q, reduction="batchmean") * temperature**2
