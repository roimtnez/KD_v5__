"""Probability-space expertise targets for server-side knowledge distillation.

This module deliberately has no CLI, dataset, filesystem, or article dependency.
It consumes teacher logits, labels, and teacher--class authority and returns a
normalized probability target plus diagnostics.  A probability target can be
fed to the repository's existing logit-based KD loss through ``T * log(q)``;
softmaxing those pseudo-logits at temperature ``T`` recovers ``q``.

The authority matrix may be binary today or non-negative/soft in a future
label-efficient protocol.  A teacher with zero total authority is rejected by
default: silently replacing its invalid supported distribution would change the
scientific estimand.  Callers may explicitly request ``zero_support="full"`` for
a separately named ablation, but Article 1 uses the fail-closed default.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import numpy as np

EPS = 1e-12
ARTICLE1_METHODS = (
    "feddf",
    "expert_full",
    "expert_support",
    "oracle_full",
    "oracle_maskgated_full",
    "oracle_maskgated_support",
    "feddf_support_only",
)

# The four arms of Article 1, Experiment 2.  Keep the historical
# ``feddf_support_only`` spelling as an accepted compatibility alias, but use
# the explicit name below in new artifacts: it says that *all* teachers remain
# selected while only their output support changes.
ARTICLE1_EXPERIMENT2_METHODS = (
    "feddf",
    "all_teachers_support",
    "expert_full",
    "expert_support",
)
_SUPPORTED_METHODS = ARTICLE1_METHODS + ("all_teachers_support",)


@dataclass(frozen=True)
class TargetBuildResult:
    """One normalized target and its selection/mechanism diagnostics."""

    probabilities: np.ndarray
    pseudo_logits: np.ndarray
    selected_teachers: np.ndarray
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class ExpertiseEstimateResult:
    """Teacher--class competence estimated on an external calibration set."""

    authority: np.ndarray
    posterior_mean: np.ndarray
    class_counts: np.ndarray
    correct_counts: np.ndarray
    diagnostics: Mapping[str, Any]


def estimate_expertise_from_logits(
    teacher_logits: np.ndarray,
    labels: np.ndarray,
    *,
    mode: Literal["soft", "hard"] = "soft",
    threshold: float = 0.7,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    min_class_examples: int = 5,
) -> ExpertiseEstimateResult:
    """Estimate authority without using proxy or test labels.

    The caller is responsible for supplying a *disjoint, declared competence
    calibration set*.  This function deliberately only receives logits and
    labels so it cannot accidentally select an evaluation split by name.
    ``soft`` returns a Beta-posterior mean; ``hard`` thresholds that mean and
    requires at least ``min_class_examples`` observations for the class.
    """
    logits, y, _ = _validate_inputs(teacher_logits, labels, None)
    if mode not in {"soft", "hard"}:
        raise ValueError("mode must be 'soft' or 'hard'")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0,1]")
    if prior_alpha <= 0 or prior_beta <= 0:
        raise ValueError("Beta prior parameters must be positive")
    if min_class_examples < 1:
        raise ValueError("min_class_examples must be positive")

    _, k, c = logits.shape
    counts = np.bincount(y, minlength=c).astype(np.int64)
    correct = np.zeros((k, c), dtype=np.int64)
    predictions = logits.argmax(axis=2)
    for cls in range(c):
        rows = y == cls
        if rows.any():
            correct[:, cls] = (predictions[rows] == cls).sum(axis=0)
    posterior = (correct + float(prior_alpha)) / (
        counts[None, :] + float(prior_alpha) + float(prior_beta)
    )
    if mode == "soft":
        authority = posterior
    else:
        authority = ((posterior >= threshold) & (counts[None, :] >= min_class_examples)).astype(np.float64)
    diagnostics = {
        "authority_source": f"estimated_{mode}",
        "estimator": "beta_binomial_posterior_mean",
        "threshold": float(threshold),
        "prior_alpha": float(prior_alpha),
        "prior_beta": float(prior_beta),
        "min_class_examples": int(min_class_examples),
        "calibration_examples": int(len(y)),
        "calibration_class_counts": counts.astype(int).tolist(),
        "zero_authority_teachers": int((authority.sum(axis=1) == 0).sum()),
    }
    return ExpertiseEstimateResult(
        authority=authority.astype(np.float32),
        posterior_mean=posterior.astype(np.float32),
        class_counts=counts,
        correct_counts=correct,
        diagnostics=diagnostics,
    )


def _validate_inputs(
    teacher_logits: np.ndarray,
    labels: np.ndarray,
    authority: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    logits = np.asarray(teacher_logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if logits.ndim != 3 or min(logits.shape) <= 0:
        raise ValueError("teacher_logits must have non-empty shape [N,K,C]")
    if not np.isfinite(logits).all():
        raise ValueError("teacher_logits must be finite")
    if y.shape != (logits.shape[0],):
        raise ValueError("labels must have shape [N]")
    if (y < 0).any() or (y >= logits.shape[2]).any():
        raise ValueError("labels must be in [0,C)")
    if authority is None:
        return logits, y, None
    weights = np.asarray(authority, dtype=np.float64)
    if weights.shape != logits.shape[1:]:
        raise ValueError(f"authority must have shape [K,C]={logits.shape[1:]}")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("authority must be finite and non-negative")
    return logits, y, weights


def temperature_softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Stable softmax over the last axis at a strictly positive temperature."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    if z.ndim < 1 or not np.isfinite(z).all():
        raise ValueError("logits must be a finite array")
    z = z - z.max(axis=-1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=-1, keepdims=True)


def apply_expertise_support(
    probabilities: np.ndarray,
    authority: np.ndarray,
    *,
    zero_support: Literal["error", "full"] = "error",
) -> tuple[np.ndarray, np.ndarray]:
    """Apply teacher-specific authority before aggregation and renormalize.

    Args:
        probabilities: ``[..., K, C]`` normalized teacher distributions.
        authority: non-negative ``[K,C]`` authority. Binary masks implement the
            Article-1 definition exactly; soft values implement ``A * p``.
        zero_support: explicit policy for teachers whose authority row sums to
            zero. ``error`` is the Article-1 policy and prevents NaNs.

    Returns:
        ``(supported_probabilities, retained_mass)`` where retained mass has
        shape ``[...,K]`` and is measured before renormalization.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    a = np.asarray(authority, dtype=np.float64)
    if p.ndim < 2 or a.shape != p.shape[-2:]:
        raise ValueError("probabilities [...,K,C] and authority [K,C] are required")
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    if not np.allclose(p.sum(axis=-1), 1.0, atol=1e-7, rtol=1e-7):
        raise ValueError("probabilities must sum to one over classes")
    if not np.isfinite(a).all() or (a < 0).any():
        raise ValueError("authority must be finite and non-negative")
    empty = a.sum(axis=1) <= 0
    if empty.any() and zero_support == "error":
        ids = np.flatnonzero(empty).tolist()
        raise ValueError(f"teachers with zero expertise support: {ids}")
    if zero_support not in {"error", "full"}:
        raise ValueError("zero_support must be 'error' or 'full'")
    effective = a.copy()
    if empty.any():
        effective[empty] = 1.0
    masked = p * effective
    retained = masked.sum(axis=-1)
    if (retained <= 0).any():
        raise ValueError("authority/probability product has zero mass")
    supported = masked / retained[..., None]
    return supported, retained


def _historical_fallback(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Study-I empty-selection fallback: softmax(mean teacher logits / T)."""
    return temperature_softmax(logits.mean(axis=1), temperature)


def _selection(
    method: str,
    logits: np.ndarray,
    labels: np.ndarray,
    authority: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    n, k, _ = logits.shape
    all_teachers = np.ones((n, k), dtype=bool)
    if method in {"feddf", "all_teachers_support", "feddf_support_only"}:
        return all_teachers, None
    predicted_correct = logits.argmax(axis=2) == labels[:, None]
    if method == "oracle_full":
        return predicted_correct, predicted_correct
    if authority is None:
        raise ValueError(f"{method} requires authority [K,C]")
    knows_true = authority[:, labels].T > 0
    if method in {"expert_full", "expert_support"}:
        return knows_true, knows_true
    if method in {"oracle_maskgated_full", "oracle_maskgated_support"}:
        return predicted_correct & knows_true, predicted_correct
    raise ValueError(f"unknown method {method!r}; expected one of {_SUPPORTED_METHODS}")


def _nan_mechanism_metrics() -> dict[str, float]:
    return {
        "mean_teacher_mass_removed_by_support": float("nan"),
        "mean_target_mass_removed_before_renormalization": float("nan"),
        "fraction_selected_teachers_with_partial_support": float("nan"),
        "mean_support_size": float("nan"),
    }


def build_server_expertise_target(
    teacher_logits: np.ndarray,
    labels: np.ndarray,
    authority: np.ndarray | None,
    *,
    method: str,
    temperature: float = 8.0,
    zero_support: Literal["error", "full"] = "error",
) -> TargetBuildResult:
    """Build one Article-1 target while preserving the Study-I empty fallback.

    ``expert_full`` and ``expert_support`` share the exact same selection matrix.
    Likewise, the two mask-gated ORACLE variants share their selection matrix.
    Support is always applied per teacher before the selected teachers are
    averaged.  ``oracle_full`` does not require an authority matrix.
    """
    if method not in _SUPPORTED_METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {_SUPPORTED_METHODS}")
    logits, y, weights = _validate_inputs(teacher_logits, labels, authority)
    full = temperature_softmax(logits, temperature)
    selected, pre_gate = _selection(method, logits, y, weights)
    counts = selected.sum(axis=1)
    fallback = counts == 0
    fallback_probs = _historical_fallback(logits, temperature)

    uses_support = method in {
        "all_teachers_support", "expert_support", "oracle_maskgated_support",
        "feddf_support_only",
    }
    supported = retained = None
    if weights is not None:
        supported, retained = apply_expertise_support(
            full, weights, zero_support=zero_support,
        )
    if uses_support and supported is None:
        raise ValueError(f"{method} requires authority [K,C]")
    teacher_distributions = supported if uses_support else full
    summed = (teacher_distributions * selected[..., None]).sum(axis=1)
    averaged = summed / np.maximum(counts[:, None], 1)
    # FedDF is retained as the Study-I reference: mean in logit space,
    # followed by the KD temperature softmax. The expertise comparisons use
    # the probability-space equations stated in the Article-1 protocol.
    if method == "feddf":
        averaged = fallback_probs
    target = np.where(fallback[:, None], fallback_probs, averaged)
    target = target / target.sum(axis=1, keepdims=True)
    if not np.isfinite(target).all() or (target < 0).any():
        raise AssertionError("constructed target is not a finite probability distribution")

    diag: dict[str, Any] = {
        "method": method,
        "target_semantics": "support" if uses_support else "full",
        "fallback_policy": "historical_feddf_logit_mean",
        "fallback_count": int(fallback.sum()),
        "fallback_rate": float(fallback.mean()),
        "mean_selected_teachers": float(counts.mean()),
        "min_selected_teachers": int(counts.min()),
        "max_selected_teachers": int(counts.max()),
    }
    if pre_gate is None:
        diag.update({
            "fallback_before_gate_count": 0,
            "fallback_after_new_gate_count": 0,
        })
    else:
        pre_empty = pre_gate.sum(axis=1) == 0
        diag.update({
            "fallback_before_gate_count": int(pre_empty.sum()),
            "fallback_after_new_gate_count": int((fallback & ~pre_empty).sum()),
        })

    if weights is None or retained is None:
        diag.update(_nan_mechanism_metrics())
    else:
        event = selected
        valid_rows = ~fallback
        support_size = (weights > 0).sum(axis=1)
        if event.any():
            diag["mean_teacher_mass_removed_by_support"] = float((1.0 - retained)[event].mean())
            diag["fraction_selected_teachers_with_partial_support"] = float(
                ((support_size[None, :] < logits.shape[2]) & event).sum() / event.sum()
            )
            diag["mean_support_size"] = float(
                np.broadcast_to(support_size[None, :], event.shape)[event].mean()
            )
        else:
            diag.update(_nan_mechanism_metrics())
        if valid_rows.any():
            # Mean mass of the selected, masked mixture before its row-wise
            # renormalization. This sample-weighted quantity differs from the
            # event-weighted teacher metric when selection cardinality varies.
            retained_target = (
                retained * selected
            ).sum(axis=1) / np.maximum(counts, 1)
            diag["mean_target_mass_removed_before_renormalization"] = float(
                (1.0 - retained_target[valid_rows]).mean()
            )

    pseudo_logits = (float(temperature) * np.log(np.clip(target, EPS, 1.0))).astype(np.float32)
    return TargetBuildResult(
        probabilities=target.astype(np.float32),
        pseudo_logits=pseudo_logits,
        selected_teachers=selected,
        diagnostics=diag,
    )


def target_quality_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Proper-score and structural diagnostics for probabilistic KD targets."""
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if p.ndim != 2 or y.shape != (len(p),):
        raise ValueError("probabilities [N,C] and labels [N] are required")
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1.0):
        raise ValueError("probabilities must be finite, non-negative, and normalized")
    true_p = p[np.arange(len(y)), y]
    wrong = p.copy()
    wrong[np.arange(len(y)), y] = 0.0
    max_wrong = wrong.max(axis=1)
    one_hot = np.zeros_like(p)
    one_hot[np.arange(len(y)), y] = 1.0
    return {
        "target_argmax_accuracy": float((p.argmax(axis=1) == y).mean()),
        "target_nll": float(-np.log(np.clip(true_p, EPS, 1.0)).mean()),
        "target_brier": float(np.square(p - one_hot).sum(axis=1).mean()),
        "target_entropy": float(-(p * np.log(np.clip(p, EPS, 1.0))).sum(axis=1).mean()),
        "mean_probability_true_class": float(true_p.mean()),
        "true_class_probability": float(true_p.mean()),
        "median_probability_true_class": float(np.median(true_p)),
        "mean_max_wrong_class_probability": float(max_wrong.mean()),
        "mean_true_class_margin": float((true_p - max_wrong).mean()),
        "target_wrong_argmax_rate": float((p.argmax(axis=1) != y).mean()),
    }


def paired_target_quality_metrics(
    probabilities: np.ndarray,
    reference_probabilities: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """Measure a target transformation relative to the same FedDF target."""
    p = np.asarray(probabilities, dtype=np.float64)
    reference = np.asarray(reference_probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if p.shape != reference.shape or p.ndim != 2 or y.shape != (len(p),):
        raise ValueError("candidate/reference targets [N,C] and labels [N] are required")
    for name, value in (("probabilities", p), ("reference_probabilities", reference)):
        if (not np.isfinite(value).all() or (value < 0).any()
                or not np.allclose(value.sum(axis=1), 1.0, atol=1e-7, rtol=1e-7)):
            raise ValueError(f"{name} must be finite normalized probabilities")
    rows = np.arange(len(y))
    delta_true = p[rows, y] - reference[rows, y]
    midpoint = 0.5 * (p + reference)
    kl_p_ref = (p * (np.log(np.clip(p, EPS, 1.0)) - np.log(np.clip(reference, EPS, 1.0)))).sum(axis=1)
    js = 0.5 * (
        (p * (np.log(np.clip(p, EPS, 1.0)) - np.log(np.clip(midpoint, EPS, 1.0)))).sum(axis=1)
        + (reference * (np.log(np.clip(reference, EPS, 1.0)) - np.log(np.clip(midpoint, EPS, 1.0)))).sum(axis=1)
    )
    return {
        "mean_true_class_probability_delta_vs_feddf": float(delta_true.mean()),
        "fraction_true_class_probability_increased_vs_feddf": float((delta_true > 0).mean()),
        "mean_l1_distance_vs_feddf": float(np.abs(p - reference).sum(axis=1).mean()),
        "mean_kl_target_to_feddf": float(kl_p_ref.mean()),
        "mean_js_target_vs_feddf": float(js.mean()),
    }
