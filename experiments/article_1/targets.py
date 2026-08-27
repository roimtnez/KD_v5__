"""Pure target construction and hypothesis diagnostics for Article 1."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.article_1.protocol import ALL_METHODS


EPS = 1e-12
# Energy is a mask-free, sample-adaptive reference.  These are intentionally
# fixed here (rather than tuned per dataset) to preserve a single comparison.
ENERGY_WEIGHT_TEMPERATURE = 1.0
ENERGY_SELECTION_TEMPERATURE = 1.0


@dataclass(frozen=True)
class TargetResult:
    probabilities: np.ndarray
    selected_teachers: np.ndarray
    metrics: dict


def softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive")
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    if z.ndim < 2 or not np.isfinite(z).all():
        raise ValueError("logits must be finite")
    z -= z.max(axis=-1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=-1, keepdims=True)


def nested_balanced_order(labels: np.ndarray, seed: int) -> np.ndarray:
    """Return one class-balanced order whose prefixes define every proxy size."""
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or len(labels) == 0 or (labels < 0).any():
        raise ValueError("labels must be a non-empty non-negative vector")
    classes = int(labels.max()) + 1
    rng = np.random.default_rng(int(seed))
    queues = [rng.permutation(np.flatnonzero(labels == cls)).tolist() for cls in range(classes)]
    if any(not values for values in queues):
        raise ValueError("the proxy source must contain every class")
    order: list[int] = []
    while any(queues):
        for cls in rng.permutation(classes):
            if queues[int(cls)]:
                order.append(queues[int(cls)].pop())
    result = np.asarray(order, dtype=np.int64)
    if len(result) != len(labels) or len(np.unique(result)) != len(labels):
        raise AssertionError("nested proxy order is not a permutation")
    return result


def authority_from_holdout(
    holdout_accuracy: np.ndarray,
    holdout_counts: np.ndarray,
    threshold: float,
) -> np.ndarray:
    accuracy = np.asarray(holdout_accuracy, dtype=np.float64)
    counts = np.asarray(holdout_counts, dtype=np.int64)
    if accuracy.shape != counts.shape or accuracy.ndim != 2:
        raise ValueError("holdout accuracy/counts must have shape [K,C]")
    if not 0 <= threshold <= 1 or (counts < 0).any():
        raise ValueError("invalid threshold or holdout counts")
    return ((accuracy >= threshold) & (counts > 0)).astype(np.uint8)


def _validate(
    teacher_logits: np.ndarray,
    labels: np.ndarray,
    authority: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits = np.asarray(teacher_logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    authority = np.asarray(authority, dtype=np.uint8)
    if logits.ndim != 3 or min(logits.shape) <= 0:
        raise ValueError("teacher logits must have shape [N,K,C]")
    if labels.shape != (len(logits),) or authority.shape != logits.shape[1:]:
        raise ValueError("labels [N] and authority [K,C] must align with logits")
    if not np.isfinite(logits).all() or not np.isin(authority, (0, 1)).all():
        raise ValueError("teacher logits/authority are invalid")
    if (labels < 0).any() or (labels >= logits.shape[2]).any():
        raise ValueError("labels outside class range")
    return logits, labels, authority


def _quality(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    p = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if p.ndim != 2 or labels.shape != (len(p),) or not np.allclose(p.sum(1), 1.0):
        raise ValueError("target probabilities must be normalized [N,C]")
    rows = np.arange(len(labels))
    true_probability = p[rows, labels]
    one_hot = np.zeros_like(p)
    one_hot[rows, labels] = 1.0
    return {
        "target_accuracy": float((p.argmax(1) == labels).mean()),
        "target_nll": float(-np.log(np.clip(true_probability, EPS, 1.0)).mean()),
        "target_brier": float(np.square(p - one_hot).sum(1).mean()),
        "target_true_probability": float(true_probability.mean()),
        "target_entropy": float(-(p * np.log(np.clip(p, EPS, 1.0))).sum(1).mean()),
        "target_nonzero_classes_mean": float((p > 0.0).sum(axis=1).mean()),
        "target_nonzero_classes_min": int((p > 0.0).sum(axis=1).min()),
    }


def _supported_teacher_probabilities(
    probabilities: np.ndarray,
    authority: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    masked = probabilities * authority[None, :, :]
    retained = masked.sum(axis=2)
    supported = probabilities.copy()
    valid = retained > EPS
    supported[valid] = masked[valid] / retained[valid, None]
    return supported, retained


def _mean_selected(distributions: np.ndarray, selected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts = selected.sum(axis=1)
    summed = (distributions * selected[:, :, None]).sum(axis=1)
    averaged = summed / np.maximum(counts[:, None], 1)
    return averaged, counts


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    return -(probabilities * np.log(np.clip(probabilities, EPS, 1.0))).sum(axis=1)


def _power_transform(probabilities: np.ndarray, exponent: np.ndarray) -> np.ndarray:
    """Apply p -> p**exponent / sum(p**exponent), stably per row."""
    log_p = np.log(np.clip(np.asarray(probabilities, dtype=np.float64), EPS, 1.0))
    scaled = log_p * np.asarray(exponent, dtype=np.float64)[:, None]
    scaled -= scaled.max(axis=1, keepdims=True)
    transformed = np.exp(scaled)
    return transformed / transformed.sum(axis=1, keepdims=True)


def entropy_match_full_target(
    full_probabilities: np.ndarray,
    reference_probabilities: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Match full-target entropy and true-class probability per example.

    The true-class mass is fixed to the support target.  The remaining mass is
    distributed over *all* non-true classes in the full target's order, with a
    single tail power chosen to match entropy.  Thus the control removes the
    label-fidelity and sharpness differences while retaining full-target
    inter-class support and relative tail structure.
    """
    full = np.asarray(full_probabilities, dtype=np.float64)
    reference = np.asarray(reference_probabilities, dtype=np.float64)
    if full.ndim != 2 or full.shape != reference.shape:
        raise ValueError("full and reference probabilities must have the same [N,C] shape")
    if not np.allclose(full.sum(axis=1), 1.0) or not np.allclose(reference.sum(axis=1), 1.0):
        raise ValueError("entropy matching requires normalized probability rows")
    labels = np.asarray(labels, dtype=np.int64)
    if labels.shape != (len(full),) or (labels < 0).any() or (labels >= full.shape[1]).any():
        raise ValueError("labels must align with probability rows")

    desired = _entropy(reference)
    classes = full.shape[1]
    result = np.empty_like(full)
    exponent = np.empty(len(full), dtype=np.float64)
    rows = np.arange(len(full))
    true_mass = reference[rows, labels]
    one_hot = true_mass >= 1.0 - 1e-10
    if one_hot.any():
        result[one_hot] = 0.0
        result[rows[one_hot], labels[one_hot]] = 1.0
        exponent[one_hot] = np.inf

    active = ~one_hot
    if active.any():
        active_full = full[active].copy()
        active_desired = desired[active]
        active_labels = labels[active]
        active_indices = np.flatnonzero(active)
        active_rows = np.arange(len(active_full))
        active_full[active_rows, active_labels] = 0.0
        # Exact equal tail logits contain no class-relation ordering.  Break
        # only those numerical ties deterministically so their entropy can be
        # matched below the tied-top limit; non-tied relations are unchanged.
        tail_mask = np.ones_like(active_full, dtype=bool)
        tail_mask[active_rows, active_labels] = False
        ties = active_full[:, :, None] == active_full[:, None, :]
        ties[:, np.arange(classes), np.arange(classes)] = False
        tied_tail = ties.any(axis=2) & tail_mask
        active_full += 1e-8 * np.arange(classes, dtype=np.float64)[None, :] * tied_tail
        active_full /= active_full.sum(axis=1, keepdims=True)
        active_true_mass = true_mass[active]

        def tail_target(power: np.ndarray) -> np.ndarray:
            transformed_tail = _power_transform(active_full, power)
            transformed_tail[active_rows, active_labels] = 0.0
            transformed_tail /= transformed_tail.sum(axis=1, keepdims=True)
            target = transformed_tail * (1.0 - active_true_mass)[:, None]
            target[active_rows, active_labels] = active_true_mass
            return target

        # A support target with one non-true class reaches the binary-entropy
        # lower bound exactly.  It is the alpha->infinity limit, so encode it
        # directly instead of treating a finite bisection bracket as a failure.
        minimum = np.zeros_like(active_full)
        minimum[active_rows, active_labels] = active_true_mass
        tail_argmax = active_full.argmax(axis=1)
        minimum[active_rows, tail_argmax] = 1.0 - active_true_mass
        # The remaining gap can be below floating-point resolution after a
        # support mask leaves an effectively single-class tail.  Treat it as
        # the analytic boundary; the saved diagnostic reports the residual.
        at_lower_bound = active_desired <= _entropy(minimum) + 1e-8
        if at_lower_bound.any():
            result[active_indices[at_lower_bound]] = minimum[at_lower_bound]
            exponent[active_indices[at_lower_bound]] = np.inf

        active_full = active_full[~at_lower_bound]
        active_desired = active_desired[~at_lower_bound]
        active_labels = active_labels[~at_lower_bound]
        active_true_mass = active_true_mass[~at_lower_bound]
        active_indices = active_indices[~at_lower_bound]
        active_rows = np.arange(len(active_full))
        if not len(active_full):
            achieved = _entropy(result)
            finite = exponent[np.isfinite(exponent)]
            return result, {
                "entropy_match_reference": "expert_v2",
                "entropy_match_mean_abs_error": float(np.abs(achieved - desired).mean()),
                "entropy_match_max_abs_error": float(np.abs(achieved - desired).max()),
                "entropy_match_mean_power": float(finite.mean()) if len(finite) else None,
                "entropy_match_max_power": float(finite.max()) if len(finite) else None,
                "entropy_match_one_hot_limit_rows": int(one_hot.sum()),
                "entropy_match_true_probability_mean_abs_error": float(np.abs(result[rows, labels] - true_mass).mean()),
                "entropy_match_true_probability_max_abs_error": float(np.abs(result[rows, labels] - true_mass).max()),
            }

        # Entropy is continuous and decreasing in the power.  A shared
        # bracketing loop keeps the solve vectorized and deterministic.
        low = np.zeros(len(active_full), dtype=np.float64)
        high = np.ones(len(active_full), dtype=np.float64)
        for _ in range(60):
            too_soft = _entropy(tail_target(high)) > active_desired
            if not too_soft.any():
                break
            high[too_soft] *= 2.0
        else:  # pragma: no cover - malformed numerical input safeguard
            raise RuntimeError("could not bracket requested target entropy")
        for _ in range(80):
            middle = (low + high) / 2.0
            too_soft = _entropy(tail_target(middle)) > active_desired
            low[too_soft] = middle[too_soft]
            high[~too_soft] = middle[~too_soft]
        fitted_power = (low + high) / 2.0
        result[active_indices] = tail_target(fitted_power)
        exponent[active_indices] = fitted_power

    achieved = _entropy(result)
    finite = exponent[np.isfinite(exponent)]
    return result, {
        "entropy_match_reference": "expert_v2",
        "entropy_match_mean_abs_error": float(np.abs(achieved - desired).mean()),
        "entropy_match_max_abs_error": float(np.abs(achieved - desired).max()),
        "entropy_match_mean_power": float(finite.mean()) if len(finite) else None,
        "entropy_match_max_power": float(finite.max()) if len(finite) else None,
        "entropy_match_one_hot_limit_rows": int(one_hot.sum()),
        "entropy_match_true_probability_mean_abs_error": float(
            np.abs(result[rows, labels] - true_mass).mean()
        ),
        "entropy_match_true_probability_max_abs_error": float(
            np.abs(result[rows, labels] - true_mass).max()
        ),
    }


def _energy_weighted_logits(teacher_logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return logit-space free-energy aggregation and its sample-wise weights.

    For teacher k, -E_k(x)=T logsumexp(z_k(x)/T).  A softmax over these
    quantities produces an adaptive weight for each teacher.  Neither the
    holdout authority nor the proxy label participates in this calculation.
    Logit-space averaging makes the result directly comparable to FedDF.
    """
    logits = np.asarray(teacher_logits, dtype=np.float64)
    scaled = logits / ENERGY_WEIGHT_TEMPERATURE
    maximum = scaled.max(axis=2, keepdims=True)
    neg_energy = ENERGY_WEIGHT_TEMPERATURE * (
        maximum[..., 0] + np.log(np.exp(scaled - maximum).sum(axis=2))
    )
    scores = neg_energy / ENERGY_SELECTION_TEMPERATURE
    scores -= scores.max(axis=1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=1, keepdims=True)
    return np.einsum("nk,nkc->nc", weights, logits), weights


def build_target(
    teacher_logits: np.ndarray,
    labels: np.ndarray,
    authority: np.ndarray,
    *,
    method: str,
    temperature: float = 8.0,
) -> TargetResult:
    """Build one target with explicit, minimal fallback semantics.

    Empty EXPERT/ORACLE selections fall back to FedDF. ORACLE-v2 is exactly
    one-hot on samples with at least one correct teacher and uses the same
    explicit FedDF fallback otherwise.
    """
    if method not in ALL_METHODS:
        raise ValueError(f"unknown Article-1 method {method!r}")
    logits, labels, authority = _validate(teacher_logits, labels, authority)
    n, k, c = logits.shape
    teacher_probabilities = softmax(logits, temperature)
    feddf = softmax(logits.mean(axis=1), temperature)
    all_teachers = np.ones((n, k), dtype=bool)
    knows_true = authority[:, labels].T.astype(bool)
    correct = logits.argmax(axis=2) == labels[:, None]
    one_hot = np.zeros((n, c), dtype=np.float64)
    one_hot[np.arange(n), labels] = 1.0
    supported, retained = _supported_teacher_probabilities(teacher_probabilities, authority)

    uses_support = False
    method_extras: dict = {}
    if method == "feddf":
        selected = all_teachers
        target = feddf
    elif method == "energy":
        selected = all_teachers
        energy_logits, energy_weights = _energy_weighted_logits(logits)
        target = softmax(energy_logits, temperature)
    elif method == "support_only":
        empty_teachers = np.flatnonzero(authority.sum(axis=1) == 0)
        selected = all_teachers
        target, _ = _mean_selected(supported, selected)
        uses_support = True
    elif method == "expert_full":
        selected = knows_true
        target, _ = _mean_selected(teacher_probabilities, selected)
    elif method == "expert_v2":
        selected = knows_true
        target, _ = _mean_selected(supported, selected)
        uses_support = True
    elif method == "expert_full_entropy_matched":
        selected = knows_true
        full_target, _ = _mean_selected(teacher_probabilities, selected)
        support_target, _ = _mean_selected(supported, selected)
        empty = selected.sum(axis=1) == 0
        # Both selector variants use the same FedDF fallback, so entropy
        # matching remains well-defined in an uncovered class.
        full_target[empty] = feddf[empty]
        support_target[empty] = feddf[empty]
        target, method_extras = entropy_match_full_target(full_target, support_target, labels)
    elif method == "oracle_full":
        selected = correct
        target, _ = _mean_selected(teacher_probabilities, selected)
    elif method == "oracle_v2":
        selected = correct
        target = one_hot.copy()
    elif method in {"supervised_matched", "supervised_standard"}:
        selected = np.zeros((n, k), dtype=bool)
        target = one_hot
    else:  # pragma: no cover - guarded above
        raise AssertionError(method)

    selected_count = selected.sum(axis=1)
    selection_method = method not in {"feddf", "support_only", "supervised_matched", "supervised_standard"}
    fallback = selected_count == 0 if selection_method else np.zeros(n, dtype=bool)
    if fallback.any():
        target[fallback] = feddf[fallback]
    target /= target.sum(axis=1, keepdims=True)

    event = selected
    outside = 1.0 - retained
    teacher_removed = float(outside[event].mean()) if uses_support and event.any() else None
    target_removed = None
    if uses_support and (~fallback).any():
        retained_by_sample = (retained * selected).sum(axis=1) / np.maximum(selected_count, 1)
        target_removed = float((1.0 - retained_by_sample[~fallback]).mean())
    metrics = {
        "method": method,
        **_quality(target, labels),
        "fallback_count": int(fallback.sum()),
        "fallback_rate": float(fallback.mean()),
        "mean_selected_teachers": float(selected_count.mean()),
        "mass_removed": target_removed,
        "teacher_mass_removed": teacher_removed,
        "support_zero_teacher_fallback_count": int((authority.sum(axis=1) == 0).sum()),
        "oracle_no_correct_count": int((correct.sum(axis=1) == 0).sum()),
        "oracle_no_correct_rate": float((correct.sum(axis=1) == 0).mean()),
    }
    if method == "energy":
        # Effective number is only a descriptive diagnostic; the target itself
        # is built exclusively from teacher logits.
        effective = 1.0 / np.square(energy_weights).sum(axis=1)
        metrics.update({
            "aggregation": "samplewise_free_energy_logit_weighted",
            "energy_weight_temperature": ENERGY_WEIGHT_TEMPERATURE,
            "energy_selection_temperature": ENERGY_SELECTION_TEMPERATURE,
            "mean_effective_teachers": float(effective.mean()),
        })
    metrics.update(method_extras)
    if method == "oracle_v2":
        hits = ~fallback
        metrics.update({
            "one_hot_when_oracle_available": bool(
                np.array_equal(target[hits], one_hot[hits]) if hits.any() else True
            ),
            "non_one_hot_rows_when_oracle_available": int(
                (~np.isclose(target[hits].max(axis=1), 1.0)).sum() if hits.any() else 0
            ),
        })
    return TargetResult(target.astype(np.float32), selected, metrics)


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3 or np.std(left[valid]) <= EPS or np.std(right[valid]) <= EPS:
        return None
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def expertise_mechanism(
    teacher_logits: np.ndarray,
    labels: np.ndarray,
    authority: np.ndarray,
    *,
    temperature: float = 8.0,
) -> tuple[dict[str, np.ndarray], dict]:
    """Measure the out-of-expertise mass injected by EXPERT-selected teachers."""
    logits, labels, authority = _validate(teacher_logits, labels, authority)
    probabilities = softmax(logits, temperature)
    selected = authority[:, labels].T.astype(bool)
    counts = selected.sum(axis=1)
    fallback = counts == 0
    teacher_outside = (probabilities * (1 - authority)[None, :, :]).sum(axis=2)
    selected_mean = (teacher_outside * selected).sum(axis=1) / np.maximum(counts, 1)
    selected_mean[fallback] = np.nan
    expert = build_target(logits, labels, authority, method="expert_full", temperature=temperature)
    rows = np.arange(len(labels))
    true_probability = expert.probabilities[rows, labels]
    nll = -np.log(np.clip(true_probability, EPS, 1.0))
    correct = expert.probabilities.argmax(axis=1) == labels
    observations = {
        "selected_mean_outside_mass": selected_mean.astype(np.float32),
        "selected_teacher_count": counts.astype(np.int16),
        "target_true_probability": true_probability.astype(np.float32),
        "target_nll": nll.astype(np.float32),
        "target_correct": correct.astype(np.uint8),
        "fallback": fallback.astype(np.uint8),
    }
    valid = ~fallback
    summary = {
        "mechanism_examples": int(len(labels)),
        "mechanism_nonfallback_examples": int(valid.sum()),
        "mean_selected_outside_mass": float(np.nanmean(selected_mean)) if valid.any() else None,
        "median_selected_outside_mass": float(np.nanmedian(selected_mean)) if valid.any() else None,
        "corr_outside_mass_target_nll": _correlation(selected_mean, nll),
        "corr_outside_mass_true_probability": _correlation(selected_mean, true_probability),
        "corr_outside_mass_target_correct": _correlation(selected_mean, correct.astype(float)),
    }
    return observations, summary


def mask_generalization_metrics(
    authority: np.ndarray,
    holdout_accuracy: np.ndarray,
    holdout_counts: np.ndarray,
    test_accuracy: np.ndarray,
    test_counts: np.ndarray,
    threshold: float,
) -> dict:
    """Evaluate a holdout-built mask on local test without changing the mask."""
    authority = np.asarray(authority, dtype=bool)
    holdout_accuracy = np.asarray(holdout_accuracy, dtype=np.float64)
    test_accuracy = np.asarray(test_accuracy, dtype=np.float64)
    observed = (np.asarray(holdout_counts) > 0) & (np.asarray(test_counts) > 0)
    expert = authority & observed
    test_pass = (test_accuracy >= threshold) & expert
    return {
        "mask_threshold": float(threshold),
        "mask_density": float(authority.mean()),
        "mask_support_per_teacher": authority.sum(axis=1).astype(int).tolist(),
        "mask_coverage_per_class": authority.sum(axis=0).astype(int).tolist(),
        "mask_uncovered_classes": np.flatnonzero(authority.sum(axis=0) == 0).astype(int).tolist(),
        "mask_zero_support_teachers": np.flatnonzero(authority.sum(axis=1) == 0).astype(int).tolist(),
        "expert_entries_with_local_test": int(expert.sum()),
        "expert_holdout_accuracy": float(holdout_accuracy[expert].mean()) if expert.any() else None,
        "expert_local_test_accuracy": float(test_accuracy[expert].mean()) if expert.any() else None,
        "expert_local_test_pass_rate": float(test_pass.sum() / expert.sum()) if expert.any() else None,
        "expert_holdout_to_test_gap": (
            float((test_accuracy[expert] - holdout_accuracy[expert]).mean()) if expert.any() else None
        ),
        "holdout_test_class_accuracy_correlation": _correlation(
            holdout_accuracy[observed], test_accuracy[observed]
        ),
    }
