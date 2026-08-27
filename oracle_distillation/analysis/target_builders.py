"""KD target builders — single module for every KD target.

Covers both families behind one ``TargetBuilder`` interface + ``get_target_builder``
factory, and owns the actual target math (this module absorbed the former
``personal_targets_unified.py``):

  * Global (one per aggregation method): feddf / consensus / oracle / expert.
    Precomputed on the GPU by ``ProxyAnalysisBuilder`` and stored in
    ``proxy_analysis.npz``; the global builders are *accessors* over that array.
    ``GLOBAL_METHOD_TO_KEY`` is the single source of truth for the
    method→array-key mapping.

  * Personal (one per client k):
      personal_expert  — teacher_k if argmax(teacher_k) in known_k else global expert
      personal_oracle  — teacher_k if y_true in known_k else global oracle
      personal_class_mask — known slots controlled by teacher_k but only with the
        probability mass the global model assigns to known_k; unknown slots stay
        global. Always warm-started from the global expert student by KdRunner.

All builders return [N_proxy, C] float32 logits.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import numpy as np

from oracle_distillation.utils import softmax_np

EPS = 1e-8

# Single source of truth for the global method -> stored-array key mapping.
GLOBAL_METHOD_TO_KEY: dict[str, str] = {
    "feddf":      "avg_logits",
    "consensus":  "consensus_avg_logits",
    "oracle":     "oracle_avg_logits",
    "expert":     "expert_avg_logits",
    # mask-free one-shot bases (privacy-friendly; no competence mask on server)
    "confidence": "confidence_avg_logits",
    "energy":     "energy_avg_logits",
}

SUPPORTED_PERSONAL_VARIANTS = (
    "personal_expert",
    "personal_oracle",
    "personal_class_mask",
)

_PERSONAL_ALIASES = {
    "expert": "personal_expert",
    "oracle": "personal_oracle",
    "class_mask": "personal_class_mask",
}


# ===========================================================================
# Proxy context
# ===========================================================================

@dataclass
class ProxyContext:
    """Light wrapper over a proxy_analysis.npz dict, with typed accessors."""
    data: Mapping[str, np.ndarray]

    @property
    def n_teachers(self) -> int:
        return int(self.data["teacher_logits_cache"].shape[1])

    @property
    def num_classes(self) -> int:
        return int(self.data["teacher_logits_cache"].shape[2])

    def array(self, key: str) -> np.ndarray:
        if key not in self.data:
            raise KeyError(f"proxy context has no '{key}'. Available: {sorted(self.data.keys())}")
        return np.asarray(self.data[key], dtype=np.float32)

    def known_mask(self, k: int) -> np.ndarray:
        return np.asarray(self.data["teacher_knows_class_mask"][k]).astype(bool)


# ===========================================================================
# Low-level helpers (operate on the analysis dict)
# ===========================================================================

def _safe_log(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, EPS, 1.0))


def _teacher_k_logits(analysis_data: Mapping[str, np.ndarray], k: int) -> np.ndarray:
    return analysis_data["teacher_logits_cache"][:, k, :].astype(np.float32)


def _known_mask(analysis_data: Mapping[str, np.ndarray], k: int) -> np.ndarray:
    return analysis_data["teacher_knows_class_mask"][k].astype(bool)


def _resolve_global_logits(analysis_data: Mapping[str, np.ndarray], prefer: str = "expert") -> np.ndarray:
    """Resolve a global target from proxy_analysis.npz, with fallbacks."""
    candidates = {
        "expert": "expert_avg_logits", "oracle": "oracle_avg_logits",
        "avg": "avg_logits", "consensus": "consensus_avg_logits", "feddf": "avg_logits",
        "confidence": "confidence_avg_logits", "energy": "energy_avg_logits",
    }
    primary = candidates.get(prefer, "expert_avg_logits")
    fallback_order = ["expert_avg_logits", "oracle_avg_logits", "avg_logits", "consensus_avg_logits"]
    for key in [primary] + [k for k in fallback_order if k != primary]:
        if key in analysis_data:
            return analysis_data[key].astype(np.float32)
    if "teacher_logits_cache" in analysis_data:
        return analysis_data["teacher_logits_cache"].mean(axis=1).astype(np.float32)
    raise KeyError(
        "No global logits found. Expected one of: expert_avg_logits, oracle_avg_logits, "
        "avg_logits, consensus_avg_logits, or teacher_logits_cache."
    )


def _check_client_id(analysis_data: Mapping[str, np.ndarray], k: int) -> None:
    n_teachers = int(analysis_data["teacher_logits_cache"].shape[1])
    if not (0 <= k < n_teachers):
        raise ValueError(f"client/teacher k={k} out of range [0, {n_teachers - 1}]")


def _canonical_variant(variant: str) -> str:
    v = _PERSONAL_ALIASES.get(variant, variant)
    if v not in SUPPORTED_PERSONAL_VARIANTS:
        raise ValueError(f"Unknown personal target variant {variant!r}. "
                         f"Supported: {SUPPORTED_PERSONAL_VARIANTS}")
    return v


# ===========================================================================
# Personal target math
# ===========================================================================

def build_personal_expert_target(analysis_data: Mapping[str, np.ndarray], k: int) -> np.ndarray:
    """Route to teacher_k if it predicts a known class, else the global expert target."""
    _check_client_id(analysis_data, k)
    z_teacher = _teacher_k_logits(analysis_data, k)
    known = _known_mask(analysis_data, k)
    pred_k = analysis_data["teacher_preds_cache"][:, k].astype(np.intp)
    routes_local = known[pred_k]
    fallback = _resolve_global_logits(analysis_data, prefer="expert")
    return np.where(routes_local[:, None], z_teacher, fallback).astype(np.float32)


def build_personal_oracle_target(analysis_data: Mapping[str, np.ndarray], k: int) -> np.ndarray:
    """Route to teacher_k if the GT class is known, else the global oracle target."""
    _check_client_id(analysis_data, k)
    z_teacher = _teacher_k_logits(analysis_data, k)
    known = _known_mask(analysis_data, k)
    if "y_true_proxy" not in analysis_data:
        raise KeyError("personal_oracle requires y_true_proxy in analysis_data")
    y = analysis_data["y_true_proxy"].astype(np.intp)
    routes_local = known[y]
    fallback = _resolve_global_logits(analysis_data, prefer="oracle")
    return np.where(routes_local[:, None], z_teacher, fallback).astype(np.float32)


def build_class_mask_target(
    analysis_data: Mapping[str, np.ndarray], k: int, T: float = 8.0, global_source: str = "expert",
    *, z_global_override: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-class class_mask target (teacher controls known slots within the global budget).

    ``z_global_override``, when given, replaces the cached ``*_avg_logits`` ensemble
    array as the anchor (e.g. the real forward-pass logits of the distilled global
    student.pt — see ``personal_class_mask_real`` / run_personal_kd.py).
    """
    _check_client_id(analysis_data, k)
    z_teacher = _teacher_k_logits(analysis_data, k)
    z_global = (z_global_override if z_global_override is not None
                else _resolve_global_logits(analysis_data, prefer=global_source))
    known = _known_mask(analysis_data, k)
    # Algebraic no-op: with zero or one known class there is nothing the local
    # teacher can reorder.  Return the anchor logits literally so a second KD pass
    # is byte-for-byte identical to proxy_plain (avoids EPS/log round-off drift).
    if int(known.sum()) <= 1:
        return z_global.copy().astype(np.float32)

    p_global = softmax_np(z_global, T=T)
    p_teacher = softmax_np(z_teacher, T=T)
    p_teacher_known = p_teacher * known[None, :]
    p_teacher_known = p_teacher_known / (p_teacher_known.sum(axis=1, keepdims=True) + EPS)
    p_known_mass = p_global[:, known].sum(axis=1, keepdims=True)
    p_target = np.where(known[None, :], p_known_mass * p_teacher_known, p_global).astype(np.float32)
    p_target = p_target / (p_target.sum(axis=1, keepdims=True) + EPS)
    return (_safe_log(p_target) * float(T)).astype(np.float32)


def build_hierarchical_class_mask_target(
    z_global: np.ndarray,
    z_teacher: np.ndarray,
    known_fine: np.ndarray,
    fine_to_coarse: np.ndarray,
    *,
    T: float = 8.0,
    full_coarse_budget: bool = False,
) -> np.ndarray:
    """Double-depth (CIFAR-100) class_mask target.

    The teacher reorders probability mass only *within* each known superclass; the
    coarse (per-superclass) split is frozen to the global's. Reduces **bit-for-bit**
    to :func:`build_class_mask_target` when all known fines map to a single
    superclass (one group → the flat formula). See docs/audit/CLASS_MASK_TARGETS_FINDINGS.md.

    Per known superclass ``s`` with ``K_s = known_fine ∩ fines(s)``:
        budget_s := Σ_{c∈K_s} P_global(c)        (competence-scoped, default)
                    Σ_{c∈s}   P_global(c)        (full_coarse_budget=True)
        t_s      := normalize(P_teacher[K_s])
        target(c∈K_s) := budget_s · t_s(c)
    Every fine not in any ``K_s`` keeps its global mass. Same T / EPS / log·T
    conventions as the flat builder so the loss recovers ``softmax(target/T)``.
    """
    z_global = np.asarray(z_global, dtype=np.float32)
    z_teacher = np.asarray(z_teacher, dtype=np.float32)
    known = np.asarray(known_fine).astype(bool)
    f2c = np.asarray(fine_to_coarse).astype(np.intp)

    p_global = softmax_np(z_global, T=T)
    if not known.any():
        return z_global.copy()
    p_teacher = softmax_np(z_teacher, T=T)

    p_target = p_global.copy()
    for s in np.unique(f2c[known]):
        K_s = known & (f2c == s)                                  # known fines of superclass s
        if full_coarse_budget:
            sib = (f2c == s)
            budget_s = p_global[:, sib].sum(axis=1, keepdims=True)
            p_target[:, sib] = 0.0                                # unknown siblings' mass is absorbed
        else:
            budget_s = p_global[:, K_s].sum(axis=1, keepdims=True)  # competence-scoped (default)
        t_s = p_teacher * K_s[None, :]
        t_s = t_s / (t_s.sum(axis=1, keepdims=True) + EPS)        # teacher shape within K_s
        p_target[:, K_s] = budget_s * t_s[:, K_s]
    p_target = p_target / (p_target.sum(axis=1, keepdims=True) + EPS)
    return (_safe_log(p_target) * float(T)).astype(np.float32)


def build_predicted_superclass_class_mask_target(
    z_global: np.ndarray,
    z_teacher: np.ndarray,
    knows_class: np.ndarray,
    fine_to_coarse: np.ndarray,
    *,
    T: float = 8.0,
) -> np.ndarray:
    """Predicted-superclass CIFAR-100 class_mask target (2-layer expertise).

    Two-layer scheme:
      * Layer 1 (superclass) is already encoded in ``z_global`` (the ``expert``
        student, built by pooling teachers competent on each superclass).
      * Layer 2 (fine) is applied here per proxy sample ``x`` using the teacher's
        **fine** competence ``knows_class`` (accuracy mask, ``teacher_knows_class_mask[k]``):

            p = argmax(P_global)                # the global student's predicted fine class
            s = superclass(p)
            if knows_class[p]:                  # teacher is a fine-expert on the predicted class
                budget = Σ_{c∈fines(s)} P_global(c)        # the global's whole mass on superclass s
                target(c) = budget · normalize(P_teacher over fines(s))   for every c∈fines(s)
            else:
                target = P_global

    So expertise on the *predicted fine class* lets the local teacher re-sort the
    **entire** superclass's probability mass (all 5 fines); the coarse split across
    superclasses is untouched (rows still sum to 1). Deployable: ``p`` comes from the
    global student, never from the ground-truth label.
    """
    z_global = np.asarray(z_global, dtype=np.float32)
    z_teacher = np.asarray(z_teacher, dtype=np.float32)
    knows = np.asarray(knows_class).astype(bool)                  # [C]
    f2c = np.asarray(fine_to_coarse).astype(np.intp)              # [C]

    p_global = softmax_np(z_global, T=T)                          # [N, C]
    if not knows.any():
        return (_safe_log(p_global) * float(T)).astype(np.float32)
    p_teacher = softmax_np(z_teacher, T=T)

    pred = p_global.argmax(axis=1)                                # [N] global's predicted fine class
    gate = knows[pred]                                            # [N] teacher expert on the predicted class
    s_pred = f2c[pred]                                            # [N] predicted superclass

    p_target = p_global.copy()
    for s in np.unique(s_pred[gate]):
        sib = np.where(f2c == s)[0]                              # all fines of superclass s
        rows = gate & (s_pred == s)                              # samples whose predicted super is s
        if not rows.any() or len(sib) == 0:
            continue
        budget = p_global[np.ix_(rows, sib)].sum(axis=1, keepdims=True)   # global's mass on s
        t = p_teacher[np.ix_(rows, sib)]
        t = t / (t.sum(axis=1, keepdims=True) + EPS)             # teacher shape within s
        p_target[np.ix_(rows, sib)] = budget * t
    p_target = p_target / (p_target.sum(axis=1, keepdims=True) + EPS)
    return (_safe_log(p_target) * float(T)).astype(np.float32)


def _class_mask_target(
    analysis_data: Mapping[str, np.ndarray], k: int, *,
    T: float = 8.0, global_source: str = "expert",
    fine_to_coarse: Optional[np.ndarray] = None,
    known_fine: Optional[np.ndarray] = None,
    full_coarse_budget: bool = False,
    z_global_override: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Dispatch class_mask. Flat datasets (``fine_to_coarse`` None) keep the per-class
    builder. CIFAR-100 uses the **predicted-superclass** builder driven by the accuracy
    fine mask (``teacher_knows_class_mask[k]``). ``known_fine``/``full_coarse_budget``
    are accepted for back-compat but unused on the CIFAR-100 path. ``z_global_override``
    (real student.pt logits) takes precedence over the cached ``*_avg_logits`` array on
    both the flat and hierarchical paths.
    """
    if fine_to_coarse is None:
        return build_class_mask_target(analysis_data, k, T=T, global_source=global_source,
                                        z_global_override=z_global_override)
    _check_client_id(analysis_data, k)
    z_global = (z_global_override if z_global_override is not None
                else _resolve_global_logits(analysis_data, prefer=global_source))
    z_teacher = _teacher_k_logits(analysis_data, k)
    knows_class = _known_mask(analysis_data, k)   # accuracy fine mask (teacher_knows_class_mask[k])
    return build_predicted_superclass_class_mask_target(
        z_global, z_teacher, knows_class, fine_to_coarse, T=T,
    )


_PERSONAL_BUILDERS = {
    "personal_expert": build_personal_expert_target,
    "personal_oracle": build_personal_oracle_target,
    "personal_class_mask": build_class_mask_target,
}


def build_personal_target(
    analysis_data: Mapping[str, np.ndarray], k: int, variant: str, *,
    fine_to_coarse: Optional[np.ndarray] = None,
    known_fine: Optional[np.ndarray] = None,
    full_coarse_budget: bool = False,
    **kwargs,
) -> np.ndarray:
    """Build one personal KD target [N, C] for client k (accepts aliases).

    For ``personal_class_mask`` a non-None ``fine_to_coarse`` selects the
    hierarchical (double-depth) target; otherwise the flat one. ``known_fine``
    optionally overrides the competence mask.
    """
    v = _canonical_variant(variant)
    if v in {"personal_expert", "personal_oracle"}:
        return _PERSONAL_BUILDERS[v](analysis_data, k)
    return _class_mask_target(
        analysis_data, k, fine_to_coarse=fine_to_coarse, known_fine=known_fine,
        full_coarse_budget=full_coarse_budget, **kwargs,
    )


# ===========================================================================
# Personal target diagnostics (footprint, routing)
# ===========================================================================

def _footprint_metrics(
    analysis_data: Mapping[str, np.ndarray], k: int, variant: str,
    T: float = 8.0, global_source: str = "expert", *,
    fine_to_coarse: Optional[np.ndarray] = None,
    known_fine: Optional[np.ndarray] = None,
    full_coarse_budget: bool = False,
    z_global_override: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    known = (np.asarray(known_fine).astype(bool) if known_fine is not None
             else _known_mask(analysis_data, k))
    n_known = int(known.sum())
    y_true = analysis_data["y_true_proxy"].astype(np.intp)
    z_personal = build_personal_target(
        analysis_data, k, variant, T=T, global_source=global_source,
        fine_to_coarse=fine_to_coarse, known_fine=known_fine, full_coarse_budget=full_coarse_budget,
        z_global_override=z_global_override,
    )
    z_global = (z_global_override if z_global_override is not None
                else _resolve_global_logits(analysis_data, prefer=global_source))
    argmax_personal = z_personal.argmax(axis=1)
    argmax_global = z_global.argmax(axis=1)
    flipped = argmax_personal != argmax_global
    correct_global = argmax_global == y_true
    correct_personal = argmax_personal == y_true
    p_global = softmax_np(z_global)
    budget_known = float(p_global[:, known].sum(axis=1).mean()) if known.any() else 0.0
    # No-op iff the teacher has nothing to reorder: every competence "fragment" is a
    # singleton. Flat = one group over all known (→ n_known==1); hierarchical = one
    # group per known superclass (→ dir_fine's size-1 fragments are all no-ops).
    if fine_to_coarse is not None and known.any():
        f2c = np.asarray(fine_to_coarse).astype(np.intp)
        frag_max = max(int((known & (f2c == s)).sum()) for s in np.unique(f2c[known]))
        is_noop = int(frag_max <= 1)
    else:
        is_noop = int(n_known == 1)
    return {
        "distill_target_flips":          int(flipped.sum()),
        "distill_footprint_correction":  int((flipped & ~correct_global & correct_personal).sum()),
        "distill_footprint_regression":  int((flipped & correct_global & ~correct_personal).sum()),
        "distill_budget_known_mean":     budget_known,
        "distill_is_noop":               is_noop,
    }


def personal_target_diagnostics(
    analysis_data: Mapping[str, np.ndarray], k: int, variant: str,
    T: float = 8.0, global_source: str = "expert", *,
    fine_to_coarse: Optional[np.ndarray] = None,
    known_fine: Optional[np.ndarray] = None,
    full_coarse_budget: bool = False,
    z_global_override: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Lightweight diagnostics for a personal target variant.

    ``known_fine`` (when given) overrides the competence mask so the reported
    ``known_classes``/``frac_*``/footprint reflect the same mask the target used
    (CIFAR-100 structural mask). ``fine_to_coarse`` selects the hierarchical target.
    ``z_global_override``, when given, is the anchor actually used to build the
    target (e.g. real student.pt logits for ``personal_class_mask_real``) — kept
    consistent with the target so the diagnostics don't silently read a different
    global source.
    """
    v = _canonical_variant(variant)
    known = (np.asarray(known_fine).astype(bool) if known_fine is not None
             else _known_mask(analysis_data, k))
    diag: Dict[str, object] = {
        "k": int(k), "variant": v,
        "n_known_classes": int(known.sum()),
        "known_classes": [int(c) for c in np.where(known)[0]],
    }
    if v == "personal_expert":
        pred_k = analysis_data["teacher_preds_cache"][:, k].astype(np.intp)
        diag["frac_local_routed"] = float(known[pred_k].mean())
    elif v == "personal_oracle":
        y = analysis_data["y_true_proxy"].astype(np.intp)
        diag["frac_local_routed"] = float(known[y].mean())
    else:
        z_global = (z_global_override if z_global_override is not None
                    else _resolve_global_logits(analysis_data, prefer=global_source))
        p_global = softmax_np(z_global, T=T)
        diag["frac_known_mass"] = float(p_global[:, known].sum(axis=1).mean()) if known.any() else 0.0
    diag.update(_footprint_metrics(
        analysis_data, k, v, T=T, global_source=global_source,
        fine_to_coarse=fine_to_coarse, known_fine=known_fine, full_coarse_budget=full_coarse_budget,
        z_global_override=z_global_override,
    ))
    return diag


# ===========================================================================
# TargetBuilder interface + builders
# ===========================================================================

class TargetBuilder(ABC):
    is_personal: bool = False
    name: str = ""

    @abstractmethod
    def build(self, ctx: ProxyContext, k: Optional[int] = None, **kwargs) -> np.ndarray:
        """Return the [N_proxy, C] float32 target logits."""


class _GlobalBuilder(TargetBuilder):
    is_personal = False

    def build(self, ctx: ProxyContext, k: Optional[int] = None, **kwargs) -> np.ndarray:
        return ctx.array(GLOBAL_METHOD_TO_KEY[self.name])


class FeddfBuilder(_GlobalBuilder):      name = "feddf"
class ConsensusBuilder(_GlobalBuilder):  name = "consensus"
class OracleBuilder(_GlobalBuilder):     name = "oracle"
class ExpertBuilder(_GlobalBuilder):     name = "expert"
class ConfidenceBuilder(_GlobalBuilder): name = "confidence"
class EnergyBuilder(_GlobalBuilder):     name = "energy"


class PersonalExpertBuilder(TargetBuilder):
    is_personal = True
    name = "personal_expert"

    def build(self, ctx: ProxyContext, k: Optional[int] = None, **kwargs) -> np.ndarray:
        return build_personal_expert_target(ctx.data, int(k))


class PersonalOracleBuilder(TargetBuilder):
    is_personal = True
    name = "personal_oracle"

    def build(self, ctx: ProxyContext, k: Optional[int] = None, **kwargs) -> np.ndarray:
        return build_personal_oracle_target(ctx.data, int(k))


class ClassMaskBuilder(TargetBuilder):
    is_personal = True
    name = "personal_class_mask"

    def build(self, ctx: ProxyContext, k: Optional[int] = None, *,
              T: float = 8.0, global_source: str = "expert",
              fine_to_coarse: Optional[np.ndarray] = None,
              known_fine: Optional[np.ndarray] = None,
              full_coarse_budget: bool = False,
              z_global_override: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        # fine_to_coarse=None → flat (cifar/mnist/fmnist, byte-identical);
        # non-None → hierarchical double-depth (cifar100). z_global_override (when
        # given) replaces the cached *_avg_logits ensemble array as the anchor.
        return _class_mask_target(
            ctx.data, int(k), T=T, global_source=global_source,
            fine_to_coarse=fine_to_coarse, known_fine=known_fine,
            full_coarse_budget=full_coarse_budget, z_global_override=z_global_override,
        )


_REGISTRY: dict[str, type[TargetBuilder]] = {
    "feddf": FeddfBuilder, "consensus": ConsensusBuilder, "oracle": OracleBuilder, "expert": ExpertBuilder,
    "confidence": ConfidenceBuilder, "energy": EnergyBuilder,
    "personal_expert": PersonalExpertBuilder, "personal_oracle": PersonalOracleBuilder,
    "personal_class_mask": ClassMaskBuilder,
}


def get_target_builder(name: str) -> TargetBuilder:
    """Return a TargetBuilder instance for a global method or personal variant."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown target {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def is_personal_target(name: str) -> bool:
    return name in _REGISTRY and _REGISTRY[name].is_personal
