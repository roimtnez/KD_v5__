"""ProxyAnalysisBuilder — build proxy_analysis.npz (REFACTOR_PLAN, full consolidation).

Single home for "run the K teachers over the proxy once and build every KD target".
Absorbs the former ``analysis/features.py`` (the GPU pass that produces the 4 global
targets + caches) and ``analysis/proxy_analysis.py`` (mask resolution, diagnostics,
artifact writing). Behaviour is preserved: ``proxy_analysis.npz`` /
``proxy_diagnostics.json`` are byte-compatible with the legacy code (rebuild parity
verified against committed teachers in tests/test_proxy_analysis_builder.py).

Targets produced (one [N, C] logit tensor each):
    feddf      -> avg_logits              (uniform teacher mean)
    consensus  -> consensus_avg_logits    (confidence-weighted majority)
    oracle     -> oracle_avg_logits       (teachers matching GT; upper bound)
    expert     -> expert_avg_logits       (teachers competent on GT by mask)
    confidence -> confidence_avg_logits   (MSP-weighted, mask-free one-shot base)
    energy     -> energy_avg_logits       (free-energy-weighted, mask-free one-shot base)

The last two are privacy-friendly mask-free ensemble aggregators: they weight the
K teachers per-sample from their own logits only (no `teacher_knows_class_mask`
needed on the server). With the weight-temperature ``tau`` swept, ``energy``
interpolates between a soft mask-free ``expert`` (tau->0, hard-select the
lowest-energy / most in-distribution teacher) and plain FedDF ``avg`` (tau->inf).

Mask origin: the teacher side stores raw local-test per-class accuracies in
`teachers_manifest.json`. The server then chooses the experiment threshold
(e.g. 0.7 today, 0.6 tomorrow) and derives `teacher_knows_class_mask` from
those accuracies.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import yaml

from data.dataset_config import get_dataset_config
from oracle_distillation.teachers import list_teachers, load_teachers
from oracle_distillation.utils import resolve_device, softmax_np


# ===========================================================================
# Mask-free ensemble aggregators (confidence / energy one-shot bases)
# ===========================================================================
# Reduction space for the weighted ensemble: must match how `avg_logits` is
# built (a plain mean *of logits*), so the new bases live on the same scale.
AVG_SPACE = "logit"
# Default weight temperatures. WEIGHT_T scales the per-teacher softmax/energy;
# CONF_TAU / ENERGY_TAU sharpen the across-teacher weighting (tau->0 hard-selects
# one teacher, tau->inf -> uniform = FedDF mean).
WEIGHT_T = 1.0
CONF_TAU = 1.0
ENERGY_TAU = 1.0


def _softmax_axis(z: np.ndarray, T: float = 1.0, axis: int = -1) -> np.ndarray:
    z = z.astype(np.float32) / float(T)
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def _logsumexp_axis(z: np.ndarray, axis: int = -1) -> np.ndarray:
    m = z.max(axis=axis, keepdims=True)
    out = m + np.log(np.exp(z - m).sum(axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def confidence_weights(tlc: np.ndarray, T: float = 1.0, tau: float = 1.0) -> np.ndarray:
    """Per-sample, per-teacher weights from max-softmax-prob (MSP). [N,K,C]->[N,K].

    Caveat: closed-world teachers are over-confident on OOD (the very pathology),
    so MSP is a leaky competence signal; prefer ``energy_weights`` as the base.
    """
    p = _softmax_axis(tlc, T=T, axis=2)          # [N,K,C]
    conf = p.max(axis=2)                          # [N,K]
    return _softmax_axis(conf, T=tau, axis=1)     # normalize across teachers


def energy_weights(tlc: np.ndarray, T: float = 1.0, tau: float = 1.0) -> np.ndarray:
    """Per-sample, per-teacher weights from the (negative) free energy. [N,K,C]->[N,K].

    Energy E_k(x) = -T * logsumexp(z_k(x)/T); low energy = in-distribution for k.
    We weight by softmax over teachers of T*logsumexp, so a teacher that
    recognizes x dominates. tau->0: hard pick of the lowest-energy teacher (soft
    'expert', no mask); tau->inf: uniform (FedDF mean).
    """
    neg_energy = T * _logsumexp_axis(tlc / float(T), axis=2)   # [N,K]  (= -E_k)
    return _softmax_axis(neg_energy, T=tau, axis=1)


def weighted_ensemble_logits(tlc: np.ndarray, w: np.ndarray, space: str = AVG_SPACE) -> np.ndarray:
    """Combine [N,K,C] teacher logits with [N,K] weights into [N,C].

    ``space="logit"`` = weighted mean of logits (matches ``avg_logits``);
    ``space="prob"`` = weighted mean of probabilities, returned as logits.
    """
    if space == "logit":
        return np.einsum("nk,nkc->nc", w, tlc)
    if space == "prob":
        p = _softmax_axis(tlc, T=1.0, axis=2)
        pe = np.einsum("nk,nkc->nc", w, p)
        return np.log(np.clip(pe, 1e-8, 1.0))
    raise ValueError(f"space must be 'logit' or 'prob', got {space!r}")


# ===========================================================================
# Teacher pass over the proxy: build all KD targets in one shot
# ===========================================================================

def _run_teachers_collect(teachers, loader, device, *, logits_dtype_cache=np.float16):
    """Run teachers on the loader and return (teacher_logits, teacher_preds, y_true).

    Returns:
        teacher_logits: np.ndarray, shape [N, K, C], dtype logits_dtype_cache
        teacher_preds: np.ndarray, shape [N, K], dtype uint8
        y_true: np.ndarray, shape [N], dtype int64
    """
    teacher_logits_list = []
    teacher_preds_list = []
    y_list = []
    use_amp = device.type == "cuda"
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16):
                batch_logits = torch.stack([t(x).float() for t in teachers], dim=0)  # [K, B, C]
            preds = batch_logits.argmax(dim=2)  # [K, B]

            # convert to numpy with same memory layout as legacy code
            logits_np = batch_logits.detach().permute(1, 0, 2).cpu().numpy().astype(logits_dtype_cache, copy=False)  # [B, K, C]
            preds_np = preds.detach().cpu().numpy().T.astype(np.uint8, copy=False)  # [B, K]
            y_np = y.detach().cpu().numpy()

            teacher_logits_list.append(logits_np)
            teacher_preds_list.append(preds_np)
            y_list.append(y_np)

    teacher_logits = np.concatenate(teacher_logits_list, axis=0)
    teacher_preds = np.concatenate(teacher_preds_list, axis=0)
    y_true = np.concatenate(y_list, axis=0).astype(np.int64)
    return teacher_logits, teacher_preds, y_true


def _super_competence_mask(
    class_mask: np.ndarray, fine_to_coarse: np.ndarray
) -> np.ndarray:
    """Derive a superclass competence mask [K, S] from the per-fine mask [K, C].

    A teacher knows superclass ``s`` iff it is competent on at least one of ``s``'s
    fine siblings. The union of superclass-experts therefore covers all the fine
    siblings, so pooling them lets the expert ensemble discriminate *within* the
    superclass — the "experts on insects vote on which insect" behaviour.
    """
    cm = np.asarray(class_mask).astype(bool)
    f2c = np.asarray(fine_to_coarse).astype(np.intp)
    n_super = int(f2c.max()) + 1
    out = np.zeros((cm.shape[0], n_super), dtype=bool)
    for s in range(n_super):
        sib = np.where(f2c == s)[0]
        if len(sib):
            out[:, s] = cm[:, sib].any(axis=1)
    return out


def _compute_global_targets(
    teacher_logits: np.ndarray, teacher_preds: np.ndarray, y_true: np.ndarray,
    teacher_knows_class_mask: np.ndarray, *, logits_dtype_targets=np.float32,
    fine_to_coarse: Optional[np.ndarray] = None,
    teacher_knows_super_mask: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Given full cached teacher logits/preds produce the same target dict as
    the legacy `extract_proxy_cache` but in a vectorized post-hoc manner.

    When ``fine_to_coarse`` is given (CIFAR-100), the **expert** aggregation pools
    teachers competent on the sample's true *superclass* (via
    :func:`_super_competence_mask`) instead of its exact fine class; flat datasets
    keep the per-fine behaviour.
    """
    # teacher_logits: [N, K, C], teacher_preds: [N, K], y_true: [N]
    N, K, C = teacher_logits.shape
    t_logits = teacher_logits.astype(np.float32)

    # avg logits
    avg_logits = t_logits.mean(axis=1).astype(logits_dtype_targets)

    # consensus: average teacher softmax (per-teacher) then majority
    probs_flat = softmax_np(t_logits.reshape(-1, C))  # [(N*K), C]
    probs = probs_flat.reshape(N, K, C)
    avg_probs = probs.mean(axis=1)  # [N, C]
    majority = avg_probs.argmax(axis=1)
    cons_mask = (teacher_preds == majority[:, None])  # [N, K]
    cons_count = cons_mask.sum(axis=1)
    cons_logits = (t_logits * cons_mask[:, :, None]).sum(axis=1) / np.maximum(cons_count[:, None], 1)
    consensus_avg_logits = cons_logits.astype(logits_dtype_targets)

    # oracle: teachers that predicted the true label
    oracle_mask = (teacher_preds == y_true[:, None])
    oracle_count = oracle_mask.sum(axis=1)
    oracle_sum = (t_logits * oracle_mask[:, :, None]).sum(axis=1)
    oracle_logits = oracle_sum / np.maximum(oracle_count[:, None], 1)
    has_oracle = oracle_count > 0
    oracle_avg_logits = np.where(has_oracle[:, None], oracle_logits, avg_logits).astype(logits_dtype_targets)

    # expert: teachers competent for the sample. Per-fine by default; for CIFAR-100
    # (fine_to_coarse given) competence is taken at the *superclass* level — the
    # server pools teachers expert on the sample's superclass. ``teacher_knows_super_mask``
    # is the **direct** coarse-accuracy mask (a teacher knows superclass s iff its
    # local-test super-accuracy on s ≥ threshold); if not supplied we fall back to
    # deriving it from the fine mask (knows ≥1 sibling) for backward compatibility.
    if fine_to_coarse is not None:
        if teacher_knows_super_mask is None:
            teacher_knows_super_mask = _super_competence_mask(teacher_knows_class_mask, fine_to_coarse)
        teacher_knows_super_mask = np.asarray(teacher_knows_super_mask).astype(bool)
        super_of = np.asarray(fine_to_coarse).astype(np.intp)[y_true]   # [N] true superclass
        mask_kb = teacher_knows_super_mask[:, super_of].T.astype(bool)  # [N, K]
    else:
        teacher_knows_super_mask = None
        # teacher_knows_class_mask: [K, C] -> mask_kb for all samples: [N, K]
        mask_kb = teacher_knows_class_mask[:, y_true].T.astype(bool)    # [N, K]
    expert_count = mask_kb.sum(axis=1)
    expert_sum = (t_logits * mask_kb[:, :, None]).sum(axis=1)
    expert_logits = expert_sum / np.maximum(expert_count[:, None], 1)
    has_expert = expert_count > 0
    expert_avg_logits = np.where(has_expert[:, None], expert_logits, avg_logits).astype(logits_dtype_targets)

    # mask-free one-shot bases: weight teachers per-sample from their own logits
    # only (no GT, no competence mask). Same logit-space reduction as avg_logits.
    confidence_avg_logits = weighted_ensemble_logits(
        t_logits, confidence_weights(t_logits, T=WEIGHT_T, tau=CONF_TAU), space=AVG_SPACE,
    ).astype(logits_dtype_targets)
    energy_avg_logits = weighted_ensemble_logits(
        t_logits, energy_weights(t_logits, T=WEIGHT_T, tau=ENERGY_TAU), space=AVG_SPACE,
    ).astype(logits_dtype_targets)

    # preds and other simple diagnostics
    vote_pred = avg_probs.argmax(axis=1).astype(np.uint8)
    avg_pred = avg_logits.argmax(axis=1).astype(np.uint8)
    oracle_pred = oracle_avg_logits.argmax(axis=1).astype(np.uint8)
    expert_pred = expert_avg_logits.argmax(axis=1).astype(np.uint8)

    conf_avgT = softmax_np(avg_logits).max(axis=1).astype(np.float32)
    correct_vote = (vote_pred == y_true).astype(np.uint8)
    correct_avg = (avg_pred == y_true).astype(np.uint8)

    expert_count_per_sample = expert_count.astype(np.int32)
    oracle_expert_agree = (oracle_pred == expert_pred).astype(np.uint8)

    # oracle-expert KL
    p_oracle = softmax_np(oracle_avg_logits.astype(np.float32)).clip(1e-12, 1.0)
    p_expert = softmax_np(expert_avg_logits.astype(np.float32)).clip(1e-12, 1.0)
    oracle_expert_kl = (p_oracle * (np.log(p_oracle) - np.log(p_expert))).sum(axis=1).astype(np.float32)

    # teacher-level accuracy metrics
    correct_matrix = (teacher_preds == y_true[:, None])  # [N, K]
    teacher_acc = correct_matrix.sum(axis=0).astype(np.float32) / max(N, 1)
    class_counts = np.bincount(y_true, minlength=C)
    teacher_class_correct = np.zeros((K, C), dtype=np.int64)
    for c in range(C):
        mask_c = (y_true == c)
        if mask_c.any():
            teacher_class_correct[:, c] = (teacher_preds[mask_c] == c).sum(axis=0).astype(np.int64)
    denom = np.maximum(class_counts[None, :], 1)
    teacher_class_acc = (teacher_class_correct / denom).astype(np.float32)

    return {
        "avg_logits": avg_logits,
        "consensus_avg_logits": consensus_avg_logits,
        "oracle_avg_logits": oracle_avg_logits,
        "expert_avg_logits": expert_avg_logits,
        "confidence_avg_logits": confidence_avg_logits,
        "energy_avg_logits": energy_avg_logits,
        "vote_pred": vote_pred,
        "avg_pred": avg_pred,
        "oracle_pred": oracle_pred,
        "expert_pred": expert_pred,
        "conf_avgT": conf_avgT,
        "correct_vote": correct_vote,
        "correct_avg": correct_avg,
        "expert_count_per_sample": expert_count_per_sample,
        "oracle_expert_agree": oracle_expert_agree,
        "oracle_expert_kl": oracle_expert_kl,
        "teacher_acc": teacher_acc,
        "teacher_class_acc": teacher_class_acc,
        "teacher_knows_super_mask": teacher_knows_super_mask,  # [K,S] or None (flat)
        "teacher_preds_cache": teacher_preds,
        "teacher_logits_cache": teacher_logits,
    }


# ===========================================================================
# Mask resolution
# ===========================================================================

def _teachers_fingerprint(teacher_paths) -> str:
    h = hashlib.sha1()
    for path in teacher_paths:
        stat = path.stat()
        h.update(str(path).encode("utf-8"))
        h.update(str(stat.st_size).encode("utf-8"))
        h.update(str(int(stat.st_mtime)).encode("utf-8"))
    return h.hexdigest()[:12]


def _load_teacher_logit_caches(
    cache_paths, *, dataset: str, seed: int, proxy_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Load the checkpoint-free teacher surface and verify its provenance.

    Each cache is emitted immediately after training one teacher.  Exact checks
    on seed, dataset, client id, proxy indices and labels prevent a cache from a
    different experimental seed from being silently reused.
    """
    ordered = sorted(Path(path) for path in cache_paths)
    logits_by_teacher = []
    labels_ref: Optional[np.ndarray] = None
    expected_idx = np.asarray(proxy_idx, dtype=np.int64)
    seen_cids = []
    for path in ordered:
        with np.load(path, allow_pickle=False) as cache:
            required = {"logits", "labels", "proxy_idx", "seed", "dataset", "cid"}
            missing = required - set(cache.files)
            if missing:
                raise ValueError(f"teacher cache {path} lacks {sorted(missing)}")
            cache_seed = int(np.asarray(cache["seed"]).item())
            cache_dataset = str(np.asarray(cache["dataset"]).item())
            cid = int(np.asarray(cache["cid"]).item())
            cache_idx = np.asarray(cache["proxy_idx"], dtype=np.int64)
            labels = np.asarray(cache["labels"], dtype=np.int64)
            logits = np.asarray(cache["logits"])
        if cache_seed != seed or cache_dataset != dataset:
            raise ValueError(
                f"teacher cache provenance mismatch at {path}: "
                f"seed={cache_seed}, dataset={cache_dataset!r}; "
                f"expected seed={seed}, dataset={dataset!r}"
            )
        if not np.array_equal(cache_idx, expected_idx):
            raise ValueError(f"teacher cache proxy_idx mismatch at {path}")
        if logits.ndim != 2 or logits.shape[0] != len(expected_idx):
            raise ValueError(f"invalid teacher logits shape {logits.shape} at {path}")
        if labels.shape != (len(expected_idx),):
            raise ValueError(f"invalid teacher labels shape {labels.shape} at {path}")
        if labels_ref is None:
            labels_ref = labels
        elif not np.array_equal(labels, labels_ref):
            raise ValueError(f"teacher cache labels disagree at {path}")
        seen_cids.append(cid)
        logits_by_teacher.append(logits.astype(np.float16, copy=False))
    if seen_cids != list(range(len(ordered))):
        raise ValueError(f"teacher cache cids must be contiguous from zero, got {seen_cids}")
    if not logits_by_teacher or labels_ref is None:
        raise ValueError("no teacher proxy-logit caches found")
    # Per-cache [N,C] -> canonical [N,K,C].
    teacher_logits = np.stack(logits_by_teacher, axis=1)
    return teacher_logits, labels_ref, _teachers_fingerprint(ordered)


def _list_teacher_sources(teachers_dir: Path) -> tuple[list[Path], list[Path]]:
    """Prefer checkpoint-free caches without probing for absent checkpoints."""
    cache_paths = sorted(Path(teachers_dir).glob("cid_*_proxy_logits.npz"))
    if cache_paths:
        return cache_paths, []
    return [], list_teachers(Path(teachers_dir))


def _proxy_idx_fingerprint(proxy_idx: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(proxy_idx, dtype=np.int64))
    return hashlib.sha256(array.tobytes()).hexdigest()


def build_expert_mask_from_manifest(
    teachers_dir: Path, *, num_teachers: int, num_classes: int,
    threshold: float,
) -> np.ndarray:
    """teacher_knows_class_mask [K, C] from teachers_manifest.json per-class accuracy.

    The manifest only stores raw local-test accuracies; thresholding is applied
    here so different experiments can choose different cutoffs without retraining
    teachers.
    """
    manifest_path = Path(teachers_dir).parent / "teachers_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing teachers_manifest.json at {manifest_path}. "
            "Required to build the expert mask from per-class accuracies."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list):
        raise ValueError(f"{manifest_path} must contain a list, got {type(manifest).__name__}")

    mask = np.zeros((num_teachers, num_classes), dtype=np.int8)
    seen = set()
    for entry in manifest:
        cid = int(entry.get("cid", -1))
        if not (0 <= cid < num_teachers):
            continue
        # Look inside the 'crea' block for raw per-class accuracies.
        metrics = entry.get("crea") or entry.get("crea ")
        if not isinstance(metrics, dict):
            raise ValueError(f"cid={cid}: no 'crea' block found in manifest entry.")
        acc_per_class = metrics.get("local_test_acc_per_class")
        source = "local_test"
        if not isinstance(acc_per_class, list):
            acc_per_class = metrics.get("hold_acc_per_class")
            source = "holdout_fallback"
        if not isinstance(acc_per_class, list):
            raise ValueError(
                f"cid={cid}: neither 'local_test_acc_per_class' nor 'hold_acc_per_class' found."
            )
        for c, acc in enumerate(acc_per_class[:num_classes]):
            if acc is not None and float(acc) >= threshold:
                mask[cid, c] = 1
        seen.add(cid)
        if source == "holdout_fallback":
            import warnings
            warnings.warn(
                f"cid={cid}: using holdout acc for teacher_knows_class_mask "
                "(no local_test_acc_per_class — regenerate teachers with local_test_frac>0).",
                stacklevel=3,
            )
    missing = set(range(num_teachers)) - seen
    if missing:
        raise ValueError(f"Manifest is missing entries for cids {sorted(missing)} (expected {num_teachers})")
    return mask


def build_super_mask_from_manifest(
    teachers_dir: Path, *, num_teachers: int, num_super: int, threshold: float,
) -> Optional[np.ndarray]:
    """teacher_knows_super_mask [K, S] from teachers_manifest.json per-superclass accuracy.

    A teacher knows superclass ``s`` iff its **coarse** local-test accuracy on ``s``'s
    images (predicted superclass == s) is ``>= threshold`` — i.e. it is competent at
    *recognising the superclass*. This is the server-side competence used by the
    ``expert`` aggregation (the server only needs the [K, S] superclass mask).

    Entries are non-null only for superclasses the teacher actually has data on
    (``local_test_acc_per_super`` is None elsewhere), so a teacher can only be a
    super-expert on a superclass it owns and classifies well. Returns ``None`` if the
    manifest predates per-super accuracies (caller then derives from the fine mask).
    """
    manifest_path = Path(teachers_dir).parent / "teachers_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing teachers_manifest.json at {manifest_path}.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    mask = np.zeros((num_teachers, num_super), dtype=np.int8)
    seen = set()
    for entry in manifest:
        cid = int(entry.get("cid", -1))
        if not (0 <= cid < num_teachers):
            continue
        metrics = entry.get("crea")
        if not isinstance(metrics, dict):
            raise ValueError(f"cid={cid}: no 'crea' block in manifest entry.")
        acc_per_super = metrics.get("local_test_acc_per_super")
        if not isinstance(acc_per_super, list):
            return None  # old manifest without per-super accuracies → caller derives from fine mask
        for s, acc in enumerate(acc_per_super[:num_super]):
            if acc is not None and float(acc) >= threshold:
                mask[cid, s] = 1
        seen.add(cid)
    missing = set(range(num_teachers)) - seen
    if missing:
        raise ValueError(f"Manifest missing entries for cids {sorted(missing)} (expected {num_teachers})")
    return mask


# ==========================================================================
# Proxy loader
# ==========================================================================

    # NOTE: proxy-loading is now centralized in `DatasetConfig.make_proxy_loader`.
    # The old `load_proxy_data` wrapper was removed — callers should call the
    # dataset config directly. This keeps dataset-specific proxy handling in one
    # place and removes branching from the analysis builder.

# ===========================================================================
# Diagnostics + artifact writers
# ===========================================================================

def _per_class_acc(preds: np.ndarray, y_true: np.ndarray, num_classes: int) -> np.ndarray:
    acc = np.full(num_classes, np.nan, dtype=np.float32)
    for c in range(num_classes):
        mask = y_true == c
        if mask.sum() > 0:
            acc[c] = float((preds[mask] == c).mean())
    return acc


def _compute_proxy_diagnostics(
    extracted: Dict, y_true: np.ndarray, mask: np.ndarray, *,
    num_classes: int, fine_to_coarse: Optional[np.ndarray] = None,
) -> Dict:
    diag: Dict = {}
    n = len(y_true)
    K = mask.shape[0]

    method_preds = {
        "feddf": extracted["avg_pred"], "consensus": extracted["vote_pred"],
        "oracle": extracted["oracle_pred"], "expert": extracted["expert_pred"],
    }
    for mname, preds in method_preds.items():
        acc_c = _per_class_acc(preds, y_true, num_classes)
        diag[f"per_class_acc_{mname}"] = acc_c.tolist()
        diag[f"acc_{mname}"] = float(np.nanmean(acc_c))

    if fine_to_coarse is not None:
        n_super = int(fine_to_coarse.max()) + 1
        coarse_labels = fine_to_coarse[y_true]
        for mname, preds in method_preds.items():
            coarse_preds = fine_to_coarse[preds.astype(int)]
            super_acc = _per_class_acc(coarse_preds, coarse_labels, n_super)
            diag[f"per_super_acc_{mname}"] = super_acc.tolist()

    all_pairs = [
        ("feddf", "expert"), ("feddf", "consensus"), ("feddf", "oracle"),
        ("oracle", "expert"), ("oracle", "consensus"),
    ]
    for ma, mb in all_pairs:
        pa, pb = method_preds[ma], method_preds[mb]
        differ = pa != pb
        n_differ = int(differ.sum())
        wrong_a = pa[differ] != y_true[differ] if n_differ > 0 else np.array([], dtype=bool)
        right_b = pb[differ] == y_true[differ] if n_differ > 0 else np.array([], dtype=bool)
        n_corrections = int((wrong_a & right_b).sum()) if n_differ > 0 else 0
        n_regressions = int((~wrong_a & ~right_b).sum()) if n_differ > 0 else 0
        diag[f"disagree_{ma}_vs_{mb}"] = {
            "n_differ": n_differ,
            "frac_differ": round(n_differ / max(n, 1), 4),
            "n_corrections_toward_b": n_corrections,
            "n_regressions_from_a": n_regressions,
        }

    ec = extracted["expert_count_per_sample"]
    diag["expert_count_hist"] = np.bincount(ec, minlength=K + 1).tolist()
    diag["frac_zero_expert"] = float((ec == 0).mean())
    diag["mean_expert_count"] = float(ec.mean())

    oracle_logits = softmax_np(extracted["oracle_avg_logits"])
    oracle_pred_hard = extracted["oracle_pred"]
    method_logit_keys = {"feddf": "avg_logits", "consensus": "consensus_avg_logits", "expert": "expert_avg_logits"}
    for mname, lkey in method_logit_keys.items():
        m_probs = softmax_np(extracted[lkey])
        m_pred = m_probs.argmax(axis=1)
        agree = float((m_pred == oracle_pred_hard).mean())
        kl = float((oracle_logits * (np.log(oracle_logits + 1e-12) - np.log(m_probs + 1e-12))).sum(axis=1).mean())
        diag[f"oracle_vs_{mname}_agree"] = agree
        diag[f"oracle_vs_{mname}_kl"] = kl
    diag["oracle_vs_oracle_agree"] = 1.0
    diag["oracle_vs_oracle_kl"] = 0.0
    diag["oracle_expert_agree_rate"] = float(extracted["oracle_expert_agree"].mean())
    diag["oracle_expert_kl_mean"] = float(extracted["oracle_expert_kl"].mean())

    known_mass_frac = []
    routing_to_teacher = []
    global_probs = softmax_np(extracted["expert_avg_logits"])  # invariant in k
    for k in range(K):
        known_k = mask[k].astype(bool)
        n_known_classes = int(known_k.sum())
        routed = known_k[y_true.astype(int)]
        frac_routed = float(routed.mean())
        routing_to_teacher.append(round(frac_routed, 4))
        p_known_mass = global_probs[:, known_k].sum(axis=1)
        known_mass_frac.append({
            "cid": k, "n_known_classes": n_known_classes,
            "frac_proxy_routed_to_teacher": frac_routed,
            "p_known_mass_mean": round(float(p_known_mass.mean()), 4),
            "p_known_mass_std": round(float(p_known_mass.std()), 4),
        })
    diag["per_client_routing"] = known_mass_frac
    diag["n_known_classes_per_client"] = mask.sum(axis=1).tolist()
    return diag


# ===========================================================================
# Builder
# ===========================================================================

class ProxyAnalysisBuilder:
    """Build proxy_analysis.npz (+ diagnostics) for one teacher set over the proxy."""

    def __init__(
        self, analysis_dir: Path, *, data_dir: Path, dataset: str,
        partition_dir: Optional[Path] = None, out_dir: Optional[Path] = None,
        proxy_split_npz: Optional[Path] = None,
        batch_size: int = 256, num_workers: int = 2, seed: int = 42, device: str = "auto",
        compressed: bool = True,
    ) -> None:
        self.teachers_dir = Path(analysis_dir)
        self.data_dir = Path(data_dir)
        self.dataset = dataset
        self.partition_dir = partition_dir
        self.proxy_split_npz = (
            Path(proxy_split_npz) if proxy_split_npz is not None
            else get_dataset_config(dataset).proxy_path_for_seed(seed, data_dir)
        )
        self.out_dir = Path(out_dir) if out_dir else self.teachers_dir / "proxy_analysis"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.device = device
        self.compressed = compressed
        self.expert_threshold = get_dataset_config(self.dataset).expert_acc_class_threshold

    def _resolve_mask(self, num_teachers: int, num_classes: int) -> np.ndarray:
        # Mask resolution is unified across datasets: derive the mask from raw
        # local-test accuracies in `teachers_manifest.json` using the configured
        # experiment threshold. The threshold is intentionally not baked into
        # teacher training so different experiments can choose 0.7, 0.6, etc.
        mask = build_expert_mask_from_manifest(
            self.teachers_dir, num_teachers=num_teachers, num_classes=num_classes,
            threshold=self.expert_threshold,
        )
        print(f"[*] Expert mask (threshold={self.expert_threshold}): {int(mask.sum())} entries. "
              f"Experts per class: {mask.sum(axis=0).tolist()}")
        return mask

    def _resolve_super_mask(self, num_teachers: int, num_super: int) -> Optional[np.ndarray]:
        """Direct coarse-accuracy superclass competence mask [K, S] (CIFAR-100)."""
        sm = build_super_mask_from_manifest(
            self.teachers_dir, num_teachers=num_teachers, num_super=num_super,
            threshold=self.expert_threshold,
        )
        if sm is None:
            print("[*] Super-expert mask: per-super accuracies absent → deriving from fine mask")
        else:
            print(f"[*] Super-expert mask (threshold={self.expert_threshold}): {int(sm.sum())} entries. "
                  f"Experts per superclass: {sm.sum(axis=0).tolist()}")
        return sm

    def build(self, *, force: bool = False) -> Dict:
        if not self.teachers_dir.is_dir():
            raise FileNotFoundError(f"teachers dir not found: {self.teachers_dir}")
        out_dir = self.out_dir
        out_npz = out_dir / "proxy_analysis.npz"

        if not self.proxy_split_npz.is_file():
            raise FileNotFoundError(
                f"seed-specific proxy split not found: {self.proxy_split_npz}"
            )
        with np.load(self.proxy_split_npz, allow_pickle=True) as split:
            if "proxy_idx" not in split.files:
                raise ValueError(f"proxy split lacks proxy_idx: {self.proxy_split_npz}")
            expected_proxy_idx = np.asarray(split["proxy_idx"], dtype=np.int64)
            if self.seed != 42 and self.dataset in {"mnist", "fmnist", "cifar"}:
                if "seed" not in split.files:
                    raise ValueError(f"new proxy split lacks seed metadata: {self.proxy_split_npz}")
                split_seed = int(np.asarray(split["seed"]).item())
                if split_seed != self.seed:
                    raise ValueError(
                        f"proxy split seed mismatch {split_seed}!={self.seed}: "
                        f"{self.proxy_split_npz}"
                    )
        proxy_fp = _proxy_idx_fingerprint(expected_proxy_idx)

        # ``list_teachers`` deliberately raises when no .pt exists.  Do not call
        # it when the canonical checkpoint-free caches are present.
        cache_paths, teacher_paths = _list_teacher_sources(self.teachers_dir)
        if cache_paths:
            # Load once here both to establish the fingerprint used by the
            # idempotence check and to fail before accepting stale output.
            cached_teacher_logits, cached_labels, fp = _load_teacher_logit_caches(
                cache_paths, dataset=self.dataset, seed=self.seed,
                proxy_idx=expected_proxy_idx,
            )
        elif teacher_paths:
            cached_teacher_logits = cached_labels = None
            fp = _teachers_fingerprint(teacher_paths)
        else:
            raise FileNotFoundError(
                f"No teacher proxy-logit caches or checkpoints in {self.teachers_dir}"
            )
        if self.partition_dir is not None:
            config_path = Path(self.partition_dir) / "config.yaml"
            if not config_path.is_file():
                raise ValueError(f"partition lacks config.yaml: {self.partition_dir}")
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            partition_seed = int(config.get("seed", -1))
            partition_proxy = Path(str(config.get("proxy_split_npz", ""))).name
            partition_proxy_fp = str(config.get("proxy_idx_sha256", ""))
            if partition_seed != self.seed:
                raise ValueError(
                    f"partition/proxy-analysis seed mismatch {partition_seed}!={self.seed}"
                )
            if partition_proxy != self.proxy_split_npz.name:
                raise ValueError(
                    "partition/proxy-analysis split mismatch: "
                    f"{partition_proxy!r}!={self.proxy_split_npz.name!r}"
                )
            if self.seed != 42 and partition_proxy_fp != proxy_fp:
                raise ValueError(
                    "partition/proxy-analysis proxy_idx hash mismatch: "
                    f"{partition_proxy_fp!r}!={proxy_fp!r}"
                )

        if not force and out_npz.is_file():
            try:
                with np.load(out_npz, mmap_mode="r", allow_pickle=True) as existing:
                    cached_seed = int(np.asarray(existing["seed"]).item())
                    cached_proxy_fp = str(np.asarray(existing["proxy_idx_sha256"]).item())
                    if (str(existing["teacher_fingerprint"][0]) == fp
                            and str(existing["dataset"][0]) == self.dataset
                            and float(existing["expert_threshold"][0]) == self.expert_threshold
                            and cached_seed == self.seed
                            and cached_proxy_fp == proxy_fp):
                        print(f"[SKIP] proxy_analysis up to date at {out_npz}")
                        return {"skipped": True, "out_dir": str(out_dir)}
                print(f"[INFO] proxy_analysis at {out_npz} does not match current → rebuilding")
            except Exception as e:
                print(f"[WARN] could not read existing cache ({e}); rebuilding")

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        device_t = resolve_device(self.device)

        # Load proxy via the dataset config (centralized proxy handling).
        ds_cfg = get_dataset_config(self.dataset)
        loader, proxy_idx, y_true_proxy = ds_cfg.make_proxy_loader(
            self.data_dir, proxy_path=self.proxy_split_npz,
            batch_size=self.batch_size, num_workers=self.num_workers,
        )
        observed_proxy_fp = _proxy_idx_fingerprint(proxy_idx)
        if observed_proxy_fp != proxy_fp:
            raise ValueError(
                f"proxy loader changed proxy_idx: {observed_proxy_fp}!={proxy_fp}"
            )

        ds_cfg = get_dataset_config(self.dataset)
        num_classes = ds_cfg.num_classes
        fine_to_coarse: Optional[np.ndarray] = None
        try:
            with np.load(self.proxy_split_npz, allow_pickle=True) as _proxy_npz:
                if "fine_to_coarse" in _proxy_npz:
                    fine_to_coarse = _proxy_npz["fine_to_coarse"].astype(np.int64)
                if fine_to_coarse is not None:
                    print(f"[*] fine_to_coarse loaded: {len(fine_to_coarse)} classes → "
                          f"{int(fine_to_coarse.max()) + 1} superclasses")
        except Exception:
            pass

        num_teachers = len(cache_paths) if cache_paths else len(teacher_paths)
        mask = self._resolve_mask(num_teachers, num_classes)
        super_mask = None
        if fine_to_coarse is not None:
            super_mask = self._resolve_super_mask(num_teachers, int(fine_to_coarse.max()) + 1)

        # Prefer the compact, checkpoint-free surface emitted during teacher
        # training.  Old runs remain rebuildable from checkpoints.
        if cached_teacher_logits is not None:
            teacher_logits = cached_teacher_logits
            teacher_preds = teacher_logits.argmax(axis=2).astype(np.uint8)
            y_collected = cached_labels
        else:
            teachers = load_teachers(teacher_paths, self.dataset, device_t)
            for teacher in teachers:
                teacher.eval()
            teacher_logits, teacher_preds, y_collected = _run_teachers_collect(
                teachers, loader, device_t,
            )
        if not np.array_equal(np.asarray(y_true_proxy), np.asarray(y_collected)):
            raise ValueError("teacher cache/checkpoint labels do not match the seeded proxy loader")
        y_true_proxy = np.asarray(y_collected, dtype=np.int64)
        extracted = _compute_global_targets(
            teacher_logits, teacher_preds, y_true_proxy, mask, fine_to_coarse=fine_to_coarse,
            teacher_knows_super_mask=super_mask,
        )

        proxy_diag = _compute_proxy_diagnostics(
            extracted, y_true_proxy.astype(np.int64), mask,
            num_classes=num_classes, fine_to_coarse=fine_to_coarse,
        )

        arrays = {
            "teacher_fingerprint": np.array([fp]),
            "dataset": np.array([self.dataset]),
            "seed": np.array(int(self.seed), dtype=np.int64),
            "proxy_split_file": np.array([self.proxy_split_npz.name]),
            "proxy_idx_sha256": np.array([proxy_fp]),
            "expert_threshold": np.array([float(self.expert_threshold)]),
            "proxy_idx": proxy_idx.astype(np.int32, copy=False),
            "y_true_proxy": y_true_proxy.astype(np.uint8, copy=False),
            "vote_pred": extracted["vote_pred"], "avg_pred": extracted["avg_pred"],
            "oracle_pred": extracted["oracle_pred"], "expert_pred": extracted["expert_pred"],
            "avg_logits": extracted["avg_logits"],
            "consensus_avg_logits": extracted["consensus_avg_logits"],
            "oracle_avg_logits": extracted["oracle_avg_logits"],
            "expert_avg_logits": extracted["expert_avg_logits"],
            "confidence_avg_logits": extracted["confidence_avg_logits"],
            "energy_avg_logits": extracted["energy_avg_logits"],
            "conf_avgT": extracted["conf_avgT"],
            "correct_vote": extracted["correct_vote"], "correct_avg": extracted["correct_avg"],
            "expert_count_per_sample": extracted["expert_count_per_sample"],
            "oracle_expert_agree": extracted["oracle_expert_agree"],
            "oracle_expert_kl": extracted["oracle_expert_kl"],
            "teacher_acc": extracted["teacher_acc"],
            "teacher_class_acc": extracted["teacher_class_acc"],
            "teacher_knows_class_mask": mask,
            "teacher_preds_cache": extracted["teacher_preds_cache"],
            "teacher_logits_cache": extracted["teacher_logits_cache"],
        }
        if fine_to_coarse is not None:
            arrays["proxy_coarse_labels"] = fine_to_coarse[y_true_proxy.astype(int)].astype(np.uint8)
            arrays["fine_to_coarse"] = fine_to_coarse.astype(np.int64)
            if extracted.get("teacher_knows_super_mask") is not None:
                # [K, S] superclass competence used for the expert aggregation
                arrays["teacher_knows_super_mask"] = extracted["teacher_knows_super_mask"]

        out_dir.mkdir(parents=True, exist_ok=True)
        saver = np.savez_compressed if self.compressed else np.savez
        saver(out_dir / "proxy_analysis.npz", **arrays)
        proxy_diag["provenance"] = {
            "dataset": self.dataset,
            "seed": int(self.seed),
            "proxy_split_file": self.proxy_split_npz.name,
            "proxy_idx_sha256": proxy_fp,
            "teacher_fingerprint": fp,
        }
        (out_dir / "proxy_diagnostics.json").write_text(json.dumps(proxy_diag, indent=2), encoding="utf-8")

        return {"summary": self._summary(arrays, extracted, proxy_diag, num_classes, y_true_proxy, mask),
                "skipped": False}

    @staticmethod
    def _summary(arrays, extracted, proxy_diag, num_classes, y_true_proxy, mask) -> Dict:
        _nan = float("nan")
        _method_csv: Dict = {}
        for _m in ("feddf", "consensus", "oracle", "expert"):
            _dis = proxy_diag.get(f"disagree_feddf_vs_{_m}", {})
            _n_dif = _dis.get("n_differ", 0)
            _n_tot = int(len(y_true_proxy))
            _corr_b = _dis.get("n_corrections_toward_b", 0)
            _per_super = proxy_diag.get(f"per_super_acc_{_m}")
            _target_super_acc = float(np.nanmean(_per_super)) if _per_super is not None else _nan
            _method_csv[_m] = {
                "distill_argmax_disagree_vs_feddf": 0.0 if _m == "feddf" else float(_n_dif / max(_n_tot, 1)),
                "distill_argmax_disagree_correction_frac":
                    float(_corr_b / max(_n_dif, 1)) if _n_dif > 0 and _m != "feddf" else _nan,
                "distill_oracle_argmax_agreement": proxy_diag.get(f"oracle_vs_{_m}_agree", _nan),
                "distill_oracle_mean_kl": proxy_diag.get(f"oracle_vs_{_m}_kl", _nan),
                "distill_expert_card_mean": proxy_diag.get("mean_expert_count", _nan) if _m == "expert" else _nan,
                "distill_frac_zero_teachers": proxy_diag.get("frac_zero_expert", _nan) if _m == "expert" else _nan,
                "distill_target_super_acc": _target_super_acc,
            }
        return {
            "dataset": str(arrays["dataset"][0]),
            "num_classes": num_classes,
            "mean_conf_avgT": float(arrays["conf_avgT"].mean()),
            "acc_feddf": float(extracted["correct_avg"].mean()),
            "acc_vote": float(extracted["correct_vote"].mean()),
            "acc_oracle": proxy_diag.get("acc_oracle", _nan),
            "acc_expert": proxy_diag.get("acc_expert", _nan),
            "frac_zero_expert": proxy_diag.get("frac_zero_expert", _nan),
            "oracle_expert_agree_rate": proxy_diag.get("oracle_expert_agree_rate", _nan),
            "n_experts_per_class": mask.sum(axis=0).tolist(),
            "per_method": _method_csv,
        }


def build_proxy_analysis(
    analysis_dir: Path, *, data_dir: Path, dataset: str,
    partition_dir: Optional[Path] = None, out_dir: Optional[Path] = None,
    proxy_split_npz: Optional[Path] = None,
    batch_size: int = 256, num_workers: int = 2, seed: int = 42, device: str = "auto",
    compressed: bool = True, force: bool = False,
) -> Dict:
    """Functional entry point (kept for callers). Delegates to ProxyAnalysisBuilder."""
    return ProxyAnalysisBuilder(
        analysis_dir, data_dir=data_dir, dataset=dataset, partition_dir=partition_dir,
        out_dir=out_dir, proxy_split_npz=proxy_split_npz,
        batch_size=batch_size, num_workers=num_workers, seed=seed,
        device=device, compressed=compressed,
    ).build(force=force)
