#!/usr/bin/env python3
"""run_personal_kd.py — personal KD runner.

Place under: oracle_distillation/cli/run_personal_kd.py

This runner trains one personal student per client from targets built on top
of an existing proxy_analysis.npz. All KD training goes through
``oracle_distillation.distill.kd_runner.KdRunner``.

Supported personal methods
--------------------------
  personal_expert
      v1 hard-routing baseline (no warm-start):
      teacher_k if argmax(teacher_k) in known_k else global expert.

  personal_oracle
      non-deployable upper bound (no warm-start):
      teacher_k if y_true in known_k else global oracle.

  personal_class_mask
      teacher_k controls only known-class slots; global controls unknown slots.
      Warm-started from the SAME global checkpoint selected by --global_source
      (oracle/expert/feddf/...) — warm-start and the class_mask anchor always
      point at one selected method, never two different ones. LwF has been
      removed from the pipeline.

      Two independent choices, both selected by CLI flags:
        --global_source        which method is "the global model" (expert/
                                oracle/feddf/consensus/confidence/energy).
        --global_logits_kind    for that method, whether the anchor is the
                                cached ensemble average ("ensemble", default —
                                {method}_avg_logits from proxy_analysis.npz) or
                                the REAL forward-pass logits of the distilled
                                {method}/student.pt ("distilled" — read from
                                student_logits.npz, computed automatically by
                                KdRunner.distill_global / backfilled by
                                scripts/build_student_logits_cache.py).

      On-disk layout is keyed by --global_source: outputs land under
      ``<rel>/personal_class_mask/<global_source>/cXXX/`` (NOT under
      ``distillation/personal_class_mask/``), so sweeping --global_source over
      multiple values (feddf/energy/consensus/confidence/expert/oracle) never
      clobbers a sibling run — each gets its own metrics (and optional checkpoint) and
      --skip_if_done works correctly per global_source.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from oracle_distillation.analysis.target_builders import (
    get_target_builder, ProxyContext, personal_target_diagnostics,
)
from oracle_distillation.config import ExecutionSpec, build_execution_spec
from oracle_distillation.checkpoints import load_checkpoint_state, save_model_checkpoint
from oracle_distillation.test_outputs import save_test_outputs
from oracle_distillation.distill.kd_runner import KdRunner
from oracle_distillation.metrics_io import (
    append_row, resume_csv_rows, PERSONAL_CSV_COLUMNS, PERSONAL_CSV_KEY,
)
from oracle_distillation.utils import (
    resolve_device, set_seed, ece_np, entropy_mean_np, softmax_np,
    per_class_accuracy, client_label, collect_logits,
)
from o3_local.teacher_trainer import load_client_npz
from oracle_distillation.run_paths import Paths, discover_proxy_jobs
from oracle_distillation.experiment_paths import assert_seed_contained
from oracle_distillation.cli.common_args import add_training_args
from data.dataset_config import get_dataset_config

# personal_<name> → unified target variant. class_mask warm-starts from
# whichever method --global_source selects; expert/oracle are target-only
# baselines with no warm-start.
_PERSONAL_TO_VARIANT: Dict[str, str] = {
    "personal_expert": "personal_expert",
    "personal_oracle": "personal_oracle",
    "personal_class_mask": "personal_class_mask",
}
GLOBAL_METHOD_DEFAULT = "expert"

# --global_source value -> distillation method-dir name, for loading the REAL
# global student.pt checkpoint ("avg" is the array-key alias for "feddf"; every
# other source name doubles as its own method dir).
_GLOBAL_SOURCE_TO_METHOD_DIR: Dict[str, str] = {
    "expert": "expert", "oracle": "oracle", "feddf": "feddf",
    "consensus": "consensus", "confidence": "confidence", "energy": "energy",
}


# ===========================================================================
# Per-class evaluation
# ===========================================================================

@torch.no_grad()
def _eval_comprehensive(
        student: torch.nn.Module,
        test_loader: DataLoader,
        device: torch.device,
        num_classes: int,
        known_mask: np.ndarray,
        fine_to_coarse: Optional[np.ndarray] = None,
        global_student: Optional[torch.nn.Module] = None,
) -> Dict[str, Any]:
    """Single-pass comprehensive evaluation returning all v5 metrics."""
    student.eval()
    if global_student is not None:
        global_student.eval()

    all_logits_p: list = []
    all_logits_g: list = []
    all_labels: list = []

    for x, y in test_loader:
        x = x.to(device, non_blocking=True)
        lp = student(x).cpu()
        all_logits_p.append(lp)
        all_labels.append(y)
        if global_student is not None:
            lg = global_student(x).cpu()
            all_logits_g.append(lg)

    logits_p = torch.cat(all_logits_p, dim=0).numpy().astype(np.float32)
    labels   = torch.cat(all_labels,   dim=0).numpy().astype(np.int64)
    logits_g = torch.cat(all_logits_g, dim=0).numpy().astype(np.float32) if all_logits_g else None

    pred_p = logits_p.argmax(axis=1)
    pred_g = logits_g.argmax(axis=1) if logits_g is not None else None

    # Per-class accuracy (NaN for classes absent from the test set)
    per_class = per_class_accuracy(pred_p, labels, num_classes)
    overall = float((pred_p == labels).mean())

    # Known / unknown split (based on GT class)
    known_of_gt = known_mask[labels]
    k_idx = known_of_gt
    u_idx = ~known_of_gt
    known_acc   = float((pred_p[k_idx] == labels[k_idx]).mean()) if k_idx.any()  else float("nan")
    unknown_acc = float((pred_p[u_idx] == labels[u_idx]).mean()) if u_idx.any()  else float("nan")
    gap_ku = (known_acc - unknown_acc) if not (np.isnan(known_acc) or np.isnan(unknown_acc)) else float("nan")

    # Global (expert) student's accuracy restricted to this client's known classes —
    # the deployable baseline that distill_teacher_known_acc / model_known_acc are
    # compared against. Same eval surface (global test set, GT in known classes).
    global_known_acc = (
        float((pred_g[k_idx] == labels[k_idx]).mean())
        if (pred_g is not None and k_idx.any()) else float("nan")
    )

    known_ece     = ece_np(logits_p[k_idx], labels[k_idx]) if k_idx.any()  else float("nan")
    unknown_ece   = ece_np(logits_p[u_idx], labels[u_idx]) if u_idx.any()  else float("nan")
    known_entropy = entropy_mean_np(logits_p[k_idx])        if k_idx.any()  else float("nan")
    unk_entropy   = entropy_mean_np(logits_p[u_idx])        if u_idx.any()  else float("nan")

    # Superclass metrics (hierarchical datasets only). Definitions match the teacher
    # and global-student paths:
    #   model_super_acc[s]    = COARSE accuracy = P(predicted superclass == s | true s)
    #   fine_in_super[s]      = fine accuracy among samples whose superclass is correct
    _nan = float("nan")
    super_acc_per_super: Optional[list] = None
    fine_in_super_per_super: Optional[list] = None
    if fine_to_coarse is not None:
        n_super = int(fine_to_coarse.max()) + 1
        coarse_labels = fine_to_coarse[labels]
        coarse_pred   = fine_to_coarse[pred_p]
        coarse_correct = coarse_pred == coarse_labels
        per_super_acc = np.full(n_super, np.nan)
        per_super_fine = np.full(n_super, np.nan)
        for s in range(n_super):
            in_s = coarse_labels == s
            if in_s.any():
                per_super_acc[s] = float((coarse_pred[in_s] == s).mean())   # coarse correctness
                cc = in_s & coarse_correct
                if cc.any():
                    per_super_fine[s] = float((pred_p[cc] == labels[cc]).mean())
        model_super_acc = float(np.nanmean(per_super_acc))
        fine_in_super   = float((pred_p[coarse_correct] == labels[coarse_correct]).mean()) \
                          if coarse_correct.any() else _nan
        super_acc_per_super     = [None if np.isnan(v) else round(float(v), 4) for v in per_super_acc]
        fine_in_super_per_super = [None if np.isnan(v) else round(float(v), 4) for v in per_super_fine]
    else:
        model_super_acc = _nan
        fine_in_super   = _nan

    # Flip analysis vs global student
    if pred_g is not None:
        flipped = pred_p != pred_g
        flip_rate = float(flipped.mean())
        kp = known_mask[pred_p]
        kg = known_mask[pred_g]
        within_known  = int((flipped & kp  & kg).sum())
        K2U           = int((flipped & ~kp & kg).sum())
        U2K           = int((flipped & kp  & ~kg).sum())
        U2Uprime      = int((flipped & ~kp & ~kg).sum())
        correct_p = pred_p == labels
        correct_g = pred_g == labels
        correction     = int((flipped & correct_p  & ~correct_g).sum())
        regression     = int((flipped & ~correct_p & correct_g).sum())
        neutral_churn  = int((flipped & ~(correct_p ^ correct_g)).sum())
        wk_corr = int((flipped & kp & correct_p  & ~correct_g).sum())
        wk_reg  = int((flipped & kp & ~correct_p & correct_g).sum())
        wk_net  = wk_corr - wk_reg
        probs_p = softmax_np(logits_p)
        probs_g = softmax_np(logits_g)
        if flipped.any():
            dprob_gt_flip = float(
                (probs_p[flipped, labels[flipped]] - probs_g[flipped, labels[flipped]]).mean()
            )
        else:
            dprob_gt_flip = _nan
        flip_dict = {
            "distill_flip_rate":      flip_rate,
            "distill_within_known":   within_known,
            "distill_K2U":            K2U,
            "distill_U2K":            U2K,
            "distill_U2Uprime":       U2Uprime,
            "distill_correction":     correction,
            "distill_regression":     regression,
            "distill_neutral_churn":  neutral_churn,
            "distill_wk_net":         wk_net,
            "distill_dprob_gt_flip":  dprob_gt_flip,
        }
    else:
        flip_dict = {k: _nan for k in (
            "distill_flip_rate", "distill_within_known", "distill_K2U", "distill_U2K",
            "distill_U2Uprime", "distill_correction", "distill_regression",
            "distill_neutral_churn", "distill_wk_net", "distill_dprob_gt_flip",
        )}

    return {
        "model_test_acc":           overall,
        "model_test_acc_per_class": per_class.tolist(),
        "model_known_acc":          known_acc,
        "model_unknown_acc":        unknown_acc,
        "model_gap_ku":             gap_ku,
        "model_known_ece":          known_ece,
        "model_unknown_ece":        unknown_ece,
        "model_known_entropy":      known_entropy,
        "model_unknown_entropy":    unk_entropy,
        "model_super_acc":          model_super_acc,
        "model_fine_in_super_acc":  fine_in_super,
        "model_super_acc_per_super":         super_acc_per_super,
        "model_fine_in_super_acc_per_super": fine_in_super_per_super,
        "distill_global_known_acc": global_known_acc,
        **flip_dict,
    }


@torch.no_grad()
def _eval_local_test(
        student: torch.nn.Module,
        train_ds,
        local_test_idx: Optional[np.ndarray],
        device: torch.device,
        batch_size: int,
) -> Dict[str, float]:
    """Evaluate the personal student on the client's held-out local-test split.

    ``local_test_idx`` indexes the full train dataset (same convention as teacher
    training). Returns NaNs when the partition carries no local-test split
    (e.g. CIFAR-100, SPLIT[1]=0).
    """
    _nan = float("nan")
    if local_test_idx is None or len(local_test_idx) == 0:
        return {"local_test_acc": _nan, "model_local_test_ece": _nan,
                "model_local_test_entropy": _nan}
    loader = DataLoader(
        Subset(train_ds, [int(i) for i in local_test_idx]),
        batch_size=batch_size, shuffle=False, num_workers=2,
        pin_memory=(device.type == "cuda"),
    )
    logits, labels = collect_logits(student, loader, device)
    return {
        "local_test_acc":           float((logits.argmax(axis=1) == labels).mean()),
        "model_local_test_ece":     ece_np(logits, labels),
        "model_local_test_entropy": entropy_mean_np(logits),
    }


# ===========================================================================
# Discovery
# ===========================================================================

# ===========================================================================
# Warm-start: locate and load the global student checkpoint
# ===========================================================================

def _global_student_path(paths: Paths, rel: str, global_source: str, seed: int = 42) -> Path:
    """Checkpoint path for a --global_source value (e.g. "avg" -> distillation/feddf/)."""
    method_dir = _GLOBAL_SOURCE_TO_METHOD_DIR.get(global_source, global_source)
    return paths.method_dir_for_rel(rel, method_dir, seed) / "student.pt"


def _maybe_warm_start(
        student: torch.nn.Module,
        warm_start_method: Optional[str],
        paths: Paths,
        rel: str,
        device: torch.device,
        seed: int = 42,
) -> bool:
    """Load the required global student state_dict into ``student``."""
    if not warm_start_method:
        return False
    ckpt = _global_student_path(paths, rel, warm_start_method, seed=seed)
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"required global warm-start checkpoint not found: {ckpt}. "
            "Phase A is checkpoint-free by default; rerun the required global "
            "source with --save_models before personal/Phase-B training."
        )
    state = load_checkpoint_state(ckpt, map_location="cpu")
    student.load_state_dict(state, strict=True)
    student.to(device)
    return True


# ===========================================================================
# Distilled global logits — --global_logits_kind distilled anchor
# ===========================================================================

@torch.no_grad()
def _distilled_global_logits(
        paths: Paths, rel: str, global_source: str, seed: int,
        student_arch: str, num_classes: int, train_ds, proxy_idx: np.ndarray,
        device: torch.device, batch_size: int = 256,
) -> np.ndarray:
    """REAL forward-pass logits of the distilled global student.pt (selected by
    ``global_source``) over the proxy, in ``proxy_idx`` order — same order as
    ``teacher_logits_cache``, so the result drops in as ``z_global_override``
    without remapping.

    Fast path: read ``student_logits.npz`` (cached automatically by
    ``KdRunner.distill_global``, or backfilled by
    ``scripts/build_student_logits_cache.py``) and select the rows matching
    ``proxy_idx``. Falls back to a live forward pass (and writes the cache, so the
    next call is fast) if the cache is missing — e.g. checkpoints trained before
    this feature existed and not yet backfilled.
    """
    method_dir = _GLOBAL_SOURCE_TO_METHOD_DIR.get(global_source, global_source)
    out_dir = paths.method_dir_for_rel(rel, method_dir, seed)
    ckpt = out_dir / "student.pt"
    cache_path = out_dir / "student_logits.npz"
    proxy_idx = np.asarray(proxy_idx).astype(np.int64)
    if cache_path.is_file():
        cache = np.load(cache_path)
        cached_idx = cache["proxy_idx"].astype(np.int64)
        if np.array_equal(cached_idx, proxy_idx):
            return cache["proxy_logits"].astype(np.float32)
        # Different proxy ordering/subset than expected — fall through to a live
        # forward pass rather than risk silently misaligned rows.
        print(f"    [WARN] {cache_path} proxy_idx mismatch; recomputing live.")

    if not ckpt.is_file():
        raise FileNotFoundError(
            f"no aligned student_logits.npz and no distilled global checkpoint at "
            f"{ckpt} (global_source={global_source!r} -> method dir {method_dir!r}). "
            "Rerun global KD with --save_models to permit a live reconstruction."
        )

    print(f"    [INFO] No usable student_logits cache at {cache_path}; "
          "computing live forward pass (consider running "
          "scripts/build_student_logits_cache.py to backfill).")
    from oracle_distillation.models import build_model as _bm
    from oracle_distillation.utils import forward_logits_for_indices
    model = _bm(student_arch, num_classes=num_classes).to(device)
    model.load_state_dict(load_checkpoint_state(ckpt, map_location=device), strict=True)
    model.eval()
    logits = forward_logits_for_indices(model, train_ds, proxy_idx, device, batch_size)
    try:
        from oracle_distillation.distill.kd_runner import save_student_logits_cache
        partition_dir = paths.partition_dir_for_rel(rel, seed)
        save_student_logits_cache(
            student=model, ds_full=train_ds, proxy_idx=proxy_idx,
            partition_dir=partition_dir, out_dir=out_dir, device=device, batch_size=batch_size,
        )
    except Exception as e:
        print(f"    [WARN] could not self-heal student_logits cache: {e}")
    return logits


# ===========================================================================
# Build target tensor
# ===========================================================================

def _build_target(
        analysis_data: Dict[str, np.ndarray],
        k: int,
        variant: str,
        *,
        T: float,
        global_source: str,
        fine_to_coarse: Optional[np.ndarray] = None,
        known_fine: Optional[np.ndarray] = None,
        full_coarse_budget: bool = False,
        z_global_override: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build the personal target via the unified TargetBuilder.

    For ``personal_class_mask`` a non-None ``fine_to_coarse`` selects the
    hierarchical (double-depth) CIFAR-100 target; otherwise the flat one.
    ``z_global_override`` (real distilled student.pt logits, when
    ``--global_logits_kind distilled``) is ignored by the expert/oracle variants.
    """
    builder = get_target_builder(variant)
    return builder.build(
        ProxyContext(analysis_data), k, T=T, global_source=global_source,
        fine_to_coarse=fine_to_coarse, known_fine=known_fine,
        full_coarse_budget=full_coarse_budget, z_global_override=z_global_override,
    )


# ===========================================================================
# Routing diagnostics (v1 + v2)
# ===========================================================================

def _routing_diag(
        analysis_data: Dict[str, np.ndarray],
        k: int,
        variant: str,
        *,
        T: float,
        global_source: str,
        fine_to_coarse: Optional[np.ndarray] = None,
        known_fine: Optional[np.ndarray] = None,
        full_coarse_budget: bool = False,
        z_global_override: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Full diagnostics dict from personal_target_diagnostics (v5).

    Forwards the hierarchical inputs so the footprint/known reporting matches the
    target actually built (CIFAR-100 structural mask + per-superclass budget).
    ``z_global_override`` keeps the diagnostics consistent with the distilled-logits
    anchor when ``--global_logits_kind distilled``.
    """
    return personal_target_diagnostics(
        analysis_data, k, variant, T=T, global_source=global_source,
        fine_to_coarse=fine_to_coarse, known_fine=known_fine,
        full_coarse_budget=full_coarse_budget, z_global_override=z_global_override,
    )


def _teacher_known_acc(manifest_path: Path, k: int, known_classes: List[int]) -> float:
    """Read per-class local_test_acc for teacher k from teachers_manifest.json."""
    if not manifest_path.is_file() or not known_classes:
        return float("nan")
    try:
        manifest = json.loads(manifest_path.read_text())
        entry = next(
            (e for e in manifest if e.get("cid") == k or e.get("client_id") == k),
            None,
        )
        if entry is None:
            return float("nan")
        per_class = entry.get("local_test_acc_per_class")
        if per_class is None:
            return float("nan")
        accs = [per_class[c] for c in known_classes if c < len(per_class)]
        return float(np.mean(accs)) if accs else float("nan")
    except Exception:
        return float("nan")

# ===========================================================================
# One client × one variant
# ===========================================================================

def _run_one_client(
        *,
        k: int,
        analysis_data: Dict[str, np.ndarray],
        personal_method: str,
        variant: str,
        spec: ExecutionSpec,
        skip_if_done: bool,
        paths: Paths,
        rel: str,
        device: torch.device,
        test_loader: DataLoader,
        train_ds,
        global_csv: Path,
        all_rows: list,
        full_coarse_budget: bool = False,
        z_global_override: Optional[np.ndarray] = None,
        global_logits_kind: str = "ensemble",
) -> Optional[dict]:
    dataset = paths.dataset
    data_dir = paths.data_dir
    runs_root = paths.runs_root
    c_label = client_label(k)
    # personal_class_mask is keyed by global_source (rel/personal_class_mask/<g>/cXXX) —
    # it's the one personal method swept over g, and the flat distillation/<method>/cXXX
    # layout would clobber across g. Other personal methods (one config each) keep the
    # flat layout.
    if personal_method == "personal_class_mask":
        out_dir = paths.personal_class_mask_dir_for_rel(rel, spec.global_source, spec.seed) / c_label
    else:
        out_dir = paths.method_dir_for_rel(rel, personal_method, spec.seed) / c_label
    metrics_path = out_dir / "metrics.json"

    if skip_if_done and metrics_path.exists() and (out_dir / "test_outputs.npz").exists():
        print(f"    [SKIP] {personal_method}/seed_{spec.seed}/{c_label}")
        return None

    t0 = time.time()

    # Warm-start policy lives in MethodPolicy (spec.warm_start_method): class_mask
    # warm-starts from the global expert student; expert/oracle are target-only
    # baselines with no warm-start (warm_start_method is None for them).
    warm_start_method = spec.warm_start_method

    # ---- CIFAR-100 2-layer class_mask input -------------------------------
    # Passing fine_to_coarse selects the predicted-superclass builder, which drives
    # the fine layer from the *accuracy* mask teacher_knows_class_mask (knows_class) —
    # no structural known_fine needed. None for flat datasets / other variants ->
    # flat path unchanged. (hier_known stays None: the builder uses the accuracy mask.)
    fine_to_coarse: Optional[np.ndarray] = analysis_data.get("fine_to_coarse")
    hier_f2c: Optional[np.ndarray] = None
    hier_known: Optional[np.ndarray] = None
    if variant == "personal_class_mask" and dataset == "cifar100" and fine_to_coarse is not None:
        hier_f2c = np.asarray(fine_to_coarse)

    # ---- 1. Build personal target tensor + diagnostics ---------------------
    target = _build_target(
        analysis_data, k, variant,
        T=spec.temperature, global_source=spec.global_source,
        fine_to_coarse=hier_f2c, known_fine=hier_known, full_coarse_budget=full_coarse_budget,
        z_global_override=z_global_override,
    )

    diag = _routing_diag(
        analysis_data, k, variant,
        T=spec.temperature, global_source=spec.global_source,
        fine_to_coarse=hier_f2c, known_fine=hier_known, full_coarse_budget=full_coarse_budget,
        z_global_override=z_global_override,
    )
    known_classes: List[int] = diag.get("known_classes", [])
    n_known = diag.get("n_known_classes", len(known_classes))

    # Superclass metadata from fine_to_coarse (CIFAR-100 only)
    if fine_to_coarse is not None:
        known_super_ids = sorted(set(int(fine_to_coarse[c]) for c in known_classes))
        n_known_super = len(known_super_ids)
    else:
        known_super_ids = []
        n_known_super = 0

    out_dir.mkdir(parents=True, exist_ok=True)
    student_path = out_dir / "student.pt"

    # ---- 2. Build (optionally warm-started) student + train via KdRunner ----
    from oracle_distillation.models import build_model, dataset_default_arch
    _nc = get_dataset_config(dataset).num_classes
    student_arch = spec.student_arch or dataset_default_arch(dataset)
    student = build_model(student_arch, num_classes=_nc).to(device)
    warm_started = _maybe_warm_start(
        student, warm_start_method, paths, rel, device, seed=spec.seed,
    )
    ws_state = ({_k: _v.detach().clone() for _k, _v in student.state_dict().items()}
                if warm_started else None)

    kd_runner = KdRunner(device=device, seed=spec.seed)
    run_metrics: Dict[str, float] = kd_runner.distill_personal(
        spec, student=student, dataset=dataset, data_dir=data_dir,
        proxy_idx=analysis_data["proxy_idx"], target=target,
    )
    run_metrics["warm_started"] = bool(warm_started)

    warmstart_drift_l2: float = float("nan")
    if ws_state is not None:
        drift_sq = sum(
            float(((v - ws_state[_k]).float() ** 2).sum().item())
            for _k, v in student.state_dict().items()
        )
        warmstart_drift_l2 = float(drift_sq ** 0.5)

    if spec.save_models:
        save_model_checkpoint(
            student_path, student,
            architecture=student_arch, dataset=dataset, regime=rel,
            method=f"{spec.method}_cid{k:03d}", seed=spec.seed, config=spec,
        )
    warm_started = bool(run_metrics.get("warm_started", False))

    # ---- 3. Load global expert student for flip analysis -------------------
    _global_ckpt = _global_student_path(paths, rel, "expert", seed=spec.seed)
    global_student_eval: Optional[torch.nn.Module] = None
    if _global_ckpt.is_file():
        try:
            from oracle_distillation.models import build_model as _bm
            _nc = get_dataset_config(dataset).num_classes
            global_student_eval = _bm(spec.student_arch, num_classes=_nc).to(device)
            global_student_eval.load_state_dict(
                load_checkpoint_state(_global_ckpt, map_location="cpu"), strict=True
            )
            global_student_eval.eval()
            for _p in global_student_eval.parameters():
                _p.requires_grad = False
        except Exception:
            global_student_eval = None

    # ---- 5. Comprehensive evaluation ---------------------------------------
    from oracle_distillation.metrics import parse_group_alpha

    _num_classes = get_dataset_config(dataset).num_classes
    known_mask_arr = analysis_data["teacher_knows_class_mask"][k].astype(bool)

    eval_result = _eval_comprehensive(
        student=student,
        test_loader=test_loader,
        device=device,
        num_classes=_num_classes,
        known_mask=known_mask_arr,
        fine_to_coarse=fine_to_coarse,
        global_student=global_student_eval,
    )
    save_test_outputs(
        out_dir / "test_outputs.npz", model=student, loader=test_loader,
        device=device, dataset=dataset, method=f"{spec.method}_cid{k:03d}",
        seed=spec.seed, expected_accuracy=eval_result["model_test_acc"],
    )

    # ---- 5b. Local-test metrics from the partition NPZ ---------------------
    partition_dir = paths.partition_dir_for_rel(rel, spec.seed)
    try:
        _, _, local_test_idx, _ = load_client_npz(partition_dir, k)
    except FileNotFoundError:
        local_test_idx = None
    lt_metrics = _eval_local_test(student, train_ds, local_test_idx, device, spec.batch_size)

    # ---- 6. Teacher known acc from manifest --------------------------------
    manifest_path = paths.teachers_dir_for_rel(rel, spec.seed) / "teachers_manifest.json"
    teacher_known_acc = _teacher_known_acc(manifest_path, k, known_classes)

    total_time = time.time() - t0

    # ---- 7. Metrics JSON (verbose) -----------------------------------------
    full_metrics: dict = {
        **run_metrics,
        **eval_result,
        **lt_metrics,
        "cid": k,
        "global_source": spec.global_source,
        "global_logits_kind": global_logits_kind,
        "known_classes": known_classes,
        "n_known_classes": n_known,
        "n_known_super": n_known_super,
        "known_super": known_super_ids,
        "total_time_s": total_time,
        **diag,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w") as f:
        json.dump(full_metrics, f, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else float(x))

    # ---- 8. CSV row (v5 schema) -------------------------------------------
    group_alpha = parse_group_alpha(rel)
    _nan = float("nan")

    row = {
        # metadata
        "group_alpha":      group_alpha,
        "personal_method":  personal_method,
        "proxy_dataset":    dataset,
        "seed":             spec.seed,
        "cid":              k,
        "global_source":    spec.global_source,
        "global_logits_kind": global_logits_kind,
        "n_known_classes":  n_known,
        "known_classes":    json.dumps(known_classes),
        "n_known_super":    n_known_super,
        "known_super":      json.dumps(known_super_ids),
        # model
        "model_test_acc":           eval_result["model_test_acc"],
        "model_known_acc":          eval_result["model_known_acc"],
        "model_unknown_acc":        eval_result["model_unknown_acc"],
        "model_gap_ku":             eval_result["model_gap_ku"],
        "model_test_acc_per_class": eval_result["model_test_acc_per_class"],
        "model_super_acc":          eval_result["model_super_acc"],
        "model_fine_in_super_acc":  eval_result["model_fine_in_super_acc"],
        "model_super_acc_per_super":         eval_result.get("model_super_acc_per_super"),
        "model_fine_in_super_acc_per_super": eval_result.get("model_fine_in_super_acc_per_super"),
        "local_test_acc":           lt_metrics["local_test_acc"],
        "model_local_test_ece":     lt_metrics["model_local_test_ece"],
        "model_local_test_entropy": lt_metrics["model_local_test_entropy"],
        "model_known_ece":          eval_result["model_known_ece"],
        "model_unknown_ece":        eval_result["model_unknown_ece"],
        "model_known_entropy":      eval_result["model_known_entropy"],
        "model_unknown_entropy":    eval_result["model_unknown_entropy"],
        # distill — existing
        "distill_frac_known_mass":   diag.get("frac_known_mass",   _nan),
        "distill_frac_local_routed": diag.get("frac_local_routed", _nan),
        # distill — footprint (v5)
        "distill_target_flips":         diag.get("distill_target_flips",         _nan),
        "distill_footprint_correction": diag.get("distill_footprint_correction", _nan),
        "distill_footprint_regression": diag.get("distill_footprint_regression", _nan),
        "distill_budget_known_mean":    diag.get("distill_budget_known_mean",    _nan),
        "distill_is_noop":              diag.get("distill_is_noop",              _nan),
        # distill — flip analysis (v5)
        **{k_: eval_result.get(k_, _nan) for k_ in (
            "distill_flip_rate", "distill_within_known", "distill_K2U", "distill_U2K",
            "distill_U2Uprime", "distill_correction", "distill_regression",
            "distill_neutral_churn", "distill_wk_net", "distill_dprob_gt_flip",
        )},
        "distill_warmstart_drift_l2": warmstart_drift_l2,
        # distill — predictor (v5)
        "distill_teacher_known_acc": teacher_known_acc,
        "distill_global_known_acc":  eval_result.get("distill_global_known_acc", _nan),
        # meta
        "warm_started":  warm_started,
        "total_time_s":  total_time,
        "timestamp":     time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    append_row(global_csv, row, PERSONAL_CSV_KEY, PERSONAL_CSV_COLUMNS)
    all_rows.append(row)

    _m_acc = eval_result["model_test_acc"]
    _k_acc = eval_result["model_known_acc"]
    _u_acc = eval_result["model_unknown_acc"]
    _fkm = diag.get("frac_known_mass", _nan)
    _flr = diag.get("frac_local_routed", _nan)
    _fkm_s = f"{_fkm:.2f}" if not (isinstance(_fkm, float) and np.isnan(_fkm)) else "NaN"
    _flr_s = f"{_flr:.2f}" if not (isinstance(_flr, float) and np.isnan(_flr)) else "NaN"
    print(
        f"    [OK] {personal_method}/{c_label} "
        f"acc={_m_acc:.4f} known={_k_acc:.4f} unk={_u_acc:.4f} "
        f"ws={warm_started} flr={_flr_s} fkm={_fkm_s}"
    )
    return row

# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Per-client second distillation (personal KD v2)")
    ap.add_argument("--work_root", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--datasets", nargs="*", default=["cifar"],
                    choices=["cifar", "cifar100", "cinic", "mnist", "fmnist"])
    ap.add_argument("--personal_methods", nargs="*",
                    default=["personal_class_mask"],
                    help=f"Methods. Supported: {sorted(_PERSONAL_TO_VARIANT)}")
    ap.add_argument("--rel_filter", nargs="*", default=None)
    ap.add_argument("--skip_if_done", action="store_true")

    # Training
    add_training_args(ap, val_frac_default=0.15)

    ap.add_argument("--global_source", type=str, default=GLOBAL_METHOD_DEFAULT,
                    choices=["expert", "oracle", "feddf", "consensus", "confidence", "energy"],
                    help="Which global method is 'the global model': both the warm-start "
                         "checkpoint and the class_mask anchor are this same method.")
    ap.add_argument("--global_logits_kind", type=str, default="ensemble",
                    choices=["ensemble", "distilled"],
                    help="For the selected --global_source, anchor the class_mask target on "
                         "the cached ensemble average ('ensemble', default — {method}_avg_logits "
                         "from proxy_analysis.npz) or the REAL forward-pass logits of the "
                         "distilled {method}/student.pt ('distilled' — student_logits.npz).")
    ap.add_argument("--full_coarse_budget", action="store_true",
                    help="CIFAR-100 hierarchical class_mask only: budget per superclass = "
                         "mass of ALL siblings (aggressive, option ii) instead of only the "
                         "known ones (default, option i). See CLASS_MASK_TARGETS_FINDINGS.md.")
    args = ap.parse_args()

    # Validation
    for pm in args.personal_methods:
        if pm not in _PERSONAL_TO_VARIANT:
            sys.exit(f"[FATAL] Unknown personal method {pm!r}. "
                     f"Supported: {sorted(_PERSONAL_TO_VARIANT)}")
    return args


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    args = parse_args()
    primary_ds = (args.datasets or ["cifar"])[0]
    if getattr(args, "arch", None):
        from data.dataset_config import set_dataset_arch
        for _ds in (args.datasets or [primary_ds]):
            set_dataset_arch(_ds, args.arch)  # match teacher/student arch used at step 1-4
        print(f"[arch] backbone overridden -> {args.arch} for {args.datasets or [primary_ds]}")
    if args.student_arch is None:
        from oracle_distillation.models import dataset_default_arch
        args.student_arch = dataset_default_arch(primary_ds)
    set_seed(args.seed)
    device = resolve_device(args.device)

    work_root = Path(args.work_root).resolve()
    if args.seed != 42 or "experiments" in work_root.parts:
        assert_seed_contained(work_root, args.seed)
    data_dir = Path(args.data_dir).resolve()
    runs_root = work_root / "runs"
    global_csv = work_root / "personal_kd_v2_results.csv"

    # Test loaders, train datasets (for the local-test split), proxy splits — registry-driven
    test_loaders: Dict[str, DataLoader] = {}
    train_datasets: Dict[str, object] = {}
    proxy_splits: Dict[str, Path] = {}
    for ds in (args.datasets or ["cifar"]):
        ds_cfg = get_dataset_config(ds)
        test_loaders[ds] = ds_cfg.make_test_loader(data_dir, batch_size=256, num_workers=2)
        # CLEAN transform (same as the teachers' local-test eval): local_test_acc
        # must not be measured on random augmentations.
        train_datasets[ds] = ds_cfg.load_train_eval_dataset(data_dir)
        proxy_splits[ds] = ds_cfg.proxy_path_for_seed(args.seed, data_dir)

    jobs = discover_proxy_jobs(runs_root, args.datasets, args.seed, rel_filter=args.rel_filter)
    if not jobs:
        sys.exit("[FATAL] No proxy_analysis.npz files found.")

    print(f"[INFO] Found {len(jobs)} (rel, dataset) pairs. "
          "class_mask warm-starts from the global expert student; KdRunner handles all KD.")

    # Resume CSV — dedup on the same canonical key that append_row() uses.
    all_rows: list = resume_csv_rows(global_csv, PERSONAL_CSV_KEY)

    for rel, dataset, analysis_npz in jobs:
        print(f"\n=== {rel} / {dataset} ===")

        src = np.load(analysis_npz, allow_pickle=True)
        analysis_data = {k: src[k] for k in src.files}
        n_teachers = int(analysis_data["teacher_logits_cache"].shape[1])
        print(f"  proxy N={len(analysis_data['proxy_idx'])}, K={n_teachers}")

        proxy_split = proxy_splits.get(dataset)
        if proxy_split is None or not proxy_split.exists():
            print(f"  [SKIP] No proxy split found for dataset={dataset!r}")
            continue
        with np.load(proxy_split, allow_pickle=True) as expected_split:
            expected_proxy_idx = np.asarray(expected_split["proxy_idx"], dtype=np.int64)
        observed_proxy_idx = np.asarray(analysis_data["proxy_idx"], dtype=np.int64)
        if not np.array_equal(observed_proxy_idx, expected_proxy_idx):
            raise RuntimeError(
                f"seed={args.seed} proxy_analysis/proxy split mismatch for {dataset}: "
                f"{analysis_npz} vs {proxy_split}"
            )

        test_loader = test_loaders.get(dataset)
        paths = Paths(work_root=work_root, data_dir=data_dir, dataset=dataset)

        for personal_method in args.personal_methods:
            variant = _PERSONAL_TO_VARIANT[personal_method]
            print(f"  -- method={personal_method}, global_source={args.global_source}, variant={variant})")

            spec = build_execution_spec(
                args=args, dataset=dataset, method=personal_method,
                is_personal=True, global_source=args.global_source, min_delta=1e-4,
            )

            # --global_logits_kind distilled: the anchor is the REAL global
            # student.pt forward pass over the proxy (read from its cache when
            # available). Same for every client in this (rel, dataset) job, so
            # compute it once here, not per-k.
            z_global_override: Optional[np.ndarray] = None
            if variant == "personal_class_mask" and args.global_logits_kind == "distilled":
                try:
                    z_global_override = _distilled_global_logits(
                        paths, rel, spec.global_source, spec.seed,
                        spec.student_arch, get_dataset_config(dataset).num_classes,
                        train_datasets.get(dataset), analysis_data["proxy_idx"],
                        device, batch_size=256,
                    )
                except FileNotFoundError as e:
                    print(f"    [SKIP] {personal_method}: {e}")
                    continue

            for k in range(n_teachers):
                try:
                    _run_one_client(
                        k=k,
                        analysis_data=analysis_data,
                        personal_method=personal_method,
                        variant=variant,
                        spec=spec,
                        skip_if_done=args.skip_if_done,
                        paths=paths,
                        rel=rel,
                        device=device,
                        test_loader=test_loader,
                        train_ds=train_datasets.get(dataset),
                        global_csv=global_csv,
                        all_rows=all_rows,
                        full_coarse_budget=args.full_coarse_budget,
                        z_global_override=z_global_override,
                        global_logits_kind=args.global_logits_kind,
                    )
                except KeyboardInterrupt:
                    print("\n[INTERRUPTED]")
                    return
                except Exception as e:
                    print(f"    [ERROR] k={k} {personal_method}: {e}")
                    import traceback
                    traceback.print_exc()

    print(f"\n[DONE] Results: {global_csv}")


if __name__ == "__main__":
    main()
