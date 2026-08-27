#!/usr/bin/env python3
"""run_transfer_set_ablation.py — Phase A: transfer-set ablation for personal class_mask.

Same ``class_mask`` mechanism as the incumbent personal KD (budget-preserving target:
the teacher re-sorts the mass on the client's *known* classes, the global keeps the
*unknown*; ALWAYS warm-started from the global ``expert`` student). The **only**
variable under test is the **transfer set** of the 2nd distillation:

    proxy       — public proxy set (the incumbent transfer set)
    proxy_plain — proxy set with PLAIN expert target (expert logits, no class_mask
                  resorting). Isolates mechanism effect from "a 2nd pass over proxy".
    local       — the client's own train_idx (known classes only)
    mix         — local train_idx ∪ proxy
    local_lwf   — local train_idx, with an LwF anchor (expert_global) on proxy inputs

Design decisions (audited in Phase 0; do NOT change these without re-reading the spec):

  * **P_global = expert_global STUDENT output**, recomputed by forward pass on *every*
    transfer set (including proxy). This is Roi's choice: the transfer set is then the
    sole variable. NOTE the on-disk incumbent ``personal_class_mask`` instead used the
    expert *ensemble* logits (``expert_avg_logits``) as anchor — that historical row is
    reported separately (arm ``proxy_ondisk_ensemble``) for cross-check, not rebuilt.

  * **The target FORMULA is reused verbatim** (``analysis.target_builders._class_mask_target``);
    only its *inputs* (``z_global``, ``z_teacher``) are recomputed per image on the chosen
    transfer set. The proxy_analysis.npz caches are aligned to ``proxy_idx`` only, so for
    local/mix we run two forward passes (expert_global student + teacher_k). The competence
    masks (``teacher_knows_class_mask``, ``fine_to_coarse``) are image-independent and reused.

  * proxy_idx and every client's train_idx index the SAME train set (load_train_eval_dataset)
    and are DISJOINT (verified), so ``mix = concat(local, proxy)`` needs no remapping.

CSV column blocks (strictly separated, never mixed):

  g_*          — expert_global on the global test set (fixed reference, same across all arms
                 for a given (dataset, regime, cid)).
  p_*          — personalized student on the global test set (the variable under comparison).
  *_localsurf  — per-arm student on the client's local held-out test split (never mixed
                 with 1 or 2; unknown_acc is NaN in single/multi by construction).

Acceptance assertion: after building the CSV, ``_assert_g_invariant`` checks that
``g_test_acc``, ``g_known_acc``, ``g_unknown_acc`` are numerically identical (atol 1e-4)
across arms for every (dataset, regime, cid). A failure means the expert_global eval is
not reproducible — indicates a code bug.

Usage
-----
    # cifar10 (scratch, seed 42) — reset old buggy data, rerun full
    python -m oracle_distillation.cli.run_transfer_set_ablation \\
        --work_root OUTPUTS/experiments/study_i/seed_42/raw_work --data_dir data --datasets cifar \\
        --rel_filter single multi iid alpha0p1 alpha0p5 alpha1p0 --reset

    # cifar100
    python -m oracle_distillation.cli.run_transfer_set_ablation \\
        --work_root OUTPUTS/companion/cifar100/seed_42/raw_work --data_dir data --datasets cifar100 \\
        --rel_filter single_super multi_super iid dir_coarse dir_fine --reset

    # rebuild the CSV/table from existing metrics.json without retraining
    python -m oracle_distillation.cli.run_transfer_set_ablation \\
        --work_root OUTPUTS/experiments/study_i/seed_42/raw_work --data_dir data --datasets cifar --table_only
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data.dataset_config import get_dataset_config, set_dataset_arch
from o3_local.teacher_trainer import load_client_npz
from oracle_distillation.analysis.target_builders import (
    _class_mask_target, personal_target_diagnostics,
)
from oracle_distillation.cli.common_args import add_training_args
from oracle_distillation.cli.run_personal_kd import _eval_comprehensive
from oracle_distillation.config import build_execution_spec, ExecutionSpec
from oracle_distillation.checkpoints import (
    checkpoint_sha256, load_checkpoint_state, save_model_checkpoint, state_dict_sha256,
)
from oracle_distillation.test_outputs import save_test_outputs
from oracle_distillation.distill import losses
from oracle_distillation.distill.kd_runner import (
    KdRunner, IndexedSubset, _make_optimizer, _split_train_val, _build_pos_of_global,
)
from oracle_distillation.metrics import parse_group_alpha
from oracle_distillation.models import build_model
from oracle_distillation.cli.run_personal_kd import _GLOBAL_SOURCE_TO_METHOD_DIR
from oracle_distillation.run_paths import Paths, discover_proxy_jobs
from oracle_distillation.experiment_paths import assert_seed_contained
from oracle_distillation.provenance import write_execution_provenance
from oracle_distillation.utils import (
    resolve_device, set_seed, softmax_np, collect_logits, client_label, write_csv,
)

ARMS = ("proxy", "proxy_plain", "local", "mix", "local_lwf")
ABLATION_SUBDIR = "transfer_ablation"
INVARIANT_ATOL = 1e-4
G_INVARIANT_ATOL = 1e-4  # tolerance for cross-arm g_* consistency check


def derive_paired_train_seed(
    seed: int, dataset: str, regime: str, global_source: str, cid: int,
) -> int:
    """Stable common-random-number seed for a Phase-B causal pair."""
    key = f"phase_b_crn_v1|{seed}|{dataset}|{regime}|{global_source}|{cid}"
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:4], "little")


# ===========================================================================
# Forward passes (recompute target inputs on a chosen transfer set)
# ===========================================================================

@torch.inference_mode()
def _forward_logits(
    model: torch.nn.Module, ds, idx: np.ndarray, device: torch.device, batch_size: int = 256,
    num_workers: int = 2,
) -> np.ndarray:
    """Return [len(idx), C] float32 logits of ``model`` over Subset(ds, idx) (clean, ordered)."""
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


def _load_teacher(teachers_dir: Path, k: int, arch: str, num_classes: int, device) -> torch.nn.Module:
    ckpt = teachers_dir / f"cid_{k:03d}.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"Missing teacher checkpoint: {ckpt}")
    m = build_model(arch, num_classes=num_classes).to(device)
    m.load_state_dict(load_checkpoint_state(ckpt, map_location=device), strict=True)
    m.eval()
    return m


def _load_student(ckpt: Path, arch: str, num_classes: int, device) -> torch.nn.Module:
    if not ckpt.is_file():
        raise FileNotFoundError(f"Missing student checkpoint: {ckpt}")
    m = build_model(arch, num_classes=num_classes).to(device)
    m.load_state_dict(load_checkpoint_state(ckpt, map_location=device), strict=True)
    return m


# ===========================================================================
# Synthetic analysis dict — lets us reuse the EXACT target formula on new images
# ===========================================================================

def _synth_analysis(
    z_global: np.ndarray, z_teacher: np.ndarray, k: int, K: int,
    teacher_knows_class_mask: np.ndarray, y_true: np.ndarray,
    fine_to_coarse: Optional[np.ndarray],
) -> Dict[str, np.ndarray]:
    """Build a proxy-analysis-shaped dict aligned to a transfer set, so the unchanged
    builder (``_class_mask_target``) reads our recomputed logits.

    ``expert_avg_logits`` is set to the expert_global STUDENT logits (the anchor of this
    ablation). ``teacher_logits_cache[:, k, :]`` holds teacher-k's logits; other teacher
    slots are zeros (unused — the class_mask builder only reads teacher k and the masks).
    """
    n, C = z_global.shape
    tlc = np.zeros((n, K, C), dtype=np.float32)
    tlc[:, k, :] = z_teacher.astype(np.float32)
    d: Dict[str, np.ndarray] = {
        "teacher_logits_cache": tlc,
        "expert_avg_logits": z_global.astype(np.float32),
        "teacher_knows_class_mask": np.asarray(teacher_knows_class_mask),
        "y_true_proxy": np.asarray(y_true).astype(np.int64),
    }
    if fine_to_coarse is not None:
        d["fine_to_coarse"] = np.asarray(fine_to_coarse).astype(np.int64)
    return d


def _build_target(
    synth: Dict[str, np.ndarray], k: int, *, T: float, fine_to_coarse: Optional[np.ndarray],
    global_source: str = "expert",
) -> np.ndarray:
    """The incumbent class_mask target, byte-for-byte formula, on the synthetic inputs."""
    return _class_mask_target(
        synth, k, T=T, global_source=global_source, fine_to_coarse=fine_to_coarse,
    )


# ===========================================================================
# Invariant verification (correctness test demanded by the spec)
# ===========================================================================

def verify_invariants(
    target_logits: np.ndarray, z_global: np.ndarray, known: np.ndarray,
    fine_to_coarse: Optional[np.ndarray], *, T: float, atol: float = INVARIANT_ATOL,
) -> Tuple[bool, Dict[str, float]]:
    """Check the budget-preserving invariants on the *recovered* target distribution.

    p_target = softmax(target/T) is exactly what the KD loss matches. Checks:
      * every row sums to 1;
      * flat: unknown slots == P_global elementwise, and Σ_known p_target == Σ_known P_global;
      * hierarchical (CIFAR-100): per-superclass totals == P_global per row (the coarse
        split is frozen; this subsumes "unknown superclasses untouched").
    """
    p_target = softmax_np(target_logits, T=T)
    p_global = softmax_np(z_global, T=T)
    known = np.asarray(known).astype(bool)
    errs: Dict[str, float] = {
        "row_sum_max_dev": float(np.abs(p_target.sum(axis=1) - 1.0).max()),
    }
    if fine_to_coarse is None:
        unk = ~known
        errs["unknown_slots_max_dev"] = (
            float(np.abs(p_target[:, unk] - p_global[:, unk]).max()) if unk.any() else 0.0
        )
        if known.any():
            kt = p_target[:, known].sum(axis=1)
            kg = p_global[:, known].sum(axis=1)
            errs["known_budget_max_dev"] = float(np.abs(kt - kg).max())
        else:
            errs["known_budget_max_dev"] = 0.0
    else:
        f2c = np.asarray(fine_to_coarse).astype(np.intp)
        dev = 0.0
        for s in np.unique(f2c):
            sib = f2c == s
            dev = max(dev, float(np.abs(p_target[:, sib].sum(axis=1) - p_global[:, sib].sum(axis=1)).max()))
        errs["super_total_max_dev"] = dev
    ok = all(v <= atol for v in errs.values())
    return ok, errs


# ===========================================================================
# local_lwf training loop (LwF was removed from the codebase — reimplemented here)
# ===========================================================================

def _distill_local_lwf(
    spec: ExecutionSpec, *, student: torch.nn.Module, ds_full,
    local_idx: np.ndarray, local_target: np.ndarray,
    proxy_idx: np.ndarray, z_anchor_proxy: np.ndarray,
    device: torch.device, lwf_weight: float, seed: int,
) -> Dict[str, Any]:
    """Refine ``student`` on the LOCAL transfer set (class_mask target) while LwF-anchoring
    its outputs on PROXY inputs to the frozen expert_global student (``z_anchor_proxy``).

    Per local batch we also pull one proxy batch and add ``lwf_weight * KD(student(x_proxy),
    expert_global(x_proxy))`` — the standard LwF regularizer, here the only way unknown-class
    *inputs* enter the local refinement. Mirrors ``distill_personal``: AdamW, cosine LR,
    early-stop on the local KD val loss.
    """
    T = spec.temperature
    tr_idx, val_idx = _split_train_val(local_idx, val_frac=spec.val_frac, min_val=spec.min_val, seed=seed)
    pin = device.type == "cuda"
    tr_loader = DataLoader(IndexedSubset(ds_full, tr_idx), batch_size=spec.batch_size,
                           shuffle=True, num_workers=2, pin_memory=pin)
    val_loader = (DataLoader(IndexedSubset(ds_full, val_idx), batch_size=spec.batch_size,
                             shuffle=False, num_workers=2, pin_memory=pin)
                  if len(val_idx) > 0 else None)
    proxy_loader = DataLoader(IndexedSubset(ds_full, proxy_idx), batch_size=spec.batch_size,
                              shuffle=True, num_workers=2, pin_memory=pin)

    pos_local = _build_pos_of_global(len(ds_full), local_idx).to(device)
    pos_proxy = _build_pos_of_global(len(ds_full), proxy_idx).to(device)
    local_t = torch.from_numpy(local_target.astype(np.float32)).to(device)
    anchor_t = torch.from_numpy(z_anchor_proxy.astype(np.float32)).to(device)

    opt = _make_optimizer(spec, student)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=spec.epochs)

    best_val, best_epoch, best_state, wait, stopped = float("inf"), -1, None, 0, False
    MIN_ROUNDS = 10
    for ep in range(1, spec.epochs + 1):
        student.train()
        proxy_iter = itertools.cycle(proxy_loader)
        for x, _y, g in tr_loader:
            x = x.to(device, non_blocking=True)
            g = g.to(device, non_blocking=True)
            loss = losses.loss_soft_kd(student(x), local_t[pos_local[g]], T)
            xp, _yp, gp = next(proxy_iter)
            xp = xp.to(device, non_blocking=True)
            gp = gp.to(device, non_blocking=True)
            loss = loss + lwf_weight * losses.loss_soft_kd(student(xp), anchor_t[pos_proxy[gp]], T)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()

        # early stop on local KD val loss (matches distill_personal monitor='val_loss')
        if val_loader is not None and spec.patience > 0:
            student.eval()
            s, n = 0.0, 0
            with torch.no_grad():
                for x, _y, g in val_loader:
                    x = x.to(device, non_blocking=True)
                    g = g.to(device, non_blocking=True)
                    s += float(losses.loss_soft_kd(student(x), local_t[pos_local[g]], T).item()) * x.size(0)
                    n += x.size(0)
            v = s / max(n, 1)
            if (best_val - v) > spec.min_delta:
                best_val, best_epoch = v, ep
                best_state = {kk: vv.detach().cpu().clone() for kk, vv in student.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= spec.patience and ep > MIN_ROUNDS:
                    stopped = True
                    break
    if best_state is not None:
        student.load_state_dict(best_state)
    return {
        "model_best_epoch": best_epoch if best_epoch > 0 else None,
        "model_stopped_early": bool(stopped),
        "distill_best_val_loss": (best_val if best_val != float("inf") else float("nan")),
        "lwf_weight": float(lwf_weight),
    }


# ===========================================================================
# Evaluation — local surface (global surface reuses _eval_comprehensive)
# ===========================================================================

@torch.no_grad()
def _eval_local_surface(
    student: torch.nn.Module, ds_eval, local_test_idx: Optional[np.ndarray],
    known_mask: np.ndarray, device: torch.device, batch_size: int = 256,
) -> Dict[str, float]:
    """Known/unknown accuracy on the client's held-out local-test split (its own distribution).

    For single/multi the local split has no unknown-class samples, so ``localsurf_unknown_acc``
    is NaN by construction — that is itself informative (the local surface cannot measure
    unknown retention). ``local_test_idx`` is disjoint from train_idx (3-way partition split),
    so this is leakage-free for the local/mix arms too.
    """
    _nan = float("nan")
    if local_test_idx is None or len(local_test_idx) == 0:
        return {"localsurf_overall_acc": _nan, "localsurf_known_acc": _nan,
                "localsurf_unknown_acc": _nan, "localsurf_gap_ku": _nan, "localsurf_n": 0}
    loader = DataLoader(Subset(ds_eval, [int(i) for i in local_test_idx]),
                        batch_size=batch_size, shuffle=False, num_workers=2,
                        pin_memory=(device.type == "cuda"))
    logits, labels = collect_logits(student, loader, device)
    pred = logits.argmax(axis=1)
    known = np.asarray(known_mask).astype(bool)
    kog = known[labels]
    unk = ~kog
    return {
        "localsurf_overall_acc": float((pred == labels).mean()),
        "localsurf_known_acc": float((pred[kog] == labels[kog]).mean()) if kog.any() else _nan,
        "localsurf_unknown_acc": float((pred[unk] == labels[unk]).mean()) if unk.any() else _nan,
        "localsurf_gap_ku": (float((pred[kog] == labels[kog]).mean()) - float((pred[unk] == labels[unk]).mean()))
                            if (kog.any() and unk.any()) else _nan,
        "localsurf_n": int(len(labels)),
    }


# ===========================================================================
# Row assembly
# ===========================================================================

def _row_from_metrics(m: Dict[str, Any]) -> Dict[str, Any]:
    """Flat CSV row (all three surface blocks + key metadata) from a metrics.json dict."""
    g = m.get
    return {
        "dataset": g("dataset"), "regime": g("regime"), "arm": g("arm"),
        "seed": g("seed"), "cid": g("cid"), "global_source": g("global_source"),
        "n_known_classes": g("n_known_classes"), "n_known_super": g("n_known_super"),
        "transfer_n": g("transfer_n"), "lwf_weight": g("lwf_weight"),
        # ---- Block 1: GLOBAL surface — expert_global (fixed reference, same across arms) ----
        "g_known_acc":   g("g_known_acc"),
        "g_unknown_acc": g("g_unknown_acc"),
        "g_gap_ku":      g("g_gap_ku"),
        "g_test_acc":    g("g_test_acc"),
        # ---- Block 2: PERSONALIZED student — evaluated on the global test set ----
        "p_known_acc":        g("p_known_acc"),
        "p_unknown_acc":      g("p_unknown_acc"),
        "p_gap_ku":           g("p_gap_ku"),
        "p_test_acc":         g("p_test_acc"),
        "p_global_known_acc": g("p_global_known_acc"),
        # ---- Block 3: LOCAL surface — student on client's held-out test split ----
        "localsurf_known_acc":   g("localsurf_known_acc"),
        "localsurf_unknown_acc": g("localsurf_unknown_acc"),
        "localsurf_gap_ku":      g("localsurf_gap_ku"),
        "localsurf_overall_acc": g("localsurf_overall_acc"),
        # ---- bookkeeping ----
        "invariants_ok": g("invariants_ok"), "warm_started": g("warm_started"),
        "model_best_epoch": g("model_best_epoch"), "model_stopped_early": g("model_stopped_early"),
        "train_seed": g("train_seed"), "train_order_sha256": g("train_order_sha256"),
        "train_epochs_completed": g("train_epochs_completed"),
        "initial_state_sha256": g("initial_state_sha256"),
        "final_state_sha256": g("final_state_sha256"),
        "total_time_s": g("total_time_s"),
    }


CSV_COLUMNS = [
    "dataset", "regime", "arm", "seed", "cid", "global_source", "n_known_classes", "n_known_super",
    "transfer_n", "lwf_weight",
    # Block 1: expert_global reference (fixed — identical across arms per (dataset, regime, cid))
    "g_known_acc", "g_unknown_acc", "g_gap_ku", "g_test_acc",
    # Block 2: personalized student on global test
    "p_known_acc", "p_unknown_acc", "p_gap_ku", "p_test_acc", "p_global_known_acc",
    # Block 3: personalized student on local surface
    "localsurf_known_acc", "localsurf_unknown_acc", "localsurf_gap_ku", "localsurf_overall_acc",
    "invariants_ok", "warm_started", "model_best_epoch", "model_stopped_early",
    "train_seed", "train_order_sha256", "train_epochs_completed",
    "initial_state_sha256", "final_state_sha256", "total_time_s",
]


# ===========================================================================
# Acceptance assertion
# ===========================================================================

def _assert_g_invariant(rows: List[Dict[str, Any]], atol: float = G_INVARIANT_ATOL) -> None:
    """Assert that g_* columns are identical across arms for each (dataset, regime, cid, global_source).

    The "g_*" block is the frozen ``global_source`` model's own eval (expert / feddf / energy /
    ...), recomputed identically for every arm within that global_source — so it must be
    numerically identical *within* a (dataset, regime, cid, global_source) group. It is NOT
    expected to match across different ``global_source`` values (those are different baseline
    models by design; comparing them here would be conflating "different arms of the same
    global" with "different globals"). A violation indicates a code bug (e.g. eval called on
    the wrong model, or the wrong values stored in the g_* keys).
    """
    import pandas as pd
    df = pd.DataFrame(rows)
    g_cols = ["g_test_acc", "g_known_acc", "g_unknown_acc"]
    group_keys = ["dataset", "regime", "cid", "global_source"]
    failures = []
    for key, grp in df.groupby(group_keys):
        for col in g_cols:
            vals = grp[col].dropna()
            if len(vals) < 2:
                continue
            try:
                fvals = [float(v) for v in vals]
            except (TypeError, ValueError):
                continue
            span = max(fvals) - min(fvals)
            if span > atol:
                failures.append(
                    f"  {dict(zip(group_keys, key))} {col}: range={span:.6f}  "
                    f"vals={[f'{v:.6f}' for v in fvals]}"
                )
    if failures:
        raise AssertionError(
            "g_* invariant violated — same-global_source eval differs across arms:\n"
            + "\n".join(failures)
            + "\n\nThis is a bug: within a fixed global_source, g_* must be the same "
              "fixed model eval for all arms."
        )
    n_groups = df.groupby(group_keys).ngroups
    print(f"[ASSERT OK] g_* invariant: identical across arms (atol={atol}) "
          f"for all {n_groups} (dataset, regime, cid, global_source) groups.")


# ===========================================================================
# Per (regime, client) driver
# ===========================================================================

def _arm_out_dir(paths: Paths, rel: str, arm: str, k: int, seed: int,
                 global_source: str = "expert", output_root: Optional[Path] = None) -> Path:
    if output_root is not None:
        return (Path(output_root) / "checkpoints" / global_source / rel
                / arm / client_label(k))
    return (paths.run_dir_for_rel(rel)
            / ABLATION_SUBDIR / global_source / arm / client_label(k))


def _process_client(
    *, k: int, rel: str, regime: str, dataset: str, arms: List[str],
    analysis: Dict[str, np.ndarray], paths: Paths, spec: ExecutionSpec,
    ds_full, test_loader, device: torch.device, arch: str, num_classes: int,
    lwf_weight: float, skip_if_done: bool, global_source: str = "expert",
    output_root: Optional[Path] = None,
) -> None:
    seed = spec.seed
    K = int(analysis["teacher_logits_cache"].shape[1])
    mask_kc = np.asarray(analysis["teacher_knows_class_mask"])
    known_k = mask_kc[k].astype(bool)
    fine_to_coarse = analysis.get("fine_to_coarse")
    f2c = np.asarray(fine_to_coarse).astype(np.int64) if fine_to_coarse is not None else None
    n_known = int(known_k.sum())
    n_known_super = (len(sorted(set(int(f2c[c]) for c in np.where(known_k)[0]))) if f2c is not None else 0)

    proxy_idx = np.asarray(analysis["proxy_idx"]).astype(int)
    partition_dir = paths.partition_dir_for_rel(rel, seed)
    train_idx, _hold, local_test_idx, _kf = load_client_npz(partition_dir, k)
    train_idx = np.asarray(train_idx).astype(int)

    # Decide which arms still need work
    todo = []
    for arm in arms:
        arm_dir = _arm_out_dir(paths, rel, arm, k, seed, global_source, output_root)
        mp = arm_dir / "metrics.json"
        if skip_if_done and mp.exists() and (arm_dir / "test_outputs.npz").exists():
            print(f"      [skip] {arm}/{client_label(k)} (done)")
            continue
        todo.append(arm)
    need_expert = not (skip_if_done and (_arm_out_dir(
        paths, rel, "global", k, seed, global_source, output_root,
    ) / "metrics.json").exists())
    if not todo and not need_expert:
        return

    # Labels for any transfer index (from the dataset targets)
    targets_all = np.asarray(getattr(ds_full, "targets"), dtype=np.int64)

    # ---- global student: warm-start source AND the P_global anchor ----
    method_dir = _GLOBAL_SOURCE_TO_METHOD_DIR.get(global_source, global_source)
    expert_ckpt = paths.method_dir_for_rel(rel, method_dir, seed) / "student.pt"
    expert_student = _load_student(expert_ckpt, arch, num_classes, device)
    for p in expert_student.parameters():
        p.requires_grad = False
    expert_state = {kk: vv.detach().cpu().clone() for kk, vv in expert_student.state_dict().items()}

    # ---- evaluate expert_global ONCE on global test — this is the fixed g_* reference ----
    ev_expert_g = _eval_comprehensive(expert_student, test_loader, device, num_classes, known_k,
                                      fine_to_coarse=f2c, global_student=expert_student)

    # ---- expert_global baseline row, once per client ----
    if need_expert:
        ev_l_expert = _eval_local_surface(expert_student, ds_full, local_test_idx, known_k,
                                          device, spec.batch_size)
        _write_metrics(paths, rel, "global", k, seed, dataset, regime, n_known, n_known_super,
                       transfer_n=0, invariants_ok=None, warm_started=True, lwf_weight=float("nan"),
                       global_source=global_source,
                       run_metrics={"model_best_epoch": None, "model_stopped_early": False},
                       eval_expert_g=ev_expert_g, eval_personal_g=ev_expert_g,
                       eval_local=ev_l_expert, total_time_s=0.0,
                       global_checkpoint=expert_ckpt,
                       output_root=output_root)
        print(f"      [ok] global/{client_label(k)} "
              f"g_known={ev_expert_g['model_known_acc']:.4f} g_unk={ev_expert_g['model_unknown_acc']:.4f}")

    if not todo:
        return

    # ---- recompute target inputs ONCE on local ∪ proxy, slice per arm ----
    union_idx = np.concatenate([train_idx, proxy_idx])  # disjoint → already unique
    row_of = {int(g): i for i, g in enumerate(union_idx)}
    teacher = _load_teacher(paths.teachers_dir_for_rel(rel, seed), k, arch, num_classes, device)
    z_global_U = _forward_logits(expert_student, ds_full, union_idx, device, spec.batch_size)
    z_teacher_U = _forward_logits(teacher, ds_full, union_idx, device, spec.batch_size)
    del teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()

    def _slice(idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = np.array([row_of[int(g)] for g in idx], dtype=np.int64)
        return z_global_U[rows], z_teacher_U[rows], targets_all[idx]

    # Precompute the LwF anchor on proxy (expert_global logits) once, if needed.
    z_anchor_proxy = None
    if "local_lwf" in todo:
        zg_p, _zt_p, _y_p = _slice(proxy_idx)
        z_anchor_proxy = zg_p

    for arm in todo:
        t0 = time.time()
        # Transfer index for this arm
        if arm in ("proxy", "proxy_plain"):
            T_idx = proxy_idx
        elif arm in ("local", "local_lwf"):
            T_idx = train_idx
        elif arm == "mix":
            T_idx = union_idx
        else:
            raise ValueError(f"unknown arm {arm!r}")

        zg, zt, y_T = _slice(T_idx)

        if arm == "proxy_plain":
            # Plain expert target: expert student logits as soft KD target (no class_mask resorting).
            # Lets us isolate the mechanism effect vs "a 2nd pass over proxy".
            target = zg.copy()
            ok, inv, diag = None, {}, None
        else:
            synth = _synth_analysis(zg, zt, k, K, mask_kc, y_T, f2c)
            target = _build_target(synth, k, T=spec.temperature, fine_to_coarse=f2c,
                                   global_source=global_source)
            ok, inv = verify_invariants(target, zg, known_k, f2c, T=spec.temperature)
            if not ok:
                print(f"      [INVARIANT FAIL] {arm}/{client_label(k)}: {inv}")
                raise RuntimeError(f"Target invariants violated for {arm}/{client_label(k)}: {inv}")
            diag = personal_target_diagnostics(synth, k, "personal_class_mask", T=spec.temperature,
                                               global_source=global_source, fine_to_coarse=f2c)

        # Warm-start a fresh student from expert_global, then refine
        student = build_model(arch, num_classes=num_classes).to(device)
        student.load_state_dict(expert_state, strict=True)
        student.to(device)
        initial_state_sha256 = state_dict_sha256(student.state_dict())
        train_seed = derive_paired_train_seed(seed, dataset, regime, global_source, k)
        set_seed(train_seed)

        if arm == "local_lwf":
            run_metrics = _distill_local_lwf(
                spec, student=student, ds_full=ds_full, local_idx=train_idx, local_target=target,
                proxy_idx=proxy_idx, z_anchor_proxy=z_anchor_proxy, device=device,
                lwf_weight=lwf_weight, seed=seed,
            )
        else:
            kd = KdRunner(device=device, seed=train_seed)
            run_metrics = kd.distill_personal(
                spec, student=student, dataset=dataset, data_dir=paths.data_dir,
                proxy_idx=T_idx, target=target,
            )
            run_metrics["lwf_weight"] = float("nan")
        run_metrics["train_seed"] = train_seed
        run_metrics["initial_state_sha256"] = initial_state_sha256
        run_metrics["final_state_sha256"] = state_dict_sha256(student.state_dict())

        # Evaluate personalized student on the global test set (Block 2: p_*)
        ev_personal_g = _eval_comprehensive(student, test_loader, device, num_classes, known_k,
                                            fine_to_coarse=f2c, global_student=expert_student)
        # Evaluate personalized student on the local surface (Block 3: *_localsurf)
        ev_l = _eval_local_surface(student, ds_full, local_test_idx, known_k, device, spec.batch_size)

        out_dir = _arm_out_dir(paths, rel, arm, k, seed, global_source, output_root)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_test_outputs(
            out_dir / "test_outputs.npz", model=student, loader=test_loader,
            device=device, dataset=dataset,
            method=f"phase_b_{global_source}_{arm}_cid{k:03d}", seed=seed,
            expected_accuracy=ev_personal_g["model_test_acc"],
        )
        if spec.save_models:
            save_model_checkpoint(
                out_dir / "student.pt", student,
                architecture=arch, dataset=dataset, regime=regime,
                method=f"phase_b_{global_source}_{arm}_cid{k:03d}",
                seed=seed, config=spec,
            )
        _write_metrics(paths, rel, arm, k, seed, dataset, regime, n_known, n_known_super,
                       transfer_n=int(len(T_idx)), invariants_ok=(bool(ok) if ok is not None else None),
                       warm_started=True, lwf_weight=run_metrics.get("lwf_weight", float("nan")),
                       global_source=global_source,
                       run_metrics=run_metrics, eval_expert_g=ev_expert_g,
                       eval_personal_g=ev_personal_g, eval_local=ev_l,
                       total_time_s=time.time() - t0, invariants=inv, diag=diag,
                       global_checkpoint=expert_ckpt,
                       output_root=output_root)
        print(f"      [ok] {arm}/{client_label(k)} N={len(T_idx)} "
              f"g_known={ev_expert_g['model_known_acc']:.4f} "
              f"p_known={ev_personal_g['model_known_acc']:.4f} "
              f"p_unk={ev_personal_g['model_unknown_acc']:.4f} "
              f"p_gap={ev_personal_g['model_gap_ku']:.4f} "
              f"ls_known={ev_l['localsurf_known_acc']:.4f} "
              f"inv_ok={ok} ({time.time() - t0:.0f}s)")
        del student
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _write_metrics(
    paths: Paths, rel: str, arm: str, k: int, seed: int, dataset: str, regime: str,
    n_known: int, n_known_super: int, *, transfer_n: int, invariants_ok: Optional[bool],
    warm_started: bool, lwf_weight: float, global_source: str, run_metrics: Dict[str, Any],
    eval_expert_g: Dict[str, Any],    # Block 1: expert_global on global test (fixed g_*)
    eval_personal_g: Dict[str, Any],  # Block 2: personalized student on global test (p_*)
    eval_local: Dict[str, Any],       # Block 3: personalized student on local surface (*_localsurf)
    total_time_s: float,
    invariants: Optional[Dict[str, float]] = None,
    diag: Optional[Dict[str, Any]] = None,
    global_checkpoint: Optional[Path] = None,
    output_root: Optional[Path] = None,
) -> None:
    out_dir = _arm_out_dir(paths, rel, arm, k, seed, global_source, output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    m: Dict[str, Any] = {
        "dataset": dataset, "regime": regime, "rel": rel, "arm": arm, "seed": seed, "cid": k,
        "global_source": global_source,
        "global_checkpoint": str(global_checkpoint) if global_checkpoint else None,
        "global_test_outputs": (
            str(Path(global_checkpoint).parent / "test_outputs.npz")
            if global_checkpoint else None
        ),
        "test_outputs": (
            str(Path(global_checkpoint).parent / "test_outputs.npz")
            if arm == "global" and global_checkpoint
            else str(out_dir / "test_outputs.npz")
        ),
        "n_known_classes": n_known, "n_known_super": n_known_super, "transfer_n": transfer_n,
        "lwf_weight": lwf_weight, "warm_started": warm_started,
        "invariants_ok": invariants_ok, "invariants": invariants,
        "total_time_s": total_time_s,
        **{kk: run_metrics.get(kk) for kk in (
            "model_best_epoch", "model_stopped_early", "distill_best_val_loss",
            "train_seed", "train_order_sha256", "train_epochs_completed",
            "initial_state_sha256", "final_state_sha256",
        )},
        # Block 1: expert_global on global test (fixed reference — same for every arm)
        "g_known_acc":   eval_expert_g.get("model_known_acc"),
        "g_unknown_acc": eval_expert_g.get("model_unknown_acc"),
        "g_gap_ku":      eval_expert_g.get("model_gap_ku"),
        "g_test_acc":    eval_expert_g.get("model_test_acc"),
        # Block 2: personalized student on global test
        "p_known_acc":         eval_personal_g.get("model_known_acc"),
        "p_unknown_acc":       eval_personal_g.get("model_unknown_acc"),
        "p_gap_ku":            eval_personal_g.get("model_gap_ku"),
        "p_test_acc":          eval_personal_g.get("model_test_acc"),
        "p_global_known_acc":  eval_personal_g.get("distill_global_known_acc"),
        "p_super_acc":         eval_personal_g.get("model_super_acc"),
        "p_fine_in_super_acc": eval_personal_g.get("model_fine_in_super_acc"),
        # Block 3: personalized student on local surface
        **{kk: eval_local.get(kk) for kk in ("localsurf_overall_acc", "localsurf_known_acc",
                                              "localsurf_unknown_acc", "localsurf_gap_ku",
                                              "localsurf_n")},
    }
    if diag is not None:
        m["target_diag"] = {kk: diag.get(kk) for kk in (
            "n_known_classes", "frac_known_mass", "distill_target_flips",
            "distill_footprint_correction", "distill_footprint_regression",
            "distill_budget_known_mean", "distill_is_noop",
        )}
    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(m, f, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else float(x))
    test_outputs_path = Path(m["test_outputs"]) if m.get("test_outputs") else None
    manifest = {
        "schema_version": 1,
        "experiment_id": "phase_b_classmask_crn",
        "seed": seed,
        "dataset": dataset,
        "regime": regime,
        "global_source": global_source,
        "cid": k,
        "arm": m["arm"],
        "global_checkpoint": str(global_checkpoint) if global_checkpoint else None,
        "global_checkpoint_sha256": (
            checkpoint_sha256(global_checkpoint) if global_checkpoint and global_checkpoint.exists() else None
        ),
        "metrics_sha256": checkpoint_sha256(metrics_path),
        "test_outputs": str(test_outputs_path) if test_outputs_path else None,
        "test_outputs_sha256": (
            checkpoint_sha256(test_outputs_path)
            if test_outputs_path is not None and test_outputs_path.exists() else None
        ),
        "train_seed": run_metrics.get("train_seed"),
        "train_order_sha256": run_metrics.get("train_order_sha256"),
        "initial_state_sha256": run_metrics.get("initial_state_sha256"),
        "final_state_sha256": run_metrics.get("final_state_sha256"),
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


# ===========================================================================
# Table / CSV
# ===========================================================================

def _collect_rows(
    work_root: Path, datasets: List[str],
    global_sources: Optional[List[str]] = None,
    output_root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Collect ablation rows from every metrics.json under the runs tree.

    ``rglob`` sweeps *all* ``transfer_ablation/<global_source>/...`` subdirs, so the
    runs tree may yield several globals at once (e.g. a leftover ``expert`` run beside
    the current ``feddf`` one). ``global_sources`` (if given) restricts the export to
    those values. The value is read from metrics.json, falling back to the path segment
    for older runs that predate the ``global_source`` key — recovery happens *before*
    filtering, so stale rows are dropped rather than emitted with a blank global_source.
    """
    gs_filter = set(global_sources) if global_sources else None
    rows: List[Dict[str, Any]] = []
    search_root = Path(output_root) / "checkpoints" if output_root is not None else work_root / "runs"
    pattern = "metrics.json" if output_root is not None else f"{ABLATION_SUBDIR}/*/*/c*/metrics.json"
    for mp in sorted(search_root.rglob(pattern)):
        try:
            m = json.loads(mp.read_text())
        except Exception:
            continue
        if datasets and m.get("dataset") not in datasets:
            continue
        if not m.get("global_source"):
            # Older runs (pre-ablation-over-mask-free-globals) predate the
            # ``global_source`` key in metrics.json; the value is still recoverable
            # from the on-disk path: .../transfer_ablation/<global_source>/<arm>/c*/metrics.json
            m = dict(m)
            m["global_source"] = mp.parent.parent.parent.name
        if gs_filter is not None and m.get("global_source") not in gs_filter:
            continue
        rows.append(_row_from_metrics(m))
    return rows


def _print_table(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("[table] no rows.")
        return
    arms_order = ["global", "proxy", "proxy_plain", "local", "local_lwf", "mix"]

    def keyf(r):
        return (str(r["dataset"]), str(r["regime"]),
                arms_order.index(r["arm"]) if r["arm"] in arms_order else 99, r["cid"])

    rows = sorted(rows, key=keyf)
    print("\n=== Per-arm × per-client (g_*: expert fixed | p_*: personal on global | localsurf) ===")
    hdr = (f"{'dataset':9} {'regime':14} {'arm':14} {'cid':>3} {'nK':>3} {'N':>6} "
           f"| {'g_known':>8} {'g_unk':>7} "
           f"| {'p_known':>8} {'p_unk':>7} {'p_gap':>7} {'p_acc':>7} "
           f"| {'ls_known':>8} {'ls_unk':>7} {'ls_acc':>7} {'inv':>4}")
    print(hdr)
    print("-" * len(hdr))

    def fmt(v):
        try:
            return f"{float(v):.4f}" if v is not None and not (isinstance(v, float) and np.isnan(v)) else "  NaN "
        except (TypeError, ValueError):
            return str(v)

    for r in rows:
        print(f"{str(r['dataset']):9} {str(r['regime']):14} {str(r['arm']):14} "
              f"{r['cid']:>3} {str(r['n_known_classes']):>3} {str(r['transfer_n']):>6} "
              f"| {fmt(r['g_known_acc']):>8} {fmt(r['g_unknown_acc']):>7} "
              f"| {fmt(r['p_known_acc']):>8} {fmt(r['p_unknown_acc']):>7} "
              f"{fmt(r['p_gap_ku']):>7} {fmt(r['p_test_acc']):>7} "
              f"| {fmt(r['localsurf_known_acc']):>8} {fmt(r['localsurf_unknown_acc']):>7} "
              f"{fmt(r['localsurf_overall_acc']):>7} {str(r['invariants_ok']):>4}")

    # Per-arm means (per dataset×regime) on the personalized global surface
    print("\n=== Per-arm means by dataset×regime (p_* = personalized on global test) ===")
    print(f"{'dataset':9} {'regime':14} {'arm':14} {'g_known':>8} {'p_known':>8} {'p_unk':>7} {'p_gap':>7} {'p_acc':>7}")
    grp: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        grp.setdefault((str(r["dataset"]), str(r["regime"]), str(r["arm"])), []).append(r)

    def meanf(rs, key):
        vals = [float(x[key]) for x in rs if x[key] is not None and not (isinstance(x[key], float) and np.isnan(float(x[key])))]
        return float(np.mean(vals)) if vals else float("nan")

    for key in sorted(grp, key=lambda t: (t[0], t[1], arms_order.index(t[2]) if t[2] in arms_order else 99)):
        rs = grp[key]
        print(f"{key[0]:9} {key[1]:14} {key[2]:14} "
              f"{meanf(rs, 'g_known_acc'):8.4f} {meanf(rs, 'p_known_acc'):8.4f} "
              f"{meanf(rs, 'p_unknown_acc'):7.4f} {meanf(rs, 'p_gap_ku'):7.4f} {meanf(rs, 'p_test_acc'):7.4f}")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Phase A: transfer-set ablation for personal class_mask")
    ap.add_argument("--work_root", default=None,
                    help="Legacy combined input/output root. Prefer --input_work_root + --output_root.")
    ap.add_argument("--input_work_root", default=None,
                    help="Phase-A raw_work containing partitions/runs/checkpoints.")
    ap.add_argument("--output_root", default=None,
                    help="Phase-first Phase-B seed root; inputs are never written here.")
    ap.add_argument("--results_csv", default=None)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--datasets", nargs="*", default=["cifar"],
                    choices=["cifar", "cifar100", "cinic", "mnist", "fmnist"])
    ap.add_argument("--rel_filter", nargs="*", default=["single"],
                    help="Substrings selecting regimes (e.g. single multi iid alpha0p1).")
    ap.add_argument("--arms", nargs="*", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--cids", nargs="*", type=int, default=None,
                    help="Restrict to these client ids (default: all teachers).")
    ap.add_argument("--global_source", type=str, default="expert",
                    choices=list(_GLOBAL_SOURCE_TO_METHOD_DIR.keys()),
                    help="Which global student to use as warm-start and P_global anchor "
                         "(default: expert). E.g. feddf, energy, consensus.")
    ap.add_argument("--lwf_weight", type=float, default=1.0,
                    help="Per-batch weight of the LwF anchor term (local_lwf arm).")
    ap.add_argument("--skip_if_done", action="store_true")
    ap.add_argument("--reset", action="store_true",
                    help="Delete existing transfer_ablation/ outputs before rerunning. "
                         "Use this to purge old buggy data. Incompatible with --skip_if_done.")
    ap.add_argument("--table_only", action="store_true",
                    help="Only (re)build the CSV/table from existing metrics.json.")
    ap.add_argument("--csv_all_globals", action="store_true",
                    help="When (re)building the CSV, include EVERY global_source found under "
                         "runs/ (default: restrict the export to --global_source). The runs "
                         "tree can hold leftover ablations from other globals; the default "
                         "keeps the export scoped to the one under test.")
    ap.add_argument("--dry_run", action="store_true",
                    help="List jobs, reference rows and trained arms without writing artifacts.")
    ap.add_argument("--allow_exploratory_arms", action="store_true",
                    help="Explicitly allow local/local_lwf/mix in a phase-first output root. "
                         "Reserved for the storage-heavy six-arm replication deferred to the end.")
    add_training_args(ap, val_frac_default=0.15)
    args = ap.parse_args()
    if args.reset and args.skip_if_done:
        ap.error("--reset and --skip_if_done are mutually exclusive.")
    if not args.input_work_root and not args.work_root:
        ap.error("one of --input_work_root or legacy --work_root is required")
    return args


def main() -> None:
    args = parse_args()
    work_root = Path(args.input_work_root or args.work_root).resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else None
    if args.seed != 42 or "experiments" in work_root.parts:
        assert_seed_contained(work_root, args.seed)
    if output_root is not None:
        assert_seed_contained(output_root, args.seed)
        excluded = sorted(set(args.arms) - {"proxy_plain", "proxy"})
        if excluded and not args.allow_exploratory_arms:
            raise SystemExit(
                f"phase-first causal replication permits only proxy_plain/proxy training; got {excluded}"
            )
    data_dir = Path(args.data_dir).resolve()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_csv = (Path(args.results_csv).resolve() if args.results_csv else
               ((output_root / "raw_results" / "results.csv") if output_root is not None else
                work_root / f"transfer_set_ablation_results_{timestamp}.csv"))
    if not args.dry_run:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_root is not None and not args.dry_run:
        write_execution_provenance(
            output_root / "configs",
            name=f"phase_b_{args.global_source}_seed_{args.seed}",
            experiment_id=("phase_b_six_arm_exploratory" if args.allow_exploratory_arms
                           else "phase_b_classmask_causal"),
            seed=args.seed,
            args=args,
            input_paths={"phase_a_raw_work": str(work_root), "data_dir": str(data_dir)},
            output_paths={"output_root": str(output_root), "results_csv": str(out_csv)},
            repo_root=Path(__file__).resolve().parents[2],
        )

    # By default the CSV export is scoped to the global under test; --csv_all_globals
    # restores the old "collect every global_source in the tree" behaviour.
    gs_filter = None if args.csv_all_globals else [args.global_source]

    if args.arch:
        for ds in args.datasets:
            set_dataset_arch(ds, args.arch)
        print(f"[arch] backbone overridden -> {args.arch} for {args.datasets}")

    if args.table_only:
        rows = _collect_rows(work_root, args.datasets, gs_filter, output_root)
        write_csv(rows, out_csv, fieldnames=CSV_COLUMNS)
        _print_table(rows)
        try:
            _assert_g_invariant(rows)
        except AssertionError as e:
            print(f"[WARN] {e}")
        print(f"\n[done] {len(rows)} rows -> {out_csv}")
        return

    set_seed(args.seed)
    device = resolve_device(args.device)

    jobs = discover_proxy_jobs(work_root / "runs", args.datasets, args.seed, rel_filter=args.rel_filter)
    if not jobs:
        sys.exit(f"[FATAL] No proxy_analysis.npz under {work_root}/runs for datasets={args.datasets} "
                 f"rel_filter={args.rel_filter} seed={args.seed}")
    print(f"[INFO] {len(jobs)} (rel,dataset) jobs; arms={args.arms}; seed={args.seed}; device={device}")

    if args.dry_run:
        n_clients = len(args.cids) if args.cids is not None else 10
        print(f"[DRY RUN] global reference rows={len(jobs) * n_clients}; "
              f"trained tasks={len(jobs) * n_clients * len(args.arms)}; output_root={output_root or work_root}")
        for rel, dataset, _ in jobs:
            for cid in (args.cids if args.cids is not None else range(10)):
                print(_arm_out_dir(
                    Paths(work_root=work_root, data_dir=data_dir, dataset=dataset),
                    rel, "global", cid, args.seed, args.global_source, output_root,
                ))
                for arm in args.arms:
                    print(_arm_out_dir(
                        Paths(work_root=work_root, data_dir=data_dir, dataset=dataset),
                        rel, arm, cid, args.seed, args.global_source, output_root,
                    ))
        return

    if args.reset:
        for rel, dataset, _ in jobs:
            paths = Paths(work_root=work_root, data_dir=data_dir, dataset=dataset)
            ablation_root = ((output_root / "checkpoints" / args.global_source / rel)
                             if output_root is not None else
                             (paths.run_dir_for_rel(rel)
                              / ABLATION_SUBDIR / args.global_source))
            if ablation_root.exists():
                shutil.rmtree(ablation_root)
                print(f"  [reset] deleted {ablation_root}")

    test_loaders: Dict[str, Any] = {}
    train_eval_ds: Dict[str, Any] = {}
    for ds in args.datasets:
        cfg = get_dataset_config(ds)
        test_loaders[ds] = cfg.make_test_loader(data_dir, batch_size=256, num_workers=2)
        train_eval_ds[ds] = cfg.load_train_eval_dataset(data_dir)

    for rel, dataset, analysis_npz in jobs:
        cfg = get_dataset_config(dataset)
        arch = cfg.arch
        num_classes = cfg.num_classes
        regime = rel.split("/")[-1].split("__", 1)[0]
        print(f"\n=== {rel} / {dataset} (arch={arch}) ===")

        analysis = {k: v for k, v in np.load(analysis_npz, allow_pickle=True).items()}
        K = int(analysis["teacher_logits_cache"].shape[1])
        cids = args.cids if args.cids is not None else list(range(K))

        paths = Paths(work_root=work_root, data_dir=data_dir, dataset=dataset)
        spec = build_execution_spec(args=args, dataset=dataset, method="personal_class_mask",
                                    is_personal=True, global_source=args.global_source, min_delta=1e-4)

        for k in cids:
            try:
                _process_client(
                    k=k, rel=rel, regime=regime, dataset=dataset, arms=args.arms,
                    analysis=analysis, paths=paths, spec=spec, ds_full=train_eval_ds[dataset],
                    test_loader=test_loaders[dataset], device=device, arch=arch,
                    num_classes=num_classes, lwf_weight=args.lwf_weight, skip_if_done=args.skip_if_done,
                    global_source=args.global_source,
                    output_root=output_root,
                )
            except KeyboardInterrupt:
                print("\n[INTERRUPTED]")
                rows = _collect_rows(work_root, args.datasets, gs_filter, output_root)
                write_csv(rows, out_csv, fieldnames=CSV_COLUMNS)
                return
            except Exception as e:
                print(f"    [ERROR] k={k} rel={rel}: {e}")
                import traceback
                traceback.print_exc()

    rows = _collect_rows(work_root, args.datasets, gs_filter, output_root)
    write_csv(rows, out_csv, fieldnames=CSV_COLUMNS)
    _print_table(rows)
    try:
        _assert_g_invariant(rows)
    except AssertionError as e:
        print(f"\n[ASSERT FAILED] {e}")
        sys.exit(1)
    print(f"\n[done] {len(rows)} rows -> {out_csv}")


if __name__ == "__main__":
    main()
