"""KD engine — both distillation paths over a precomputed proxy target.

A KD run trains a student to match a ``[N_proxy, C]`` target-logits tensor on the
proxy. The teachers are already collapsed into that target by ProxyAnalysisBuilder
/ TargetBuilder, so "how many teachers" is not a KD-time concern.

``KdRunner`` owns both entry points as instance methods, deliberately symmetric:
  * ``distill_personal(spec, student=..., dataset=..., data_dir=..., proxy_idx=...,
    target=...)`` — per-client KD on a personal target; the student is
    warm-started by the caller (class_mask is ALWAYS initialised from the global
    expert student). Cosine LR, early-stop on validation KD loss. The caller owns
    persistence (no student.pt / metrics.json from here).
  * ``distill_global(spec, paths=..., outputs=...)`` — global offline KD for
    feddf/consensus/oracle/expert. No scheduler, early-stop on validation accuracy.
    Owns the full lifecycle: dataset/target loading, training, test eval,
    extended diagnostics, and persistence (student.pt / metrics.json / manifest).
  * ``run(spec, ...)`` — single spec-driven dispatcher; routes to one of the above
    based on ``spec.is_personal``.

Both use the single KD loss ``losses.loss_soft_kd`` and share one training loop,
``_fit``: an ``IndexedSubset`` proxy loader + a ``[N_proxy, C]`` target tensor
looked up per batch via ``pos_of_global``. ``_fit`` is parameterised by
``scheduler`` and ``monitor``:
  * global   -> ``scheduler=None``,  ``monitor='val_acc'`` .
  * personal -> cosine scheduler,    ``monitor='val_loss'``.

Collaborating modules (split out of this file):
  * ``distill/kd_targets.py``  — ``KdTargets``: load/hold the target + alignment.
  * ``distill/kd_metrics.py``  — post-hoc evaluators (distillation quality,
    extended student metrics).
  * ``distill/losses.py``      — the single KD loss.
"""
from __future__ import annotations

import json
import hashlib
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from data.dataset_config import get_dataset_config
from oracle_distillation.distill import losses
from oracle_distillation.distill.kd_metrics import (
    evaluate_distillation_quality, eval_student_extended,
)
from oracle_distillation.distill.kd_targets import KdTargets, load_proxy_indices
from oracle_distillation.analysis.target_builders import GLOBAL_METHOD_TO_KEY
from oracle_distillation.config import ExecutionSpec
from oracle_distillation.checkpoints import checkpoint_sha256, save_model_checkpoint
from oracle_distillation.models import build_model, dataset_default_arch
from oracle_distillation.test_outputs import save_test_outputs
from oracle_distillation.utils import (
    eval_ce_acc, set_seed,
)


class KdRunner:
    """Single KD engine for global and personal (warm-start) distillation.

    Both paths share the ``_fit`` loop below over an ``IndexedSubset`` proxy loader
    + a ``[N_proxy, C]`` target tensor (looked up per batch via ``pos_of_global``).
    """

    def __init__(self, *, device: torch.device, seed: int = 42):
        self.device = device
        self.seed = seed

    # ------------------------------------------------------------- dispatch
    def run(
        self, spec: ExecutionSpec, *,
        paths: Optional["KdRunPaths"] = None,
        outputs: Optional["KdRunOutputs"] = None,
        analysis_dir: Optional[Path] = None,
        student: Optional[torch.nn.Module] = None,
        dataset: Optional[str] = None,
        data_dir: Optional[Path] = None,
        proxy_idx: Optional[np.ndarray] = None,
        target: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Single spec-driven entry point. Dispatches on ``spec.is_personal``.

        * global   -> requires ``paths`` + ``outputs`` (offline KD, full persistence).
        * personal -> requires ``student`` + ``dataset`` + ``data_dir`` + ``proxy_idx``
          + ``target`` (the caller owns warm-start, diagnostics and persistence).
        """
        if spec.is_personal:
            if any(x is None for x in (student, dataset, data_dir, proxy_idx, target)):
                raise ValueError(
                    "personal run requires student/dataset/data_dir/proxy_idx/target"
                )
            return self.distill_personal(
                spec, student=student, dataset=dataset, data_dir=data_dir,
                proxy_idx=proxy_idx, target=target,
            )
        if paths is None or outputs is None:
            raise ValueError("global run requires paths and outputs")
        return self.distill_global(spec, paths=paths, outputs=outputs, analysis_dir=analysis_dir)

    # -------------------------------------------------------------- personal
    def distill_personal(
        self, spec: ExecutionSpec, *, student: torch.nn.Module,
        dataset: str, data_dir: Path,
        proxy_idx: np.ndarray, target: np.ndarray, num_workers: int = 2,
        labels_for_diagnostics: bool = True,
    ) -> Dict[str, float]:
        """Train ``student`` (already warm-started or fresh) on the personal target.

        Cosine LR; early-stop on validation KD loss when ``spec.val_frac>0``. Uses the
        shared ``_fit`` engine. The caller owns student construction/warm-start and
        checkpoint saving. All training knobs come from ``spec``.
        """
        # Clean transform: the proxy is a transfer set; the student must match the
        # cached targets (extracted on the clean proxy), not random augmentations.
        ds_full = get_dataset_config(dataset).load_train_eval_dataset(data_dir)
        train_loader, val_loader, pos_of_global = _build_indexed_loaders(
            ds_full, proxy_idx, val_frac=spec.val_frac, min_val=spec.min_val,
            seed=self.seed, batch_size=spec.batch_size,
            train_workers=num_workers, val_workers=num_workers,
            pin_memory=(self.device.type == "cuda"),
            # Personal KD runs for many epochs over one fixed transfer set.
            # Keeping CUDA-loader workers alive avoids respawning them per epoch.
            persistent_workers=(num_workers > 0),
        )
        target_tensor = torch.from_numpy(np.asarray(target, dtype=np.float32)).to(self.device)
        pos_of_global = pos_of_global.to(self.device)

        student.train()
        opt = _make_optimizer(spec, student)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=spec.epochs)
        history, best_epoch, stopped_early, best_val = _fit(
            student=student, train_loader=train_loader, val_loader=val_loader,
            target_tensor=target_tensor, pos_of_global=pos_of_global, device=self.device,
            epochs=spec.epochs, temperature=spec.temperature, opt=opt, scheduler=sched,
            patience=spec.patience, min_delta=spec.min_delta, monitor="val_loss",
            labels_for_diagnostics=labels_for_diagnostics,
        )

        last = history[-1] if history else {}
        out: Dict[str, Any] = {
            "distill_train_kd_loss_final": last.get("distill_train_kd_loss", float("nan")),
            "model_best_epoch": best_epoch if (val_loader is not None and best_epoch > 0) else None,
            "model_stopped_early": bool(stopped_early),
            "train_order_sha256": train_loader.sampler.order_sha256,
            "train_epochs_completed": len(history),
            "train_steps_planned": int(spec.epochs * len(train_loader)),
            "train_steps_completed": int(len(history) * len(train_loader)),
            "train_batch_size": int(spec.batch_size),
            "train_examples": int(len(train_loader.dataset)),
            "train_num_workers": int(num_workers),
        }
        if val_loader is not None:
            out["distill_val_loss_final"] = last.get("distill_val_kd", float("nan"))
            out["distill_best_val_loss"] = best_val
        return out

    # ----------------------------------------------------------------- global
    def distill_global(
        self, spec: ExecutionSpec, *, paths: "KdRunPaths", outputs: "KdRunOutputs",
        analysis_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Global offline KD (feddf/consensus/oracle/expert) on a precomputed target.

        Mirrors ``distill_personal``'s shape: resolve dataset/target/loaders, train
        via the shared ``_fit`` engine, then (unlike personal) own test evaluation,
        extended diagnostics and persistence — there is no external caller for the
        global path. All training knobs come from ``spec``; re-seeds per call so
        each method's student gets an identical, reproducible init.
        """
        # Resolve the student arch default once so every downstream consumer
        # (model build, manifest, metrics) sees the concrete arch.
        if spec.student_arch is None:
            spec = replace(spec, student_arch=dataset_default_arch(spec.proxy_dataset))
        set_seed(self.seed)
        device = self.device
        method = spec.method.lower()

        if method not in METHOD_TO_TARGET_KEY:
            raise ValueError(
                f"Method '{method}' not supported by the offline runner. "
                f"Supported: {list(METHOD_TO_TARGET_KEY)}"
            )

        print(f"[*] KD method='{method}' dataset='{spec.proxy_dataset}' device='{device}'")

        # ---------------- Dataset --------------------------------------------
        ds_cfg = get_dataset_config(spec.proxy_dataset)
        # Clean transform: proxy = transfer set; student matches targets cached on
        # the clean proxy, so its inputs must not be randomly augmented.
        ds_full = ds_cfg.load_train_eval_dataset(paths.data_dir)
        test_loader = ds_cfg.make_test_loader(paths.data_dir, batch_size=256, num_workers=2)

        proxy_indices = load_proxy_indices(paths.proxy_split_npz)
        train_loader, val_loader, _ = _build_indexed_loaders(
            ds_full, proxy_indices, val_frac=spec.val_frac, min_val=spec.min_val,
            seed=self.seed, batch_size=(128 if spec.proxy_dataset == "cifar100" else spec.batch_size),
            train_workers=6, val_workers=2, pin_memory=(device.type == "cuda"),
            persistent_workers=True,
        )
        train_idx, _ = _split_train_val(
            proxy_indices, val_frac=spec.val_frac, min_val=spec.min_val, seed=self.seed,
        )

        # ---------------- Offline KD target ----------------------------------
        adir = Path(analysis_dir) if analysis_dir is not None else Path(paths.analysis_dir)
        targets = KdTargets.from_analysis_npz(adir, method, dataset=spec.proxy_dataset, device=device)
        target_tensor = targets.logits
        proxy_idx_in_npz = targets.proxy_idx
        _fine_to_coarse = targets.fine_to_coarse

        # y_true_proxy for distillation quality diagnostics (-1 sentinel if absent)
        if targets.y_true is not None:
            y_true_proxy = torch.from_numpy(targets.y_true).to(device)
        else:
            y_true_proxy = torch.full((len(proxy_idx_in_npz),), -1, dtype=torch.int64, device=device)

        # Sanity: the npz must cover all proxy indices we will consume
        pos_of_global = _build_pos_of_global(len(ds_full), proxy_idx_in_npz).to(device)
        missing = (pos_of_global[torch.from_numpy(train_idx).to(device)] < 0).any()
        if missing.item():
            raise RuntimeError(
                "Some train_idx are not present in proxy_analysis.npz proxy_idx. "
                "The proxy split used for analysis and for this KD run must match."
            )

        # ---------------- Train ----------------------------------------------
        student = _build_student(spec.student_arch, num_classes=ds_cfg.num_classes).to(device)

        t0 = time.time()
        opt = _make_optimizer(spec, student)
        history, best_epoch, stopped_early, _ = _fit(
            student=student,
            train_loader=train_loader,
            val_loader=val_loader,
            target_tensor=target_tensor,
            pos_of_global=pos_of_global,
            device=device,
            epochs=spec.epochs,
            temperature=spec.temperature,
            opt=opt,
            scheduler=None,
            patience=spec.patience,
            min_delta=spec.min_delta,
            monitor="val_acc",
        )
        total_time = time.time() - t0

        # ---------------- Distillation diagnostics ---------------------------
        distill_quality = evaluate_distillation_quality(
            student=student,
            target_tensor=target_tensor,
            y_true_proxy=y_true_proxy,
            pos_of_global=pos_of_global,
            train_loader=train_loader,
            val_loader=val_loader,
            temperature=spec.temperature,
            device=device,
        )

        # ---------------- Test eval + persist --------------------------------
        model_test_ce, model_test_acc = eval_ce_acc(student, test_loader, device=device)
        outputs.student_path.parent.mkdir(parents=True, exist_ok=True)
        save_test_outputs(
            outputs.student_path.parent / "test_outputs.npz",
            model=student, loader=test_loader, device=device,
            dataset=spec.proxy_dataset, method=method, seed=spec.seed,
            expected_accuracy=model_test_acc,
        )
        if spec.save_models:
            save_model_checkpoint(
                outputs.student_path, student,
                architecture=str(spec.student_arch),
                dataset=spec.proxy_dataset,
                regime=outputs.student_path.parents[2].name,
                method=method,
                seed=spec.seed,
                config=spec,
            )

        # Cache the freshly-trained student's real logits over proxy + local pool
        # This cache is retained even when model checkpoints are disabled: it is
        # an analysis artifact, not a serialized model.
        try:
            cache_path = save_student_logits_cache(
                student=student, ds_full=ds_full, proxy_idx=proxy_indices,
                partition_dir=paths.partition_dir, out_dir=outputs.student_path.parent,
                device=device, batch_size=spec.batch_size,
            )
            if cache_path is not None:
                print(f"[*] student_logits cache -> {cache_path}")
        except Exception as e:
            print(f"[WARN] student_logits cache failed: {e}")

        # ---------------- Extended Phase-5 metrics ---------------------------
        # Full-proxy loader — plain (x,y) batches (Subset, not IndexedSubset)
        _proxy_all_loader = DataLoader(
            Subset(ds_full, proxy_idx_in_npz.tolist()),
            batch_size=spec.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=(device.type == "cuda"),
        )

        extended = eval_student_extended(
            student=student,
            test_loader=test_loader,
            proxy_all_loader=_proxy_all_loader,
            target_tensor=target_tensor,
            device=device,
            num_classes=ds_cfg.num_classes,
            fine_to_coarse=_fine_to_coarse,
        )

        _write_manifest(outputs, spec, adir / "proxy_analysis.npz")

        return _save_metrics(
            outputs=outputs,
            cfg=spec,
            proxy_train_n=len(train_idx),
            model_test_acc=model_test_acc,
            model_test_ce=model_test_ce,
            total_time_s=total_time,
            best_epoch=best_epoch,
            stopped_early=stopped_early,
            history=history,
            distill_quality=distill_quality,
            extended=extended,
        )


# ===========================================================================
# Global offline KD engine (formerly distill/runner.py)
# ===========================================================================

MIN_ROUNDS = 10

# Single source of truth lives in analysis.target_builders.
METHOD_TO_TARGET_KEY: Dict[str, str] = GLOBAL_METHOD_TO_KEY


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class KdRunPaths:
    proxy_split_npz: Path
    data_dir: Path
    analysis_dir: Path            # directory containing proxy_analysis.npz
    # teachers_root kept for API compatibility but UNUSED (runner is offline)
    teachers_root: Optional[Path] = None
    # Partition dir (clients/cXXX.npz) for this rel — used only to build the
    # local-train-pool index for the student_logits.npz cache (see
    # save_student_logits_cache). None disables the cache (e.g. tests).
    partition_dir: Optional[Path] = None


@dataclass
class KdRunOutputs:
    student_path: Path
    metrics_path: Path


# ---------------------------------------------------------------------------
# Student logits cache — REAL forward-pass logits of the distilled student,
# saved alongside student.pt so later consumers (e.g. personal_class_mask's
# --global_logits_kind distilled) can read them instead of re-running the model.
# ---------------------------------------------------------------------------

def save_student_logits_cache(
    *, student: torch.nn.Module, ds_full: Dataset, proxy_idx: np.ndarray,
    partition_dir: Optional[Path], out_dir: Path, device: torch.device,
    batch_size: int = 256,
) -> Optional[Path]:
    """Cache ``student``'s own logits over the proxy and the local training pool
    (union of every client's ``train_idx`` under ``partition_dir``) as
    ``student_logits.npz`` in ``out_dir`` (alongside ``student.pt``/``metrics.json``).

    Returns the written path, or ``None`` if ``partition_dir`` is unavailable (no
    client splits to build the local pool from — caching is then skipped, not an
    error). proxy_idx and the local pool must be disjoint (enforced by partition
    design); raises if they aren't, since that would corrupt the cache semantics.
    """
    if partition_dir is None or not Path(partition_dir).is_dir():
        return None
    from o3_local.teacher_trainer import list_client_ids, load_client_npz
    from oracle_distillation.utils import forward_logits_for_indices

    client_ids = list_client_ids(partition_dir)
    local_parts = [load_client_npz(partition_dir, cid)[0] for cid in client_ids]  # train_idx
    local_idx = np.unique(np.concatenate(local_parts)).astype(np.int64)
    proxy_idx = np.asarray(proxy_idx).astype(np.int64)

    overlap = np.intersect1d(proxy_idx, local_idx)
    if overlap.size:
        raise RuntimeError(
            f"proxy_idx and local train_idx overlap ({overlap.size} indices) under "
            f"{partition_dir} — partition/proxy split is inconsistent."
        )

    proxy_logits = forward_logits_for_indices(student, ds_full, proxy_idx, device, batch_size)
    local_logits = forward_logits_for_indices(student, ds_full, local_idx, device, batch_size)

    out_path = Path(out_dir) / "student_logits.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        proxy_idx=proxy_idx.astype(np.int32), proxy_logits=proxy_logits.astype(np.float32),
        local_idx=local_idx.astype(np.int32), local_logits=local_logits.astype(np.float32),
    )
    return out_path


# ---------------------------------------------------------------------------
# Indexed subset (batch yields gidx so we can look up the cached target)
# ---------------------------------------------------------------------------

class IndexedSubset(Dataset):
    def __init__(self, base: Dataset, indices: np.ndarray):
        self.base = base
        self.indices = indices.astype(np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        gidx = int(self.indices[i])
        x, y = self.base[gidx]
        return x, y, gidx


# ---------------------------------------------------------------------------
# Proxy split (target IO lives in distill/kd_targets.py)
# ---------------------------------------------------------------------------

def _split_train_val(
    proxy_indices: np.ndarray,
    *,
    val_frac: float,
    min_val: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if val_frac <= 0.0:
        return proxy_indices, np.array([], dtype=np.int64)
    rng = np.random.default_rng(seed)
    idx = proxy_indices.copy()
    rng.shuffle(idx)
    n = len(idx)
    n_val = max(int(round(n * val_frac)), min_val)
    n_val = min(n_val, n)
    return idx[n_val:], idx[:n_val]


def _build_pos_of_global(n_total: int, proxy_indices: np.ndarray) -> torch.Tensor:
    """Map global dataset idx -> position in proxy_indices (-1 if not present)."""
    pos = np.full((n_total,), -1, dtype=np.int64)
    pos[proxy_indices] = np.arange(len(proxy_indices), dtype=np.int64)
    return torch.from_numpy(pos)


class _HashedRandomSampler(Sampler[int]):
    """Deterministic shuffle sampler that records the order actually emitted."""

    def __init__(self, data_source: Dataset, *, seed: int):
        self.data_source = data_source
        self.generator = torch.Generator().manual_seed(int(seed))
        self._digest = hashlib.sha256()
        self.epochs_emitted = 0

    def __iter__(self):
        order = torch.randperm(len(self.data_source), generator=self.generator)
        self._digest.update(self.epochs_emitted.to_bytes(8, "little"))
        self._digest.update(order.numpy().tobytes())
        self.epochs_emitted += 1
        return iter(order.tolist())

    def __len__(self) -> int:
        return len(self.data_source)

    @property
    def order_sha256(self) -> str:
        return self._digest.hexdigest()


def _build_indexed_loaders(
    ds_full: Dataset,
    proxy_indices: np.ndarray,
    *,
    val_frac: float,
    min_val: int,
    seed: int,
    batch_size: int,
    train_workers: int,
    val_workers: int,
    pin_memory: bool,
    persistent_workers: bool = False,
) -> Tuple[DataLoader, Optional[DataLoader], torch.Tensor]:
    """Shared train/val ``IndexedSubset`` loaders + ``pos_of_global`` (CPU tensor).

    Used by both ``distill_personal`` and ``distill_global``; they differ only in
    worker counts/persistence, passed in by the caller.
    """
    train_idx, val_idx = _split_train_val(proxy_indices, val_frac=val_frac, min_val=min_val, seed=seed)
    train_data = IndexedSubset(ds_full, train_idx)
    train_sampler = _HashedRandomSampler(train_data, seed=seed)
    worker_generator = torch.Generator().manual_seed(int(seed) ^ 0x5EED5EED)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, sampler=train_sampler,
        num_workers=train_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers and train_workers > 0,
        generator=worker_generator,
    )
    val_loader = None
    if len(val_idx) > 0:
        val_loader = DataLoader(
            IndexedSubset(ds_full, val_idx), batch_size=batch_size, shuffle=False,
            num_workers=val_workers, pin_memory=pin_memory,
            persistent_workers=persistent_workers and val_workers > 0,
        )
    pos_of_global = _build_pos_of_global(len(ds_full), proxy_indices)
    return train_loader, val_loader, pos_of_global


# ---------------------------------------------------------------------------
# Model / optim
# ---------------------------------------------------------------------------

def _build_student(student_arch: str, num_classes: int = 10) -> torch.nn.Module:
    return build_model(student_arch, num_classes=num_classes)


def _make_optimizer(spec: ExecutionSpec, student: torch.nn.Module) -> torch.optim.Optimizer:
    opt = spec.optimizer.lower()
    if opt == "adamw":
        return torch.optim.AdamW(student.parameters(), lr=spec.lr, weight_decay=spec.weight_decay)
    if opt == "adam":
        return torch.optim.Adam(student.parameters(), lr=spec.lr, weight_decay=spec.weight_decay)
    raise ValueError(f"Unsupported optimizer: {spec.optimizer}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train_one_epoch(
    *,
    student: torch.nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    target_tensor: torch.Tensor,        # [N_proxy, C] on device
    pos_of_global: torch.Tensor,        # [N_dataset] on device
    temperature: float,
    device: torch.device,
    labels_for_diagnostics: bool = True,
) -> Dict[str, float]:
    """One training epoch. Returns per-epoch metric dict with canonical names."""
    student.train()
    total_kd_loss = total_ce = total_acc = 0.0
    n = 0

    for x, y, gidx in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        gidx = gidx.to(device, non_blocking=True)

        student_logits = student(x)

        # Look up the precomputed teacher target for this batch
        pos = pos_of_global[gidx]                       # [B]
        target_logits = target_tensor[pos].to(device, non_blocking=True).detach()            # [B, C]

        loss = losses.loss_soft_kd(
            student_logits, target_logits, temperature, reduction="mean",
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        bs = y.size(0)
        n += bs
        total_kd_loss += float(loss.detach().item()) * bs
        if labels_for_diagnostics:
            # Labels are diagnostic only and never enter the KD objective.  Privacy-
            # constrained local runs can disable even these readouts so proxy labels
            # are not consumed anywhere in the adaptation phase.
            with torch.no_grad():
                total_ce  += float(torch.nn.functional.cross_entropy(student_logits, y).item()) * bs
                total_acc += float((student_logits.argmax(1) == y).float().mean().item()) * bs

    return {
        "distill_train_kd_loss": total_kd_loss / max(n, 1),
        "model_proxy_train_ce":  total_ce / max(n, 1) if labels_for_diagnostics else float("nan"),
        "model_proxy_train_acc": total_acc / max(n, 1) if labels_for_diagnostics else float("nan"),
    }


@torch.no_grad()
def _evaluate_ce(
    student: torch.nn.Module,
    loader: Optional[DataLoader],
    device: torch.device,
) -> Dict[str, float]:
    """CE + accuracy on a val loader. Returns NaN when loader is None."""
    _nan = float("nan")
    if loader is None:
        return {"model_proxy_val_ce": _nan, "model_proxy_val_acc": _nan}
    student.eval()
    total_ce = total_acc = 0.0
    n = 0
    for batch in loader:
        x, y = batch[0], batch[1]
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = student(x)
        bs = y.size(0)
        n += bs
        total_ce  += float(torch.nn.functional.cross_entropy(logits, y).item()) * bs
        total_acc += float((logits.argmax(1) == y).float().mean().item()) * bs
    return {
        "model_proxy_val_ce":  total_ce  / max(n, 1),
        "model_proxy_val_acc": total_acc / max(n, 1),
    }


@torch.no_grad()
def _evaluate_val_kd(student, loader, target_tensor, pos_of_global, temperature, device) -> float:
    """Mean KD loss on a val fold (IndexedSubset yielding x, y, gidx). NaN if None."""
    if loader is None:
        return float("nan")
    student.eval()
    s = 0.0
    n = 0
    for batch in loader:
        x, y, gidx = batch[0], batch[1], batch[2]
        x = x.to(device, non_blocking=True)
        gidx = gidx.to(device, non_blocking=True)
        tgt = target_tensor[pos_of_global[gidx]]
        s += float(losses.loss_soft_kd(student(x), tgt, temperature).item()) * y.size(0)
        n += y.size(0)
    return s / max(n, 1)


def _fit(
    *,
    student: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    target_tensor: torch.Tensor,
    pos_of_global: torch.Tensor,
    device: torch.device,
    epochs: int,
    temperature: float,
    opt: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    patience: int = 0,
    min_delta: float = 0.0,
    monitor: str = "val_acc",
    labels_for_diagnostics: bool = True,
) -> Tuple[List[Dict[str, float]], int, bool, float]:
    """Shared KD training loop for both distillation paths.

    ``monitor='val_acc'`` + ``scheduler=None``  -> the global engine.
    ``monitor='val_loss'`` + cosine scheduler    -> the personal engine.
    Returns (history, best_epoch, stopped_early, best_metric).
    """
    history: List[Dict[str, float]] = []
    best_epoch = -1
    best_state: Optional[Dict[str, torch.Tensor]] = None
    wait = 0
    stopped_early = False
    higher_better = (monitor == "val_acc")
    best_metric = -1.0 if higher_better else float("inf")

    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr = _train_one_epoch(
            student=student, loader=train_loader, opt=opt,
            target_tensor=target_tensor, pos_of_global=pos_of_global,
            temperature=temperature, device=device,
            labels_for_diagnostics=labels_for_diagnostics,
        )
        if monitor == "val_acc":
            val = _evaluate_ce(student, val_loader, device)
            metric = val["model_proxy_val_acc"]
            row = {"epoch": ep, "sec": time.time() - t0, **tr, **val}
            mstr = f"ValAcc={metric:.4f}" if not np.isnan(metric) else "ValAcc=NaN"
        else:
            metric = _evaluate_val_kd(student, val_loader, target_tensor,
                                      pos_of_global, temperature, device)
            row = {"epoch": ep, "sec": time.time() - t0, **tr, "distill_val_kd": metric}
            mstr = f"ValKD={metric:.4f}" if not np.isnan(metric) else "ValKD=NaN"
        history.append(row)
        if scheduler is not None:
            scheduler.step()
        print(f"Ep {ep:03d}: KDLoss={row['distill_train_kd_loss']:.4f} "
              f"TrainAcc={row['model_proxy_train_acc']:.4f} {mstr} "
              f"({row['sec']:.1f}s) Wait {wait}/{patience}")

        if val_loader is not None and patience > 0:
            improved = (metric == metric) and (
                (metric - best_metric) > min_delta if higher_better
                else (best_metric - metric) > min_delta
            )
            if improved:
                best_metric = metric
                best_epoch = ep
                best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience and ep > MIN_ROUNDS:
                    stopped_early = True
                    break

    if best_state is not None and val_loader is not None and patience > 0:
        student.load_state_dict(best_state)
    return history, best_epoch, stopped_early, best_metric


# ---------------------------------------------------------------------------
# Evaluation/metrics live in distill/kd_metrics.py
#   (evaluate_distillation_quality, eval_student_extended, eval_per_class_acc)
# ---------------------------------------------------------------------------


def _write_manifest(
    outputs: KdRunOutputs, spec: ExecutionSpec, proxy_analysis_path: Path,
) -> None:
    """Write run_manifest.json alongside metrics.json for implementation-level fields."""
    manifest = {
        "student_arch": spec.student_arch,
        "patience":     spec.patience,
        "min_delta":    spec.min_delta,
        "val_frac":     spec.val_frac,
        "min_val":      spec.min_val,
        "device":       spec.device,
        "proxy_analysis": str(proxy_analysis_path),
        "proxy_analysis_sha256": hashlib.sha256(proxy_analysis_path.read_bytes()).hexdigest(),
        "checkpoint_persisted": bool(spec.save_models),
        "checkpoint_path": str(outputs.student_path) if spec.save_models else None,
        "checkpoint_sha256": (
            checkpoint_sha256(outputs.student_path) if spec.save_models else None
        ),
        "test_outputs": str(outputs.metrics_path.parent / "test_outputs.npz"),
        "test_outputs_sha256": hashlib.sha256(
            (outputs.metrics_path.parent / "test_outputs.npz").read_bytes()
        ).hexdigest(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = outputs.metrics_path.parent / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))


def _save_metrics(
    *,
    outputs: KdRunOutputs,
    cfg: ExecutionSpec,
    proxy_train_n: int,
    model_test_acc: float,
    model_test_ce: float,
    total_time_s: float,
    best_epoch: int,
    stopped_early: bool,
    history: List[Dict[str, float]],
    distill_quality: Dict[str, float],
    extended: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build, persist, and return the final metrics dict (v5 canonical names)."""
    _nan = float("nan")
    last = history[-1] if history else {}
    ext  = extended or {}

    # --- v5 canonical names: super_acc as scalars, renamed keys -------------
    _super_acc_list = ext.get("model_super_acc")  # list or None
    _intra_list     = ext.get("model_fine_intra_super_acc")  # list or None
    model_super_acc_scalar      = float(np.mean(_super_acc_list)) if _super_acc_list is not None else _nan
    model_fine_in_super_acc_scalar = float(np.mean(_intra_list)) if _intra_list is not None else _nan

    # distill_student_target_agreement: prefer full-proxy (extended) over train-fold
    _sta = ext.get(
        "distill_student_target_agree",
        distill_quality.get("distill_student_target_agreement_train", _nan),
    )

    metrics: Dict[str, Any] = {
        # --- metadata ---
        "method":         cfg.method,
        "proxy_dataset":  cfg.proxy_dataset,
        "seed":           cfg.seed,
        "temperature":    cfg.temperature,
        "lr":             cfg.lr,
        "weight_decay":   cfg.weight_decay,
        "epochs":         cfg.epochs,
        "batch_size":     cfg.batch_size,
        "optimizer":      cfg.optimizer,
        "proxy_train_n":  proxy_train_n,
        "total_time_s":   float(total_time_s),
        # v5 canonical names for CSV
        "model_best_epoch":   int(best_epoch) if best_epoch >= 0 else None,
        "model_stopped_early": bool(stopped_early),
        # --- model metrics (classifier quality) ---
        "model_test_acc":              float(model_test_acc),
        "model_test_loss":             float(model_test_ce),
        "model_proxy_train_acc_final": last.get("model_proxy_train_acc", _nan),
        "model_proxy_train_ce_final":  last.get("model_proxy_train_ce",  _nan),
        "model_proxy_val_acc_final":   last.get("model_proxy_val_acc",   _nan),
        "model_proxy_val_ce_final":    last.get("model_proxy_val_ce",    _nan),
        "distill_train_kd_loss_final": last.get("distill_train_kd_loss", _nan),
        # --- distillation diagnostics (full dict; verbose keys kept for JSON) ---
        **distill_quality,
        # --- v5 canonical distill names ---
        "distill_target_acc": distill_quality.get("distill_target_train_acc", _nan),
        "distill_student_target_agreement": float(_sta),
        # --- extended Phase-5 metrics (per-class, ECE, …) ---
        **ext,
        # --- v5 canonical model names (override verbose ones from ext) ---
        "model_test_entropy":      ext.get("model_test_entropy_mean", _nan),
        "model_super_acc":         model_super_acc_scalar,
        "model_fine_in_super_acc": model_fine_in_super_acc_scalar,
        # keep verbose lists in JSON under their original names for analysis
        "model_super_acc_per_super":          _super_acc_list,
        "model_fine_in_super_acc_per_super":  _intra_list,
    }
    # proxy-vs-test accuracy gap
    if "model_proxy_acc" in ext:
        metrics["model_proxy_test_gap"] = float(model_test_acc) - ext["model_proxy_acc"]

    outputs.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with outputs.metrics_path.open("w") as f:
        json.dump({**metrics, "history": history}, f, indent=2, default=str)
    return metrics
