"""TeacherTrainer — self-contained local-teacher training (REFACTOR_PLAN phase 3).

This module is the single home for the "train local teachers" stage. It absorbs
the per-client training loop, the early-stopping training routine (formerly
``o3_local/train_test.py``) and all the partition-loading / evaluation /
diagnostics helpers (formerly scattered in ``o3_local/run_train_teachers_local.py``),
reusing the shared numeric utilities in ``oracle_distillation.utils``.

Per client:
    train_idx       -> training
    holdout_idx     -> validation set for early stopping
    local_test_idx  -> clean evaluation (feeds teacher_knows_class_mask)
    full test set   -> global evaluation

Behaviour is a faithful consolidation of the legacy code (same operation order
and RNG usage), so manifests/checkpoints are unchanged. Locked by
tests/test_teacher_trainer.py (manifest schema + opt-in real 1-client train).

``main(...)`` keeps the old function-call entry point used by
``run_dirichlet_alphas_methods``.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from oracle_distillation.models import build_model
from oracle_distillation.checkpoints import checkpoint_sha256, save_model_checkpoint
from oracle_distillation.utils import (
    set_seed, resolve_device, collect_logits, ece_np, entropy_mean_np, client_npz_name,
)
from data.dataset_config import get_dataset_config


# ===========================================================================
# Partition IO
# ===========================================================================

def load_client_npz(
    partition_dir: Path, cid: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_idx, holdout_idx, local_test_idx, known_fine).

    local_test_idx is empty for old partitions that lack the key.
    known_fine is empty for 10-class partitions that don't carry explicit masks.
    """
    import warnings
    p = Path(partition_dir) / "clients" / client_npz_name(cid)
    if not p.exists():
        raise FileNotFoundError(f"Missing client split: {p}")
    d = np.load(p)
    train_idx = d["train_idx"].astype(int)
    hold_idx = d["holdout_idx"].astype(int)
    if "local_test_idx" in d:
        local_test_idx = d["local_test_idx"].astype(int)
    else:
        warnings.warn(
            f"Partition {p} has no 'local_test_idx' — using empty array (old format). "
            "Regenerate partitions with local_test_frac>0 for per-client test sets.",
            stacklevel=2,
        )
        local_test_idx = np.array([], dtype=int)
    # known_fine may be a boolean competence mask (bool[C], CIFAR-100 k10) or an
    # explicit list of class ids; downstream expects a list of class ids.
    if "known_fine" in d:
        kf = d["known_fine"]
        known_fine = (np.where(kf)[0] if kf.dtype == bool else kf).astype(int)
    else:
        known_fine = np.array([], dtype=int)
    return train_idx, hold_idx, local_test_idx, known_fine


def list_client_ids(partition_dir: Path) -> List[int]:
    cdir = Path(partition_dir) / "clients"
    files = sorted(cdir.glob("c*.npz"))
    if not files:
        raise FileNotFoundError(f"No client files found in {cdir}")
    return [int(f.stem[1:]) for f in files]


def make_loaders_from_indices(
    train_ds_full, eval_ds_full, train_idx: np.ndarray, hold_idx: np.ndarray,
    batch_size: int, num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    """Train loader (augmented ``train_ds_full``) + holdout loader (CLEAN
    ``eval_ds_full``). The two datasets differ only when ``train_transform`` adds
    augmentation (CIFAR-100); validation/early-stopping must see clean images."""
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        Subset(train_ds_full, train_idx.tolist()),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin,
    )
    hold_loader = DataLoader(
        Subset(eval_ds_full, hold_idx.tolist()),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    return train_loader, hold_loader


def _to_json_list(arr: np.ndarray) -> List:
    return [float(x) if not math.isnan(x) else None for x in arr]


# ===========================================================================
# Evaluation + diagnostics
# ===========================================================================

@torch.no_grad()
def test_with_per_class_accuracy(
    net: torch.nn.Module, loader: DataLoader, device, num_classes: int = 10,
    fine_to_coarse: Optional[np.ndarray] = None,
) -> Tuple[float, float, np.ndarray, Optional[np.ndarray]]:
    """Return ``(avg_loss, accuracy, per_class_acc[C], per_super_acc[S] | None)``.

    ``per_class_acc`` is the fine-grained per-class accuracy (NaN for absent classes).
    When ``fine_to_coarse`` (int[C] fine→superclass map) is given, ``per_super_acc``
    is the per-superclass accuracy: for each true superclass ``s``, the fraction of
    its samples whose *predicted* superclass equals ``s`` (i.e. coarse-level
    correctness). It is ``None`` for flat datasets.
    """
    device = torch.device(device) if isinstance(device, str) else device
    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss().to(device)

    has_super = fine_to_coarse is not None
    if has_super:
        f2c = np.asarray(fine_to_coarse).astype(np.intp)
        n_super = int(f2c.max()) + 1
        correct_per_super = np.zeros(n_super, dtype=np.int64)
        total_per_super = np.zeros(n_super, dtype=np.int64)

    running_loss = 0.0
    correct_global = total_global = n_batches = 0
    correct_per_class = np.zeros(num_classes, dtype=np.int64)
    total_per_class = np.zeros(num_classes, dtype=np.int64)

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        loss = criterion(outputs, labels)

        running_loss += float(loss.item())
        n_batches += 1

        preds = outputs.argmax(dim=1)
        correct_global += int((preds == labels).sum().item())
        total_global += int(labels.size(0))

        labels_np = labels.cpu().numpy()
        preds_np = preds.cpu().numpy()
        for c in np.unique(labels_np):
            mask = labels_np == c
            total_per_class[c] += int(mask.sum())
            correct_per_class[c] += int((preds_np[mask] == c).sum())

        if has_super:
            super_true = f2c[labels_np]
            super_pred = f2c[np.clip(preds_np, 0, len(f2c) - 1)]
            for s in np.unique(super_true):
                m = super_true == s
                total_per_super[s] += int(m.sum())
                correct_per_super[s] += int((super_pred[m] == s).sum())

    avg_loss = running_loss / max(1, n_batches)
    accuracy = correct_global / max(1, total_global)
    per_class_acc = np.full(num_classes, np.nan, dtype=float)
    valid = total_per_class > 0
    per_class_acc[valid] = correct_per_class[valid] / total_per_class[valid]

    per_super_acc = None
    if has_super:
        per_super_acc = np.full(n_super, np.nan, dtype=float)
        v = total_per_super > 0
        per_super_acc[v] = correct_per_super[v] / total_per_super[v]
    return float(avg_loss), float(accuracy), per_class_acc, per_super_acc


def _confusion_known(
    logits: np.ndarray, labels: np.ndarray, known_fine: np.ndarray,
) -> List[List[int]]:
    """Confusion matrix restricted to known classes (extra 'other' column)."""
    if len(known_fine) == 0:
        return []
    idx_map = {int(c): i for i, c in enumerate(known_fine)}
    n_k = len(known_fine)
    mat = np.zeros((n_k, n_k + 1), dtype=np.int64)
    preds = logits.argmax(axis=1)
    for true_label, pred in zip(labels, preds):
        if int(true_label) not in idx_map:
            continue
        row = idx_map[int(true_label)]
        col = idx_map.get(int(pred), n_k)  # n_k = "other"
        mat[row, col] += 1
    return mat.tolist()


def _argmax_dist(logits: np.ndarray, num_classes: int) -> List[int]:
    """Histogram of argmax predictions (collapse detector)."""
    counts = np.bincount(logits.argmax(axis=1), minlength=num_classes)
    return counts.tolist()


def _teacher_diagnostics(
    net: torch.nn.Module, local_test_loader: Optional[DataLoader],
    proxy_loader: Optional[DataLoader], known_fine: np.ndarray,
    num_classes: int, device, proxy_logits: Optional[np.ndarray] = None,
) -> dict:
    """Per-teacher diagnostic block (ECE/entropy, known confusion, proxy argmax)."""
    diag: dict = {}
    if local_test_loader is not None:
        logits_lt, labels_lt = collect_logits(net, local_test_loader, device)
        diag["local_test_ece"] = ece_np(logits_lt, labels_lt)
        diag["local_test_entropy"] = entropy_mean_np(logits_lt)
        if len(known_fine) > 0:
            mask_known = np.isin(labels_lt, known_fine)
            if mask_known.sum() > 0:
                logits_k = logits_lt[mask_known]
                labels_k = labels_lt[mask_known]
                diag["local_test_ece_known"] = ece_np(logits_k, labels_k)
                diag["local_test_entropy_known"] = entropy_mean_np(logits_k)
                diag["local_test_conf_matrix_known"] = _confusion_known(logits_k, labels_k, known_fine)
                diag["known_fine"] = [int(c) for c in known_fine]
    if proxy_loader is not None:
        logits_px = proxy_logits
        if logits_px is None:
            logits_px, _ = collect_logits(net, proxy_loader, device)
        diag["proxy_argmax_dist"] = _argmax_dist(logits_px, num_classes)
    return diag


# ===========================================================================
# Training loop (with optional early stopping + LR scheduling)
# ===========================================================================

@torch.no_grad()
def _eval_loss(model, loader, loss_fn, device) -> float:
    model.eval()
    running, n = 0.0, 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        running += float(loss_fn(logits, labels).item())
        n += 1
    return running / max(1, n)


def _train_model(
    model: torch.nn.Module,
    train_loader,
    epochs: int,
    lr: float,
    device,
    optimizer=None,
    scheduler=None,
    val_loader=None,
    patience: int = 3,
    min_delta: float = 1e-3,
    warmup_epochs: int = 1,
    target_loss: Optional[float] = None,
    loss_fn: Optional[torch.nn.Module] = None,
) -> Tuple[float, int, Dict[str, Any], torch.nn.Module]:
    """Train with optional early stopping. Returns (best_loss, epochs_ran, history, model).

    Monitors val_loss if val_loader is given, else train_loss.
    """
    model = model.to(device)
    opt = optimizer if optimizer is not None else torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = loss_fn if loss_fn is not None else torch.nn.CrossEntropyLoss()

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    bad_epochs = 0
    history: Dict[str, Any] = {"train_loss": [], "val_loss": [], "monitor": []}

    for epoch in range(epochs):
        model.train()
        running, n = 0.0, 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            opt.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            opt.step()
            running += float(loss.item())
            n += 1

        train_loss = running / max(1, n)
        history["train_loss"].append(train_loss)

        if val_loader is not None:
            val_loss = _eval_loss(model, val_loader, loss_fn, device)
            history["val_loss"].append(val_loss)
            monitor, monitor_name = val_loss, "val_loss"
        else:
            monitor, monitor_name = train_loss, "train_loss"

        history["monitor"].append(monitor)
        if scheduler is not None:
            scheduler.step()
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}  {monitor_name}={monitor:.4f}")

        if target_loss is not None and monitor <= target_loss:
            model.epochs_ran = epoch + 1
            return float(monitor), epoch + 1, history, model

        if patience > 0 and (epoch + 1) >= warmup_epochs:
            if monitor < (best_loss - min_delta):
                best_loss = monitor
                best_state = copy.deepcopy(model.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    model.load_state_dict(best_state)
                    model.epochs_ran = epoch + 1
                    return float(best_loss), epoch + 1, history, model

        if (epoch + 1) < warmup_epochs and monitor < best_loss:
            best_loss = monitor
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.epochs_ran = epochs
    return float(best_loss), epochs, history, model


# ===========================================================================
# Trainer
# ===========================================================================

class TeacherTrainer:
    """Train one teacher per client for a partition and emit the manifest."""

    def __init__(
        self,
        dataset: str,
        data_dir: str,
        *,
        device: str = "auto",
        seed: int = 42,
        batch_size: int = -1,     # -1 -> DatasetConfig.teacher_batch_size
        epochs: int = -1,         # -1 -> DatasetConfig.teacher_epochs
        lr: float = -1.0,         # -1 -> DatasetConfig.teacher_lr
        es_patience: int = -1,
        es_min_delta: float = 5e-4,
        num_workers: int = 2,
        save_models: bool = False,
    ) -> None:
        self.dataset = dataset
        self.data_dir: Path = Path(data_dir)
        self.seed = seed
        set_seed(seed)
        self.device = resolve_device(device)

        self.ds_cfg = get_dataset_config(dataset)
        self.batch_size = self.ds_cfg.teacher_batch_size if batch_size < 0 else batch_size
        self.epochs = self.ds_cfg.teacher_epochs if epochs < 0 else epochs
        self.lr = self.ds_cfg.teacher_lr if lr < 0 else lr
        self.es_patience = self.ds_cfg.teacher_patience if es_patience < 0 else es_patience
        self.es_min_delta = es_min_delta
        self.num_workers = num_workers
        self.save_models = bool(save_models)

        self.num_classes = self.ds_cfg.num_classes
        self.model_arch = self.ds_cfg.arch

        self.partition_dir: Optional[Path] = None
        self.train_ds_full = None
        self.test_loader: Optional[DataLoader] = None
        self.proxy_loader: Optional[DataLoader] = None
        self.fine_to_coarse: Optional[np.ndarray] = None   # set for CIFAR-100 (per-superclass metrics)
        self.client_ids: List[int] = []
        self._init_state: Optional[dict] = None
        self._last_ckpt_path: Optional[Path] = None
        self._last_proxy_cache_path: Optional[Path] = None
        self.proxy_idx: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ setup
    def prepare(self, partition_dir) -> None:
        partition_dir = Path(partition_dir)
        if not (partition_dir / "clients").exists():
            raise FileNotFoundError(f"partition_dir must contain clients/: {partition_dir}")
        self.partition_dir = partition_dir

        data_dir = self.data_dir
        self.train_ds_full = self.ds_cfg.load_train_dataset(data_dir)        # augmented (teacher SGD)
        self.eval_ds_full = self.ds_cfg.load_train_eval_dataset(data_dir)    # CLEAN (holdout/local-test/proxy)
        self.test_loader = self.ds_cfg.make_test_loader(
            data_dir, batch_size=self.batch_size, num_workers=self.num_workers
        )
        self.client_ids = list_client_ids(partition_dir)

        self.proxy_loader = None
        _proxy_npz_path = self.ds_cfg.proxy_path_for_seed(self.seed, data_dir)
        config_path = partition_dir / "config.yaml"
        if config_path.is_file():
            partition_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            partition_seed = int(partition_cfg.get("seed", -1))
            recorded_proxy = Path(str(partition_cfg.get("proxy_split_npz", ""))).name
            if partition_seed != int(self.seed):
                raise ValueError(
                    f"teacher/partition seed mismatch {self.seed}!={partition_seed}: {partition_dir}"
                )
            if recorded_proxy != _proxy_npz_path.name:
                raise ValueError(
                    "teacher/partition proxy split mismatch: "
                    f"{_proxy_npz_path.name!r}!={recorded_proxy!r}"
                )
        if _proxy_npz_path.exists():
            with np.load(_proxy_npz_path, allow_pickle=True) as _proxy_npz:
                _proxy_idx = _proxy_npz["proxy_idx"].astype(int)
                self.proxy_idx = _proxy_idx.astype(np.int64, copy=False)
                expected_hash = hashlib.sha256(
                    np.ascontiguousarray(_proxy_idx.astype(np.int64)).tobytes()
                ).hexdigest()
                recorded_hash = str(partition_cfg.get("proxy_idx_sha256", "")) if config_path.is_file() else ""
                if self.seed != 42 and recorded_hash != expected_hash:
                    raise ValueError(
                        "teacher/partition proxy_idx hash mismatch: "
                        f"{expected_hash!r}!={recorded_hash!r}"
                    )
                if "fine_to_coarse" in _proxy_npz:
                    self.fine_to_coarse = _proxy_npz["fine_to_coarse"].astype(int)
            self.proxy_loader = DataLoader(
                Subset(self.eval_ds_full, _proxy_idx.tolist()),
                batch_size=self.batch_size, shuffle=False,
                num_workers=self.num_workers, pin_memory=torch.cuda.is_available(),
            )
            print(f"[INFO] Proxy loader ready: {len(_proxy_idx)} samples for argmax diagnostics")
            # fine→superclass map (CIFAR-100): enables per-superclass accuracy.
            if self.fine_to_coarse is not None:
                print(f"[INFO] fine_to_coarse loaded → per-superclass metrics enabled "
                      f"({int(self.fine_to_coarse.max()) + 1} superclasses)")

        _template = build_model(self.model_arch, num_classes=self.num_classes)
        self._init_state = copy.deepcopy(_template.state_dict())
        del _template
        print(f"[INFO] Shared initial weights captured "
              f"(arch={self.model_arch}, num_classes={self.num_classes}, seed={self.seed})")

    # ------------------------------------------------------------ per client
    def train_client(self, cid: int, teachers_dir: Path) -> Dict:
        """Train/evaluate one client and persist its proxy-logit cache."""
        teachers_dir = Path(teachers_dir)
        ckpt_path = teachers_dir / f"cid_{cid:03d}.pt"

        partition_dir = self.partition_dir
        test_loader = self.test_loader
        assert partition_dir is not None
        assert test_loader is not None
        train_idx, hold_idx, local_test_idx, known_fine = load_client_npz(partition_dir, cid)
        train_loader, hold_loader = make_loaders_from_indices(
            self.train_ds_full, self.eval_ds_full, train_idx, hold_idx,
            self.batch_size, self.num_workers,
        )

        print(f"\n[Train] cid={cid}  n_train={len(train_idx)}  n_hold={len(hold_idx)}"
              f"  lr={self.lr}  epochs={self.epochs}  patience={self.es_patience}")
        net = build_model(self.model_arch, num_classes=self.num_classes)
        net.load_state_dict(copy.deepcopy(self._init_state))  # same start for every client
        opt = self.ds_cfg.make_optimizer(net.parameters())
        sched = self.ds_cfg.make_scheduler(opt, epochs=self.epochs)
        _, epochs_ran, _, net = _train_model(
            net, train_loader=train_loader, epochs=self.epochs, lr=self.lr,
            device=self.device, optimizer=opt, scheduler=sched,
            val_loader=hold_loader, patience=self.es_patience, min_delta=self.es_min_delta,
            loss_fn=self.ds_cfg.make_criterion(),
        )

        # -- Evaluation 1: local_test_idx (clean transform via eval_ds_full) --
        if len(local_test_idx) > 0:
            local_test_loader_cid = DataLoader(
                Subset(self.eval_ds_full, local_test_idx.tolist()),
                batch_size=self.batch_size, shuffle=False,
                num_workers=self.num_workers, pin_memory=torch.cuda.is_available(),
            )
            local_test_loss, local_test_acc, local_test_acc_per_class, local_test_acc_per_super = \
                test_with_per_class_accuracy(net, local_test_loader_cid, self.device,
                                             num_classes=self.num_classes,
                                             fine_to_coarse=self.fine_to_coarse)
            local_test_acc_list = _to_json_list(local_test_acc_per_class)
            local_test_super_list = _to_json_list(local_test_acc_per_super) \
                if local_test_acc_per_super is not None else None
        else:
            local_test_loader_cid = None
            local_test_loss = local_test_acc = float("nan")
            local_test_acc_list = [None] * self.num_classes
            local_test_super_list = None

        # -- Evaluation 2: global test set --
        test_loss, test_acc, test_acc_per_class_global, test_acc_per_super_global = \
            test_with_per_class_accuracy(net, test_loader, self.device,
                                         num_classes=self.num_classes,
                                         fine_to_coarse=self.fine_to_coarse)

        # Persist per-teacher proxy logits so the server aggregation stage does
        # not require a model checkpoint. This keeps the default pipeline fully
        # executable with --save_models disabled.
        proxy_logits = None
        proxy_labels = None
        if self.proxy_loader is not None:
            proxy_logits, proxy_labels = collect_logits(net, self.proxy_loader, self.device)

        # -- Phase-3 diagnostics --
        diag = _teacher_diagnostics(
            net, local_test_loader_cid, self.proxy_loader, known_fine,
            self.num_classes, self.device, proxy_logits=proxy_logits,
        )

        proxy_cache_path = teachers_dir / f"cid_{cid:03d}_proxy_logits.npz"
        if proxy_logits is None or proxy_labels is None or self.proxy_idx is None:
            raise RuntimeError("teacher proxy-logit cache cannot be built: proxy loader is unavailable")
        np.savez_compressed(
            proxy_cache_path,
            logits=proxy_logits.astype(np.float16),
            labels=proxy_labels.astype(np.int64),
            proxy_idx=self.proxy_idx.astype(np.int64),
            seed=np.array(self.seed, dtype=np.int64),
            dataset=np.array(self.dataset),
            cid=np.array(cid, dtype=np.int64),
        )
        if self.save_models:
            save_model_checkpoint(
                ckpt_path, net,
                architecture=self.model_arch,
                dataset=self.dataset,
                regime=self.partition_dir.name,
                method="local_teacher",
                seed=self.seed,
                config={
                    "cid": cid,
                    "epochs_max": self.epochs,
                    "epochs_ran": epochs_ran,
                    "batch_size": self.batch_size,
                    "lr": self.lr,
                    "holdout_patience": self.es_patience,
                },
            )

        row = {
            "cid": cid,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "seed": self.seed,
            "device": str(self.device),
            "epochs_max": self.epochs,
            "epochs_ran": epochs_ran,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "n_train": int(len(train_idx)),
            "n_holdout": int(len(hold_idx)),
            "n_local_test": int(len(local_test_idx)),
            "n_final_train": int(len(train_idx)),
            "local_test_loss": float(local_test_loss) if not math.isnan(local_test_loss) else None,
            "local_test_acc": float(local_test_acc) if not math.isnan(local_test_acc) else None,
            "local_test_acc_per_class": local_test_acc_list,
            "local_test_acc_per_super": local_test_super_list,
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
            "test_acc_per_class": _to_json_list(test_acc_per_class_global),
            "test_acc_per_super": (_to_json_list(test_acc_per_super_global)
                                   if test_acc_per_super_global is not None else None),
            # Emit diagnostics and the raw local-test per-class accuracies;
            # the server-side proxy analysis step decides the threshold and
            # derives the expert mask for each experiment.
            **diag,
        }

        lt = f"{local_test_acc:.4f}" if not math.isnan(local_test_acc) else "N/A"
        print(f"[OK] cid={cid}  epochs={epochs_ran}  lr={self.lr}"
              f"  local_test_acc={lt}  global_test_acc={test_acc:.4f}")
        self._last_ckpt_path = ckpt_path if self.save_models else None
        self._last_proxy_cache_path = proxy_cache_path
        return row

    # ----------------------------------------------------------- orchestrate
    def run(self, partition_dir, out_root) -> List[dict]:
        self.prepare(partition_dir)
        out_dir = Path(out_root)
        teachers_dir = out_dir / "teachers"
        teachers_dir.mkdir(parents=True, exist_ok=True)

        manifest: List[dict] = []
        for cid in self.client_ids:
            row = self.train_client(cid, teachers_dir)
            manifest.append({
                "cid": cid,
                "ckpt": (os.path.relpath(self._last_ckpt_path, out_dir)
                         if self._last_ckpt_path is not None else None),
                "ckpt_sha256": (checkpoint_sha256(self._last_ckpt_path)
                                if self._last_ckpt_path is not None else None),
                "proxy_logits": os.path.relpath(self._last_proxy_cache_path, out_dir),
                "crea": row,
            })

        manifest = sorted(manifest, key=lambda x: x["cid"])
        (out_dir / "teachers_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        run_cfg = {
            "partition_dir": os.path.relpath(self.partition_dir, Path.cwd()),
            "data_dir": os.path.relpath(self.data_dir, Path.cwd()),
            "dataset": self.dataset,
            "seed": self.seed,
            "device": str(self.device),
            "epochs_max": self.epochs,
            "lr": self.lr,
            "es_patience": self.es_patience,
            "es_min_delta": self.es_min_delta,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "num_clients": len(self.client_ids),
            "save_models": self.save_models,
        }
        (out_dir / "run_config.yaml").write_text(
            yaml.safe_dump(run_cfg, sort_keys=False), encoding="utf-8"
        )

        print(f"\n[DONE] Saved teacher caches to: {teachers_dir}")
        print(f"[DONE] Manifest: {out_dir / 'teachers_manifest.json'}")
        return manifest


# ===========================================================================
# Manifest → teachers_results.csv rows
# ===========================================================================

def load_manifest(manifest_path: Path) -> List[dict]:
    """Read a teachers_manifest.json (list of {cid, ckpt, crea})."""
    return json.loads(Path(manifest_path).read_text())


def teacher_csv_rows(
    manifest: List[dict], *, group_alpha: str, proxy_dataset: str, seed: int,
) -> List[Dict[str, Any]]:
    """Flatten a teachers manifest into teachers_results.csv rows (one per cid).

    The manifest (teachers_manifest.json) remains the canonical per-run record;
    this projects the queryable subset into the shared top-level CSV. Used by both
    the pipeline (run_dirichlet_alphas_methods) and scripts/build_teachers_results_csv.py.
    """
    rows: List[Dict[str, Any]] = []
    for entry in sorted(manifest, key=lambda e: e.get("cid", 0)):
        crea = entry.get("crea", {})
        rows.append({
            "group_alpha":        group_alpha,
            "proxy_dataset":      proxy_dataset,
            "seed":               seed,
            "cid":                entry.get("cid"),
            "n_train":            crea.get("n_train"),
            "n_holdout":          crea.get("n_holdout"),
            "n_local_test":       crea.get("n_local_test"),
            "epochs_ran":         crea.get("epochs_ran"),
            "epochs_max":         crea.get("epochs_max"),
            "local_test_acc":     crea.get("local_test_acc"),
            "local_test_loss":    crea.get("local_test_loss"),
            "test_acc":           crea.get("test_acc"),
            "test_loss":          crea.get("test_loss"),
            # per-class (fine, 100) + per-superclass (coarse, 20; None off CIFAR-100)
            "local_test_acc_per_class": crea.get("local_test_acc_per_class"),
            "test_acc_per_class":       crea.get("test_acc_per_class"),
            "local_test_acc_per_super": crea.get("local_test_acc_per_super"),
            "test_acc_per_super":       crea.get("test_acc_per_super"),
            "local_test_ece":     crea.get("local_test_ece"),
            "local_test_entropy": crea.get("local_test_entropy"),
            "timestamp":          crea.get("timestamp"),
        })
    return rows


# ===========================================================================
# Function-call entry point (kept for run_dirichlet_alphas_methods)
# ===========================================================================

def main(
    partition_dir: str,
    data_dir: str = "../data",
    out_root: str = "./outputs/h3_teachers",
    device: str = "auto",
    seed: int = 42,
    batch_size: int = -1,
    num_workers: int = 2,
    epochs: int = -1,
    lr: float = -1.0,
    es_patience: int = -1,
    es_min_delta: float = 5e-4,
    dataset: str = "cifar",
    save_models: bool = False,
) -> None:
    """Train N local teachers for a partition_dir (thin wrapper over TeacherTrainer)."""
    TeacherTrainer(
        dataset=dataset, data_dir=data_dir, device=device, seed=seed,
        batch_size=batch_size, epochs=epochs, lr=lr,
        es_patience=es_patience, es_min_delta=es_min_delta, num_workers=num_workers,
        save_models=save_models,
    ).run(partition_dir, out_root)
