#!/usr/bin/env python3
"""Article 1, Experiment 2: selection versus output-support aggregation.

This runner intentionally owns only the four-way mechanistic comparison:
FedDF, support restriction with all teachers, full-vector EXPERT, and
support-restricted EXPERT.  It consumes immutable teacher proxy logits and an
Experiment-1 holdout-authority artifact; it never reads local-test outputs or
a historical ``teacher_knows_class_mask``.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.dataset_config import get_dataset_config
from oracle_distillation.targets.expertise import (
    ARTICLE1_EXPERIMENT2_METHODS,
    build_server_expertise_target,
    paired_target_quality_metrics,
    target_quality_metrics,
)
from oracle_distillation.utils import resolve_device
from experiments.article_1_server_expertise.artifacts import sha256_file
from experiments.article_1_server_expertise.config import Article1PilotConfig
from experiments.article_1_server_expertise.proxy_designs import build_proxy_design
from experiments.article_1_server_expertise.protocols import ProxyDesignSpec, TrainingBudgetSpec
from experiments.article_1_server_expertise.storage import canonical_hash, jsonable, save_npz_once, write_json_once
from experiments.article_1_server_expertise.training import initial_state, train_student


EXPERIMENT_ID = "article1_experiment2_selection_support_v1"


def _scalar_string(value: np.ndarray) -> str:
    return str(np.asarray(value).item())


def _load_proxy(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        required = {"proxy_idx", "y_true_proxy", "teacher_logits_cache", "teacher_fingerprint"}
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"proxy source lacks {sorted(missing)}: {path}")
        result = {key: np.asarray(payload[key]) for key in required}
    logits = result["teacher_logits_cache"]
    labels = result["y_true_proxy"].astype(np.int64)
    indices = result["proxy_idx"].astype(np.int64)
    if logits.ndim != 3 or labels.shape != (len(logits),) or indices.shape != (len(logits),):
        raise ValueError("proxy source must contain logits [N,K,C], labels [N], and proxy_idx [N]")
    if (not np.isfinite(logits).all() or len(np.unique(indices)) != len(indices)
            or (labels < 0).any() or (labels >= logits.shape[2]).any()):
        raise ValueError("invalid proxy source logits, labels, or indices")
    fingerprint = _scalar_string(result["teacher_fingerprint"])
    if not fingerprint:
        raise ValueError("proxy source teacher_fingerprint must be non-empty")
    return {"logits": logits, "labels": labels, "indices": indices, "teacher_fingerprint": fingerprint}


def _load_holdout_authority(path: Path, *, expected_k: int, expected_c: int,
                            source_sha: str) -> tuple[np.ndarray, dict]:
    """Load Experiment-1's clean train/holdout/test authority artifact."""
    provenance_path = Path(path).with_name("provenance.json")
    if not provenance_path.is_file():
        raise FileNotFoundError(f"authority provenance missing: {provenance_path}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("source_proxy_sha256") != source_sha:
        raise ValueError("holdout authority does not match the proxy-logit source")
    mask_source = str(provenance.get("mask_source", ""))
    if not mask_source.startswith("holdout_acc_per_class>="):
        raise ValueError("Experiment 2 requires Experiment-1 holdout-derived authority, never local-test authority")
    with np.load(path, allow_pickle=False) as payload:
        required = {"authority", "source_proxy_sha256"}
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"authority artifact lacks {sorted(missing)}: {path}")
        authority = np.asarray(payload["authority"], dtype=np.float64)
        artifact_source_sha = _scalar_string(payload["source_proxy_sha256"])
    if authority.shape != (expected_k, expected_c) or not np.isin(authority, [0, 1]).all():
        raise ValueError("authority must be a binary [K,C] holdout mask")
    if artifact_source_sha != source_sha or (authority.sum(axis=1) == 0).any():
        raise ValueError("authority artifact has incompatible source or empty teacher support")
    return authority.astype(np.float32), {"authority_artifact": str(path), "authority_sha256": sha256_file(path), **provenance}


def _base(args: argparse.Namespace, *, design: ProxyDesignSpec, source_sha: str,
          calibration: dict, authority: np.ndarray) -> dict:
    return {
        "article": "article_1_server_expertise",
        "experiment": "experiment_2_selection_vs_support",
        "experiment_version": EXPERIMENT_ID,
        "dataset": args.dataset,
        "seed": int(args.seed),
        "regime": args.regime,
        "temperature": float(args.temperature),
        "source_proxy_analysis": str(args.source_proxy_analysis),
        "source_proxy_sha256": source_sha,
        "proxy_design": design.to_dict(),
        "expertise_protocol": "train__optimization_holdout__test",
        "authority_semantics": "hard_thresholded_accuracy_on_experiment1_holdout",
        "authority_sha256": canonical_hash({"authority": authority.tolist()}),
        **calibration,
    }


def _root(args: argparse.Namespace, design: ProxyDesignSpec) -> Path:
    token = canonical_hash(design.to_dict())[:10]
    return (Path(args.output_root) / "experiment_2" / args.dataset / f"seed_{args.seed}"
            / args.regime / f"N{design.size}_balanced_{token}")


def _inputs(args: argparse.Namespace):
    source = _load_proxy(args.source_proxy_analysis)
    design = ProxyDesignSpec(size=args.proxy_size, composition="balanced", seed=args.seed)
    selected = build_proxy_design(source["labels"], design)
    logits = source["logits"][selected.positions]
    labels = source["labels"][selected.positions]
    indices = source["indices"][selected.positions]
    source_sha = sha256_file(args.source_proxy_analysis)
    authority, authority_meta = _load_holdout_authority(
        args.authority_npz, expected_k=logits.shape[1], expected_c=logits.shape[2], source_sha=source_sha,
    )
    return logits, labels, indices, design, selected.diagnostics, authority, {
        **authority_meta,
    }, source_sha


def targets_stage(args: argparse.Namespace) -> Path:
    logits, labels, indices, design, design_diag, authority, calibration, source_sha = _inputs(args)
    base = _root(args, design)
    common = _base(args, design=design, source_sha=source_sha, calibration=calibration, authority=authority)
    write_json_once(base / "protocol.json", {
        **common, **design_diag, "methods": list(ARTICLE1_EXPERIMENT2_METHODS),
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    save_npz_once(base / "expertise.npz", authority=authority.astype(np.float32),
                  authority_sha256=np.array(calibration["authority_sha256"]))
    results = {}
    for method in ARTICLE1_EXPERIMENT2_METHODS:
        results[method] = build_server_expertise_target(
            logits, labels, authority, method=method, temperature=args.temperature,
        )
    all_selected = results["feddf"].selected_teachers
    if not np.array_equal(all_selected, results["all_teachers_support"].selected_teachers):
        raise AssertionError("all-teacher support changed teacher selection")
    if not np.array_equal(results["expert_full"].selected_teachers, results["expert_support"].selected_teachers):
        raise AssertionError("EXPERT support changed teacher selection")
    for method, result in results.items():
        record = {
            **common, **design_diag, **dict(result.diagnostics),
            **target_quality_metrics(result.probabilities, labels),
            **paired_target_quality_metrics(result.probabilities, results["feddf"].probabilities, labels),
            "method": method,
            "proxy_size": int(len(labels)),
            "student_training_status": "not_started",
            "config_hash": canonical_hash({**common, "method": method, "proxy_design": design.to_dict()}),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        target_dir = base / "targets" / method
        save_npz_once(target_dir / "target.npz", probabilities=result.probabilities,
                      pseudo_logits=result.pseudo_logits,
                      selected_teachers=result.selected_teachers.astype(np.uint8),
                      proxy_idx=indices, labels=labels, authority=authority.astype(np.float32))
        write_json_once(target_dir / "diagnostics.json", record)
    return base


def students_stage(args: argparse.Namespace, device: torch.device) -> Path:
    _, _, _, design, _, _, _, _ = _inputs(args)
    base = _root(args, design)
    protocol_path = base / "protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError("targets are required before students; run --stage targets")
    student_init_seed = args.seed if args.student_init_seed is None else args.student_init_seed
    batch_order_seed = args.seed if args.batch_order_seed is None else args.batch_order_seed
    cfg = replace(Article1PilotConfig(), seed=args.seed, dataset=args.dataset, proxy_size=args.proxy_size,
                  student_init_seed=student_init_seed, batch_order_seed=batch_order_seed,
                  temperature=args.temperature)
    budget = TrainingBudgetSpec(mode=args.training_mode, epochs=cfg.epochs, updates=args.fixed_updates)
    ds_cfg = get_dataset_config(args.dataset)
    train_dataset = ds_cfg.load_train_eval_dataset(Path("data"))
    test_dataset = ds_cfg.load_test_dataset(Path("data"))
    if args.smoke:
        test_dataset = Subset(test_dataset, list(range(min(64, len(test_dataset)))))
    test_loader = DataLoader(test_dataset, batch_size=64 if args.smoke else 256, shuffle=False, num_workers=0)
    init, init_hash = initial_state(cfg.student_arch, ds_cfg.num_classes, cfg.student_init_seed)
    pair_id = canonical_hash({"base": str(base), "init": init_hash, "order_seed": cfg.batch_order_seed,
                              "budget": budget.to_dict()})
    for method in ARTICLE1_EXPERIMENT2_METHODS:
        target_path = base / "targets" / method / "target.npz"
        out = base / "students" / method
        if not target_path.is_file():
            raise FileNotFoundError(target_path)
        if (out / "metrics.json").is_file():
            existing = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
            if existing.get("target_sha256") != sha256_file(target_path) or existing.get("paired_run_id") != pair_id:
                raise FileExistsError(f"existing student has incompatible provenance: {out}")
            continue
        with np.load(target_path, allow_pickle=False) as target:
            indices = np.asarray(target["proxy_idx"])
            probabilities = np.asarray(target["probabilities"])
        planned = 1 if args.smoke else budget.planned_updates(len(indices), cfg.batch_size)
        result = train_student(arch=cfg.student_arch, num_classes=ds_cfg.num_classes, init_state=init,
                               init_hash=init_hash, train_dataset=train_dataset, test_loader=test_loader,
                               dataset_indices=indices, target_probabilities=probabilities, device=device,
                               seed=cfg.batch_order_seed, batch_size=cfg.batch_size,
                               learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay,
                               epochs=cfg.epochs, temperature=cfg.temperature, max_updates=planned)
        target_diag = json.loads((base / "targets" / method / "diagnostics.json").read_text(encoding="utf-8"))
        write_json_once(out / "metrics.json", {
            **target_diag, **result.metrics, "optimization_budget": budget.to_dict(),
            "paired_run_id": pair_id, "student_artifact_role": "new_paired_student_from_reused_teacher_logits",
            "target_artifact": str(target_path), "target_sha256": sha256_file(target_path),
            "smoke": bool(args.smoke), "created_at": datetime.now(timezone.utc).isoformat(),
        })
        save_npz_once(out / "test_outputs.npz", logits=result.test_logits, labels=result.test_labels)
    metadata = [json.loads((base / "students" / method / "metrics.json").read_text())
                for method in ARTICLE1_EXPERIMENT2_METHODS]
    for field in ("student_init_sha256", "train_order_sha256", "total_updates", "paired_run_id"):
        if len({row[field] for row in metadata}) != 1:
            raise AssertionError(f"four-way paired invariant failed: {field}")
    return base


def aggregate_stage(args: argparse.Namespace) -> Path:
    _, _, _, design, _, _, _, _ = _inputs(args)
    base = _root(args, design)
    rows = []
    for method in ARTICLE1_EXPERIMENT2_METHODS:
        target = json.loads((base / "targets" / method / "diagnostics.json").read_text())
        rows.append({**target, "record_type": "target"})
        student = base / "students" / method / "metrics.json"
        if student.is_file():
            rows.append({**json.loads(student.read_text()), "record_type": "student"})
    output = base / "results.csv"
    if output.exists():
        raise FileExistsError(f"immutable aggregate already exists: {output}")
    fields = sorted(set().union(*(row.keys() for row in rows)))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: jsonable(value) for key, value in row.items()} for row in rows])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("targets", "students", "aggregate", "all"), required=True)
    parser.add_argument("--source-proxy-analysis", type=Path, required=True)
    parser.add_argument("--authority-npz", type=Path, required=True,
                        help="Experiment-1 holdout_authority/.../authority.npz")
    parser.add_argument("--output-root", type=Path, default=Path("OUTPUTS/experiments/article1_experiment2_v1"))
    parser.add_argument("--dataset", default="cifar")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--regime", required=True)
    parser.add_argument("--proxy-size", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=8.0)
    parser.add_argument("--student-init-seed", type=int,
                        help="Common initialization seed; defaults to --seed.")
    parser.add_argument("--batch-order-seed", type=int,
                        help="Common deterministic batch-order seed; defaults to --seed.")
    parser.add_argument("--training-mode", choices=("fixed_epochs", "fixed_updates"), default="fixed_epochs")
    parser.add_argument("--fixed-updates", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage in {"targets", "all"}:
        targets_stage(args)
    if args.stage in {"students", "all"}:
        students_stage(args, resolve_device(args.device))
    if args.stage in {"aggregate", "all"}:
        print(f"[OK] aggregate: {aggregate_stage(args)}")


if __name__ == "__main__":
    main()
