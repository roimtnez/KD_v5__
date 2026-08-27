#!/usr/bin/env python3
"""Manifest-driven Article-1 proxy/expertise matrix runner.

This runner consumes a *read-only* logits cache and creates immutable target,
split and student artifacts.  It deliberately has no option for label-light
protocols or local personalization: all labels here are either proxy labels
needed by the server-side estimator or labels on a declared competence
calibration set.
"""
from __future__ import annotations

import argparse
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
    ARTICLE1_METHODS,
    build_server_expertise_target,
    estimate_expertise_from_logits,
    target_quality_metrics,
)
from oracle_distillation.utils import resolve_device
from experiments.article_1_server_expertise.artifacts import sha256_file
from experiments.article_1_server_expertise.config import Article1PilotConfig
from experiments.article_1_server_expertise.protocols import (
    ExpertiseEstimateSpec,
    ProxyDesignSpec,
    TrainingBudgetSpec,
)
from experiments.article_1_server_expertise.proxy_designs import build_proxy_design
from experiments.article_1_server_expertise.storage import canonical_hash, save_npz_once, write_json_once
from experiments.article_1_server_expertise.training import initial_state, train_student


def _load_source(path: Path) -> dict[str, np.ndarray]:
    """Load only the stable cache interface, independently of legacy layout."""
    with np.load(path, allow_pickle=False) as source:
        required = {"proxy_idx", "y_true_proxy", "teacher_logits_cache", "teacher_knows_class_mask"}
        missing = required - set(source.files)
        if missing:
            raise ValueError(f"{path} lacks required cache fields {sorted(missing)}")
        data = {key: np.asarray(source[key]) for key in required}
    logits, labels, authority = (
        data["teacher_logits_cache"], data["y_true_proxy"], data["teacher_knows_class_mask"],
    )
    if logits.ndim != 3 or labels.shape != (len(logits),) or authority.shape != logits.shape[1:]:
        raise ValueError("source cache must contain [N,K,C] logits, [N] labels and [K,C] authority")
    if not np.isfinite(logits).all() or not np.isfinite(authority).all() or (authority < 0).any():
        raise ValueError("source logits/authority must be finite and authority non-negative")
    if (labels < 0).any() or (labels >= logits.shape[2]).any():
        raise ValueError("source labels outside logits class range")
    if len(np.unique(data["proxy_idx"])) != len(data["proxy_idx"]):
        raise ValueError("source proxy_idx must be unique")
    return data


def _load_calibration(path: Path, expected_k: int, expected_c: int) -> tuple[np.ndarray, np.ndarray]:
    """Read logits from a separately supplied competence-calibration cache."""
    if "localtest" in path.name.lower() or "local_test" in path.name.lower():
        raise ValueError(
            "refusing historical local-test logits as competence calibration; "
            "they are an evaluation surface in the preserved Study-I runs"
        )
    with np.load(path, allow_pickle=False) as source:
        logits_key = "teacher_logits" if "teacher_logits" in source else "teacher_localtest_logits"
        labels_key = "labels" if "labels" in source else "y_localtest"
        if logits_key not in source or labels_key not in source:
            raise ValueError("calibration cache needs teacher_logits/labels (or legacy-compatible aliases)")
        logits = np.asarray(source[logits_key])
        labels = np.asarray(source[labels_key], dtype=np.int64)
    if logits.ndim != 3 or logits.shape[1:] != (expected_k, expected_c) or labels.shape != (len(logits),):
        raise ValueError("calibration cache dimensions do not match source teachers/classes")
    return logits, labels


def _design_id(spec: ProxyDesignSpec) -> str:
    return f"N{spec.size}_{spec.composition}_{canonical_hash(spec.to_dict())[:10]}"


def _authority_id(spec: ExpertiseEstimateSpec) -> str:
    return f"{spec.source}_{canonical_hash(spec.to_dict())[:10]}"


def _root(output: Path, dataset: str, seed: int, regime: str, design: ProxyDesignSpec,
          estimate: ExpertiseEstimateSpec) -> Path:
    return Path(output) / "matrix_v1" / dataset / f"seed_{seed}" / regime / _design_id(design) / _authority_id(estimate)


def _authority(
    source: dict[str, np.ndarray], estimate: ExpertiseEstimateSpec, calibration: Path | None,
    known_authority_role: str,
) -> tuple[np.ndarray, dict, dict[str, np.ndarray]]:
    estimate.validate()
    if estimate.source == "known":
        return source["teacher_knows_class_mask"], {
            "authority_source": "known",
            "authority_semantics": "source_teacher_class_authority",
            "known_authority_role": known_authority_role,
            "scientific_status": (
                "clean_candidate_pending_split_provenance_audit"
                if known_authority_role == "clean_disjoint_competence" else
                "legacy_leakage_affected_not_final_publication_evidence"
            ),
            "calibration_used": False,
        }, {}
    if calibration is None:
        raise ValueError("estimated expertise requires --calibration-npz")
    logits, labels = _load_calibration(calibration, source["teacher_logits_cache"].shape[1], source["teacher_logits_cache"].shape[2])
    result = estimate_expertise_from_logits(
        logits, labels,
        mode="soft" if estimate.source == "estimated_soft" else "hard",
        threshold=estimate.threshold, prior_alpha=estimate.prior_alpha,
        prior_beta=estimate.prior_beta, min_class_examples=estimate.min_class_examples,
    )
    if (result.authority.sum(axis=1) == 0).any():
        raise ValueError(
            "estimated authority contains teachers without any supported class; "
            "adjust calibration/threshold instead of silently restoring full support"
        )
    return result.authority, {
        **dict(result.diagnostics),
        "authority_semantics": "teacher_class_beta_posterior" if estimate.source == "estimated_soft" else "teacher_class_thresholded_posterior",
        "calibration_used": True,
        "calibration_path": str(calibration),
        "calibration_sha256": sha256_file(calibration),
        "calibration_role": estimate.calibration_role,
        "scientific_status": "clean_candidate_pending_calibration_split_audit",
    }, {
        "posterior_mean": result.posterior_mean,
        "class_counts": result.class_counts,
        "correct_counts": result.correct_counts,
    }


def targets_stage(args: argparse.Namespace) -> list[Path]:
    source = _load_source(args.source_proxy_analysis)
    design_spec = ProxyDesignSpec(
        size=args.proxy_size, composition=args.composition, seed=args.seed,
        dropped_classes=tuple(args.dropped_classes), long_tail_ratio=args.long_tail_ratio,
    )
    estimate = ExpertiseEstimateSpec(
        source=args.authority_source, threshold=args.expertise_threshold,
        prior_alpha=args.prior_alpha, prior_beta=args.prior_beta,
        min_class_examples=args.min_class_examples, calibration_role=args.calibration_role,
    )
    design = build_proxy_design(source["y_true_proxy"], design_spec)
    base = _root(args.output_root, args.dataset, args.seed, args.regime, design_spec, estimate)
    source_sha = sha256_file(args.source_proxy_analysis)
    proxy_idx = source["proxy_idx"][design.positions]
    labels = source["y_true_proxy"][design.positions]
    logits = source["teacher_logits_cache"][design.positions]
    authority, authority_meta, authority_arrays = _authority(
        source, estimate, args.calibration_npz, args.known_authority_role,
    )
    protocol = {
        "article": "article_1_server_expertise",
        "matrix_version": "article1_matrix_v1",
        "dataset": args.dataset, "seed": args.seed, "regime": args.regime,
        "temperature": args.temperature, "source_proxy_analysis": str(args.source_proxy_analysis),
        "source_proxy_sha256": source_sha, "proxy_design": design_spec.to_dict(),
        "training_budget": {"declared": args.training_mode, "fixed_updates": args.fixed_updates},
        "expertise_estimation": estimate.to_dict(),
    }
    write_json_once(base / "protocol.json", {**protocol, **design.diagnostics, **authority_meta,
                                               "created_at": datetime.now(timezone.utc).isoformat()})
    save_npz_once(base / "proxy_design.npz", positions=design.positions, proxy_idx=proxy_idx,
                  labels=labels, source_proxy_sha256=np.array(source_sha))
    if authority_arrays:
        save_npz_once(base / "estimated_authority.npz", authority=authority, **authority_arrays)
    paths = []
    for method in args.methods:
        result = build_server_expertise_target(logits, labels, authority, method=method, temperature=args.temperature)
        record = {
            **protocol, **design.diagnostics, **authority_meta, **dict(result.diagnostics),
            **target_quality_metrics(result.probabilities, labels),
            "method": method, "proxy_size": int(len(labels)),
            "config_hash": canonical_hash({**protocol, "method": method}),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        method_dir = base / "targets" / method
        save_npz_once(method_dir / "target.npz", probabilities=result.probabilities,
                      pseudo_logits=result.pseudo_logits,
                      selected_teachers=result.selected_teachers.astype(np.uint8),
                      proxy_idx=proxy_idx.astype(np.int64), labels=labels.astype(np.int64),
                      authority=authority.astype(np.float32))
        write_json_once(method_dir / "diagnostics.json", record)
        paths.append(method_dir / "target.npz")
    for left, right in (("expert_full", "expert_support"), ("oracle_maskgated_full", "oracle_maskgated_support")):
        if left in args.methods and right in args.methods:
            with np.load(base / "targets" / left / "target.npz") as a, np.load(base / "targets" / right / "target.npz") as b:
                if not np.array_equal(a["selected_teachers"], b["selected_teachers"]):
                    raise AssertionError(f"selection changed within causal pair {left}/{right}")
    return paths


def students_stage(args: argparse.Namespace, device: torch.device) -> None:
    source = _load_source(args.source_proxy_analysis)
    design_spec = ProxyDesignSpec(size=args.proxy_size, composition=args.composition, seed=args.seed,
                                  dropped_classes=tuple(args.dropped_classes), long_tail_ratio=args.long_tail_ratio)
    estimate = ExpertiseEstimateSpec(source=args.authority_source, threshold=args.expertise_threshold,
                                    prior_alpha=args.prior_alpha, prior_beta=args.prior_beta,
                                    min_class_examples=args.min_class_examples, calibration_role=args.calibration_role)
    base = _root(args.output_root, args.dataset, args.seed, args.regime, design_spec, estimate)
    cfg = replace(Article1PilotConfig(), seed=args.seed, dataset=args.dataset, proxy_size=args.proxy_size,
                  student_init_seed=args.seed, batch_order_seed=args.seed, temperature=args.temperature)
    budget = TrainingBudgetSpec(mode=args.training_mode, epochs=cfg.epochs, updates=args.fixed_updates)
    ds_cfg = get_dataset_config(args.dataset)
    train_dataset = ds_cfg.load_train_eval_dataset(Path("data"))
    test_dataset = ds_cfg.load_test_dataset(Path("data"))
    if args.smoke:
        test_dataset = Subset(test_dataset, list(range(min(64, len(test_dataset)))))
    test_loader = DataLoader(test_dataset, batch_size=64 if args.smoke else 256, shuffle=False, num_workers=0)
    init, init_hash = initial_state(cfg.student_arch, ds_cfg.num_classes, cfg.student_init_seed)
    for method in args.methods:
        target_path = base / "targets" / method / "target.npz"
        diag_path = base / "targets" / method / "diagnostics.json"
        out = base / "students" / method
        if not target_path.is_file():
            raise FileNotFoundError(f"missing target; run matrix --stage targets first: {target_path}")
        if (out / "metrics.json").is_file():
            existing = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
            if existing.get("target_sha256") != sha256_file(target_path):
                raise FileExistsError(f"existing student has different target provenance: {out}")
            if existing.get("optimization_budget") != budget.to_dict():
                raise FileExistsError(f"existing student has different optimization budget: {out}")
            continue
        with np.load(target_path, allow_pickle=False) as target:
            proxy_idx = np.asarray(target["proxy_idx"])
            probabilities = np.asarray(target["probabilities"])
        planned = 1 if args.smoke else budget.planned_updates(len(proxy_idx), cfg.batch_size)
        result = train_student(arch=cfg.student_arch, num_classes=ds_cfg.num_classes, init_state=init,
                               init_hash=init_hash, train_dataset=train_dataset, test_loader=test_loader,
                               dataset_indices=proxy_idx, target_probabilities=probabilities, device=device,
                               seed=cfg.batch_order_seed, batch_size=cfg.batch_size,
                               learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay,
                               epochs=cfg.epochs, temperature=cfg.temperature, max_updates=planned)
        target_diag = json.loads(diag_path.read_text(encoding="utf-8"))
        record = {**target_diag, **result.metrics, "optimization_budget": budget.to_dict(),
                  "student_artifact_role": "new_student_from_reused_logits",
                  "target_artifact": str(target_path), "target_sha256": sha256_file(target_path),
                  "smoke": bool(args.smoke), "created_at": datetime.now(timezone.utc).isoformat()}
        write_json_once(out / "metrics.json", record)
        save_npz_once(out / "test_outputs.npz", logits=result.test_logits, labels=result.test_labels)
    for left, right in (("expert_full", "expert_support"), ("oracle_maskgated_full", "oracle_maskgated_support")):
        if left in args.methods and right in args.methods:
            left_meta = json.loads((base / "students" / left / "metrics.json").read_text())
            right_meta = json.loads((base / "students" / right / "metrics.json").read_text())
            for field in ("student_init_sha256", "train_order_sha256", "total_updates"):
                if left_meta[field] != right_meta[field]:
                    raise AssertionError(f"paired student invariant failed for {left}/{right}: {field}")


def supervised_stage(args: argparse.Namespace, device: torch.device) -> None:
    """Train the no-distillation control on the exact same proxy design."""
    source = _load_source(args.source_proxy_analysis)
    design_spec = ProxyDesignSpec(size=args.proxy_size, composition=args.composition, seed=args.seed,
                                  dropped_classes=tuple(args.dropped_classes), long_tail_ratio=args.long_tail_ratio)
    design = build_proxy_design(source["y_true_proxy"], design_spec)
    base = (Path(args.output_root) / "matrix_v1" / args.dataset / f"seed_{args.seed}" /
            args.regime / _design_id(design_spec) / "supervised")
    cfg = replace(Article1PilotConfig(), seed=args.seed, dataset=args.dataset, proxy_size=args.proxy_size,
                  student_init_seed=args.seed, batch_order_seed=args.seed, temperature=args.temperature)
    budget = TrainingBudgetSpec(mode=args.training_mode, epochs=cfg.epochs, updates=args.fixed_updates)
    ds_cfg = get_dataset_config(args.dataset)
    train_dataset = ds_cfg.load_train_eval_dataset(Path("data"))
    test_dataset = ds_cfg.load_test_dataset(Path("data"))
    if args.smoke:
        test_dataset = Subset(test_dataset, list(range(min(64, len(test_dataset)))))
    test_loader = DataLoader(test_dataset, batch_size=64 if args.smoke else 256, shuffle=False, num_workers=0)
    init, init_hash = initial_state(cfg.student_arch, ds_cfg.num_classes, cfg.student_init_seed)
    proxy_idx = source["proxy_idx"][design.positions]
    labels = source["y_true_proxy"][design.positions]
    planned = 1 if args.smoke else budget.planned_updates(len(proxy_idx), cfg.batch_size)
    if (base / "metrics.json").is_file():
        existing = json.loads((base / "metrics.json").read_text(encoding="utf-8"))
        if existing.get("optimization_budget") != budget.to_dict():
            raise FileExistsError(f"existing supervised control has different optimization budget: {base}")
        if existing.get("proxy_design") != design_spec.to_dict():
            raise FileExistsError(f"existing supervised control has different proxy design: {base}")
        return
    result = train_student(arch=cfg.student_arch, num_classes=ds_cfg.num_classes, init_state=init,
                           init_hash=init_hash, train_dataset=train_dataset, test_loader=test_loader,
                           dataset_indices=proxy_idx, hard_labels=labels, device=device,
                           seed=cfg.batch_order_seed, batch_size=cfg.batch_size,
                           learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay,
                           epochs=cfg.epochs, temperature=cfg.temperature, max_updates=planned)
    protocol = {
        "article": "article_1_server_expertise", "matrix_version": "article1_matrix_v1",
        "dataset": args.dataset, "seed": args.seed, "regime": args.regime,
        "method": "supervised_proxy", "target_semantics": "hard_labels",
        "source_proxy_analysis": str(args.source_proxy_analysis),
        "source_proxy_sha256": sha256_file(args.source_proxy_analysis),
        "proxy_design": design_spec.to_dict(), **design.diagnostics,
        "optimization_budget": budget.to_dict(), "proxy_size": int(len(proxy_idx)),
        "labeled_proxy_size": int(len(proxy_idx)),
        "student_artifact_role": "supervised_control_no_distillation",
        "smoke": bool(args.smoke), "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json_once(base / "protocol.json", protocol)
    save_npz_once(base / "proxy_design.npz", positions=design.positions, proxy_idx=proxy_idx,
                  labels=labels, source_proxy_sha256=np.array(protocol["source_proxy_sha256"]))
    write_json_once(base / "metrics.json", {**protocol, **result.metrics})
    save_npz_once(base / "test_outputs.npz", logits=result.test_logits, labels=result.test_labels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("targets", "students", "supervised", "all"), required=True)
    parser.add_argument("--source-proxy-analysis", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("OUTPUTS/experiments/article1_support_v1"))
    parser.add_argument("--dataset", default="cifar")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--regime", default="multi")
    parser.add_argument("--proxy-size", type=int, required=True)
    parser.add_argument("--composition", choices=("balanced", "uniform", "long_tail", "reduced_coverage"), default="balanced")
    parser.add_argument("--dropped-classes", type=int, nargs="*", default=[])
    parser.add_argument("--long-tail-ratio", type=float, default=8.0)
    parser.add_argument("--authority-source", choices=("known", "estimated_soft", "estimated_hard"), default="known")
    parser.add_argument("--known-authority-role", choices=("legacy_leakage_affected", "clean_disjoint_competence"),
                        default="legacy_leakage_affected")
    parser.add_argument("--calibration-npz", type=Path)
    parser.add_argument("--calibration-role", default="external_disjoint_competence_calibration")
    parser.add_argument("--expertise-threshold", type=float, default=0.7)
    parser.add_argument("--prior-alpha", type=float, default=1.0)
    parser.add_argument("--prior-beta", type=float, default=1.0)
    parser.add_argument("--min-class-examples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=8.0)
    parser.add_argument("--methods", nargs="+", choices=ARTICLE1_METHODS, default=list(ARTICLE1_METHODS))
    parser.add_argument("--training-mode", choices=("fixed_epochs", "fixed_updates"), default="fixed_epochs")
    parser.add_argument("--fixed-updates", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage in {"targets", "all"}:
        paths = targets_stage(args)
        print(f"[OK] wrote/reused {len(paths)} Article-1 targets")
    if args.stage in {"students", "all"}:
        students_stage(args, resolve_device(args.device))
    if args.stage == "supervised":
        supervised_stage(args, resolve_device(args.device))


if __name__ == "__main__":
    main()
