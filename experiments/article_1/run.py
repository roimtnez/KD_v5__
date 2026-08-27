#!/usr/bin/env python3
"""Run the canonical Article-1 targets and paired students.

Presets keep the scientific matrix explicit:

* central: FedDF / support-only / EXPERT-full / EXPERT-v2 at N=10000.
* oracle: ORACLE-full / ORACLE-v2 / matched supervised sanity control.
* curve: historical seed-42 EXPERT-support diagnostic curve.
* proxy-efficiency: final multi-seed EXPERT-full baseline across proxy size.
* proxy-efficiency-energy: optional Energy extension, run only after baseline.
* threshold: small canonical-threshold +/-0.05 ablation for EXPERT-v2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset_config import get_dataset_config
from experiments.article_1 import OUTPUT_ROOT, PROTOCOL_VERSION
from experiments.article_1.io import sha256_array, sha256_file, write_json, write_npz
from experiments.article_1.protocol import (
    ALL_METHODS, CENTRAL_METHODS, CURVE_METHODS, DATASETS, FINAL_BASELINE_METHODS,
    MECHANISM_CONTROL_METHODS, ORACLE_METHODS,
    PROXY_EFFICIENCY_BASELINE_METHODS, PROXY_EFFICIENCY_ENERGY_METHODS,
    PROXY_SIZES, REGIMES, SEEDS, THRESHOLD_OFFSETS, StudentRecipe,
    regime_metadata, run_dir, source_dir, threshold_for,
)
from experiments.article_1.targets import (
    authority_from_holdout, build_target, expertise_mechanism,
    mask_generalization_metrics,
)
from experiments.article_1.training import initial_state, train_student
from oracle_distillation.utils import resolve_device


PRESETS = {
    "central": {"sizes": (10000,), "offsets": (0.0,), "methods": CENTRAL_METHODS},
    "oracle": {"sizes": (10000,), "offsets": (0.0,), "methods": ORACLE_METHODS},
    "curve": {"sizes": PROXY_SIZES, "offsets": (0.0,), "methods": CURVE_METHODS},
    "threshold": {"sizes": (10000,), "offsets": THRESHOLD_OFFSETS, "methods": ("expert_v2",)},
    "baseline": {"sizes": (10000,), "offsets": (0.0,), "methods": FINAL_BASELINE_METHODS},
    "mechanism": {"sizes": (10000,), "offsets": (0.0,), "methods": MECHANISM_CONTROL_METHODS},
    "proxy-efficiency": {
        "sizes": PROXY_SIZES, "offsets": (0.0,), "methods": PROXY_EFFICIENCY_BASELINE_METHODS,
    },
    "proxy-efficiency-energy": {
        "sizes": PROXY_SIZES, "offsets": (0.0,), "methods": PROXY_EFFICIENCY_ENERGY_METHODS,
    },
}


def _hash(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _load_source(root: Path, dataset: str, seed: int, regime: str) -> tuple[dict, dict[str, np.ndarray]]:
    base = source_dir(root, dataset, seed, regime)
    npz_path, json_path = base / "teacher_source.npz", base / "source.json"
    if not npz_path.is_file() or not json_path.is_file():
        raise FileNotFoundError(
            f"clean teacher source missing for {dataset}/seed_{seed}/{regime}; run prepare first"
        )
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    if metadata.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"incompatible teacher source protocol: {json_path}")
    if metadata.get("teacher_source_sha256") != sha256_file(npz_path):
        raise ValueError(f"teacher source hash mismatch: {npz_path}")
    expected_identity = (dataset, int(seed), regime)
    observed_identity = (metadata.get("dataset"), int(metadata.get("seed", -1)), metadata.get("regime"))
    if observed_identity != expected_identity:
        raise ValueError(f"teacher source identity mismatch: {observed_identity}!={expected_identity}")
    with np.load(npz_path, allow_pickle=False) as payload:
        required = {
            "proxy_idx", "proxy_labels", "proxy_logits", "proxy_order",
            "holdout_accuracy", "holdout_counts", "test_accuracy", "test_counts",
        }
        if not required.issubset(payload.files):
            raise ValueError(f"teacher source lacks {sorted(required - set(payload.files))}")
        arrays = {key: np.asarray(payload[key]) for key in required}
    return metadata, arrays


def _condition_common(
    *, dataset: str, seed: int, regime: str, threshold: float, size: int,
    temperature: float, source_meta: dict, source_arrays: dict[str, np.ndarray],
    authority: np.ndarray, positions: np.ndarray, output_root: Path,
) -> dict:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": dataset,
        "seed": int(seed),
        "regime": regime,
        **regime_metadata(regime),
        "threshold": float(threshold),
        "proxy_size": int(size),
        "temperature": float(temperature),
        "teacher_source": str(source_dir(output_root, dataset, seed, regime) / "teacher_source.npz"),
        "teacher_source_sha256": source_meta["teacher_source_sha256"],
        "teacher_fingerprint": source_meta["teacher_fingerprint"],
        "proxy_subset_sha256": sha256_array(source_arrays["proxy_idx"][positions]),
        **mask_generalization_metrics(
            authority,
            source_arrays["holdout_accuracy"], source_arrays["holdout_counts"],
            source_arrays["test_accuracy"], source_arrays["test_counts"], threshold,
        ),
    }


def targets_condition(
    *, dataset: str, seed: int, regime: str, threshold: float, size: int,
    temperature: float, output_root: Path, force: bool, methods: tuple[str, ...] | None = None,
) -> Path:
    source_meta, arrays = _load_source(output_root, dataset, seed, regime)
    if size > len(arrays["proxy_idx"]):
        raise ValueError(f"proxy size {size} exceeds source size {len(arrays['proxy_idx'])}")
    positions = arrays["proxy_order"][:size].astype(np.int64)
    indices = arrays["proxy_idx"][positions].astype(np.int64)
    labels = arrays["proxy_labels"][positions].astype(np.int64)
    logits = arrays["proxy_logits"][positions].astype(np.float32)
    authority = authority_from_holdout(arrays["holdout_accuracy"], arrays["holdout_counts"], threshold)
    destination = run_dir(output_root, dataset, seed, regime, threshold, size)
    target_path = destination / "targets.npz"
    metrics_path = destination / "target_metrics.json"
    requested_methods = tuple(methods or ALL_METHODS)
    if any(method not in ALL_METHODS for method in requested_methods):
        raise ValueError(f"unsupported target methods: {requested_methods}")

    # protocol_v1's completed corpus predates the clean Energy target.  Never
    # rewrite its immutable targets.npz: add only absent methods in a sidecar
    # with its own provenance and hashes.
    if target_path.is_file() and metrics_path.is_file() and not force:
        with np.load(target_path, allow_pickle=False) as payload:
            base_methods = {name.removeprefix("target__") for name in payload.files if name.startswith("target__")}
        missing = tuple(method for method in requested_methods if method not in base_methods)
        if not missing:
            return destination
        extension_tag = "__".join(missing)
        extension_path = destination / f"targets_extension__{extension_tag}.npz"
        extension_metrics_path = destination / f"target_metrics_extension__{extension_tag}.json"
        if extension_path.is_file() and extension_metrics_path.is_file():
            return destination
        observations, mechanism = expertise_mechanism(logits, labels, authority, temperature=temperature)
        common = _condition_common(
            dataset=dataset, seed=seed, regime=regime, threshold=threshold, size=size,
            temperature=temperature, source_meta=source_meta, source_arrays=arrays,
            authority=authority, positions=positions, output_root=output_root,
        )
        results = {
            method: build_target(logits, labels, authority, method=method, temperature=temperature)
            for method in missing
        }
        method_metrics = [{**common, **mechanism, **result.metrics} for result in results.values()]
        write_npz(
            extension_path, force=False, proxy_idx=indices, labels=labels, authority=authority,
            **{f"target__{method}": result.probabilities for method, result in results.items()},
        )
        write_json(extension_metrics_path, {
            "condition": common,
            "extension_of": str(target_path),
            "extension_of_sha256": sha256_file(target_path),
            "methods": method_metrics,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }, force=False)
        return destination

    observations, mechanism = expertise_mechanism(
        logits, labels, authority, temperature=temperature,
    )
    common = _condition_common(
        dataset=dataset, seed=seed, regime=regime, threshold=threshold, size=size,
        temperature=temperature, source_meta=source_meta, source_arrays=arrays,
        authority=authority, positions=positions, output_root=output_root,
    )
    results = {
        method: build_target(logits, labels, authority, method=method, temperature=temperature)
        for method in requested_methods
    }
    if {"feddf", "support_only"}.issubset(results) and not np.array_equal(results["feddf"].selected_teachers, results["support_only"].selected_teachers):
        raise AssertionError("support-only changed teacher selection")
    if {"expert_full", "expert_v2"}.issubset(results) and not np.array_equal(results["expert_full"].selected_teachers, results["expert_v2"].selected_teachers):
        raise AssertionError("EXPERT-v2 changed teacher selection")
    oracle_hits = results["oracle_v2"].selected_teachers.any(axis=1) if "oracle_v2" in results else None
    supervised = results["supervised_matched"].probabilities if "supervised_matched" in results else None
    if oracle_hits is not None and supervised is not None and not np.array_equal(results["oracle_v2"].probabilities[oracle_hits], supervised[oracle_hits]):
        raise AssertionError("ORACLE-v2 is not one-hot when a correct teacher exists")

    method_metrics = []
    for method, result in results.items():
        record = {**common, **mechanism, **result.metrics}
        if method == "oracle_v2" and supervised is not None:
            difference = np.abs(result.probabilities - supervised).sum(axis=1)
            record.update({
                "mean_l1_vs_supervised_onehot": float(difference.mean()),
                "rows_different_from_supervised": int((difference > 1e-7).sum()),
            })
        method_metrics.append(record)
    write_npz(
        target_path, force=force,
        proxy_idx=indices,
        labels=labels,
        authority=authority,
        **{f"target__{method}": result.probabilities for method, result in results.items()},
    )
    write_npz(destination / "mechanism.npz", force=force, **observations)
    write_json(metrics_path, {
        "condition": common,
        "mechanism": mechanism,
        "methods": method_metrics,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }, force=force)
    return destination


def _load_targets_for_methods(
    destination: Path, methods: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, dict], dict[str, Path]]:
    """Load requested targets, including immutable extension sidecars."""
    target_files = [destination / "targets.npz", *sorted(destination.glob("targets_extension__*.npz"))]
    metric_files = [destination / "target_metrics.json", *sorted(destination.glob("target_metrics_extension__*.json"))]
    if not target_files[0].is_file() or not metric_files[0].is_file():
        raise FileNotFoundError(f"targets missing; run targets stage first: {destination}")
    arrays: dict[str, np.ndarray] = {}
    target_sources: dict[str, Path] = {}
    indices = labels = None
    for path in target_files:
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as payload:
            current_indices = np.asarray(payload["proxy_idx"], dtype=np.int64)
            current_labels = np.asarray(payload["labels"], dtype=np.int64)
            if indices is None:
                indices, labels = current_indices, current_labels
            elif not np.array_equal(indices, current_indices) or not np.array_equal(labels, current_labels):
                raise ValueError(f"target extension proxy identity mismatch: {path}")
            for method in methods:
                key = f"target__{method}"
                if key in payload:
                    if method in arrays:
                        raise ValueError(f"duplicate target method {method!r} in {path}")
                    arrays[method] = np.asarray(payload[key], dtype=np.float32)
                    target_sources[method] = path
    metrics_by_method: dict[str, dict] = {}
    for path in metric_files:
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["methods"]:
            if row["method"] in methods:
                metrics_by_method[row["method"]] = row
    missing = set(methods) - set(arrays)
    if missing or set(methods) - set(metrics_by_method):
        raise FileNotFoundError(f"requested target methods missing from {destination}: {sorted(missing)}")
    assert indices is not None and labels is not None
    return indices, labels, arrays, metrics_by_method, target_sources


def students_condition(
    *, dataset: str, seed: int, regime: str, threshold: float, size: int,
    methods: tuple[str, ...], temperature: float, output_root: Path, data_dir: Path,
    device: torch.device, force: bool,
) -> Path:
    destination = run_dir(output_root, dataset, seed, regime, threshold, size)
    indices, labels, targets, metrics_by_method, target_sources = _load_targets_for_methods(destination, methods)

    config = get_dataset_config(dataset)
    recipe = StudentRecipe(arch=config.arch, temperature=temperature)
    clean_dataset = config.load_train_eval_dataset(data_dir)
    augmented_dataset = config.load_train_dataset(data_dir)
    test_loader = config.make_test_loader(data_dir, batch_size=256, num_workers=2)
    init, init_hash = initial_state(recipe.arch, config.num_classes, seed)
    paired_id = _hash({
        "dataset": dataset, "seed": seed, "regime": regime, "threshold": threshold,
        "size": size, "init": init_hash, "order_seed": seed,
        "epochs": recipe.kd_epochs, "batch_size": recipe.batch_size,
    })

    for method in methods:
        if method not in ALL_METHODS:
            raise ValueError(f"unsupported method {method!r}")
        output = destination / "students" / method / "metrics.json"
        if output.is_file() and not force:
            continue
        standard = method == "supervised_standard"
        metrics = train_student(
            arch=recipe.arch, classes=config.num_classes,
            init_state=init, init_hash=init_hash,
            train_dataset=augmented_dataset if standard else clean_dataset,
            test_loader=test_loader, proxy_indices=indices, labels=labels,
            target_probabilities=targets[method], hard_labels=standard,
            temperature=temperature,
            epochs=recipe.supervised_epochs if standard else recipe.kd_epochs,
            batch_size=recipe.batch_size, learning_rate=recipe.learning_rate,
            weight_decay=recipe.weight_decay, init_seed=seed, order_seed=seed,
            device=device, cosine=standard,
        )
        write_json(output, {
            **metrics_by_method[method], **metrics,
            "paired_run_id": None if standard else paired_id,
            "training_recipe": "supervised_standard" if standard else "paired_kd",
            "target_sha256": sha256_array(targets[method]),
            "target_artifact": str(target_sources[method]),
            "target_artifact_sha256": sha256_file(target_sources[method]),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }, force=force)

    paired_paths = [destination / "students" / method / "metrics.json"
                    for method in methods if method != "supervised_standard"]
    paired_rows = [json.loads(path.read_text()) for path in paired_paths if path.is_file()]
    if paired_rows:
        for field in ("student_init_sha256", "train_order_sha256", "total_updates", "paired_run_id"):
            if len({row[field] for row in paired_rows}) != 1:
                raise AssertionError(f"paired Article-1 methods disagree at {field}")
        pairing_path = destination / "pairing.json"
        if force or not pairing_path.exists():
            write_json(pairing_path, {
                "methods": [row["method"] for row in paired_rows],
                "student_init_sha256": paired_rows[0]["student_init_sha256"],
                "train_order_sha256": paired_rows[0]["train_order_sha256"],
                "total_updates": paired_rows[0]["total_updates"],
                "paired_run_id": paired_rows[0]["paired_run_id"],
            }, force=force)

    oracle_path = destination / "students" / "oracle_v2" / "metrics.json"
    supervised_path = destination / "students" / "supervised_matched" / "metrics.json"
    if oracle_path.is_file() and supervised_path.is_file():
        oracle = json.loads(oracle_path.read_text())
        supervised_metrics = json.loads(supervised_path.read_text())
        for field in ("student_init_sha256", "train_order_sha256", "total_updates", "paired_run_id"):
            if oracle[field] != supervised_metrics[field]:
                raise AssertionError(f"ORACLE-v2/supervised pairing failed at {field}")
        zero_fallback = int(oracle["oracle_no_correct_count"]) == 0
        same_student = oracle["student_final_sha256"] == supervised_metrics["student_final_sha256"]
        if zero_fallback and not same_student:
            raise AssertionError("identical one-hot targets produced different paired students")
        sanity_path = destination / "oracle_sanity.json"
        if force or not sanity_path.exists():
            write_json(sanity_path, {
                "oracle_no_correct_count": oracle["oracle_no_correct_count"],
                "oracle_no_correct_rate": oracle["oracle_no_correct_rate"],
                "target_rows_different": oracle.get("rows_different_from_supervised"),
                "student_accuracy_delta_oracle_v2_minus_supervised": (
                    oracle["student_test_accuracy"] - supervised_metrics["student_test_accuracy"]
                ),
                "identical_final_student": same_student,
            }, force=force)
    return destination


def _selection(args: argparse.Namespace) -> tuple[tuple[int, ...], tuple[float, ...], tuple[str, ...]]:
    preset = PRESETS[args.preset]
    sizes = tuple(args.sizes) if args.sizes else tuple(preset["sizes"])
    offsets = tuple(args.threshold_offsets) if args.threshold_offsets else tuple(preset["offsets"])
    methods = tuple(args.methods) if args.methods else tuple(preset["methods"])
    if any(size not in PROXY_SIZES for size in sizes):
        raise ValueError(f"sizes must be selected from {PROXY_SIZES}")
    return sizes, offsets, methods


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("targets", "students", "all"), required=True)
    parser.add_argument("--preset", choices=tuple(PRESETS), required=True)
    parser.add_argument("--output-root", type=Path, default=Path(OUTPUT_ROOT))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--regimes", nargs="+", choices=REGIMES, default=list(REGIMES))
    parser.add_argument("--sizes", nargs="+", type=int)
    parser.add_argument("--threshold-offsets", nargs="+", type=float)
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS)
    parser.add_argument("--temperature", type=float, default=8.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sizes, offsets, methods = _selection(args)
    cells = [
        (dataset, seed, regime, threshold_for(dataset, offset), size)
        for dataset in args.datasets for seed in args.seeds for regime in args.regimes
        for offset in offsets for size in sizes
    ]
    if args.dry_run:
        print(f"conditions={len(cells)} student_runs={len(cells) * len(methods)} methods={methods}")
        for cell in cells:
            print(cell)
        return
    device = resolve_device(args.device) if args.stage in {"students", "all"} else torch.device("cpu")
    for dataset, seed, regime, threshold, size in cells:
        if args.stage in {"targets", "all"}:
            path = targets_condition(
                dataset=dataset, seed=seed, regime=regime, threshold=threshold, size=size,
                temperature=args.temperature, output_root=args.output_root, force=args.force, methods=methods,
            )
            print(f"[TARGETS] {path}")
        if args.stage in {"students", "all"}:
            path = students_condition(
                dataset=dataset, seed=seed, regime=regime, threshold=threshold, size=size,
                methods=methods, temperature=args.temperature, output_root=args.output_root,
                data_dir=args.data_dir, device=device, force=args.force,
            )
            print(f"[STUDENTS] {path}")


if __name__ == "__main__":
    main()
