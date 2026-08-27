#!/usr/bin/env python3
"""Audit how strongly the retained Article-1 teachers can be documented.

This audit deliberately distinguishes two claims:

1. retained-artifact integrity: the migrated splits, logits and checkpoint
   hashes are internally consistent;
2. historical-training provenance: the original checkpoint, teacher manifest
   and run configuration still exist and can prove how that checkpoint was
   selected.

The first claim cannot substitute for the second.  In strict mode the command
returns a non-zero status whenever teacher regeneration is required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.article_1 import OUTPUT_ROOT, PROTOCOL_VERSION
from experiments.article_1.io import sha256_file, write_json


REQUIRED_ARRAYS = {
    "proxy_idx", "proxy_labels", "proxy_logits", "proxy_order",
    "holdout_accuracy", "holdout_counts", "test_accuracy", "test_counts",
    "train_sizes", "holdout_sizes", "test_sizes",
}


def _fingerprint(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("ascii"))
    return digest.hexdigest()


def _resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def audit_source(source_json: Path, *, repo_root: Path) -> dict:
    source_json = Path(source_json)
    source_dir = source_json.parent
    meta = json.loads(source_json.read_text(encoding="utf-8"))
    source_npz = source_dir / "teacher_source.npz"
    errors: list[str] = []

    if meta.get("protocol_version") != PROTOCOL_VERSION:
        errors.append("unexpected_protocol_version")
    if not source_npz.is_file():
        errors.append("missing_teacher_source_npz")
        arrays = {}
    else:
        if sha256_file(source_npz) != meta.get("teacher_source_sha256"):
            errors.append("teacher_source_sha256_mismatch")
        with np.load(source_npz, allow_pickle=False) as payload:
            missing_arrays = sorted(REQUIRED_ARRAYS - set(payload.files))
            if missing_arrays:
                errors.append(f"missing_arrays:{','.join(missing_arrays)}")
            arrays = {key: np.asarray(payload[key]) for key in payload.files}

    hashes = [str(value) for value in meta.get("checkpoint_sha256", [])]
    if len(hashes) != int(meta.get("num_teachers", -1)):
        errors.append("checkpoint_hash_count_mismatch")
    if hashes and _fingerprint(hashes) != meta.get("teacher_fingerprint"):
        errors.append("teacher_fingerprint_mismatch")

    split_files = sorted((source_dir / "splits").glob("client_*.npz"))
    expected_teachers = int(meta.get("num_teachers", -1))
    if len(split_files) != expected_teachers:
        errors.append("client_split_count_mismatch")
    allocated: list[np.ndarray] = []
    split_sizes = {"train_idx": [], "holdout_idx": [], "test_idx": []}
    for split_file in split_files:
        with np.load(split_file, allow_pickle=False) as payload:
            if not {"train_idx", "holdout_idx", "test_idx"}.issubset(payload.files):
                errors.append(f"incomplete_split:{split_file.name}")
                continue
            values = {name: np.asarray(payload[name], dtype=np.int64)
                      for name in split_sizes}
        for name, value in values.items():
            split_sizes[name].append(len(value))
            if len(np.unique(value)) != len(value):
                errors.append(f"duplicate_indices:{split_file.name}:{name}")
        names = list(values)
        if any(np.intersect1d(values[names[i]], values[names[j]]).size
               for i in range(3) for j in range(i + 1, 3)):
            errors.append(f"within_client_overlap:{split_file.name}")
        allocated.extend(values.values())

    if allocated:
        joined = np.concatenate(allocated)
        if len(np.unique(joined)) != len(joined):
            errors.append("between_client_or_role_overlap")
        if "proxy_idx" in arrays and np.intersect1d(joined, arrays["proxy_idx"]).size:
            errors.append("proxy_client_overlap")
    for role, array_name in (("train_idx", "train_sizes"),
                             ("holdout_idx", "holdout_sizes"),
                             ("test_idx", "test_sizes")):
        if array_name in arrays and not np.array_equal(
            np.asarray(split_sizes[role], dtype=np.int64), arrays[array_name]
        ):
            errors.append(f"{array_name}_mismatch")

    checkpoint_paths = [_resolve(repo_root, value)
                        for value in meta.get("checkpoint_paths", [])]
    checkpoints_present = len(checkpoint_paths) == expected_teachers and all(
        path.is_file() for path in checkpoint_paths
    )
    checkpoint_hashes_verified = False
    if checkpoints_present:
        checkpoint_hashes_verified = all(
            sha256_file(path) == expected
            for path, expected in zip(checkpoint_paths, hashes)
        )
        if not checkpoint_hashes_verified:
            errors.append("retained_checkpoint_sha256_mismatch")

    legacy_run = _resolve(repo_root, meta.get("legacy_run", ""))
    manifest = legacy_run / "teachers_manifest.json"
    run_config = legacy_run / "run_config.yaml"
    execution_evidence_present = (
        checkpoints_present and checkpoint_hashes_verified
        and manifest.is_file() and run_config.is_file()
    )
    artifact_integrity_verified = not errors
    strict_reusable = artifact_integrity_verified and execution_evidence_present
    return {
        "dataset": meta.get("dataset"),
        "seed": meta.get("seed"),
        "regime": meta.get("regime"),
        "teachers": expected_teachers,
        "artifact_integrity_verified": artifact_integrity_verified,
        "split_and_proxy_separation_verified": not any(
            "overlap" in error or "split" in error or "sizes_mismatch" in error
            for error in errors
        ),
        "checkpoint_hash_chain_recorded": len(hashes) == expected_teachers,
        "original_checkpoints_present": checkpoints_present,
        "original_checkpoint_hashes_verified": checkpoint_hashes_verified,
        "teacher_manifest_present": manifest.is_file(),
        "run_config_present": run_config.is_file(),
        "historical_training_execution_verified": execution_evidence_present,
        "strictly_reusable_without_retraining": strict_reusable,
        "status": "verified_reusable" if strict_reusable else (
            "invalid_artifact_chain" if errors
            else "requires_regeneration_for_strict_provenance"
        ),
        "errors": errors,
    }


def build_report(output_root: Path, *, repo_root: Path) -> dict:
    source_files = sorted((Path(output_root) / "sources").glob("**/source.json"))
    records = [audit_source(path, repo_root=repo_root) for path in source_files]
    strict = all(row["strictly_reusable_without_retraining"] for row in records)
    integrity = all(row["artifact_integrity_verified"] for row in records)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "audit_scope": "retained_teacher_sources_and_historical_training_evidence",
        "conditions": len(records),
        "teachers": sum(int(row["teachers"]) for row in records),
        "artifact_integrity_verified_conditions": sum(
            bool(row["artifact_integrity_verified"]) for row in records
        ),
        "historical_training_verified_conditions": sum(
            bool(row["historical_training_execution_verified"]) for row in records
        ),
        "strictly_reusable_conditions": sum(
            bool(row["strictly_reusable_without_retraining"]) for row in records
        ),
        "artifact_chain_status": "verified" if integrity else "invalid",
        "historical_training_status": "verified" if strict else "unverifiable",
        "decision": "reuse" if strict else "regenerate_teachers",
        "reason": (
            "Every original checkpoint, teacher manifest and run configuration is present "
            "and hash-linked to the migrated source."
            if strict else
            "Migrated logits/splits are internally valid, but the deleted original "
            "checkpoints/manifests/configs cannot prove the exact historical training run."
        ),
        "records": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path(OUTPUT_ROOT))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--report", type=Path)
    parser.add_argument("--strict", action="store_true",
                        help="exit 2 unless every historical training run is verifiable")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args.output_root, repo_root=args.repo_root.resolve())
    destination = args.report or args.output_root / "summary" / "teacher_provenance_audit.json"
    write_json(destination, report, force=True)
    print(
        f"[PROVENANCE] conditions={report['conditions']} teachers={report['teachers']} "
        f"artifact_chain={report['artifact_chain_status']} "
        f"historical_training={report['historical_training_status']} "
        f"decision={report['decision']} report={destination}"
    )
    if args.strict and report["decision"] != "reuse":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
