"""Build the non-destructive OUTPUTS experiment manifest and indexes."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from oracle_distillation.experiment_paths import (
    EXPERIMENTS,
    SEED42_ANALYSIS_SOURCES,
)


INDEX_COLUMNS = (
    "experiment_id", "phase", "seed", "dataset", "regime", "method_or_arm",
    "artifact_type", "path", "source_script", "status", "is_canonical",
    "parent_artifacts", "selection_filter", "link_target", "checksum", "created_at",
)
CHECKSUM_LIMIT_BYTES = 64 * 1024 * 1024


def _sha256(path: Path) -> str:
    if path.is_symlink():
        return ""
    if not path.is_file() or path.stat().st_size > CHECKSUM_LIMIT_BYTES:
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _created_at(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _artifact_type(path: Path) -> str:
    if path.is_symlink():
        return "symlink_directory" if path.is_dir() else "symlink_file"
    if path.is_dir():
        return "directory"
    if path.name in {"student.pt", "checkpoint.pt"} or path.suffix == ".pt":
        return "checkpoint"
    if path.name == "proxy_analysis.npz":
        return "proxy_analysis"
    if path.suffix == ".npz":
        return "array_cache"
    if path.suffix == ".csv":
        return "csv"
    if path.suffix in {".json", ".yaml", ".yml"}:
        return "provenance"
    if path.suffix == ".md":
        return "documentation"
    return path.suffix.lstrip(".") or "file"


def _metadata_from_parts(path: Path) -> tuple[str, str, str]:
    dataset = next((p for p in path.parts if p in {"mnist", "fmnist", "cifar", "cifar10"}), "")
    if dataset == "cifar10":
        dataset = "cifar"
    regime = next((p for p in path.parts if p in {
        "single", "multi", "iid", "alpha0p1", "alpha0p5", "alpha1p0",
    } or p.startswith(("alpha0p1__", "alpha0p5__", "alpha1p0__", "single__", "multi__", "iid__"))), "")
    method = next((p for p in path.parts if p in {
        "feddf", "energy", "confidence", "consensus", "expert", "oracle",
        "global", "proxy_plain", "proxy", "oracle_LN_matched",
    }), "")
    return dataset, regime, method


def _entry(
    *, experiment_id: str, seed: int, path: Path, root: Path,
    status: str, canonical: bool, selection_filter: str = "",
) -> dict[str, Any]:
    definition = EXPERIMENTS[experiment_id]
    dataset, regime, method = _metadata_from_parts(path)
    return {
        "experiment_id": experiment_id,
        "phase": definition.phase_dir,
        "seed": seed,
        "dataset": dataset,
        "regime": regime,
        "method_or_arm": method,
        "artifact_type": _artifact_type(path),
        "path": str(path.relative_to(root)).replace("\\", "/"),
        "source_script": definition.source_script,
        "status": status,
        "is_canonical": canonical,
        "parent_artifacts": list(definition.parent_experiments),
        "selection_filter": selection_filter,
        "link_target": os.readlink(path) if path.is_symlink() else "",
        "checksum": _sha256(path),
        "created_at": _created_at(path),
    }


def _artifact_tree_status(path: Path, seed_dir: Path) -> str | None:
    """Resolve explicit invalid/migration markers inherited by an artifact."""
    candidate = path.parent if path.is_file() or path.is_symlink() else path
    seed_dir = seed_dir.resolve()
    needs_migration = False
    while True:
        if (candidate / "INVALID.json").is_file():
            return "invalid"
        if (candidate / "NEEDS_MIGRATION.json").is_file():
            needs_migration = True
        if candidate.resolve() == seed_dir or candidate.parent == candidate:
            return "needs_migration" if needs_migration else None
        candidate = candidate.parent


def _is_invalidated(path: Path, seed_dir: Path) -> bool:
    """Backward-compatible predicate used by tests and external validators."""
    return _artifact_tree_status(path, seed_dir) == "invalid"


def build_experiment_manifest(repo_root: Path) -> tuple[dict[str, Any], list[str]]:
    repo_root = Path(repo_root).resolve()
    outputs = repo_root / "OUTPUTS"
    entries: list[dict[str, Any]] = []
    errors: list[str] = []

    for experiment_id, sources in SEED42_ANALYSIS_SOURCES.items():
        for source in sources:
            path = repo_root / source
            if not path.exists():
                errors.append(f"seed-42 analysis source missing: {source}")
                continue
            selection_filter = {
                "phase_a_methods": "datasets=mnist|fmnist|cifar;methods=feddf|energy|confidence|consensus|expert|oracle",
                "phase_a12_auto_v2": "N=100|250|500|1000|2500|5000;arm=auto_v2",
                "phase_a12_oracle_ln_matched": "arm=oracle_LN_matched",
                "headroom_core": "base=expert|feddf|energy",
                "phase_b_classmask_causal": "global_source=feddf|energy;arm=global|proxy_plain|proxy",
                "phase_b_six_arm_exploratory": "global_source=expert|feddf|energy;arm=global|proxy_plain|proxy|local|local_lwf|mix",
                "exp5_a12_classmask": "N=100;arm=global|proxy_plain|proxy",
            }.get(experiment_id, "")
            entries.append(_entry(
                experiment_id=experiment_id, seed=42, path=path, root=repo_root,
                status="analysis_only", canonical=True,
                selection_filter=selection_filter,
            ))

    experiments_root = outputs / "experiments"
    # A phase may contain more than one stable experiment id (Phase A1/2 does).
    # Walk every phase once and classify by its explicit artifact namespace so a
    # file can never be emitted twice under two experiment ids.
    phase_to_default: dict[str, str] = {}
    for experiment_id, definition in EXPERIMENTS.items():
        phase_to_default.setdefault(definition.phase_dir, experiment_id)
    for phase, default_experiment in phase_to_default.items():
        phase_root = experiments_root / phase
        if not phase_root.exists():
            continue
        for seed_dir in sorted(phase_root.glob("seed_*")):
            try:
                seed = int(seed_dir.name.split("_", 1)[1])
            except (IndexError, ValueError):
                errors.append(f"invalid seed directory: {seed_dir.relative_to(repo_root)}")
                continue
            for path in sorted(
                p for p in seed_dir.rglob("*")
                if p.is_file() or (p.is_symlink() and p.exists())
            ):
                relative_parts = path.relative_to(seed_dir).parts
                experiment_id = default_experiment
                if phase == "phase_a12" and relative_parts:
                    if relative_parts[0] == "oracle_ln_matched":
                        experiment_id = "phase_a12_oracle_ln_matched"
                    else:
                        experiment_id = "phase_a12_auto_v2"
                elif phase == "phase_b" and relative_parts:
                    is_exploratory_checkpoint = (
                        relative_parts[0] == "checkpoints" and len(relative_parts) >= 4 and
                        (relative_parts[1] == "expert" or
                         relative_parts[3] in {"local", "local_lwf", "mix"})
                    )
                    if (relative_parts[0] == "checkpoints_six_arm" or
                            path.name == "transfer_set_ablation_all_arms.csv" or
                            is_exploratory_checkpoint):
                        experiment_id = "phase_b_six_arm_exploratory"
                    else:
                        experiment_id = "phase_b_classmask_causal"
                entries.append(_entry(
                    experiment_id=experiment_id, seed=seed, path=path, root=repo_root,
                    status=(
                        _artifact_tree_status(path, seed_dir)
                        or ("complete" if path.name in {"metrics.json", "student.pt"} else "partial")
                    ),
                    canonical=False,
                ))

    keys = [(e["experiment_id"], e["seed"], e["path"]) for e in entries]
    if len(keys) != len(set(keys)):
        errors.append("experiment manifest contains duplicate (experiment_id, seed, path) keys")
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entries": sorted(entries, key=lambda e: (e["experiment_id"], e["seed"], e["path"])),
    }
    return manifest, errors


def write_experiment_indexes(manifest: dict[str, Any], outputs_root: Path) -> None:
    outputs_root.mkdir(parents=True, exist_ok=True)
    (outputs_root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (outputs_root / "INDEX.json").write_text(json.dumps(manifest["entries"], indent=2, sort_keys=True) + "\n")
    with (outputs_root / "INDEX.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        for entry in manifest["entries"]:
            row = dict(entry)
            row["parent_artifacts"] = "|".join(row["parent_artifacts"])
            writer.writerow({c: row.get(c, "") for c in INDEX_COLUMNS})


def validate_manifest_paths(manifest: dict[str, Any], repo_root: Path) -> list[str]:
    errors: list[str] = []
    for entry in manifest.get("entries", []):
        path = Path(repo_root) / entry["path"]
        if not path.exists():
            errors.append(f"manifest path missing: {entry['path']}")
        checksum = entry.get("checksum") or ""
        if checksum and _sha256(path) != checksum:
            errors.append(f"checksum mismatch: {entry['path']}")
    return errors
