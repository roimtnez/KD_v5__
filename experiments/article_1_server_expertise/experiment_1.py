#!/usr/bin/env python3
"""Article 1, Experiment 1: diagnose EXPERT mass outside teacher expertise.

This is intentionally an analysis-only runner.  It reads the saved proxy
teacher logits and authority mask, constructs the current probability-space
``expert_full`` target, and never trains a student.

The normal mode accepts only masks reconstructed from per-class *holdout*
accuracy.  The historical Study-I artifacts used local-test accuracy instead;
they can be inspected with ``--allow-legacy-exploratory`` but are marked as
leakage-affected in every output.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from oracle_distillation.targets.expertise import (
    EPS,
    build_server_expertise_target,
    target_quality_metrics,
    temperature_softmax,
)
from experiments.article_1_server_expertise.artifacts import sha256_file
from experiments.article_1_server_expertise.storage import canonical_hash, save_npz_once, write_json_once


# These are the registered dataset-specific thresholds, kept locally so this
# read-only diagnostic does not import torchvision merely to inspect metadata.
THRESHOLD_BY_DATASET_LABEL = {"cifar10": 0.7, "mnist": 0.9, "fmnist": 0.8, "cinic": 0.7, "cifar100": 0.4}
HOLDOUT_MASK_KEY = "holdout_acc_per_class"
LEGACY_MASK_KEY = "local_test_acc_per_class"


@dataclass(frozen=True)
class SourceArtifact:
    path: Path
    run_dir: Path
    partition_dir: Path
    dataset: str
    seed: int
    regime: str


def _seed_from_path(path: Path) -> int:
    for part in path.parts:
        if part.startswith("seed_") and part[5:].isdigit():
            return int(part[5:])
    raise ValueError(f"cannot identify seed from {path}")


def discover_sources(study_root: Path) -> list[SourceArtifact]:
    """Discover the stable Study-I proxy-cache interface without hard-coding seeds."""
    found: list[SourceArtifact] = []
    for path in sorted(Path(study_root).glob("seed_*/raw_work/runs/*/K*/*/proxy_analysis/proxy_analysis.npz")):
        run_dir = path.parent.parent
        # .../raw_work/runs/<dataset>/K10/<regime>/proxy_analysis/file.npz
        dataset = path.parents[3].name
        partition_dir = path.parents[5] / "partitions" / dataset / path.parents[2].name / run_dir.name
        found.append(SourceArtifact(
            path=path, run_dir=run_dir, partition_dir=partition_dir, dataset=dataset,
            seed=_seed_from_path(path), regime=run_dir.name.split("__", 1)[0],
        ))
    return found


def _load_cache(path: Path) -> dict[str, np.ndarray]:
    required = {"proxy_idx", "y_true_proxy", "teacher_logits_cache", "teacher_knows_class_mask"}
    with np.load(path, allow_pickle=False) as archive:
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{path} lacks {sorted(missing)}")
        data = {key: np.asarray(archive[key]) for key in required}
    logits, labels, mask = data["teacher_logits_cache"], data["y_true_proxy"], data["teacher_knows_class_mask"]
    if logits.ndim != 3 or labels.shape != (len(logits),) or mask.shape != logits.shape[1:]:
        raise ValueError(f"invalid cache dimensions in {path}")
    if not np.isfinite(logits).all() or not np.isin(mask, (0, 1)).all():
        raise ValueError(f"non-finite logits or non-binary mask in {path}")
    return data


def _rebuild_mask(manifest: list[dict], key: str, threshold: float, shape: tuple[int, int]) -> np.ndarray | None:
    rebuilt = np.zeros(shape, dtype=np.uint8)
    seen: set[int] = set()
    for entry in manifest:
        cid = int(entry.get("cid", -1))
        values = (entry.get("crea") or {}).get(key)
        if not (0 <= cid < shape[0]) or not isinstance(values, list):
            return None
        for cls, value in enumerate(values[: shape[1]]):
            if value is not None and float(value) >= threshold:
                rebuilt[cid, cls] = 1
        seen.add(cid)
    return rebuilt if seen == set(range(shape[0])) else None


def _holdout_is_independent(partition_dir: Path, n_teachers: int) -> tuple[bool, str]:
    """Confirm the intended train/holdout/local-test separation from client files."""
    client_dir = Path(partition_dir) / "clients"
    for cid in range(n_teachers):
        path = client_dir / f"c{cid:03d}.npz"
        if not path.is_file():
            return False, f"missing client partition {path}"
        with np.load(path, allow_pickle=False) as split:
            required = {"train_idx", "holdout_idx", "local_test_idx"}
            if not required.issubset(split.files):
                return False, f"{path} lacks a three-way train/holdout/local-test split"
            train = np.asarray(split["train_idx"], dtype=np.int64)
            holdout = np.asarray(split["holdout_idx"], dtype=np.int64)
            local_test = np.asarray(split["local_test_idx"], dtype=np.int64)
        if len(holdout) == 0:
            return False, f"{path} has an empty holdout"
        if (np.intersect1d(train, holdout).size or np.intersect1d(train, local_test).size
                or np.intersect1d(holdout, local_test).size):
            return False, f"{path} has overlapping train/holdout/local-test indices"
    return True, "verified disjoint train/holdout/local-test client splits"


def _prepared_authority_path(authority_root: Path | None, source_sha256: str) -> Path | None:
    if authority_root is None:
        return None
    path = Path(authority_root) / "holdout_authority" / source_sha256 / "authority.npz"
    return path if path.is_file() else None


def audit_source(source: SourceArtifact, authority_root: Path | None = None) -> dict[str, Any]:
    """Audit mask provenance and return an analysis-ready provenance record."""
    cache = _load_cache(source.path)
    mask = cache["teacher_knows_class_mask"].astype(np.uint8)
    manifest_path = source.run_dir / "teachers_manifest.json"
    record: dict[str, Any] = {
        "dataset": source.dataset, "seed": source.seed, "regime": source.regime,
        "source_proxy_analysis": str(source.path), "source_proxy_sha256": sha256_file(source.path),
        "teachers_manifest": str(manifest_path), "partition_dir": str(source.partition_dir),
        "n_proxy": int(len(cache["proxy_idx"])), "n_teachers": int(mask.shape[0]), "n_classes": int(mask.shape[1]),
        "threshold": float(THRESHOLD_BY_DATASET_LABEL.get(source.dataset, 0.7)),
        "mask_support_sizes": mask.sum(axis=1).astype(int).tolist(),
        "mask_protocol_valid": False, "mask_source": "unverified", "scientific_status": "invalid",
        "audit_message": "manifest not inspected",
    }
    if not manifest_path.is_file():
        record["audit_message"] = "teachers_manifest.json is missing"
        return record
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list):
        record["audit_message"] = "teachers_manifest.json is not a list"
        return record
    threshold = record["threshold"]
    prepared = _prepared_authority_path(authority_root, record["source_proxy_sha256"])
    if prepared is not None:
        with np.load(prepared, allow_pickle=False) as archive:
            authority = np.asarray(archive["authority"], dtype=np.uint8) if "authority" in archive else None
            source_hash = str(np.asarray(archive["source_proxy_sha256"]).item()) if "source_proxy_sha256" in archive else ""
        independent, message = _holdout_is_independent(source.partition_dir, mask.shape[0])
        if authority is not None and authority.shape == mask.shape and source_hash == record["source_proxy_sha256"]:
            record.update({
                "mask_protocol_valid": bool(independent), "mask_source": f"holdout_acc_per_class>={threshold} (prepared)",
                "authority_artifact": str(prepared),
                "scientific_status": "valid_holdout_mask" if independent else "invalid_partition_provenance",
                "audit_message": message,
            })
            return record
        record["audit_message"] = f"prepared authority is incompatible: {prepared}"
        return record
    holdout_mask = _rebuild_mask(manifest, HOLDOUT_MASK_KEY, threshold, mask.shape)
    if holdout_mask is not None and np.array_equal(holdout_mask, mask):
        independent, message = _holdout_is_independent(source.partition_dir, mask.shape[0])
        record.update({
            "mask_protocol_valid": bool(independent), "mask_source": f"{HOLDOUT_MASK_KEY}>={threshold}",
            "scientific_status": "valid_holdout_mask" if independent else "invalid_partition_provenance",
            "audit_message": message,
        })
        return record
    legacy_mask = _rebuild_mask(manifest, LEGACY_MASK_KEY, threshold, mask.shape)
    if legacy_mask is not None and np.array_equal(legacy_mask, mask):
        record.update({
            "mask_source": f"{LEGACY_MASK_KEY}>={threshold}",
            "scientific_status": "legacy_local_test_leakage_affected",
            "audit_message": (
                "mask exactly matches local-test per-class accuracy; local test is evaluation-only "
                "under Experiment 1 and cannot establish a valid expertise mask"
            ),
        })
        return record
    record["audit_message"] = (
        "cached mask cannot be reconstructed from holdout_acc_per_class or "
        "local_test_acc_per_class at the registered threshold"
    )
    return record


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return float("nan")
    xv, yv = x[valid], y[valid]
    if np.std(xv) == 0 or np.std(yv) == 0:
        return float("nan")
    return float(np.corrcoef(xv, yv)[0, 1])


def _regime_metadata(regime: str) -> dict[str, Any]:
    if regime == "iid":
        return {"regime_family": "iid", "dirichlet_alpha": None, "classes_per_client": None,
                "specialization_level": 0}
    if regime.startswith("alpha"):
        alpha = float(regime[5:].replace("p", "."))
        return {"regime_family": "dirichlet", "dirichlet_alpha": alpha, "classes_per_client": None,
                "specialization_level": None}
    if regime == "multi":
        return {"regime_family": "specialized", "dirichlet_alpha": None, "classes_per_client": "multi",
                "specialization_level": 1}
    if regime == "single":
        return {"regime_family": "specialized", "dirichlet_alpha": None, "classes_per_client": 1,
                "specialization_level": 2}
    return {"regime_family": "other", "dirichlet_alpha": None, "classes_per_client": None,
            "specialization_level": None}


def analyze_cache(cache: dict[str, np.ndarray], *, temperature: float) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Compute per-proxy and condition-level EXPERT diagnostics.

    Individual-teacher noise is ``1 - sum_c M[k,c] p[k,c|x]`` for selected
    teachers.  At target level, unsupported noise is the EXPERT target mass on
    classes unsupported by *every* selected teacher (the complement of their
    union support).  Fallback examples are retained for target-quality metrics
    but excluded from unsupported-mass summaries because they selected nobody.
    """
    logits = cache["teacher_logits_cache"]
    labels = cache["y_true_proxy"].astype(np.int64)
    authority = cache["teacher_knows_class_mask"].astype(bool)
    result = build_server_expertise_target(logits, labels, authority, method="expert_full", temperature=temperature)
    target = result.probabilities.astype(np.float64)
    selected = result.selected_teachers.astype(bool)
    counts = selected.sum(axis=1)
    fallback = counts == 0
    teacher_probs = temperature_softmax(logits, temperature)
    teacher_outside = (teacher_probs * (~authority)[None, :, :]).sum(axis=2)
    support_size = authority.sum(axis=1).astype(np.int16)
    selected_mean_outside = np.divide((teacher_outside * selected).sum(axis=1), counts,
                                      out=np.full(len(counts), np.nan), where=counts > 0)
    selected_mean_support = np.divide((support_size[None, :] * selected).sum(axis=1), counts,
                                      out=np.full(len(counts), np.nan), where=counts > 0)
    selected_singleton_fraction = np.divide(((support_size[None, :] == 1) * selected).sum(axis=1), counts,
                                            out=np.full(len(counts), np.nan), where=counts > 0)
    union_support = np.einsum("nk,kc->nc", selected.astype(np.uint8), authority.astype(np.uint8)) > 0
    target_outside_union = (target * ~union_support).sum(axis=1)
    target_outside_union[fallback] = np.nan
    # The same target compared to each selected teacher's own support is a
    # complementary target-level view; it need not vanish when union noise does.
    target_outside_each = np.einsum("nc,kc->nk", target, (~authority).astype(np.uint8))
    target_mean_outside_each = np.divide((target_outside_each * selected).sum(axis=1), counts,
                                         out=np.full(len(counts), np.nan), where=counts > 0)
    true_probability = target[np.arange(len(labels)), labels]
    target_nll_per_example = -np.log(np.clip(true_probability, EPS, 1.0))
    one_hot = np.zeros_like(target)
    one_hot[np.arange(len(labels)), labels] = 1.0
    target_brier_per_example = np.square(target - one_hot).sum(axis=1)
    target_entropy_per_example = -(target * np.log(np.clip(target, EPS, 1.0))).sum(axis=1)
    target_correct = (target.argmax(axis=1) == labels).astype(np.uint8)
    event = selected
    event_values = teacher_outside[event]
    valid = ~fallback
    quality = target_quality_metrics(target, labels)
    support_by_event = np.broadcast_to(support_size[None, :], selected.shape)
    summary: dict[str, Any] = {
        **dict(result.diagnostics), **quality,
        "n_selected_teacher_events": int(event.sum()),
        "mean_selected_teacher_out_of_expertise_mass": float(event_values.mean()) if event_values.size else float("nan"),
        "median_selected_teacher_out_of_expertise_mass": float(np.median(event_values)) if event_values.size else float("nan"),
        "p90_selected_teacher_out_of_expertise_mass": float(np.quantile(event_values, .90)) if event_values.size else float("nan"),
        "mean_selected_teacher_support_size": float(support_by_event[event].mean()) if event.any() else float("nan"),
        "mean_selected_teacher_support_fraction": float((support_by_event / authority.shape[1])[event].mean()) if event.any() else float("nan"),
        "fraction_selected_teacher_events_single_class_support": float((support_by_event == 1)[event].mean()) if event.any() else float("nan"),
        "mean_target_mass_outside_selected_union_support": float(np.nanmean(target_outside_union)) if valid.any() else float("nan"),
        "mean_target_mass_outside_each_selected_teacher_support": float(np.nanmean(target_mean_outside_each)) if valid.any() else float("nan"),
        "corr_individual_outside_vs_target_nll": _safe_corr(selected_mean_outside, target_nll_per_example),
        "corr_individual_outside_vs_true_class_probability": _safe_corr(selected_mean_outside, true_probability),
        "corr_individual_outside_vs_target_correct": _safe_corr(selected_mean_outside, target_correct),
        "corr_union_outside_vs_target_nll": _safe_corr(target_outside_union, target_nll_per_example),
        "corr_union_outside_vs_true_class_probability": _safe_corr(target_outside_union, true_probability),
        "corr_union_outside_vs_target_correct": _safe_corr(target_outside_union, target_correct),
    }
    observations = {
        "proxy_idx": cache["proxy_idx"].astype(np.int64), "label": labels,
        "selected_teacher_count": counts.astype(np.int16), "fallback": fallback.astype(np.uint8),
        "selected_mean_teacher_out_of_expertise_mass": selected_mean_outside.astype(np.float32),
        "selected_mean_support_size": selected_mean_support.astype(np.float32),
        "selected_single_class_support_fraction": selected_singleton_fraction.astype(np.float32),
        "target_mass_outside_selected_union_support": target_outside_union.astype(np.float32),
        "target_mass_outside_each_selected_teacher_support": target_mean_outside_each.astype(np.float32),
        "target_true_class_probability": true_probability.astype(np.float32),
        "target_nll": target_nll_per_example.astype(np.float32), "target_brier": target_brier_per_example.astype(np.float32),
        "target_entropy": target_entropy_per_example.astype(np.float32), "target_correct": target_correct,
    }
    return observations, summary


def _write_csv_once(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    lines: list[str] = []
    import io
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})
    payload = buffer.getvalue()
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"refusing to overwrite a different derived result: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _select(sources: Iterable[SourceArtifact], args: argparse.Namespace) -> list[SourceArtifact]:
    return [source for source in sources if (
        (not args.datasets or source.dataset in args.datasets)
        and (not args.seeds or source.seed in args.seeds)
        and (not args.regimes or source.regime in args.regimes)
    )]


def run(args: argparse.Namespace) -> Path:
    sources = _select(discover_sources(args.study_root), args)
    if not sources:
        raise FileNotFoundError(f"no proxy-analysis artifacts found below {args.study_root}")
    audits = [audit_source(source, args.authority_root) for source in sources]
    spec = {"experiment": "article_1_experiment_1", "temperature": args.temperature,
            "datasets": args.datasets, "seeds": args.seeds, "regimes": args.regimes,
            "allow_legacy_exploratory": bool(args.allow_legacy_exploratory),
            "authority_root": str(args.authority_root) if args.authority_root else None}
    root = args.output_root / f"diagnostic_{canonical_hash(spec)[:12]}"
    write_json_once(root / "provenance_audit.json", {"spec": spec, "artifacts": audits})
    _write_csv_once(root / "provenance_audit.csv", audits)
    if args.stage == "audit":
        return root
    audit_by_path = {row["source_proxy_analysis"]: row for row in audits}
    selected = [source for source in sources if audit_by_path[str(source.path)]["mask_protocol_valid"]]
    if args.allow_legacy_exploratory:
        selected = [source for source in sources if audit_by_path[str(source.path)]["scientific_status"]
                    in {"valid_holdout_mask", "legacy_local_test_leakage_affected"}]
    if not selected:
        raise ValueError(
            "no artifacts have a valid holdout-based expertise mask. Run --stage audit to inspect "
            "the provenance report, or use --allow-legacy-exploratory only for clearly labelled exploratory results."
        )
    summaries: list[dict[str, Any]] = []
    for source in selected:
        audit = audit_by_path[str(source.path)]
        cache = _load_cache(source.path)
        authority_path = audit.get("authority_artifact")
        if authority_path:
            with np.load(authority_path, allow_pickle=False) as prepared:
                cache["teacher_knows_class_mask"] = np.asarray(prepared["authority"], dtype=np.uint8)
        observations, summary = analyze_cache(cache, temperature=args.temperature)
        observation_path = root / "observations" / source.dataset / f"seed_{source.seed}" / f"{source.regime}.npz"
        save_npz_once(observation_path, **observations,
                      source_proxy_sha256=np.array(audit["source_proxy_sha256"]),
                      mask_source=np.array(audit["mask_source"]),
                      scientific_status=np.array(audit["scientific_status"]))
        summaries.append({**audit, **_regime_metadata(source.regime), **summary,
                          "observation_artifact": str(observation_path),
                          "temperature": float(args.temperature),
                          "experiment": "article_1_experiment_1"})
    summaries.sort(key=lambda row: (row["dataset"], row["seed"], row["regime"]))
    write_json_once(root / "results.json", {"spec": spec, "observations_unit": "proxy_example",
                                             "condition_results": summaries})
    _write_csv_once(root / "results.csv", summaries)
    return root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("audit", "analyze", "all"), default="all")
    parser.add_argument("--study-root", type=Path, default=Path("OUTPUTS/experiments/study_i"))
    parser.add_argument("--output-root", type=Path, default=Path("OUTPUTS/experiments/article_1_server_expertise/experiment_1"))
    parser.add_argument("--datasets", nargs="*", choices=tuple(THRESHOLD_BY_DATASET_LABEL))
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--regimes", nargs="*", choices=("iid", "alpha1p0", "alpha0p5", "alpha0p1", "multi", "single"))
    parser.add_argument("--temperature", type=float, default=8.0)
    parser.add_argument("--authority-root", type=Path,
                        help="Root containing holdout_authority/<source-sha256>/authority.npz from prepare_holdout_authority.py")
    parser.add_argument("--allow-legacy-exploratory", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = run(args)
    print(f"[OK] Article 1 Experiment 1 outputs: {root}")


if __name__ == "__main__":
    main()
