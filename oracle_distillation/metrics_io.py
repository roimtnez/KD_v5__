"""Centralized CSV IO for the KD pipeline (v5).

Single source of truth for column order and idempotent row writing.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# methods_results.csv — global KD
# ---------------------------------------------------------------------------

GLOBAL_CSV_COLUMNS: List[str] = [
    # metadata
    "group_alpha", "method", "proxy_dataset", "seed", "proxy_train_n",
    # model
    "model_test_acc", "model_test_loss",
    "model_test_acc_per_class",
    "model_test_ece", "model_test_entropy",
    "model_super_acc", "model_fine_in_super_acc",
    "model_super_acc_per_super", "model_fine_in_super_acc_per_super",
    "model_best_epoch", "model_stopped_early",
    # distill
    "distill_target_acc", "distill_target_super_acc",
    "distill_student_target_agreement",
    "distill_target_entropy_mean", "distill_target_gt_mass_mean",
    "distill_expert_card_mean", "distill_frac_zero_teachers",
    "distill_argmax_disagree_vs_feddf", "distill_argmax_disagree_correction_frac",
    "distill_oracle_argmax_agreement", "distill_oracle_mean_kl",
    # timing
    "total_time_s", "timestamp",
]

GLOBAL_CSV_KEY: List[str] = [
    "group_alpha", "method", "proxy_dataset", "seed",
]

# ---------------------------------------------------------------------------
# personal_kd_v2_results.csv — personal KD
# ---------------------------------------------------------------------------

PERSONAL_CSV_COLUMNS: List[str] = [
    # metadata
    "group_alpha", "personal_method", "proxy_dataset", "seed", "cid",
    "global_source", "global_logits_kind",
    "n_known_classes", "known_classes",
    "n_known_super", "known_super",
    # model
    "model_test_acc", "model_known_acc", "model_unknown_acc", "model_gap_ku",
    "model_test_acc_per_class",
    "model_super_acc", "model_fine_in_super_acc",
    "model_super_acc_per_super", "model_fine_in_super_acc_per_super",
    "local_test_acc",
    "model_local_test_ece", "model_local_test_entropy",
    "model_known_ece", "model_unknown_ece",
    "model_known_entropy", "model_unknown_entropy",
    # distill — existing
    "distill_frac_known_mass", "distill_frac_local_routed",
    # distill — footprint (v5)
    "distill_target_flips", "distill_footprint_correction", "distill_footprint_regression",
    "distill_budget_known_mean", "distill_is_noop",
    # distill — model-level flips (v5)
    "distill_flip_rate",
    "distill_within_known", "distill_K2U", "distill_U2K", "distill_U2Uprime",
    "distill_correction", "distill_regression", "distill_neutral_churn",
    "distill_wk_net", "distill_dprob_gt_flip",
    "distill_warmstart_drift_l2",
    # distill — predictor (v5)
    "distill_teacher_known_acc", "distill_global_known_acc",
    # meta
    "warm_started", "total_time_s", "timestamp",
]

PERSONAL_CSV_KEY: List[str] = [
    "group_alpha", "personal_method", "proxy_dataset", "seed", "cid",
    "global_source", "global_logits_kind",
]

# ---------------------------------------------------------------------------
# teachers_results.csv — per-client teacher training (one row per rel/seed/cid)
# ---------------------------------------------------------------------------

TEACHERS_CSV_COLUMNS: List[str] = [
    # metadata
    "group_alpha", "proxy_dataset", "seed", "cid",
    "n_train", "n_holdout", "n_local_test",
    # training
    "epochs_ran", "epochs_max",
    # quality (clean local-test set + global test set)
    "local_test_acc", "local_test_loss", "test_acc", "test_loss",
    # per-class accuracy lists — fine (C classes) + superclass (CIFAR-100, 20;
    # empty off CIFAR-100). Stored as JSON lists.
    "local_test_acc_per_class", "test_acc_per_class",
    "local_test_acc_per_super", "test_acc_per_super",
    # calibration / uncertainty on the local-test set
    "local_test_ece", "local_test_entropy",
    "timestamp",
]

TEACHERS_CSV_KEY: List[str] = [
    "group_alpha", "proxy_dataset", "seed", "cid",
]


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def _coerce(v: Any) -> Any:
    """Convert non-serializable types to CSV-friendly Python scalars/strings."""
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.ndarray):
        return json.dumps(v.tolist())
    if isinstance(v, list):
        return json.dumps(v)
    return v


def append_row(
    csv_path: Path,
    row: Dict[str, Any],
    key_cols: Sequence[str],
    columns: Sequence[str],
) -> None:
    """Idempotent CSV append.

    Reads existing rows, deduplicates on *key_cols* (last write wins), then
    writes back with *columns* in canonical order (overflow columns go last,
    sorted alphabetically).
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    row = {k: _coerce(v) for k, v in row.items()}

    existing: list = []
    if csv_path.is_file():
        try:
            # keep_default_na=False: empty key cells (e.g. baseline proxy_variant="")
            # must stay "" rather than becoming NaN, or dedup keys stop matching and
            # rows duplicate across resumes.
            existing = pd.read_csv(
                csv_path, dtype=str, keep_default_na=False
            ).to_dict("records")
        except Exception:
            pass
    existing.append(row)

    seen: dict = {}
    for r in existing:
        key = tuple(str(r.get(c, "")) for c in key_cols)
        seen[key] = r

    df = pd.DataFrame(list(seen.values()))
    ordered = [c for c in columns if c in df.columns]
    rest = sorted(c for c in df.columns if c not in set(columns))
    df[ordered + rest].to_csv(csv_path, index=False)


def resume_csv_rows(csv_path: Path, key_cols: Sequence[str]) -> List[Dict[str, Any]]:
    """Load existing rows from a results CSV, deduplicated on *key_cols*.

    Used by the CLI ``main()`` entrypoints to resume an in-memory row buffer
    across re-runs (the on-disk CSV itself is kept idempotent by
    ``append_row``; this just mirrors the same dedup logic for the buffer).
    Returns ``[]`` if the file doesn't exist or can't be parsed.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        return []
    try:
        raw = pd.read_csv(csv_path).to_dict(orient="records")
    except Exception:
        print(f"[WARN] Could not read {csv_path}; starting fresh.")
        return []

    seen: dict = {}
    for row in raw:
        seen[tuple(str(row.get(k, "")) for k in key_cols)] = row
    rows = list(seen.values())
    if len(rows) < len(raw):
        print(f"[WARN] {csv_path.name} tenía {len(raw)} filas; "
              f"{len(raw) - len(rows)} duplicadas eliminadas.")
    print(f"[*] Resuming with {len(rows)} rows from {csv_path}")
    return rows
