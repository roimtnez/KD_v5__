"""Read-only loaders and fail-closed checks for the three final chapter studies.

Training code must not import this module.  It is intentionally limited to artifact
schemas, analysis invariants and small derived tables used by the canonical notebooks.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from oracle_distillation.output_registry import FLAT_DATASETS, repo_root


STUDY_I_CSV = Path("ANALYSIS_v2/data/study_i/seed_42/methods_results.csv")
STUDY_II_DIR = Path("ANALYSIS/02_study_ii_label_light/outputs/confirmatory")
STUDY_III_HEADROOM_CSV = Path(
    "ANALYSIS/data/headroom/seed_42/headroom_maskfree_paper.csv"
)
STUDY_III_ABLATION_CSV = Path(
    "ANALYSIS/data/phase_b/seed_42/transfer_set_ablation_results.csv"
)

STUDY_I_METHODS = frozenset(
    {"oracle", "expert", "feddf", "energy", "confidence", "consensus"}
)
STUDY_III_ARMS = frozenset(
    {"global", "proxy_plain", "proxy", "local", "local_lwf", "mix"}
)
STUDY_III_MAIN_SOURCES = frozenset({"feddf", "energy"})
STUDY_III_REFERENCE_SOURCES = frozenset({"expert"})


@dataclass(frozen=True)
class AuditResult:
    path: Path
    rows: int
    columns: tuple[str, ...]
    datasets: tuple[str, ...]
    seeds: tuple[int, ...]


def _path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else repo_root() / value


def _display_path(path: Path) -> Path:
    try:
        return path.relative_to(repo_root())
    except ValueError:
        return path


def _require_columns(frame: pd.DataFrame, required: Iterable[str], path: Path) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing required columns: {missing}")


def _require_flat_datasets(frame: pd.DataFrame, column: str, path: Path) -> None:
    observed = set(frame[column].dropna().astype(str))
    unexpected = observed.difference(FLAT_DATASETS)
    if unexpected:
        raise ValueError(
            f"{path}: non-main datasets present in canonical artifact: {sorted(unexpected)}"
        )
    missing = set(FLAT_DATASETS).difference(observed)
    if missing:
        raise ValueError(f"{path}: missing main datasets: {sorted(missing)}")


def load_study_i(path: str | Path = STUDY_I_CSV) -> tuple[pd.DataFrame, AuditResult]:
    resolved = _path(path)
    frame = pd.read_csv(resolved)
    required = {
        "proxy_dataset", "group_alpha", "method", "seed",
        "model_test_acc", "model_test_loss",
    }
    _require_columns(frame, required, resolved)
    _require_flat_datasets(frame, "proxy_dataset", resolved)
    methods = set(frame["method"].astype(str))
    if methods != STUDY_I_METHODS:
        raise ValueError(
            f"{resolved}: expected Study-I methods {sorted(STUDY_I_METHODS)}, "
            f"observed {sorted(methods)}"
        )
    key = ["proxy_dataset", "group_alpha", "method", "seed"]
    if frame.duplicated(key).any():
        raise ValueError(f"{resolved}: duplicate Study-I keys")
    audit = AuditResult(
        _display_path(resolved), len(frame), tuple(frame.columns),
        tuple(sorted(frame["proxy_dataset"].unique())),
        tuple(sorted(int(x) for x in frame["seed"].unique())),
    )
    return frame, audit


def study_ii_status(
    directory: str | Path = STUDY_II_DIR,
) -> dict[str, object]:
    """Return explicit availability without manufacturing absent Study-II results."""
    resolved = _path(directory)
    expected = (
        "mask_authority_results.csv",
        "pseudo_label_results.csv",
        "target_fidelity_results.csv",
        "student_results.csv",
    )
    present = tuple(name for name in expected if (resolved / name).is_file())
    return {
        "directory": str(_display_path(resolved)),
        "expected": expected,
        "present": present,
        "complete": len(present) == len(expected),
        "state": "complete" if len(present) == len(expected) else "not_launched_or_incomplete",
    }


def load_study_iii_headroom(
    path: str | Path = STUDY_III_HEADROOM_CSV,
) -> tuple[pd.DataFrame, AuditResult]:
    resolved = _path(path)
    frame = pd.read_csv(resolved)
    required = {
        "seed", "dataset", "regime", "base", "anchor", "surface", "cid",
        "teacher_intra", "base_intra", "headroom",
    }
    _require_columns(frame, required, resolved)
    _require_flat_datasets(frame, "dataset", resolved)
    expected = frame["teacher_intra"] - frame["base_intra"]
    # The promoted CSV stores headroom rounded one decimal place further than a few
    # source columns; the maximum documented reconstruction difference is 1e-6.
    if not np.allclose(frame["headroom"], expected, equal_nan=True, atol=1.1e-6):
        raise ValueError(f"{resolved}: headroom != teacher_intra - base_intra")
    audit = AuditResult(
        _display_path(resolved), len(frame), tuple(frame.columns),
        tuple(sorted(frame["dataset"].unique())),
        tuple(sorted(int(x) for x in frame["seed"].unique())),
    )
    return frame, audit


def load_study_iii_ablation(
    path: str | Path = STUDY_III_ABLATION_CSV,
) -> tuple[pd.DataFrame, AuditResult]:
    resolved = _path(path)
    frame = pd.read_csv(resolved)
    required = {
        "dataset", "regime", "arm", "seed", "cid", "global_source",
        "g_known_acc", "g_unknown_acc", "g_gap_ku", "g_test_acc",
        "p_known_acc", "p_unknown_acc", "p_gap_ku", "p_test_acc",
        "localsurf_known_acc", "localsurf_unknown_acc",
        "localsurf_gap_ku", "localsurf_overall_acc", "invariants_ok",
    }
    _require_columns(frame, required, resolved)
    _require_flat_datasets(frame, "dataset", resolved)
    if len(frame) != 3240:
        raise ValueError(f"{resolved}: expected 3240 rows, observed {len(frame)}")
    if set(frame["arm"]) != STUDY_III_ARMS:
        raise ValueError(f"{resolved}: incomplete or unexpected ablation arms")
    allowed_sources = STUDY_III_MAIN_SOURCES | STUDY_III_REFERENCE_SOURCES
    if set(frame["global_source"]) != allowed_sources:
        raise ValueError(f"{resolved}: incomplete or unexpected global sources")

    key = ["dataset", "regime", "seed", "cid", "global_source"]
    if frame.duplicated(key + ["arm"]).any():
        raise ValueError(f"{resolved}: duplicate ablation keys")
    sizes = frame.groupby(key, dropna=False).size()
    if len(sizes) != 540 or not (sizes == 6).all():
        raise ValueError(f"{resolved}: expected 540 complete six-arm groups")
    if not frame["invariants_ok"].astype(bool).all():
        raise ValueError(f"{resolved}: rows with invariants_ok=False")

    global_columns = [column for column in frame if column.startswith("g_")]
    violations = sum(
        int((frame.groupby(key, dropna=False)[column].nunique(dropna=False) > 1).sum())
        for column in global_columns
    )
    if violations:
        raise ValueError(f"{resolved}: {violations} cross-arm g_* violations")

    wide = frame.pivot(index=key, columns="arm", values="p_test_acc")
    identity_error = np.abs(
        (wide["proxy"] - wide["global"])
        - (
            (wide["proxy_plain"] - wide["global"])
            + (wide["proxy"] - wide["proxy_plain"])
        )
    )
    if float(identity_error.max()) > 1e-12:
        raise ValueError(f"{resolved}: attribution identity failed")

    global_rows = frame[frame["arm"] == "global"].set_index(key).reindex(wide.index)
    if not np.allclose(
        wide["global"].to_numpy(), global_rows["g_test_acc"].to_numpy(), atol=1e-12
    ):
        raise ValueError(f"{resolved}: global arm p_test_acc differs from g_test_acc")

    audit = AuditResult(
        _display_path(resolved), len(frame), tuple(frame.columns),
        tuple(sorted(frame["dataset"].unique())),
        tuple(sorted(int(x) for x in frame["seed"].unique())),
    )
    return frame, audit


def study_iii_attribution(frame: pd.DataFrame) -> pd.DataFrame:
    """Return paired effects without mixing global, personalized and local surfaces."""
    key = ["dataset", "regime", "seed", "cid", "global_source"]
    _require_columns(frame, set(key) | {"arm", "p_test_acc"}, Path("<dataframe>"))
    wide = frame.pivot(index=key, columns="arm", values="p_test_acc")
    output = wide.reset_index()[key].copy()
    output["extra_pass_effect"] = (
        wide["proxy_plain"] - wide["global"]
    ).to_numpy()
    output["class_mask_effect"] = (
        wide["proxy"] - wide["proxy_plain"]
    ).to_numpy()
    output["total_gain"] = (wide["proxy"] - wide["global"]).to_numpy()
    if not np.allclose(
        output["total_gain"],
        output["extra_pass_effect"] + output["class_mask_effect"],
        atol=1e-12,
    ):
        raise AssertionError("attribution identity failed after pivot")
    return output
