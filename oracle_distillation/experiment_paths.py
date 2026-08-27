"""Central phase-first paths for the selective multi-seed replication.

Raw executions always live below ``OUTPUTS/experiments/<phase>/seed_<seed>``.
The retained, analysis-ready seed-42 tables live below ``ANALYSIS/data`` until
their raw checkpoints are restored and migrated into the same execution tree.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_OUTPUT_ROOT = Path("OUTPUTS")
REPLICATION_SEEDS = (42, 43, 44)
NEW_REPLICATION_SEEDS = (43, 44)


@dataclass(frozen=True)
class ExperimentDefinition:
    experiment_id: str
    phase_dir: str
    source_script: str
    parent_experiments: tuple[str, ...] = ()


EXPERIMENTS: dict[str, ExperimentDefinition] = {
    "phase_a_methods": ExperimentDefinition(
        "phase_a_methods", "study_i",
        "oracle_distillation.cli.run_dirichlet_alphas_methods",
    ),
    "phase_a12_auto_v2": ExperimentDefinition(
        "phase_a12_auto_v2", "phase_a12",
        "LEGACY_EVIDENCE (no active launcher)",
        ("phase_a_methods",),
    ),
    "phase_a12_oracle_ln_matched": ExperimentDefinition(
        "phase_a12_oracle_ln_matched", "phase_a12",
        "scripts/semisup/oracle_ln/train_oracle_ln.py",
        ("phase_a_methods", "phase_a12_auto_v2"),
    ),
    "headroom_core": ExperimentDefinition(
        "headroom_core", "headroom",
        "scripts/class_mask_base_prescreen.py",
        ("phase_a_methods", "phase_a12_auto_v2"),
    ),
    "phase_b_classmask_causal": ExperimentDefinition(
        "phase_b_classmask_causal", "phase_b",
        "oracle_distillation.cli.run_transfer_set_ablation",
        ("phase_a_methods",),
    ),
    "phase_b_six_arm_exploratory": ExperimentDefinition(
        "phase_b_six_arm_exploratory", "phase_b",
        "oracle_distillation.cli.run_transfer_set_ablation",
        ("phase_a_methods",),
    ),
    "exp5_a12_classmask": ExperimentDefinition(
        "exp5_a12_classmask", "exp5_a12_classmask",
        "scripts/semisup/exp5/semisup_exp5_classmask.py",
        ("phase_a12_auto_v2",),
    ),
}


ARTIFACT_DIRS = {
    "root": "",
    "configs": "configs",
    "raw_work": "raw_work",
    "partitions": "partitions",
    "teachers": "teachers",
    "proxy_analysis": "proxy_analysis",
    "signal_analysis": "signal_analysis",
    "auto_v2": "auto_v2",
    "students": "students",
    "baselines": "baselines",
    "oracle_ln_matched": "oracle_ln_matched",
    "targets": "targets",
    "checkpoints": "checkpoints",
    "raw_results": "raw_results",
    "readouts": "readouts",
    "logs": "logs",
    "validation": "validation",
}


SEED42_ANALYSIS_SOURCES: dict[str, tuple[str, ...]] = {
    "phase_a_methods": (
        "ANALYSIS_v2/data/study_i/seed_42/methods_results.csv",
    ),
    "phase_a12_auto_v2": (
        "ANALYSIS/data/phase_a12/seed_42/exp3/exp3_kd_students.csv",
    ),
    "phase_a12_oracle_ln_matched": (
        "ANALYSIS/data/phase_a12/seed_42/oracle_ln/oracle_ln_matched.csv",
    ),
    "headroom_core": (
        "ANALYSIS/data/headroom/seed_42/headroom_maskfree_paper.csv",
    ),
    "phase_b_classmask_causal": (
        "ANALYSIS/data/phase_b/seed_42/causal_reduced.csv",
    ),
    "phase_b_six_arm_exploratory": (
        "ANALYSIS/data/phase_b/seed_42/transfer_set_ablation_results.csv",
    ),
    "exp5_a12_classmask": (
        "ANALYSIS/data/exp5/seed_42/exp5_semisup_classmask.csv",
    ),
}


def experiment_definition(experiment: str) -> ExperimentDefinition:
    try:
        return EXPERIMENTS[str(experiment)]
    except KeyError as exc:
        raise ValueError(
            f"unknown experiment {experiment!r}; expected one of {sorted(EXPERIMENTS)}"
        ) from exc


def seed_dir(seed: int) -> str:
    seed = int(seed)
    if seed not in REPLICATION_SEEDS:
        raise ValueError(f"replication seed must be one of {REPLICATION_SEEDS}; got {seed}")
    return f"seed_{seed}"


def output_path(
    *,
    experiment: str,
    seed: int,
    artifact: str = "root",
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    parts: Iterable[str | Path] = (),
) -> Path:
    """Resolve one phase-first experiment path without touching the filesystem."""
    definition = experiment_definition(experiment)
    if artifact not in ARTIFACT_DIRS:
        raise ValueError(f"unknown artifact {artifact!r}; expected one of {sorted(ARTIFACT_DIRS)}")
    path = Path(output_root) / "experiments" / definition.phase_dir / seed_dir(seed)
    suffix = ARTIFACT_DIRS[artifact]
    if suffix:
        path /= suffix
    for part in parts:
        candidate = Path(part)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"artifact path component must be relative and contained: {part!r}")
        path /= candidate
    return path


def seed42_analysis_paths(experiment: str, *, repo_root: Path | str = Path(".")) -> tuple[Path, ...]:
    """Return retained analysis inputs for seed 42; never write to these paths."""
    experiment_definition(experiment)
    return tuple(Path(repo_root) / p for p in SEED42_ANALYSIS_SOURCES.get(experiment, ()))


def assert_seed_contained(path: Path | str, seed: int) -> None:
    """Fail closed when a new artifact path does not carry its requested seed."""
    token = seed_dir(seed)
    if token not in Path(path).parts:
        raise ValueError(f"path {path} is not contained in {token}")
