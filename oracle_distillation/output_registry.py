"""Canonical artifact paths for the final thesis analysis surface.

This module is deliberately lightweight: it only records where artifacts live
and how to validate their identity. It does not import training code or mutate
the filesystem, so scripts and notebooks can use it as a stable path registry.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

from oracle_distillation.experiment_paths import EXPERIMENTS, output_path


SCHEMA_VERSION = 1

FLAT_DATASETS: tuple[str, ...] = ("mnist", "fmnist", "cifar")
PHASEB_MAIN_GLOBAL_SOURCES: tuple[str, ...] = ("feddf", "energy")
PHASEB_REFERENCE_GLOBAL_SOURCES: tuple[str, ...] = ("expert",)


def repo_root() -> Path:
    """Return the repository root from this module location."""
    return Path(__file__).resolve().parents[1]


def resolve_repo_path(path: str | Path, root: Optional[Path] = None) -> Path:
    """Resolve a repo-relative path against *root* without requiring it to exist."""
    p = Path(path)
    if p.is_absolute():
        return p
    return (root or repo_root()) / p


@dataclass(frozen=True)
class ArtifactSpec:
    """Static description of an output artifact used by the paper/thesis.

    ``canonical_path`` is the current source of truth. ``legacy_paths`` are
    readable compatibility locations that should not be preferred for new work.
    ``source_paths`` records raw/provenance inputs when the canonical artifact is
    a staged or derived asset.
    """

    artifact_id: str
    phase: str
    artifact_type: str
    canonical_path: str
    description: str
    dataset_scope: str
    feeds: tuple[str, ...] = ()
    source_paths: tuple[str, ...] = ()
    legacy_paths: tuple[str, ...] = ()
    key_columns: tuple[str, ...] = ()
    required_columns: tuple[str, ...] = ()
    dataset_column: Optional[str] = None
    allowed_datasets: tuple[str, ...] = ()
    notes: str = ""

    def to_manifest_dict(self, root: Optional[Path] = None) -> dict:
        root = root or repo_root()
        data = asdict(self)
        # JSON serializes tuples as lists. Normalize them here so a manifest
        # rebuilt in memory compares equal to the persisted JSON during
        # ``scripts/build_output_index.py --check``.
        for field in (
            "feeds",
            "source_paths",
            "legacy_paths",
            "key_columns",
            "required_columns",
            "allowed_datasets",
        ):
            data[field] = list(data[field])
        data["schema_version"] = SCHEMA_VERSION
        data["exists"] = resolve_repo_path(self.canonical_path, root).exists()
        return data


def artifact_specs() -> tuple[ArtifactSpec, ...]:
    """Return the canonical artifact registry.

    Only artifacts consumed by the three canonical notebooks, the explicitly retained diagnostic,
    or the final multi-seed readouts belong here. Raw checkpoints live under ``OUTPUTS/``.
    """

    return (
        ArtifactSpec(
            artifact_id="phaseA.methods.flat",
            phase="phaseA",
            artifact_type="processed_csv",
            canonical_path="ANALYSIS_v2/data/study_i/seed_42/methods_results.csv",
            source_paths=("OUTPUTS/experiments/study_i/seed_42/raw_work/methods_results.csv",),
            description="Global KD results for Phase A over MNIST/FMNIST/CIFAR-10.",
            dataset_scope="flat_main",
            feeds=("Phase A", "Discussion"),
            key_columns=("proxy_dataset", "group_alpha", "method", "seed"),
            required_columns=("proxy_dataset", "group_alpha", "method", "seed", "model_test_acc"),
            dataset_column="proxy_dataset",
            allowed_datasets=FLAT_DATASETS,
            notes="Use for the thesis/paper Phase A narrative. personal_expert is not part of this CSV.",
        ),
        ArtifactSpec(
            artifact_id="phaseA12.semisup.index",
            phase="phaseA12",
            artifact_type="analysis_readme",
            canonical_path="ANALYSIS/narrative/REPO_GUIDE.md",
            description="Canonical narrative for the partially labeled proxy study (unified guide, section 3).",
            dataset_scope="flat_main",
            feeds=("Phase A1/2", "Discussion"),
            source_paths=(
                "possibly_deprecated/study_ii_auto_v2_legacy/ANALYSIS/notebooks/"
                "phase_a12_label_efficiency_readable_executed.ipynb",
            ),
            notes="Merged into the unified REPO_GUIDE.md; paired with the readable executed notebook.",
        ),
        ArtifactSpec(
            artifact_id="phaseA12.semisup.exp3_students",
            phase="phaseA12",
            artifact_type="processed_csv",
            canonical_path="ANALYSIS/data/phase_a12/seed_42/exp3/exp3_kd_students.csv",
            source_paths=("OUTPUTS/experiments/phase_a12/seed_42/students",),
            description="Student-level KD confirmation for auto_v2 semisupervised routing.",
            dataset_scope="flat_main",
            feeds=("Phase A1/2",),
            key_columns=("dataset", "regime", "N", "arm"),
            required_columns=("dataset", "regime", "N", "arm", "student_test_acc", "supervised_on_N"),
            dataset_column="dataset",
            allowed_datasets=FLAT_DATASETS,
        ),
        ArtifactSpec(
            artifact_id="phaseA12.oracle_ln.comparison",
            phase="phaseA12",
            artifact_type="processed_csv",
            canonical_path=(
                "ANALYSIS/data/phase_a12/seed_42/oracle_ln/oracle_ln_comparison.csv"
            ),
            source_paths=(
                "ANALYSIS/data/phase_a12/seed_42/oracle_ln/oracle_ln_raw_results.csv",
                "ANALYSIS/data/phase_a12/seed_42/exp3/exp3_kd_students.csv",
                "ANALYSIS/data/phase_a12/seed_42/exp4/exp4_budget_matched.csv",
                "ANALYSIS_v2/data/study_i/seed_42/methods_results.csv",
            ),
            description=(
                "Integrated label-efficiency control comparing auto_v2 with exact Phase-A "
                "ORACLE restricted to the same nested labeled slice L_N."
            ),
            dataset_scope="flat_main",
            feeds=("Phase A1/2", "Discussion"),
            key_columns=("dataset", "regime", "N"),
            required_columns=(
                "dataset", "regime", "N", "auto_v2", "oracle_LN_standard",
                "oracle_LN_matched", "supervised_on_N", "supervised_matched",
                "self_training", "expert_full_label", "oracle_full_label",
            ),
            dataset_column="dataset",
            allowed_datasets=FLAT_DATASETS,
            notes=(
                "Complete 108-cell integrated readout backed by 216 new training rows; use the "
                "associated REPORT.md for the seed-42 and single-regime caveats. The umbrella "
                "experiment ID is oracle_LN_only; this is separate from the Exp-5 bridge."
            ),
        ),
        ArtifactSpec(
            artifact_id="headroom.maskfree",
            phase="headroom",
            artifact_type="processed_csv",
            canonical_path="ANALYSIS/data/headroom/seed_42/headroom_maskfree_paper.csv",
            source_paths=("OUTPUTS/experiments/headroom/seed_42/raw_results/headroom_clients.csv",),
            description="Proxy-surface headroom screen for expert and mask-free bases.",
            dataset_scope="flat_main",
            feeds=("Headroom", "Phase B"),
            key_columns=("dataset", "regime", "base", "anchor", "surface", "cid"),
            required_columns=("dataset", "regime", "base", "anchor", "surface", "cid", "headroom"),
            dataset_column="dataset",
            allowed_datasets=FLAT_DATASETS,
            notes="Final analysis copy is restricted to MNIST/FMNIST/CIFAR-10.",
        ),
        ArtifactSpec(
            artifact_id="phaseA12.phaseB_bridge.exp5",
            phase="phaseA12_phaseB_bridge",
            artifact_type="processed_csv",
            canonical_path="ANALYSIS/data/exp5/seed_42/exp5_semisup_classmask.csv",
            source_paths=(
                "OUTPUTS/experiments/phase_a12/seed_42/students",
                "OUTPUTS/experiments/exp5_a12_classmask/seed_42/checkpoints",
            ),
            description="Exp-5 trained bridge from fixed auto_v2 N=100 globals to private local class_mask KD.",
            dataset_scope="flat_main",
            feeds=("Phase A1/2-to-Phase-B bridge", "Discussion"),
            key_columns=("dataset", "regime", "N", "seed", "cid", "arm", "global_source"),
            required_columns=(
                "dataset", "regime", "N", "seed", "cid", "arm", "global_source",
                "g_test_acc", "p_test_acc", "localsurf_overall_acc",
            ),
            dataset_column="dataset",
            allowed_datasets=FLAT_DATASETS,
            notes=(
                "Complete 540-row seed-42 matrix with 180 three-arm client groups and a passing "
                "sanity report. Interpret descriptively; smoke outputs are excluded."
            ),
        ),
        ArtifactSpec(
            artifact_id="phaseB.trained.transfer_set_ablation",
            phase="phaseB",
            artifact_type="processed_csv",
            canonical_path="ANALYSIS/data/phase_b/seed_42/transfer_set_ablation_results.csv",
            source_paths=("OUTPUTS/experiments/phase_b/seed_42/checkpoints",),
            description="Complete six-arm transfer-set ablation over flat datasets.",
            dataset_scope="flat_main",
            feeds=("Phase B", "Transfer-set ablation", "Discussion"),
            key_columns=("dataset", "regime", "arm", "seed", "cid", "global_source"),
            required_columns=(
                "dataset", "regime", "arm", "seed", "cid", "global_source",
                "g_test_acc", "p_test_acc", "localsurf_overall_acc",
            ),
            dataset_column="dataset",
            allowed_datasets=FLAT_DATASETS,
            notes=(
                "Main Phase B conclusions use feddf/energy as mask-free global_source; "
                "expert is a reference anchor. Preserve g_*, p_* and *_localsurf separation."
            ),
        ),
        ArtifactSpec(
            artifact_id="discussion.attribution.review",
            phase="discussion",
            artifact_type="report_md",
            canonical_path="ANALYSIS/narrative/REPO_GUIDE.md",
            source_paths=("ANALYSIS/data/phase_b/seed_42/transfer_set_ablation_results.csv",),
            description="Mechanism-vs-extra-pass attribution review (unified guide, section 5).",
            dataset_scope="flat_main",
            feeds=("Discussion", "Transfer-set ablation"),
        ),
        ArtifactSpec(
            artifact_id="paper.figures.current",
            phase="paper_assets",
            artifact_type="figure_dir",
            canonical_path="ANALYSIS/figures",
            source_paths=(
                "ANALYSIS/data",
                "possibly_deprecated/notebooks/ANALYSIS/notebooks/paper_v2.ipynb",
            ),
            description="Current exported notebook figures and paper tables.",
            dataset_scope="flat_main",
            feeds=("Paper figures", "Thesis figures"),
            notes=(
                "Generated by the three canonical notebooks plus the explicitly retained partition "
                "diagnostic; data never live in the figure tree."
            ),
        ),
    )


def artifacts_by_phase(phase: str) -> tuple[ArtifactSpec, ...]:
    return tuple(spec for spec in artifact_specs() if spec.phase == phase)


def artifact_by_id(artifact_id: str) -> ArtifactSpec:
    for spec in artifact_specs():
        if spec.artifact_id == artifact_id:
            return spec
    raise KeyError(f"unknown artifact_id: {artifact_id}")


def canonical_path(artifact_id: str, root: Optional[Path] = None) -> Path:
    return resolve_repo_path(artifact_by_id(artifact_id).canonical_path, root)


def existing_legacy_paths(spec: ArtifactSpec, root: Optional[Path] = None) -> tuple[Path, ...]:
    root = root or repo_root()
    return tuple(p for p in (resolve_repo_path(x, root) for x in spec.legacy_paths) if p.exists())


def iter_specs(ids: Optional[Iterable[str]] = None) -> Sequence[ArtifactSpec]:
    if ids is None:
        return artifact_specs()
    wanted = set(ids)
    return tuple(spec for spec in artifact_specs() if spec.artifact_id in wanted)


def multiseed_experiment_ids() -> tuple[str, ...]:
    """Stable IDs in the phase-first OUTPUTS registry."""
    return tuple(EXPERIMENTS)


def resolve_multiseed_output(
    experiment_id: str, seed: int, artifact: str = "root", *, root: Optional[Path] = None,
) -> Path:
    """Resolve a new multi-seed output while leaving paper paths unchanged."""
    base = (root or repo_root()) / "OUTPUTS"
    return output_path(
        experiment=experiment_id, seed=seed, artifact=artifact, output_root=base,
    )
