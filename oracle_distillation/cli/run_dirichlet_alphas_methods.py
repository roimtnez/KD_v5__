#!/usr/bin/env python3
"""
Pipeline declarativo sobre grupos de Dirichlet alpha × métodos de KD.

Flujo por config:
  1. ensure_partition   -> sweep_partitions
  2. ensure_teachers    -> run_local_teachers
  3. ensure_proxy       -> build_proxy_analysis
  4. run_methods        -> KdRunner.distill_global (por método)

Naming unificado (opción A): partitions/ y runs/ comparten el mismo path
relativo, construido por `build_partition_name`:

    partitions/<dataset>/K<N>/<tag>__<pool>-<proxy>-<holdout>/
    runs/<dataset>/K<N>/<tag>__<pool>-<proxy>-<holdout>/
        teachers/
        <proxy_dataset>/
            proxy_analysis/
            distillation/<method>/

Idempotente: cada paso se salta si su artefacto existe, salvo --force-*.
Arrancable desde un work_root vacío.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

# repo root en sys.path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from oracle_distillation.distill.kd_runner import (
    KdRunOutputs, KdRunPaths, KdRunner,
)
from oracle_distillation.config import build_execution_spec
from oracle_distillation.checkpoints import checkpoint_sha256
from oracle_distillation.analysis.proxy_analysis_builder import build_proxy_analysis
from oracle_distillation.metrics_io import (
    append_row, resume_csv_rows, GLOBAL_CSV_COLUMNS, GLOBAL_CSV_KEY,
    TEACHERS_CSV_COLUMNS, TEACHERS_CSV_KEY,
)
from oracle_distillation.utils import resolve_device
from oracle_distillation.run_paths import (
    Paths, RunConfig, build_run_configs,
    DEFAULT_HOLDOUT_FRAC, DEFAULT_LOCAL_TEST_FRAC,
)
from oracle_distillation.cli.common_args import add_training_args
from oracle_distillation.provenance import write_execution_provenance
from o3_local.teacher_trainer import (
    main as run_local_teachers, load_manifest, teacher_csv_rows,
)
from o1_partitions.partitioner import partition_exists, make_partitioner
from data.proxy_splits import FLAT_DATASETS, ensure_proxy_split

DEFAULT_ALPHAS = ["0.1", "0.5", "1.0", "iid", "single", "multi"]
DEFAULT_METHODS = ["feddf", "consensus", "oracle", "expert"]


# ---------------------------------------------------------------------------
# Global-CSV row helpers (formerly in cli/sweep_all.py)
# ---------------------------------------------------------------------------

_RUNNER_RENAME = {
    "test_acc": "model_test_acc", "test_loss": "model_test_loss",
    "best_epoch": "model_best_epoch", "stopped_early": "model_stopped_early",
}
_RUNNER_DROP_PREFIXES = ("config.",)
_RUNNER_DROP_KEYS = {"subset"}


def _clean_run_metrics(run_metrics: dict) -> dict:
    """Drop config.* / subset and rename runner metrics to canonical names."""
    clean = {}
    for k, v in run_metrics.items():
        if k in _RUNNER_DROP_KEYS or any(k.startswith(p) for p in _RUNNER_DROP_PREFIXES):
            continue
        clean[_RUNNER_RENAME.get(k, k)] = v
    return clean


def _save_runner_metrics(all_rows, row, run_metrics, global_csv_path) -> None:
    row.update(_clean_run_metrics(run_metrics))
    all_rows.append(row)
    append_row(global_csv_path, row, GLOBAL_CSV_KEY, GLOBAL_CSV_COLUMNS)


# ---------------------------------------------------------------------------
# Ensure-* stages (idempotent)
# ---------------------------------------------------------------------------

def ensure_partition(cfg: RunConfig, paths: Paths, args, *, force: bool = False) -> bool:
    # Single dispatch point: the dataset decides flat-Dirichlet vs CIFAR-100 K=20.
    part = make_partitioner(
        dataset=paths.dataset,
        group_label=cfg.group_label,
        partition_mode=cfg.partition_mode,
        alpha=cfg.alpha,
        num_clients=cfg.num_clients,
        seed=args.seed,
        data_dir=paths.data_dir,
        proxy_size=paths.proxy_size,
        holdout_frac=paths.holdout_frac,
        local_test_frac=paths.local_test_frac,
        classes_per_client=cfg.classes_per_client,
        proxy_split_npz=paths.proxy_split_npz_for_seed(args.seed),
    )
    pdir = paths.partition_dir(cfg, args.seed)
    partition_root = paths.partitions_root_for_seed(args.seed)

    action = "checking/backfilling" if not force and part.exists(partition_root) else "creating"
    print(f"  [RUN] {action} partition at {pdir}")
    try:
        created = part.ensure(partition_root, force=force, raise_on_error=True)
    except Exception as e:
        print(f"  [ERROR] partition creation failed: {e}")
        return False

    if created.resolve() != pdir.resolve():
        print(f"  [WARN] partitioner returned {created} but expected {pdir}")

    return part.exists(partition_root)


def _emit_teacher_rows(cfg: RunConfig, paths: Paths, args) -> None:
    """Project the teachers manifest into the shared top-level teachers_results.csv.

    Idempotent (keyed on group_alpha/proxy_dataset/seed/cid via append_row), so it
    is safe to call on both freshly-trained and pre-existing teacher sets.
    """
    manifest_path = paths.teachers_dir(cfg, args.seed).parent / "teachers_manifest.json"
    if not manifest_path.is_file():
        return
    csv_path = paths.work_root / "teachers_results.csv"
    rows = teacher_csv_rows(
        load_manifest(manifest_path),
        group_alpha=cfg.group_label, proxy_dataset=paths.dataset, seed=args.seed,
    )
    for row in rows:
        append_row(csv_path, row, TEACHERS_CSV_KEY, TEACHERS_CSV_COLUMNS)


def _teacher_caches_complete(tdir: Path, num_clients: int) -> bool:
    expected = {
        f"cid_{cid:03d}_proxy_logits.npz" for cid in range(int(num_clients))
    }
    observed = {path.name for path in Path(tdir).glob("cid_*_proxy_logits.npz")}
    return observed == expected


def _teacher_checkpoints_complete(tdir: Path, num_clients: int) -> bool:
    expected = {f"cid_{cid:03d}.pt" for cid in range(int(num_clients))}
    observed = {path.name for path in Path(tdir).glob("cid_*.pt")}
    return observed == expected


def _backfill_teacher_checkpoint_hashes(manifest_path: Path) -> None:
    """Upgrade a valid pre-policy manifest without retraining its teachers."""
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except Exception:
        return
    changed = False
    for entry in manifest if isinstance(manifest, list) else []:
        ckpt_rel = entry.get("ckpt")
        ckpt = Path(manifest_path).parent / ckpt_rel if ckpt_rel else None
        if ckpt is None or not ckpt.is_file():
            continue
        observed = checkpoint_sha256(ckpt)
        if entry.get("ckpt_sha256") != observed:
            entry["ckpt_sha256"] = observed
            changed = True
    if changed:
        tmp = Path(manifest_path).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        tmp.replace(manifest_path)


def _method_outputs_complete(
    exp_dir: Path, *, save_models: bool, proxy_analysis_path: Path | None = None,
) -> bool:
    required = [
        Path(exp_dir) / "metrics.json",
        Path(exp_dir) / "student_logits.npz",
        Path(exp_dir) / "test_outputs.npz",
    ]
    if save_models:
        required.append(Path(exp_dir) / "student.pt")
    if not all(path.is_file() for path in required):
        return False
    if proxy_analysis_path is not None:
        manifest_path = Path(exp_dir) / "run_manifest.json"
        if not manifest_path.is_file() or not Path(proxy_analysis_path).is_file():
            return False
        try:
            manifest = json.loads(manifest_path.read_text())
            observed = str(manifest.get("proxy_analysis_sha256", ""))
            expected = hashlib.sha256(Path(proxy_analysis_path).read_bytes()).hexdigest()
            if save_models:
                checkpoint_path = Path(exp_dir) / "student.pt"
                checkpoint_ok = (
                    bool(manifest.get("checkpoint_persisted"))
                    and str(manifest.get("checkpoint_sha256", ""))
                    == checkpoint_sha256(checkpoint_path)
                )
            else:
                checkpoint_ok = True
        except Exception:
            return False
        if observed != expected or not checkpoint_ok:
            return False
    return True


def _method_csv_row_exists(
    rows: list[dict], *, dataset: str, group_alpha: str, method: str, seed: int,
) -> bool:
    for row in rows:
        try:
            if (
                str(row.get("proxy_dataset")) == str(dataset)
                and str(row.get("group_alpha")) == str(group_alpha)
                and str(row.get("method")) == str(method)
                and int(row.get("seed", 42)) == int(seed)
            ):
                return True
        except (TypeError, ValueError):
            continue
    return False


def ensure_teachers(cfg: RunConfig, paths: Paths, args, *, force: bool = False) -> bool:
    tdir = paths.teachers_dir(cfg, args.seed)
    _manifest = tdir.parent / "teachers_manifest.json"
    caches_ok = _teacher_caches_complete(tdir, cfg.num_clients)
    checkpoints_ok = _teacher_checkpoints_complete(tdir, cfg.num_clients)
    if args.save_models and checkpoints_ok and _manifest.is_file():
        _backfill_teacher_checkpoint_hashes(_manifest)
    if (
        not force and _manifest.is_file() and caches_ok
        and (not args.save_models or checkpoints_ok)
    ):
        print(f"  [OK] teachers exist: {tdir}")
        _emit_teacher_rows(cfg, paths, args)
        return True

    if getattr(args, "require_prepared", False):
        print(
            "  [ERROR] --require_prepared: complete teacher manifest/proxy-logit "
            f"caches are required at {tdir}; refusing to retrain teachers"
        )
        return False

    pdir = paths.partition_dir(cfg, args.seed)
    if not partition_exists(pdir, cfg.num_clients):
        print(f"  [ERROR] cannot train teachers: partition missing/incomplete at {pdir}")
        return False

    print(f"  [RUN] training teachers at {tdir}")
    try:
        run_local_teachers(
            partition_dir=str(pdir),
            data_dir=str(paths.data_dir),
            out_root=str(paths.teachers_dir(cfg, args.seed).parent),
            device=args.device,
            seed=args.seed,
            batch_size=int(args.teacher_batch_size),
            num_workers=int(args.teacher_num_workers),
            epochs=int(args.teacher_epochs),
            lr=-1.0,        # sentinel → resolved from DatasetConfig inside main()
            dataset=paths.dataset,
            save_models=bool(args.save_models),
        )
    except Exception as e:
        print(f"  [ERROR] run_local_teachers failed: {e}")
        return False

    ok = _teacher_caches_complete(tdir, cfg.num_clients)
    if args.save_models:
        ok = ok and _teacher_checkpoints_complete(tdir, cfg.num_clients)
    if ok:
        _emit_teacher_rows(cfg, paths, args)
    return ok


def ensure_proxy_analysis(cfg: RunConfig, paths: Paths, args, *, force: bool = False) -> bool:
    pxdir = paths.proxy_dir(cfg, args.seed)
    npz = pxdir / "proxy_analysis.npz"
    print(f"  [RUN] checking/building proxy_analysis at {pxdir}")
    try:
        build_proxy_analysis(
            analysis_dir=paths.teachers_dir(cfg, args.seed),
            out_dir=pxdir,
            data_dir=paths.data_dir,
            dataset=paths.dataset,
            partition_dir=paths.partition_dir(cfg, args.seed),
            proxy_split_npz=paths.proxy_split_npz_for_seed(args.seed),
            batch_size=int(args.batch_size),
            num_workers=2,
            seed=int(args.seed),
            device=args.device,
            force=force,
        )
    except Exception as e:
        print(f"  [ERROR] build_proxy_analysis failed: {e}")
        return False

    return npz.exists()


def run_methods(cfg: RunConfig, paths: Paths, args, all_rows: list[dict]) -> bool:
    kd_paths = KdRunPaths(
        analysis_dir=paths.proxy_dir(cfg, args.seed),
        teachers_root=paths.teachers_dir(cfg, args.seed),
        proxy_split_npz=paths.proxy_split_npz_for_seed(args.seed),
        data_dir=paths.data_dir,
        partition_dir=paths.partition_dir(cfg, args.seed),
    )
    global_csv = paths.global_csv()
    rel = paths.rel(cfg)

    ok = True
    for method in args.methods:
        exp_dir = paths.method_dir(cfg, method, seed=args.seed)
        exp_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = exp_dir / "metrics.json"

        if (
            args.skip_if_done
            and _method_outputs_complete(
                exp_dir, save_models=bool(args.save_models),
                proxy_analysis_path=kd_paths.analysis_dir / "proxy_analysis.npz",
            )
            and _method_csv_row_exists(
                all_rows, dataset=paths.dataset, group_alpha=cfg.group_label,
                method=method, seed=args.seed,
            )
        ):
            print(f"  [SKIP] {rel} | {method}")
            continue

        print(f"  -> KD method: {method}")
        exec_spec = build_execution_spec(args=args, dataset=paths.dataset, method=method)
        adir = kd_paths.analysis_dir

        try:
            kd_runner = KdRunner(device=resolve_device(exec_spec.device), seed=exec_spec.seed)
            run_metrics = kd_runner.distill_global(
                exec_spec,
                paths=kd_paths,
                outputs=KdRunOutputs(
                    student_path=exp_dir / "student.pt",
                    metrics_path=metrics_path,
                ),
                analysis_dir=adir if adir.exists() else None,
            )

            if not _method_outputs_complete(
                exp_dir, save_models=bool(args.save_models),
                proxy_analysis_path=kd_paths.analysis_dir / "proxy_analysis.npz",
            ):
                raise RuntimeError(
                    f"incomplete KD outputs at {exp_dir}; expected metrics, logits cache"
                    + (" and checkpoint" if args.save_models else "")
                )

            # Load per-method proxy diagnostics (written by build_proxy_analysis)
            proxy_diag_path = paths.proxy_dir(cfg, args.seed) / "proxy_diagnostics.json"
            per_method_fields: dict = {}
            if proxy_diag_path.exists():
                try:
                    _pd = json.loads(proxy_diag_path.read_text())
                    per_method_fields = _pd.get("per_method", {}).get(method, {})
                except Exception:
                    pass

            row = {
                # metadata
                "group_alpha":    cfg.group_label,
                "method":         method,
                "proxy_dataset":  paths.dataset,
                "seed":           args.seed,
                "timestamp":      time.strftime("%Y-%m-%d %H:%M:%S"),
                # per-method proxy diagnostics (v5)
                **per_method_fields,
            }
            _save_runner_metrics(all_rows, row, run_metrics, global_csv)
        except Exception as e:
            print(f"  [ERROR] KD {method} failed: {e}")
            ok = False
    return ok


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def process_config(cfg: RunConfig, paths: Paths, args, all_rows: list[dict]) -> bool:
    rel = paths.rel(cfg)
    print(f"\n=== CONFIG: {rel} | group={cfg.group_label} ===")
    print(f"  partition_dir: {paths.partition_dir(cfg, args.seed)}")
    print(f"  run_dir:       {paths.run_dir(cfg)}")
    paths.run_dir(cfg).mkdir(parents=True, exist_ok=True)

    if not ensure_partition(cfg, paths, args, force=args.force_partitions):
        print(f"  [ABORT] no partition for {rel}")
        return False
    if not ensure_teachers(cfg, paths, args, force=args.force_teachers):
        print(f"  [ABORT] no teachers for {rel}")
        return False
    if not ensure_proxy_analysis(cfg, paths, args, force=args.force_proxy):
        print(f"  [ABORT] proxy_analysis unavailable for {rel}")
        return False

    if args.prepare_only:
        print(f"  [INFO] prepare_only set; skipping KD for {rel}")
        return True

    return run_methods(cfg, paths, args, all_rows)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Dirichlet-alpha × KD-method pipeline")
    ap.add_argument("--work_root", type=str, required=True,
                    help="Root for partitions/, runs/, and the global CSV.")
    ap.add_argument("--data_dir", type=str, required=True,
                    help="Base dataset directory (CIFAR).")
    ap.add_argument("--dataset", type=str, default="cifar",
                    choices=["cifar", "cifar100", "cinic", "mnist", "fmnist"],
                    help="Proxy dataset used for KD.")

    ap.add_argument("--alphas", nargs="*", default=DEFAULT_ALPHAS,
                    help="Group labels: 0.1, 0.5, 1.0, iid, single, multi, or any float.")
    ap.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    ap.add_argument("--num_clients", type=int, default=10,
                    help="Default K (single-class always uses 10).")
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="Compatibility alias for one complete replicate. The seed "
                         "must match the seed_<N> component of --work_root; inner "
                         "partitions/ and runs/ paths are deliberately seed-free.")

    ap.add_argument("--prepare_only", action="store_true",
                    help="Only partitions + teachers + proxy_analysis; no KD.")
    ap.add_argument(
        "--require_prepared", action="store_true",
        help="Do not train missing teachers. Intended for KD passes after a complete "
             "--prepare_only pass using the same checkpoint-retention setting.",
    )
    ap.add_argument("--skip_if_done", action="store_true",
                    help="Skip KD only when metrics, logits cache, CSV row and any requested "
                         "checkpoint all exist.")
    ap.add_argument("--force_partitions", action="store_true")
    ap.add_argument("--force_teachers", action="store_true")
    ap.add_argument("--force_proxy", action="store_true")
    ap.add_argument("--dry_run", action="store_true",
                    help="List seed/config/method output paths without creating artifacts.")

    add_training_args(ap, val_frac_default=0.0)
    ap.add_argument("--teacher_epochs", type=int, default=-1,
                    help="Teacher training epochs. Default: -1 = use DatasetConfig.teacher_epochs.")
    ap.add_argument("--teacher_batch_size", type=int, default=-1,
                    help="Teacher training batch size. Default: -1 = use DatasetConfig.teacher_batch_size.")
    ap.add_argument("--teacher_num_workers", type=int, default=2)
    ap.add_argument("--holdout_frac", type=float, default=DEFAULT_HOLDOUT_FRAC,
                    help="Holdout fraction used in partition naming (must match what "
                         "the partition was created with). Default 0.2.")
    ap.add_argument("--local_test_frac", type=float, default=DEFAULT_LOCAL_TEST_FRAC,
                    help="Fraction of each client's data reserved as a local test set "
                         "(never seen during training or early stopping). 0.0 = disabled.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if getattr(args, "arch", None):
        from data.dataset_config import set_dataset_arch
        set_dataset_arch(args.dataset, args.arch)  # teachers + teacher-loading + students
        print(f"[arch] {args.dataset} backbone overridden -> {args.arch}")
    if args.student_arch is None:
        from oracle_distillation.models import dataset_default_arch
        args.student_arch = dataset_default_arch(args.dataset)

    work_root = Path(args.work_root).resolve()
    csv_path = work_root / "methods_results.csv"
    all_rows: list[dict] = [] if args.dry_run else resume_csv_rows(csv_path, GLOBAL_CSV_KEY)

    configs = build_run_configs(args.alphas, args.num_clients)
    seeds = args.seeds if args.seeds is not None else [args.seed]
    if len(seeds) != len(set(seeds)):
        raise SystemExit("--seeds must not contain duplicates")
    if len(seeds) != 1:
        raise SystemExit(
            "one work_root cannot contain multiple replicas; use the selective replication "
            "planner to invoke this CLI once per seed"
        )
    seed_tokens = {part for part in work_root.parts if part.startswith("seed_")}
    if seed_tokens:
        if len(seeds) != 1 or seed_tokens != {f"seed_{int(seeds[0])}"}:
            raise SystemExit(
                "a phase-first --work_root is seed-scoped; invoke once per seed so outputs "
                "cannot be mixed (or use the replication command generator)"
            )
    elif int(seeds[0]) != 42:
        raise SystemExit(
            f"seed {seeds[0]} requires a seed-scoped --work_root containing seed_{seeds[0]}"
        )

    paths = Paths(
        work_root=work_root,
        data_dir=Path(args.data_dir).resolve(),
        dataset=args.dataset,
        holdout_frac=args.holdout_frac,
        local_test_frac=args.local_test_frac,
    )
    print(f"\n{'='*60}")
    print(f"Planning {len(seeds)} seeds × {len(configs)} configs × {len(args.methods)} methods")
    print(f"{'='*60}")
    for c in configs:
        print(f"  - {paths.rel(c)}")

    if args.dry_run:
        for seed in seeds:
            print(f"[DRY RUN] seed={seed} proxy_split={paths.proxy_split_npz_for_seed(seed)}")
            for cfg in configs:
                rel = paths.rel(cfg)
                print(f"[DRY RUN] seed={seed} partition={paths.partition_dir(cfg, seed)}")
                print(f"[DRY RUN] seed={seed} teachers={paths.teachers_dir(cfg, seed)}")
                print(f"[DRY RUN] seed={seed} proxy={paths.proxy_dir(cfg, seed)}")
                for method in args.methods:
                    print(f"[DRY RUN] seed={seed} method={method} output={paths.method_dir(cfg, method, seed)}")
        return

    work_root.mkdir(parents=True, exist_ok=True)

    failed_configs: list[str] = []
    for seed in seeds:
        args.seed = int(seed)
        if paths.dataset in FLAT_DATASETS:
            split_path = ensure_proxy_split(
                paths.dataset, args.seed, paths.data_dir, proxy_size=paths.proxy_size,
            )
            print(f"[OK] seed-specific proxy split: {split_path}")
        if args.seed != 42 or "experiments" in work_root.parts:
            config_dir = work_root.parent / "configs" if work_root.name == "raw_work" else work_root / "configs"
            stage = (
                "prepare"
                if args.prepare_only
                else "methods_" + "_".join(str(method) for method in args.methods)
            )
            write_execution_provenance(
                config_dir,
                name=f"phase_a_{args.dataset}_seed_{args.seed}_{stage}",
                experiment_id="phase_a_methods",
                seed=args.seed,
                args=args,
                input_paths={
                    "data_dir": args.data_dir,
                    "proxy_split": str(paths.proxy_split_npz_for_seed(args.seed)),
                },
                output_paths={"work_root": str(work_root)},
                repo_root=_REPO_ROOT,
            )
        print(f"\n{'='*60}\nSEED {args.seed}\n{'='*60}")
        for cfg in configs:
            try:
                if not process_config(cfg, paths, args, all_rows):
                    failed_configs.append(f"seed={args.seed}:{paths.rel(cfg)}")
            except KeyboardInterrupt:
                print("\n[INTERRUPTED] exiting.")
                raise
            except Exception as e:
                print(f"[ERROR] seed={args.seed} {paths.rel(cfg)}: {e}")
                failed_configs.append(f"seed={args.seed}:{paths.rel(cfg)}")

    print(f"\n[DONE] Global CSV: {csv_path}")
    if failed_configs:
        raise SystemExit(
            f"pipeline failed for {len(failed_configs)} config(s): "
            + ", ".join(failed_configs)
        )


if __name__ == "__main__":
    main()
