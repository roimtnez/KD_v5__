"""Deterministic distribution reports for persisted client partitions.

The report is derived exclusively from ``clients/c*.npz`` and the dataset
labels.  It never changes partition indices, so it is safe to backfill reports
for an already trained seed.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml


REPORT_SPLITS = ("train_idx", "holdout_idx", "local_test_idx")
REPORT_FILENAMES = ("counts.csv", "percent.csv")


@dataclass(frozen=True)
class ClientSummary:
    client_id: int
    n: int
    num_classes_present: int
    entropy: float
    kl_to_global: float
    js_to_global: float
    gini: float


def load_dataset_labels(dataset: str, data_dir: Path) -> np.ndarray:
    """Load the canonical train labels through the dataset registry."""
    from data.dataset_config import get_dataset_config

    ds = get_dataset_config(dataset).load_train_eval_dataset(Path(data_dir))
    targets = getattr(ds, "targets", None)
    if targets is None:
        raise ValueError(f"dataset {dataset!r} does not expose a targets array")
    return np.asarray(targets, dtype=np.int64)


def _client_files(partition_dir: Path) -> list[Path]:
    files = sorted((Path(partition_dir) / "clients").glob("c*.npz"))
    if not files:
        raise FileNotFoundError(f"no clients/c*.npz below {partition_dir}")
    return files


def available_splits(partition_dir: Path) -> tuple[str, ...]:
    files = _client_files(partition_dir)
    with np.load(files[0], allow_pickle=False) as first:
        return tuple(split for split in REPORT_SPLITS if split in first.files)


def report_complete(partition_dir: Path) -> bool:
    """Whether all CSV and plot products required by the client arrays exist."""
    partition_dir = Path(partition_dir)
    try:
        splits = available_splits(partition_dir)
    except FileNotFoundError:
        return False
    report = partition_dir / "report"
    required_csv = [report / "summary.csv"]
    required = [report / "distribution_heatmap.png", report / "dist_counts.png"]
    for split in splits:
        split_name = split.removesuffix("_idx")
        required_csv.extend(report / split_name / name for name in REPORT_FILENAMES)
    if not all(path.is_file() for path in (*required, *required_csv)):
        return False
    for path in required_csv:
        with path.open(newline="", encoding="utf-8") as handle:
            if "seed" not in next(csv.reader(handle), []):
                return False
    return True


def _counts(labels: np.ndarray, client_indices: Sequence[np.ndarray], num_classes: int) -> np.ndarray:
    matrix = np.zeros((len(client_indices), num_classes), dtype=np.int64)
    for cid, indices in enumerate(client_indices):
        indices = np.asarray(indices, dtype=np.int64)
        if indices.size:
            if indices.min() < 0 or indices.max() >= len(labels):
                raise IndexError(f"client {cid} contains an index outside the dataset label array")
            matrix[cid] = np.bincount(labels[indices], minlength=num_classes)[:num_classes]
    return matrix


def _percent(counts: np.ndarray) -> np.ndarray:
    return counts / np.maximum(1, counts.sum(axis=1, keepdims=True))


def _report_context(partition_dir: Path) -> dict[str, object]:
    config_path = Path(partition_dir) / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"partition report requires provenance config: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if "seed" not in config or "dataset" not in config:
        raise ValueError(f"partition config lacks seed/dataset provenance: {config_path}")
    return {
        "seed": int(config["seed"]),
        "dataset": str(config["dataset"]),
        "regime": Path(partition_dir).name.split("__", 1)[0],
    }


def _write_matrix(
    path: Path,
    matrix: np.ndarray,
    *,
    context: dict[str, object],
    split: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "dataset", "regime", "split", "Client ID",
                         *map(str, range(matrix.shape[1]))])
        for cid, row in enumerate(matrix):
            writer.writerow([
                context["seed"], context["dataset"], context["regime"], split, cid,
                *row.tolist(),
            ])


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    if not mask.any():
        return 0.0
    return float(np.sum(p[mask] * (np.log(p[mask]) - np.log(np.clip(q[mask], 1e-12, 1.0)))))


def _gini(values: np.ndarray) -> float:
    values = np.sort(np.asarray(values, dtype=float))
    if values.sum() == 0:
        return 0.0
    n = len(values)
    cumulative = np.cumsum(values)
    return float((n + 1 - 2 * (cumulative / cumulative[-1]).sum()) / n)


def _summaries(train_counts: np.ndarray) -> list[ClientSummary]:
    global_p = train_counts.sum(axis=0).astype(float)
    global_p /= max(1.0, global_p.sum())
    summaries: list[ClientSummary] = []
    for cid, row in enumerate(train_counts):
        n = int(row.sum())
        p = row.astype(float) / max(1, n)
        nonzero = p[p > 0]
        entropy = float(-np.sum(nonzero * np.log(nonzero))) if nonzero.size else 0.0
        midpoint = 0.5 * (p + global_p)
        summaries.append(ClientSummary(
            client_id=cid,
            n=n,
            num_classes_present=int(np.count_nonzero(row)),
            entropy=entropy,
            kl_to_global=_kl(p, global_p),
            js_to_global=0.5 * _kl(p, midpoint) + 0.5 * _kl(global_p, midpoint),
            gini=_gini(row),
        ))
    return summaries


def _write_summary(path: Path, train_counts: np.ndarray, context: dict[str, object]) -> None:
    rows = [{**context, "split": "train", **asdict(row)} for row in _summaries(train_counts)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_plots(report_dir: Path, counts_by_split: dict[str, np.ndarray]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    split_names = list(counts_by_split)
    n_splits = len(split_names)
    num_clients, num_classes = next(iter(counts_by_split.values())).shape
    height = max(4.5, num_clients * 0.38)

    fig, axes = plt.subplots(1, n_splits, figsize=(max(7, 6 * n_splits), height), squeeze=False,
                             sharey=True, constrained_layout=True)
    images = []
    for axis, split_name in zip(axes[0], split_names):
        image = axis.imshow(_percent(counts_by_split[split_name]), aspect="auto", vmin=0, vmax=1,
                            cmap="viridis", interpolation="nearest")
        images.append(image)
        axis.set_title(split_name.replace("_", " ").title())
        axis.set_xlabel("Class")
        axis.set_xticks(np.arange(num_classes))
        axis.set_xticklabels([str(i) for i in range(num_classes)], rotation=90, fontsize=7)
        axis.set_yticks(np.arange(num_clients))
    axes[0, 0].set_ylabel("Client ID")
    fig.colorbar(images[0], ax=axes.ravel().tolist(), label="Proportion", shrink=0.8)
    fig.savefig(report_dir / "distribution_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(n_splits, 1, figsize=(max(10, num_clients * 0.65), 4.2 * n_splits),
                             squeeze=False, constrained_layout=True)
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, num_classes))
    x = np.arange(num_clients)
    for axis, split_name in zip(axes[:, 0], split_names):
        bottom = np.zeros(num_clients)
        matrix = counts_by_split[split_name]
        for class_id in range(num_classes):
            axis.bar(x, matrix[:, class_id], bottom=bottom, color=colors[class_id],
                     label=str(class_id), width=0.85)
            bottom += matrix[:, class_id]
        axis.set_title(split_name.replace("_", " ").title())
        axis.set_xlabel("Client ID")
        axis.set_ylabel("Samples")
        axis.set_xticks(x)
    axes[0, 0].legend(title="Class", bbox_to_anchor=(1.01, 1), loc="upper left", ncol=2)
    fig.savefig(report_dir / "dist_counts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_partition_reports(
    partition_dir: Path,
    labels: np.ndarray,
    *,
    num_classes: int | None = None,
    include_plots: bool = True,
) -> Path:
    """Create CSV summaries and plots without modifying partition arrays."""
    partition_dir = Path(partition_dir)
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError("labels must be a one-dimensional array")
    inferred_classes = int(labels.max()) + 1 if labels.size else 0
    num_classes = int(num_classes if num_classes is not None else inferred_classes)
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    files = _client_files(partition_dir)
    splits = available_splits(partition_dir)
    context = _report_context(partition_dir)
    counts_by_split: dict[str, np.ndarray] = {}
    for split in splits:
        indices = []
        for path in files:
            with np.load(path, allow_pickle=False) as client:
                if split not in client.files:
                    raise ValueError(f"inconsistent client schema: {path} lacks {split}")
                indices.append(np.asarray(client[split], dtype=np.int64))
        name = split.removesuffix("_idx")
        counts = _counts(labels, indices, num_classes)
        counts_by_split[name] = counts
        _write_matrix(
            partition_dir / "report" / name / "counts.csv", counts,
            context=context, split=name,
        )
        _write_matrix(
            partition_dir / "report" / name / "percent.csv", _percent(counts),
            context=context, split=name,
        )

    train_counts = counts_by_split.get("train")
    if train_counts is None:
        raise ValueError(f"partition {partition_dir} has no train_idx arrays")
    report_dir = partition_dir / "report"
    _write_summary(report_dir / "summary.csv", train_counts, context)
    if include_plots:
        _write_plots(report_dir, counts_by_split)
    (report_dir / "metadata.json").write_text(json.dumps({
        "schema_version": 1,
        "source": "clients/c*.npz",
        **context,
        "num_clients": len(files),
        "num_classes": num_classes,
        "splits": list(counts_by_split),
    }, indent=2) + "\n", encoding="utf-8")
    return report_dir


def backfill_partition_reports(
    partitions_root: Path,
    dataset: str,
    data_dir: Path,
    *,
    include_plots: bool = True,
) -> list[Path]:
    """Backfill every persisted partition under a seed/dataset root."""
    labels = load_dataset_labels(dataset, data_dir)
    written: list[Path] = []
    for clients_dir in sorted(Path(partitions_root).rglob("clients")):
        partition_dir = clients_dir.parent
        if not report_complete(partition_dir):
            written.append(write_partition_reports(
                partition_dir, labels, include_plots=include_plots,
            ))
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partitions-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("mnist", "fmnist", "cifar", "cifar100"), required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    written = backfill_partition_reports(
        args.partitions_root, args.dataset, args.data_dir, include_plots=not args.no_plots,
    )
    print(f"[OK] reports written: {len(written)}")
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
