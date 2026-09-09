"""Editorial figures from published closure tables; no training or cache access."""

from __future__ import annotations

import hashlib
import json
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATASETS = ("mnist", "fmnist", "cifar")
REGIMES = ("iid", "alpha1p0", "alpha0p5", "alpha0p1", "multi", "single")
REGIME_LABELS = ("IID", "α=1", "α=.5", "α=.1", "Multi", "Single")
NAMES = {"mnist": "MNIST", "fmnist": "Fashion-MNIST", "cifar": "CIFAR-10"}
CONTRASTS = (
    "selection_logit",
    "feddf_pooling",
    "oracle_pooling",
    "expertise_gain",
    "oracle_expertise_gap",
    "support",
)
FILES = (
    "main_contrast_summary",
    "private_knowledge_summary",
    "proxy_curve_paired",
    "coverage",
)


def load_evidence(source: Path) -> dict:
    """Require complete published evidence, without claiming to re-audit caches."""
    source = Path(source)
    manifest = json.loads((source / "manifest.json").read_text())
    if any(
        manifest["closure"].get(k) != "cerrado"
        for k in ("masks", "baseline", "supervised", "proxy_curve")
    ):
        raise ValueError("Paper requires a closed main study and curve")
    data = {name: pd.read_csv(source / "tables" / f"{name}.csv") for name in FILES}
    for name, contrasts in (
        ("main_contrast_summary", CONTRASTS),
        ("private_knowledge_summary", (None,)),
    ):
        frame = data[name]
        for contrast in contrasts:
            rows = frame if contrast is None else frame[frame.contrast.eq(contrast)]
            for metric in ("student_test_accuracy", "student_test_nll"):
                key = metric if contrast is not None else "delta_" + metric
                selected = rows[rows.metric.eq(key)]
                expected = set(product(DATASETS, REGIMES))
                if (
                    len(selected) != 18
                    or set(zip(selected.dataset, selected.regime)) != expected
                ):
                    raise ValueError(
                        f"Incomplete or duplicate summary: {name}/{contrast}/{key}"
                    )
                if not (
                    selected.n.eq(3).all()
                    and selected.seeds.eq("42,43,44").all()
                    and np.isfinite(selected[["mean", "sd"]]).all().all()
                    and selected.sd.ge(0).all()
                ):
                    raise ValueError(
                        "Expected finite mean/sample SD and three paired seeds"
                    )
    curve = data["proxy_curve_paired"]
    keys = ["regime", "proxy_size", "seed"]
    expected = set(
        product(
            ("iid", "alpha0p1", "single"), (100, 500, 1000, 5000, 10000), (42, 43, 44)
        )
    )
    if (
        len(curve) != 45
        or set(map(tuple, curve[keys].values)) != expected
        or not curve.dataset.eq("cifar").all()
    ):
        raise ValueError("Expected 45 unique CIFAR pairs")
    for metric in ("student_test_accuracy", "student_test_nll"):
        columns = [metric + "_left", metric + "_right", "delta_" + metric]
        if not np.isfinite(curve[columns]).all().all() or not np.allclose(
            curve[columns[0]] - curve[columns[1]],
            curve[columns[2]],
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError("Invalid paired curve delta")
    coverage = data["coverage"]
    if len(coverage) != 54 or set(
        map(tuple, coverage[["dataset", "regime", "seed"]].values)
    ) != set(product(DATASETS, REGIMES, (42, 43, 44))):
        raise ValueError("Expected 54 source conditions")
    data["manifest"] = manifest
    data["source"] = source.resolve()
    return data


def summary_figure(rows: pd.DataFrame, panels: dict, metric: str):
    """Summary-only error bars: never invent individual seed observations."""
    scale = 100 if metric == "student_test_accuracy" else 1
    fig, axes = plt.subplots(
        len(panels), 3, figsize=(11, 2.8 * len(panels)), squeeze=False
    )
    colors = ("#2471A3", "#A04000", "#148F77")
    for i, (contrast, title) in enumerate(panels.items()):
        for j, dataset in enumerate(DATASETS):
            ax = axes[i, j]
            selected = rows[rows.dataset.eq(dataset) & rows.metric.eq(metric)]
            if contrast is not None:
                selected = selected[selected.contrast.eq(contrast)]
            selected = selected.set_index("regime").loc[list(REGIMES)]
            ax.errorbar(
                range(6),
                selected["mean"] * scale,
                yerr=selected.sd * scale,
                fmt="o",
                capsize=3,
                color=colors[j],
            )
            ax.axhline(0, color="0.5", linewidth=0.8)
            ax.set_xticks(range(6), REGIME_LABELS, rotation=25)
            ax.set_title(f"{NAMES[dataset]} | {title}", fontsize=10)
            ax.set_ylabel("Δ accuracy (pp)" if scale == 100 else "Δ NLL")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def curve_figure(curve: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), squeeze=False)
    for j, regime in enumerate(("iid", "alpha0p1", "single")):
        rows = curve[curve.regime.eq(regime)]
        for i, (metric, scale, label) in enumerate(
            (
                ("delta_student_test_accuracy", 100, "Δ accuracy (pp)"),
                ("delta_student_test_nll", 1, "Δ NLL"),
            )
        ):
            ax = axes[i, j]
            for seed, group in rows.groupby("seed"):
                group = group.sort_values("proxy_size")
                ax.plot(
                    group.proxy_size,
                    group[metric] * scale,
                    alpha=0.45,
                    linewidth=1,
                    label=str(seed),
                )
            summary = rows.groupby("proxy_size")[metric].agg(["mean", "std"])
            ax.errorbar(
                summary.index,
                summary["mean"] * scale,
                yerr=summary["std"] * scale,
                color="black",
                fmt="o-",
                capsize=3,
                label="Mean ± SD",
            )
            ax.axhline(0, color="0.5", linewidth=0.8)
            ax.set_xscale("log")
            ax.set_xlabel("Proxy examples N")
            ax.set_ylabel(label)
            ax.set_title(f"CIFAR-10 | {regime}")
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def export_paper(source: Path, output: Path) -> dict:
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Output must be separate from the source evidence directory")
    data = load_evidence(source)
    output.mkdir(parents=True, exist_ok=True)
    main = data["main_contrast_summary"]
    panels = {
        "routing": {
            "selection_logit": "ORACLE − FedDF (logit)",
            "expertise_gain": "EXPERT − FedDF (prob)",
        },
        "pooling": {
            "feddf_pooling": "FedDF: prob − logit",
            "oracle_pooling": "ORACLE: prob − logit",
        },
        "support": {"support": "EXPERT: SR − full"},
    }
    figs = {}
    for name, mapping in panels.items():
        for metric in ("student_test_accuracy", "student_test_nll"):
            figs[f"{name}_{metric}"] = summary_figure(main, mapping, metric)
    ce = data["private_knowledge_summary"].copy()
    ce["metric"] = ce.metric.str.removeprefix("delta_")
    for metric in ("student_test_accuracy", "student_test_nll"):
        figs[f"public_labels_{metric}"] = summary_figure(
            ce, {None: "EXPERT-prob − CE"}, metric
        )
    figs["proxy_curve"] = curve_figure(data["proxy_curve_paired"])
    for name, fig in figs.items():
        for ext in ("png", "pdf"):
            fig.savefig(output / f"{name}.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig)
    # Keep numerical units explicit in machine-readable tables.
    table = main[main.metric.isin(("student_test_accuracy", "student_test_nll"))].copy()
    table["unit"] = np.where(
        table.metric.eq("student_test_accuracy"), "percentage points", "NLL"
    )
    for col in ("mean", "sd"):
        table.loc[table.metric.eq("student_test_accuracy"), col] *= 100
    table.to_csv(output / "main_effects.csv", index=False)
    data["private_knowledge_summary"].to_csv(
        output / "public_label_comparison.csv", index=False
    )
    data["proxy_curve_paired"].to_csv(output / "proxy_curve_pairs.csv", index=False)
    inputs = [source / "manifest.json"] + [
        source / "tables" / f"{f}.csv" for f in FILES
    ]
    provenance = {
        "source_snapshot": data["manifest"]["snapshot_date"],
        "source_analysis_commit": data["manifest"]["analysis_commit"],
        "scope": "Editorial rendering of published summaries and curve pairs; no cache re-audit.",
        "inputs": {
            str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in inputs
        },
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return {"figures": list(figs), "output": str(output), "provenance": provenance}
