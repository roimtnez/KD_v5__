"""CSV validation, paired contrasts and figures for the one Article-1 notebook."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from article1 import DATASETS, PROTOCOL_VERSION, REGIMES, SEEDS, THRESHOLDS
from article1.experiments import ANALYSIS_BLOCKS, T8_BLOCKS

KEY = ["dataset", "regime", "seed"]
IDENTITY = KEY + ["method", "temperature"]
CRN = [
    "cache_sha256",
    "M_sha256",
    "proxy_sha256",
    "student_init_sha256",
    "batch_order_sha256",
    "updates",
]
ROUTING = [
    "fallback_count",
    "fallback_rate",
    "mean_selected_teachers",
    "effective_teachers",
]
METRICS = [
    "student_test_accuracy",
    "student_test_nll",
    "target_accuracy",
    "target_nll",
    "target_entropy",
]
SUPPORT_MASS = "pre_restriction_outside_support_mass"
DATASET_LABELS = {"mnist": "MNIST", "fmnist": "Fashion-MNIST", "cifar": "CIFAR-10"}
REGIME_LABELS = ["IID", "α=1.0", "α=0.5", "α=0.1", "Multi", "Single"]
METHOD_LABELS = {
    "feddf_logit": "FedDF",
    "expert_logit": "EXPERT",
    "oracle_logit": "ORACLE",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def paired(
    frame: pd.DataFrame, left: str, right: str, *, fields=CRN, metrics=METRICS
) -> pd.DataFrame:
    """Return left minus right; reject missing partners and recorded CRN mismatches."""
    keys = KEY + ["temperature"]
    a = frame[frame.method.eq(left)].set_index(keys)
    b = frame[frame.method.eq(right)].set_index(keys)
    require(
        len(a) > 0
        and a.index.is_unique
        and b.index.is_unique
        and set(a.index) == set(b.index),
        f"Incomplete or duplicate pairs: {left} / {right}",
    )
    result = a.join(b, lsuffix="_left", rsuffix="_right")
    for field in fields:
        require(
            result[field + "_left"].notna().all()
            and result[field + "_left"].eq(result[field + "_right"]).all(),
            f"Pair mismatch for {left}/{right}: {field}",
        )
    for metric in metrics:
        require(
            np.isfinite(
                result[[metric + "_left", metric + "_right"]].to_numpy(float)
            ).all(),
            f"Nonfinite metric: {metric}",
        )
        result["delta_" + metric] = result[metric + "_left"] - result[metric + "_right"]
    if "student_test_accuracy" in metrics:
        result["delta_accuracy_pp"] = 100 * result.delta_student_test_accuracy
    return result.reset_index()


def summarize(
    frame: pd.DataFrame, column: str = "delta_accuracy_pp", groups=KEY[:2]
) -> pd.DataFrame:
    """Mean and sample SD across paired seeds, never across clients."""
    return (
        frame.groupby(groups, observed=True)[column]
        .agg(mean="mean", sd="std", n="size")
        .reset_index()
    )


def load_results(out: Path, stage: str = "rq1") -> dict:
    """Validate CSV coverage and recorded provenance; this is NOT a cache audit."""
    from itertools import product

    out = Path(out)
    require(stage in ANALYSIS_BLOCKS, f"Unknown analysis stage: {stage}")
    input_blocks = [T8_BLOCKS[name] for name in ANALYSIS_BLOCKS[stage]]
    frames = []
    for filename, methods in input_blocks:
        frame = pd.read_csv(out / filename)
        require(
            "method" in frame and frame.method.isin(methods).all(),
            f"Unexpected methods in {filename}",
        )
        require(
            "temperature" in frame and frame.temperature.eq(8).all(),
            f"Expected only T=8 in {filename}",
        )
        frames.append(frame)
    main = pd.concat(frames, ignore_index=True)
    conditions = pd.read_csv(out / "conditions.csv")
    require(
        "protocol_version" in main and main.protocol_version.eq(PROTOCOL_VERSION).all(),
        "Results must belong exclusively to article1-v3; historical v2 remains separate",
    )
    require(
        "protocol_version" in conditions
        and conditions.protocol_version.eq(PROTOCOL_VERSION).all(),
        "Conditions must belong exclusively to article1-v3",
    )
    required = set(IDENTITY + CRN + METRICS + ROUTING + ["run_id", SUPPORT_MASS])
    require(
        required <= set(main), f"Missing result columns: {sorted(required - set(main))}"
    )
    require(
        not main.duplicated(IDENTITY).any() and not main.run_id.duplicated().any(),
        "Duplicate result identities",
    )
    t8 = main[main.temperature.eq(8)].copy()
    methods = [m for _, block_methods in input_blocks for m in block_methods]
    expected = set(product(DATASETS, REGIMES, SEEDS, methods, [8.0]))
    require(
        len(t8) == len(expected) and set(map(tuple, t8[IDENTITY].values)) == expected,
        f"Incomplete T=8 grid for {stage}: expected {len(expected)} rows",
    )
    require(
        len(conditions) == 54
        and set(map(tuple, conditions[KEY].values))
        == set(product(DATASETS, REGIMES, SEEDS)),
        "Expected 54 unique dataset/regime/seed conditions",
    )
    require(
        t8[CRN].notna().all().all()
        and t8.groupby(KEY)[CRN].nunique().eq(1).all().all(),
        "Main-grid CRN mismatch",
    )
    require(
        np.isfinite(t8[METRICS + ROUTING].to_numpy(float)).all()
        and t8.updates.eq(1200).all(),
        "Nonfinite metrics or unexpected KD budget",
    )
    for field in ["student_test_accuracy", "target_accuracy", "fallback_rate"]:
        require(t8[field].between(0, 1).all(), f"Invalid range: {field}")
    require(
        np.allclose(t8.fallback_count, 10000 * t8.fallback_rate, atol=1e-8, rtol=0),
        "Fallback count/rate mismatch",
    )
    joined = t8.merge(conditions, on=KEY, validate="many_to_one")
    require(
        joined.proxy_sha256_x.eq(joined.proxy_sha256_y).all(),
        "Conditions/results proxy mismatch",
    )
    require(
        conditions.expertise_threshold.eq(conditions.dataset.map(THRESHOLDS)).all(),
        "Thresholds differ from the fixed protocol",
    )
    require(
        np.allclose(conditions.M_density * 10, conditions.experts_per_class_mean),
        "Mask density/coverage mismatch",
    )
    context = {
        "out": out,
        "input_blocks": input_blocks,
        "t8": t8,
        "conditions": conditions,
        "temperature": None,
        "validation": {
            "level": "CSV structure and recorded pairing ONLY; caches not checked",
            "main_t8_rows": len(t8),
            "main_non_t8_rows_excluded": len(main) - len(t8),
            "missing_student_final_hashes": int(
                t8.get("student_final_sha256", pd.Series(index=t8.index, dtype=object))
                .isna()
                .sum()
            ),
        },
    }
    path = out / "results_rq2_temperature.csv"
    if stage == "temperature":
        extra = pd.read_csv(path)
        require(
            extra.dataset.eq("cifar").all()
            and extra.regime.isin(["iid", "alpha0p1", "single"]).all()
            and extra.method.isin(["expert_logit", "expert_prob"]).all()
            and extra.temperature.isin([1.0, 4.0]).all(),
            "Foreign rows in isolated temperature CSV",
        )
        combined = pd.concat([main, extra], ignore_index=True)
        require(
            not combined.duplicated(IDENTITY).any()
            and not combined.run_id.duplicated().any(),
            "Duplicate rows across temperature sources",
        )
        temp = combined[
            combined.dataset.eq("cifar")
            & combined.regime.isin(["iid", "alpha0p1", "single"])
            & combined.method.isin(["expert_logit", "expert_prob"])
            & combined.temperature.isin([1.0, 4.0, 8.0])
        ].copy()
        expected_temp = set(
            product(
                ["cifar"],
                ["iid", "alpha0p1", "single"],
                SEEDS,
                ["expert_logit", "expert_prob"],
                [1.0, 4.0, 8.0],
            )
        )
        require(
            len(temp) == 54 and set(map(tuple, temp[IDENTITY].values)) == expected_temp,
            "Incomplete 54-run temperature comparison",
        )
        require(temp.updates.eq(1200).all(), "Unexpected temperature KD budget")
        paired(temp, "expert_prob", "expert_logit", fields=CRN + ROUTING)
        context["temperature"] = temp
    return context


def comparisons(context: dict) -> dict[str, pd.DataFrame]:
    t8 = context["t8"]
    result = {
        "routing": paired(t8, "expert_logit", "feddf_logit"),
        "oracle_gap": paired(t8, "oracle_logit", "expert_logit"),
    }
    for control in ["confidence_logit", "consensus_logit", "energy_logit"]:
        result[control] = paired(t8, "expert_logit", control)
    if t8.method.eq("expert_prob").any():
        result["aggregation"] = paired(
            t8, "expert_prob", "expert_logit", fields=CRN + ROUTING
        )
    if t8.method.eq("expert_prob_sr").any():
        result["support"] = paired(
            t8, "expert_prob_sr", "expert_prob", fields=CRN + ROUTING + [SUPPORT_MASS]
        )
    if context["temperature"] is not None:
        result["temperature"] = paired(
            context["temperature"], "expert_prob", "expert_logit", fields=CRN + ROUTING
        )
    return result


def routing_trajectories(frame: pd.DataFrame) -> pd.DataFrame:
    table = frame.pivot(
        index=["dataset", "seed"], columns="regime", values="delta_accuracy_pp"
    )[list(REGIMES)]
    table["nondecreasing"] = (np.diff(table, axis=1) >= 0).all(axis=1)
    return table.reset_index()


def support_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(KEY[:2])
        .agg(
            full_accuracy_pct=("student_test_accuracy_right", lambda x: 100 * x.mean()),
            sr_accuracy_pct=("student_test_accuracy_left", lambda x: 100 * x.mean()),
            delta_pp=("delta_accuracy_pp", "mean"),
            sd_pp=("delta_accuracy_pp", "std"),
            n=("seed", "size"),
            full_student_nll=("student_test_nll_right", "mean"),
            sr_student_nll=("student_test_nll_left", "mean"),
            delta_target_nll=("delta_target_nll", "mean"),
            delta_target_entropy=("delta_target_entropy", "mean"),
            outside_support_mass=(SUPPORT_MASS + "_left", "mean"),
        )
        .reindex(pd.MultiIndex.from_product([DATASETS, REGIMES], names=KEY[:2]))
        .reset_index()
    )


def support_counts(frame: pd.DataFrame) -> dict:
    # Descriptive counts, not 54 IID replicates; ignore floating-point zero noise.
    better = frame.delta_target_nll < -1e-12
    worse = frame.delta_accuracy_pp < -1e-12
    return {
        "pairs": len(frame),
        "target_nll_improves": int(better.sum()),
        "student_accuracy_worsens": int(worse.sum()),
        "both": int((better & worse).sum()),
        "student_nll_worsens": int((frame.delta_student_test_nll > 1e-12).sum()),
    }


def _axes(ylabel: str):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True, layout="constrained")
    for ax, dataset in zip(axes, DATASETS):
        ax.set_title(DATASET_LABELS[dataset])
        ax.set_xticks(range(len(REGIMES)), REGIME_LABELS, rotation=25)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel(ylabel)
    return fig, axes


def _series(ax, rows: pd.DataFrame, column: str, color: str, scale=1.0, label=None):
    for _, seed_rows in rows.groupby("seed"):
        values = seed_rows.set_index("regime").loc[list(REGIMES), column] * scale
        ax.plot(
            range(len(REGIMES)),
            values,
            color=color,
            alpha=0.25,
            lw=0.7,
            marker="o",
            ms=3,
        )
    stats = (
        rows.groupby("regime")[column].agg(["mean", "std"]).loc[list(REGIMES)] * scale
    )
    ax.errorbar(
        range(len(REGIMES)),
        stats["mean"],
        yerr=stats["std"],
        color=color,
        marker="o",
        capsize=3,
        label=label,
    )


def plot_accuracy(t8: pd.DataFrame):
    fig, axes = _axes("Student test accuracy (%)")
    for ax, dataset in zip(axes, DATASETS):
        for method, color in zip(METHOD_LABELS, ["#333333", "#0072B2", "#D55E00"]):
            rows = t8[t8.dataset.eq(dataset) & t8.method.eq(method)]
            _series(
                ax,
                rows,
                "student_test_accuracy",
                color,
                scale=100.0,
                label=METHOD_LABELS[method],
            )
        ax.set_ylim(0, 100)
    axes[0].legend(frameon=False)
    return fig


def plot_effect(frame: pd.DataFrame, ylabel: str, column="delta_accuracy_pp"):
    fig, axes = _axes(ylabel)
    for ax, dataset in zip(axes, DATASETS):
        _series(ax, frame[frame.dataset.eq(dataset)], column, "#0072B2")
        ax.axhline(0, color="black", lw=0.8)
    return fig


def plot_temperature(frame: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for regime, label, color in zip(
        ["iid", "alpha0p1", "single"],
        ["IID", "α=0.1", "Single"],
        ["#0072B2", "#D55E00", "#009E73"],
    ):
        rows = frame[frame.regime.eq(regime)]
        stats = rows.groupby("temperature").delta_accuracy_pp.agg(["mean", "std"])
        ax.scatter(rows.temperature, rows.delta_accuracy_pp, color=color, alpha=0.5)
        ax.errorbar(
            stats.index,
            stats["mean"],
            yerr=stats["std"],
            color=color,
            marker="o",
            capsize=3,
            label=label,
        )
    ax.axhline(0, color="black", lw=0.8)
    ax.set(
        xlabel="Temperature T",
        ylabel="EXPERT-prob − EXPERT-logit (pp)",
        xticks=[1, 4, 8],
    )
    ax.legend(frameon=False)
    return fig


def export(out: Path, tables: dict[str, pd.DataFrame], figures: dict) -> None:
    """Only derived artifacts are written; never overwrite input results CSVs."""
    for folder in ["tables", "figures"]:
        (Path(out) / folder).mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(Path(out) / "tables" / f"{name}.csv", index=False)
    for name, fig in figures.items():
        for extension in ["png", "pdf"]:
            fig.savefig(
                Path(out) / "figures" / f"{name}.{extension}",
                dpi=180,
                bbox_inches="tight",
            )
