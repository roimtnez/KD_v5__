"""CSV validation, paired contrasts and figures for the one Article-1 notebook."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from article1 import DATASETS, PROTOCOL_VERSION, REGIMES, SEEDS, THRESHOLDS
from article1.experiments import ANALYSIS_BLOCKS, T8_BLOCKS
from article1.distillation import METHODS

BASELINE_DESIGNS = {
    'six': ('feddf_prob', 'expert_prob', 'expert_prob_sr', 'oracle_prob', 'feddf_logit', 'oracle_logit'),
    'ten': tuple(METHODS),
}


def baseline_design(frame, design='auto'):
    """Infer the candidate design from actual methods, then require its full grid.

    This inference never makes an incomplete grid acceptable. An explicitly
    requested historical design cannot be reduced to fit the observed rows.
    """
    require(design in ('auto', *BASELINE_DESIGNS), 'Unknown baseline design')
    observed = set(frame.method) if 'method' in frame else set()
    require(observed and observed <= set(METHODS), 'Missing or unknown baseline methods')
    if design == 'auto':
        design = 'six' if observed <= set(BASELINE_DESIGNS['six']) else 'ten'
    require(observed <= set(BASELINE_DESIGNS[design]), f'Unexpected methods for {design}-method design')
    return design

KEY = ["dataset", "regime", "seed"]
IDENTITY = KEY + ["method", "temperature"]
CRN = [
    "cache_sha256",
    "M_sha256",
    "proxy_sha256",
    "student_init_sha256",
    "batch_order_sha256",
    "updates",
    "consumed_batches_sha256",
    "proxy_labels_sha256",
    "proxy_master_sha256",
    "training_recipe_json",
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
        equal = result[field + "_left"].notna() & result[field + "_left"].eq(result[field + "_right"])
        if field == SUPPORT_MASS:
            # Undefined only when no expert was ever selected on either arm.
            equal |= (result[field+'_left'].isna() & result[field+'_right'].isna()
                      & result.fallback_rate_left.eq(1) & result.fallback_rate_right.eq(1))
        require(equal.all(), f"Pair mismatch for {left}/{right}: {field}")
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


def load_results(out: Path, stage: str = "rq1", *, design: str = 'auto') -> dict:
    """Validate CSV coverage and recorded provenance; this is NOT a cache audit."""
    from itertools import product

    out = Path(out)
    require(stage in ANALYSIS_BLOCKS, f"Unknown analysis stage: {stage}")
    input_blocks = [T8_BLOCKS[name] for name in ANALYSIS_BLOCKS[stage]]
    detected_design = None
    baseline_frame = None
    if stage == 'baseline':
        baseline_frame = pd.read_csv(out/'results_baseline.csv')
        detected_design = baseline_design(baseline_frame, design)
        input_blocks = [('results_baseline.csv', BASELINE_DESIGNS[detected_design])]
    frames = []
    for filename, methods in input_blocks:
        frame = baseline_frame.copy() if baseline_frame is not None else pd.read_csv(out / filename)
        require(
            "method" in frame and frame.method.isin(methods).all(),
            f"Unexpected methods in {filename}",
        )
        require(
            "temperature" in frame and frame.temperature.eq(8).all(),
            f"Expected only T=8 in {filename}",
        )
        if "expert_prob_sr" in methods:
            require(
                "target_revision" in frame
                and frame.loc[frame.method.eq("expert_prob_sr"), "target_revision"]
                .eq(2)
                .all(),
                "Support results require stable target revision 2",
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
            "baseline_design": detected_design,
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
    return context


def comparisons(context: dict) -> dict[str, pd.DataFrame]:
    t8 = context["t8"]
    result = {"selection_logit": paired(t8, "oracle_logit", "feddf_logit")}
    available = set(t8.method)
    if "feddf_prob" in available:
        result["feddf_pooling"] = paired(
            t8, "feddf_prob", "feddf_logit", fields=CRN + ROUTING
        )
        result["oracle_pooling"] = paired(
            t8, "oracle_prob", "oracle_logit", fields=CRN + ROUTING
        )
    if "expert_prob" in available:
        result["expertise_gain"] = paired(t8, "expert_prob", "feddf_prob")
        result["oracle_expertise_gap"] = paired(t8, "oracle_prob", "expert_prob")
    if "expert_prob_sr" in available:
        result["support"] = paired(
            t8, "expert_prob_sr", "expert_prob", fields=CRN + ROUTING + [SUPPORT_MASS]
        )
    if {"expert_prob", "expert_logit"} <= available:
        result["expert_pooling"] = paired(
            t8, "expert_prob", "expert_logit", fields=CRN + ROUTING
        )
    for control in ("confidence_logit", "consensus_logit", "energy_logit"):
        if control in available:
            result[control] = paired(t8, control, "feddf_logit")
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


def focal_comparisons(out: Path, temperatures=(8,)) -> pd.DataFrame:
    from itertools import product

    from article1.experiments import FOCAL_REGIMES

    out = Path(out)
    paths = [out / T8_BLOCKS["expertise"][0], out / "results_expert_logit_focal.csv"]
    if set(temperatures) != {8}:
        paths.append(out / "results_expert_temperature.csv")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    require(
        frame.protocol_version.eq(PROTOCOL_VERSION).all(), "Incompatible focal protocol"
    )
    frame = frame[
        frame.dataset.eq("cifar")
        & frame.regime.isin(FOCAL_REGIMES)
        & frame.method.isin(["expert_logit", "expert_prob"])
        & frame.temperature.isin(temperatures)
    ]
    expected = set(
        product(
            ["cifar"],
            FOCAL_REGIMES,
            SEEDS,
            ["expert_logit", "expert_prob"],
            temperatures,
        )
    )
    require(
        len(frame) == len(expected)
        and set(map(tuple, frame[IDENTITY].values)) == expected,
        "Incomplete focal EXPERT operator comparison",
    )
    require(frame.updates.eq(1200).all(), "Unexpected focal budget")
    return paired(frame, "expert_prob", "expert_logit", fields=CRN + ROUTING)


def definitive_snapshot(snapshot: Path, design='auto') -> dict:
    """Close a complete main design using the existing strict loader and pairer.

    Optional archival contrasts remain separate; they cannot hold a completed
    primary design open, and never become additional primary replications.
    """
    import json
    from article1.progress import (read_csv, save_table, save_figure, effect_plot,
                                   audit_conditions, progress, export_ready_blocks,
                                   pipeline_dependencies)
    out = Path(snapshot)
    audit_conditions(out)
    sources = read_csv(out/'tables/source_inventory.csv')
    require(len(sources) == 54 and sources.status.eq('present').all(), 'Incomplete source audit')
    context = load_results(out/'snapshots', stage='baseline', design=design)
    checked = read_csv(out/'tables/configured_baseline_validated.csv')
    require(len(checked) == len(context['t8']) and checked.valid.all(), 'Invalid main baseline rows')
    require(set(checked.run_id) == set(context['t8'].run_id), 'Snapshot identities differ')
    effects = comparisons(context)
    summaries = []
    for name, effect in effects.items():
        save_table(out, name+'_paired', effect)
        for metric in METRICS:
            stats = summarize(effect, 'delta_'+metric)
            stats['seeds'] = '42,43,44'  # Full-grid loader required these three seeds.
            stats['contrast'], stats['metric'] = name, metric
            summaries.append(stats)
        for metric in METRICS[:2]:
            effect_plot(out, name+'_'+metric, effect, 'delta_'+metric)
    save_table(out, 'main_contrast_summary', pd.concat(summaries, ignore_index=True))
    for metric in METRICS[:2]:
        fig, axes = _axes(metric)
        for ax, dataset in zip(axes, DATASETS):
            for method in context['t8'].method.unique():
                rows = context['t8'][context['t8'].dataset.eq(dataset) & context['t8'].method.eq(method)]
                color = ax._get_lines.get_next_color()
                _series(ax, rows, metric, color, label=method)
        axes[0].legend(fontsize=7)
        save_figure(out, 'main_'+metric, fig)
    save_table(out, 'main_target_routing', context['t8'][IDENTITY+METRICS+ROUTING+[SUPPORT_MASS]])
    manifest = json.loads((out/'manifest.json').read_text())
    manifest['main_analysis'] = dict(status='closed', **context['validation'],
                                     pairs={name: len(effect) for name, effect in effects.items()})
    manifest['exported_blocks'] = export_ready_blocks(out, read_csv(out/'tables/baseline_validated.csv'))
    manifest['pipeline_dependencies'] = pipeline_dependencies(out)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    # Reuse the partial-pair path solely for methods outside the closed primary design.
    progress(out, optional_only=True, primary_methods=set(context['t8'].method))
    return context['validation']


def expertise_relationships(snapshot):
    """Descriptive condition-level plots, without fitting or causal attribution."""
    from article1.progress import read_csv, save_table, save_figure
    out = Path(snapshot)
    cells = read_csv(out/'tables/expertise_cells.csv')
    coverage = read_csv(out/'tables/coverage.csv')
    support = cells[cells.M.eq(1)].groupby(KEY).expertise_count.median().rename('median_selected_support').reset_index()
    effects = read_csv(out/'tables/expertise_gain_paired.csv')
    if effects.empty:
        return
    rows = effects.merge(coverage, on=KEY, validate='one_to_one').merge(support, on=KEY, validate='one_to_one')
    save_table(out, 'coverage_support_results', rows)
    for feature in ('M_density', 'median_selected_support'):
        for metric in ('delta_student_test_accuracy', 'delta_student_test_nll'):
            fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
            for ax, dataset in zip(axes, DATASETS):
                selected = rows[rows.dataset.eq(dataset)]
                for regime in REGIMES:
                    group = selected[selected.regime.eq(regime)]
                    ax.scatter(group[feature], group[metric], label=regime, alpha=.75)
                ax.set(title=dataset, xlabel=feature, ylabel=metric)
            axes[0].legend(fontsize=7)
            fig.suptitle('EXPERT-prob − FedDF-prob; relaciones descriptivas, sin ajuste causal')
            save_figure(out, feature+'_'+metric, fig)
