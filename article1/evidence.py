"""Read-only Article-1 RQ1/RQ2 analysis and export utilities."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from article1.audit import audit
from article1.rq2 import validate_cells

DATASETS = ("mnist", "fmnist", "cifar")
HETEROGENEITY = ("iid", "alpha1p0", "alpha0p5", "alpha0p1", "multi", "single")
RQ2_REGIMES = ("iid", "alpha0p1", "single")
SEEDS = (42, 43, 44)
CRN = ("cache_sha256", "M_sha256", "proxy_sha256", "student_init_sha256", "batch_order_sha256", "updates")
EXPERT_ROUTING = ("fallback_rate", "mean_selected_teachers")
DATASET_LABEL = {"mnist": "MNIST", "fmnist": "Fashion-MNIST", "cifar": "CIFAR-10"}
REGIME_LABEL = {"iid": "IID", "alpha1p0": "α=1.0", "alpha0p5": "α=0.5", "alpha0p1": "α=0.1", "multi": "Multi", "single": "Single"}
COLORS = {"FedDF": "#333333", "EXPERT": "#0072B2", "ORACLE": "#D55E00", "IID": "#0072B2", "α=0.1": "#D55E00", "Single": "#009E73"}


def _root(root: Path | None = None) -> Path:
    if root is not None: return Path(root)
    return next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "OUTPUTS" / "article1" / "results.csv").is_file())


def _finite(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    if not np.isfinite(frame[columns].to_numpy(float)).all():
        raise RuntimeError(f"{label} has missing or non-finite metrics")


def _summary(frame: pd.DataFrame, groups: list[str], value: str) -> pd.DataFrame:
    return frame.groupby(groups, observed=True)[value].agg(mean="mean", sd=lambda x: x.std(ddof=1), n="count", minimum="min", maximum="max").reset_index()


def _paired(frame: pd.DataFrame, left: str, right: str, *, keys: list[str], metrics: list[str], paired_fields: tuple[str, ...] = CRN + EXPERT_ROUTING) -> pd.DataFrame:
    a = frame.loc[frame.method.eq(left), keys + metrics + list(paired_fields)].copy()
    b = frame.loc[frame.method.eq(right), keys + metrics + list(paired_fields)].copy()
    if a.duplicated(keys).any() or b.duplicated(keys).any():
        raise RuntimeError(f"duplicate rows in {left}/{right} pairing")
    joined = a.merge(b, on=keys, suffixes=("_left", "_right"), validate="one_to_one")
    unequal = [field for field in paired_fields if not joined[f"{field}_left"].eq(joined[f"{field}_right"]).all()]
    if unequal: raise RuntimeError(f"CRN mismatch for {left}/{right}: {unequal}")
    for metric in metrics:
        joined[f"delta_{metric}"] = joined[f"{metric}_left"] - joined[f"{metric}_right"]
    return joined


def load_and_validate(root: Path | None = None) -> dict:
    root = _root(root); out = root / "OUTPUTS" / "article1"
    main_path, rq2_path, conditions_path = out / "results.csv", out / "results_rq2_temperature.csv", out / "conditions.csv"
    main, new_rq2, conditions = pd.read_csv(main_path), pd.read_csv(rq2_path), pd.read_csv(conditions_path)
    required = {"dataset", "regime", "seed", "method", "temperature", "student_test_accuracy", "student_test_nll", "target_accuracy", "target_nll", "target_entropy", *CRN}
    if required - set(main) or required - set(new_rq2): raise RuntimeError("results CSV lacks the required canonical schema")
    t8 = main.loc[main.temperature.eq(8.0)].copy()
    if len(t8) != 486: raise RuntimeError(f"main T=8 grid has {len(t8)} rows, expected 486")
    if t8.duplicated(["dataset", "regime", "seed", "method", "temperature"]).any(): raise RuntimeError("duplicate main T=8 identity")
    if len(conditions) != 54 or conditions.duplicated(["dataset", "regime", "seed"]).any(): raise RuntimeError("conditions CSV must contain 54 unique conditions")
    _finite(t8, ["student_test_accuracy", "student_test_nll", "target_accuracy", "target_nll", "target_entropy", "fallback_rate", "mean_selected_teachers"], "main T=8")
    expert_t8 = _paired(t8, "expert_prob", "expert_logit", keys=["dataset", "regime", "seed"], metrics=["student_test_accuracy", "student_test_nll", "target_accuracy", "target_nll", "target_entropy"])
    if len(expert_t8) != 54: raise RuntimeError("expected 54 complete EXPERT pairs at T=8")
    # Full cache-level validation establishes immutable-cache correspondence and
    # reconstructs all 486 T=8 targets, including EXPERT routing/fallbacks.
    integrity = audit(main_path, source_root=out / "sources")
    if not integrity["ok"]: raise RuntimeError(f"main cache audit failed: {integrity['issues']}")
    rq2 = validate_cells([main_path, rq2_path], source_root=out / "sources", temperatures=(1.0, 4.0, 8.0), reject_foreign=rq2_path)
    if not rq2["ok"] or rq2["counts"].get("valid_reusable") != 54:
        raise RuntimeError(f"RQ2 temperature integrity failed: {rq2}")
    temp = pd.concat([main, new_rq2], ignore_index=True)
    temp = temp.loc[(temp.dataset.eq("cifar")) & temp.regime.isin(RQ2_REGIMES) & temp.seed.isin(SEEDS) & temp.method.isin(("expert_logit", "expert_prob")) & temp.temperature.isin((1.0, 4.0, 8.0))].copy()
    if len(temp) != 54 or temp.duplicated(["dataset", "regime", "seed", "method", "temperature"]).any(): raise RuntimeError("RQ2 sensitivity must contain 54 unique rows")
    _finite(temp, ["student_test_accuracy", "student_test_nll", "target_accuracy", "target_nll", "target_entropy", "fallback_rate", "mean_selected_teachers"], "RQ2 sensitivity")
    return {"root": root, "out": out, "t8": t8, "conditions": conditions, "expert_t8": expert_t8, "temperature": temp, "integrity": integrity, "rq2_integrity": rq2}


def _pp_summary(frame: pd.DataFrame, groups: list[str], value: str) -> pd.DataFrame:
    rows = []
    for key, values in frame.groupby(groups, observed=True)[value]:
        key = key if isinstance(key, tuple) else (key,)
        v = values.to_numpy(float); absolute = np.abs(v)
        rows.append(dict(zip(groups, key), mean_pp=float(v.mean()), sd_pp=float(v.std(ddof=1)) if len(v) > 1 else np.nan,
                         mae_pp=float(absolute.mean()), median_abs_pp=float(np.median(absolute)), max_abs_pp=float(absolute.max()),
                         within_0_5pp=int((absolute <= .5).sum()), within_1pp=int((absolute <= 1).sum()), n=int(len(v)),
                         positives=int((v > 0).sum()), negatives=int((v < 0).sum()), ties=int((v == 0).sum())))
    return pd.DataFrame(rows)


def rq1_tables(context: dict) -> dict[str, pd.DataFrame]:
    t8 = context["t8"]; keys = ["dataset", "regime", "seed"]
    gain = _paired(t8, "expert_logit", "feddf_logit", keys=keys, metrics=["student_test_accuracy"], paired_fields=CRN)
    gain["expert_minus_feddf_pp"] = 100 * gain.pop("delta_student_test_accuracy")
    trajectory = []
    for (dataset, seed), rows in gain.groupby(["dataset", "seed"], observed=True):
        values = rows.set_index("regime").loc[list(HETEROGENEITY), "expert_minus_feddf_pp"].to_numpy()
        changes = np.diff(values)
        trajectory.append({"dataset": dataset, "seed": seed, "nondecreasing": bool(np.all(changes >= 0)),
                           "decreasing_transitions": json.dumps([f"{HETEROGENEITY[i]}→{HETEROGENEITY[i + 1]}" for i, delta in enumerate(changes) if delta < 0]),
                           **{f"delta_{regime}_pp": value for regime, value in zip(HETEROGENEITY, values)}})
    effects = gain.copy()
    effect_summary = _summary(gain, ["dataset", "regime"], "expert_minus_feddf_pp").rename(columns={"mean": "mean_pp", "sd": "sd_pp", "minimum": "min_pp", "maximum": "max_pp"})
    effect_summary["positive_seeds"] = effect_summary.apply(lambda r: int((gain.loc[(gain.dataset == r.dataset) & (gain.regime == r.regime), "expert_minus_feddf_pp"] > 0).sum()), axis=1)
    oracle = _paired(t8, "oracle_logit", "expert_logit", keys=keys, metrics=["student_test_accuracy"], paired_fields=CRN)
    oracle["oracle_minus_expert_pp"] = 100 * oracle.pop("delta_student_test_accuracy")
    oracle["abs_gap_pp"] = oracle.oracle_minus_expert_pp.abs()
    gap_rows = []
    for name, values in [("overall", oracle.oracle_minus_expert_pp), *[((f"{dataset}/{regime}"), group.oracle_minus_expert_pp) for (dataset, regime), group in oracle.groupby(["dataset", "regime"], observed=True)]]:
        a = values.abs(); gap_rows.append({"scope": name, "mean_signed_pp": values.mean(), "mae_pp": a.mean(), "median_abs_pp": a.median(), "max_abs_pp": a.max(), "within_1pp": int((a <= 1).sum()), "expert_exceeds_oracle": int((values < 0).sum()), "n": len(values)})
    accuracy = _summary(t8.loc[t8.method.isin(("feddf_logit", "expert_logit", "oracle_logit"))], ["dataset", "regime", "method"], "student_test_accuracy")
    controls = _summary(t8.loc[t8.method.isin(("feddf_logit", "confidence_logit", "consensus_logit", "energy_logit", "expert_logit", "oracle_logit"))], ["dataset", "regime", "method"], "student_test_accuracy")
    control_pairs = []
    for control in ("confidence_logit", "consensus_logit", "energy_logit"):
        comparison = _paired(t8, "expert_logit", control, keys=keys, metrics=["student_test_accuracy"], paired_fields=CRN)
        comparison["control"] = control
        comparison["expert_minus_control_pp"] = 100 * comparison.pop("delta_student_test_accuracy")
        control_pairs.append(comparison[[*keys, "control", "expert_minus_control_pp"]])
    return {"accuracy": accuracy, "effects": effects, "effects_summary": effect_summary, "trajectory": pd.DataFrame(trajectory), "oracle_gap": pd.DataFrame(gap_rows), "controls": controls, "control_pairs": pd.concat(control_pairs, ignore_index=True), "gain_seed": gain}


def rq2_tables(context: dict) -> dict[str, pd.DataFrame]:
    t8 = context["t8"]; keys = ["dataset", "regime", "seed"]
    metrics = ["student_test_accuracy", "student_test_nll", "target_accuracy", "target_nll", "target_entropy"]
    paired = _paired(t8, "expert_prob", "expert_logit", keys=keys, metrics=metrics)
    paired["space_delta_pp"] = 100 * paired.pop("delta_student_test_accuracy")
    paired = paired.rename(columns={"delta_student_test_nll": "delta_student_nll", "delta_target_accuracy": "delta_target_accuracy", "delta_target_nll": "delta_target_nll", "delta_target_entropy": "delta_target_entropy"})
    condition_summary = _pp_summary(paired, ["dataset", "regime"], "space_delta_pp")
    global_summary = _pp_summary(paired.assign(scope="overall"), ["scope"], "space_delta_pp")
    oracle = _paired(t8, "oracle_prob", "oracle_logit", keys=keys, metrics=metrics)
    oracle["space_delta_pp"] = 100 * oracle.pop("delta_student_test_accuracy")
    oracle_summary = _pp_summary(oracle.assign(scope="overall"), ["scope"], "space_delta_pp")
    temp = context["temperature"]
    temp_pairs = _paired(temp, "expert_prob", "expert_logit", keys=["dataset", "regime", "seed", "temperature"], metrics=metrics)
    temp_pairs["space_delta_pp"] = 100 * temp_pairs.pop("delta_student_test_accuracy")
    temp_pairs = temp_pairs.rename(columns={"delta_student_test_nll": "delta_student_nll", "delta_target_accuracy": "delta_target_accuracy", "delta_target_nll": "delta_target_nll", "delta_target_entropy": "delta_target_entropy"})
    temp_summary = _pp_summary(temp_pairs, ["regime", "temperature"], "space_delta_pp")
    temp_summary["seed_values_pp"] = temp_summary.apply(lambda r: json.dumps(temp_pairs.loc[(temp_pairs.regime == r.regime) & (temp_pairs.temperature == r.temperature), "space_delta_pp"].tolist()), axis=1)
    return {"paired": paired, "condition_summary": condition_summary, "global_summary": global_summary, "oracle": oracle_summary, "temperature_paired": temp_pairs, "temperature_summary": temp_summary}


def _save(fig, path: Path, caption: str) -> None:
    fig.text(.5, .005, caption, ha="center", va="bottom", fontsize=8, wrap=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def figures(context: dict, rq1: dict, rq2: dict) -> list[Path]:
    fig_dir = context["out"] / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    t8, gain, paired, temp = context["t8"], rq1["gain_seed"], rq2["paired"], rq2["temperature_paired"]
    x = np.arange(len(HETEROGENEITY)); labels = [REGIME_LABEL[r] for r in HETEROGENEITY]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True, constrained_layout=False)
    method_styles = (("feddf_logit", "FedDF", "o"), ("expert_logit", "EXPERT", "s"), ("oracle_logit", "ORACLE", "^") )
    legend_handles = []
    for ax, dataset in zip(axes, DATASETS):
        d = t8[(t8.dataset == dataset) & t8.method.isin([item[0] for item in method_styles])]
        for method, label, marker in method_styles:
            rows = d[d.method == method]
            for _, seed_rows in rows.groupby("seed"):
                values = seed_rows.set_index("regime").loc[list(HETEROGENEITY), "student_test_accuracy"]
                ax.plot(x, 100 * values, color=COLORS[label], lw=.7, alpha=.25, marker=marker, ms=3)
            stats = rows.groupby("regime").student_test_accuracy.agg(["mean", "std"]).loc[list(HETEROGENEITY)]
            handle = ax.errorbar(x, 100 * stats["mean"], yerr=100 * stats["std"], color=COLORS[label], lw=2.2, marker=marker, capsize=2, label=label)
            if dataset == DATASETS[0]: legend_handles.append(handle)
        ax.set_title(DATASET_LABEL[dataset]); ax.set_xticks(x, labels, rotation=30); ax.set_ylim(0, 100); ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("Student test accuracy (%)"); fig.legend(legend_handles, ["FedDF", "EXPERT", "ORACLE"], loc="upper center", ncol=3, frameon=False); fig.subplots_adjust(top=.82, bottom=.23)
    path = fig_dir / "fig_rq1_student_accuracy"; _save(fig, path, "Student test accuracy (%) across the ordered heterogeneity regimes. Faint trajectories are individual seeds; markers and bars show mean ± SD across three seeds. Uniform categorical spacing does not imply equal quantitative heterogeneity intervals."); saved.append(path)
    gain_values = gain.expert_minus_feddf_pp.to_numpy(); gain_limit = max(abs(gain_values.min()), abs(gain_values.max())) + 5
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True, constrained_layout=False)
    for ax, dataset in zip(axes, DATASETS):
        d = gain[gain.dataset == dataset]
        for _, rows in d.groupby("seed"):
            y = rows.set_index("regime").loc[list(HETEROGENEITY), "expert_minus_feddf_pp"]; ax.plot(x, y, color="#777777", lw=.9, alpha=.55, marker="o", ms=3)
        stats = d.groupby("regime").expert_minus_feddf_pp.agg(["mean", "std"]).loc[list(HETEROGENEITY)]
        ax.errorbar(x, stats["mean"], yerr=stats["std"], color=COLORS["EXPERT"], lw=2.5, marker="o", capsize=2)
        ax.axhline(0, color="black", lw=.8); ax.set_title(DATASET_LABEL[dataset]); ax.set_xticks(x, labels, rotation=30); ax.set_ylim(-gain_limit, gain_limit); ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("EXPERT − FedDF (pp)"); fig.subplots_adjust(bottom=.23)
    path = fig_dir / "fig_rq1_expertise_gain"; _save(fig, path, "Paired EXPERT − FedDF student-accuracy gain (pp) across the ordered heterogeneity axis. Thin lines are seed trajectories; thick lines and bars are mean ± SD across three seeds; zero marks no paired gain."); saved.append(path)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True, constrained_layout=False)
    for ax, dataset in zip(axes, DATASETS):
        d = paired[paired.dataset == dataset]
        for seed, rows in d.groupby("seed"):
            y = rows.set_index("regime").loc[list(HETEROGENEITY), "space_delta_pp"]; ax.plot(x, y, color="#777777", lw=.7, alpha=.35, marker="o", ms=3)
        stats = d.groupby("regime").space_delta_pp.agg(["mean", "std"]).loc[list(HETEROGENEITY)]
        ax.errorbar(x, stats["mean"], yerr=stats["std"], color=COLORS["EXPERT"], lw=2.2, marker="o", capsize=2)
        ax.axhspan(-1, 1, color="#BBBBBB", alpha=.2); ax.axhline(0, color="black", lw=.8)
        ax.set_xticks(x, labels, rotation=30); ax.set_title(DATASET_LABEL[dataset]); ax.grid(axis="y", alpha=.25)
    span = max(abs(paired.space_delta_pp).max(), 1.0) + .75
    for ax in axes: ax.set_ylim(-span, span)
    axes[0].set_ylabel("EXPERT-prob − EXPERT-logit (pp)"); fig.subplots_adjust(bottom=.23)
    path = fig_dir / "fig_rq2_logit_probability_t8"; _save(fig, path, "Paired EXPERT-prob − EXPERT-logit student-accuracy difference at T=8 across the ordered heterogeneity axis. Thin trajectories are seeds; error bars are mean ± SD. The shaded ±1 pp band is a descriptive practical reference, not an equivalence test."); saved.append(path)
    fig, ax = plt.subplots(figsize=(7.5, 4.8)); temperatures = [1., 4., 8.]
    for regime, label in (("iid", "IID"), ("alpha0p1", "α=0.1"), ("single", "Single")):
        d = temp[temp.regime == regime]; stats = d.groupby("temperature").space_delta_pp.agg(["mean", "std"]).loc[temperatures]
        ax.errorbar(temperatures, stats["mean"], yerr=stats["std"], marker="o", lw=2, color=COLORS[label], label=label)
        for seed, rows in d.groupby("seed"): ax.scatter(rows.temperature, rows.space_delta_pp, color=COLORS[label], s=24, alpha=.6)
    ax.axhspan(-1, 1, color="#BBBBBB", alpha=.2); ax.axhline(0, color="black", lw=.8); ax.set_xticks(temperatures); ax.set_xlabel("temperature T"); ax.set_ylabel("EXPERT-prob − EXPERT-logit (pp)"); ax.legend(frameon=False); ax.grid(axis="y", alpha=.25)
    fig.subplots_adjust(bottom=.16); path = fig_dir / "fig_rq2_temperature_sensitivity"; _save(fig, path, "CIFAR-10 paired seed-level differences by temperature. Lines show mean ± SD; the shaded band is ±1 pp. Regime labels refer to positions on the common ordered heterogeneity axis."); saved.append(path)
    return saved


def claims(context: dict, rq1: dict, rq2: dict) -> str:
    trajectory = rq1["trajectory"]; gain = rq1["gain_seed"]; gap = rq1["oracle_gap"].query("scope == 'overall'").iloc[0]
    t8 = rq2["global_summary"].iloc[0]; temp = rq2["temperature_summary"]
    low_t = temp.loc[temp.temperature.eq(1.0), "mean_pp"].abs().max()
    return f"""# RQ1/RQ2 claim–evidence ledger

## RQ1 — Need for expertise

### Observación

En {int(trajectory.nondecreasing.sum())}/{len(trajectory)} trayectorias dataset–seed del eje completo IID→α=1.0→α=0.5→α=0.1→multi→single, EXPERT−FedDF es no decreciente. Las excepciones se registran explícitamente en `table_rq1_trajectory.csv`; la tendencia media aumenta a través del eje. El gap ORACLE−EXPERT medio firmado es {gap.mean_signed_pp:.3f} pp (MAE {gap.mae_pp:.3f} pp).

### Interpretación

El patrón es compatible con que el routing por expertise sea más útil cuando la especialización de clase hace menos apropiado promediar a todos los teachers. ORACLE es una referencia informada por muestra para contextualizar el routing, no un techo del estudiante.

### Claim defendible

En este protocolo, la ventaja de EXPERT sobre FedDF aumenta de forma descriptiva a través de los regímenes ordenados de heterogeneidad, aunque una trayectoria individual puede presentar una excepción local.

### Lo que no demuestra

No demuestra causalidad únicamente desde accuracy, ni que EXPERT reproduzca ORACLE, ni que las separaciones categóricas uniformes del eje correspondan a incrementos cuantitativos idénticos de heterogeneidad.

## RQ2 — Logits frente a probabilidades

### Observación

A T=8, EXPERT-prob−EXPERT-logit tiene media {t8.mean_pp:.3f} pp, SD {t8.sd_pp:.3f} pp, MAE {t8.mae_pp:.3f} pp y {int(t8.within_1pp)}/{int(t8.n)} condiciones dentro de ±1 pp. En CIFAR a T=1, el mayor valor absoluto de la media por régimen es {low_t:.3f} pp.

### Interpretación

Con routing EXPERT idéntico, el espacio de agregación tiene un efecto práctico pequeño en conjunto a T=8, pero la sensibilidad CIFAR muestra que esa similitud no se extiende automáticamente a temperaturas bajas.

### Claim defendible

Bajo la receta canónica T=8, logits y probabilidades producen resultados de estudiante mayoritariamente cercanos, pero la robustez frente a temperatura es limitada: el efecto depende del régimen y puede ser material a T=1.

### Lo que no demuestra

No demuestra equivalencia matemática entre operadores, invarianza fuera de T=1,4,8, ni una temperatura óptima. El control ORACLE a T=8 es secundario y no convierte ORACLE en un upper bound del estudiante.

## Ledger

| claim | evidence | figure/table | status | limitation |
|---|---|---|---|---|
| Expertise gain increases across the ordered heterogeneity axis | {int(trajectory.nondecreasing.sum())}/{len(trajectory)} nondecreasing seed trajectories | fig_rq1_expertise_gain; table_rq1_paired_effects | qualified | One local trajectory exception; descriptive, three seeds, protocol-specific |
| EXPERT is close to ORACLE in every condition | Oracle gap has max absolute {gap.max_abs_pp:.3f} pp | table_rq1_oracle_gap | unsupported | Do not use in narrative; gaps remain condition-dependent |
| Aggregation space is practically limited at T=8 | {int(t8.within_1pp)}/{int(t8.n)} within ±1 pp, MAE {t8.mae_pp:.3f} pp | fig_rq2_logit_probability_t8; table_rq2_t8_summary | qualified | Some condition-level effects exceed ±1 pp |
| T=8 similarity is temperature-invariant | CIFAR T=1 has up to {low_t:.3f} pp mean absolute regime effect | fig_rq2_temperature_sensitivity; table_rq2_temperature_summary | unsupported | Do not use in narrative |
"""


def run_analysis(root: Path | None = None) -> dict:
    context = load_and_validate(root); out, table_dir = context["out"], context["out"] / "tables"; table_dir.mkdir(parents=True, exist_ok=True)
    rq1, rq2 = rq1_tables(context), rq2_tables(context)
    exports = {
        "table_rq1_accuracy.csv": rq1["accuracy"], "table_rq1_paired_effects.csv": rq1["effects"], "table_rq1_paired_effects_summary.csv": rq1["effects_summary"], "table_rq1_trajectory.csv": rq1["trajectory"], "table_rq1_oracle_gap.csv": rq1["oracle_gap"], "table_rq1_internal_controls.csv": rq1["controls"], "table_rq1_expert_internal_paired.csv": rq1["control_pairs"],
        "table_rq2_t8_paired.csv": rq2["paired"], "table_rq2_t8_summary.csv": rq2["condition_summary"], "table_rq2_t8_global_summary.csv": rq2["global_summary"], "table_rq2_oracle_space_control.csv": rq2["oracle"], "table_rq2_temperature_paired.csv": rq2["temperature_paired"], "table_rq2_temperature_summary.csv": rq2["temperature_summary"],
    }
    for name, frame in exports.items(): frame.to_csv(table_dir / name, index=False)
    figure_paths = figures(context, rq1, rq2)
    claims_path = out / "rq1_rq2_claims.md"; claims_path.write_text(claims(context, rq1, rq2), encoding="utf-8")
    return {"integrity": {"main_audit": context["integrity"], "rq2": context["rq2_integrity"]}, "rq1": rq1, "rq2": rq2, "exports": [str(table_dir / name) for name in exports] + [str(path.with_suffix(ext)) for path in figure_paths for ext in (".pdf", ".png")] + [str(claims_path)]}
