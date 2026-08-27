"""Small, dependency-light readouts for independent seed replicates."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def sign_verdict(values: Iterable[float], *, atol: float = 1e-12) -> str:
    """Return the literal cross-seed sign pattern; this is not a p-value."""
    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        return "missing"
    signs = {"positive" if v > atol else "negative" if v < -atol else "zero" for v in finite}
    return signs.pop() if len(signs) == 1 else "mixed"


def aggregate_seed_metric(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    metric: str,
) -> pd.DataFrame:
    """Summarise one seed-indexed metric without inferential statistics."""
    if "seed" not in frame.columns:
        raise ValueError("seed-indexed reporting requires a 'seed' column")
    missing = [c for c in [*group_columns, metric] if c not in frame.columns]
    if missing:
        raise ValueError(f"missing columns for seed reporting: {missing}")

    rows: list[dict] = []
    for key, part in frame.groupby(list(group_columns), dropna=False, sort=True):
        seed_counts = part.groupby("seed", dropna=False).size()
        if (seed_counts > 1).any():
            raise ValueError("seed reporting requires one value per seed and reporting cell")
        values = pd.to_numeric(part[metric], errors="coerce")
        seeds = sorted({int(s) for s in part.loc[values.notna(), "seed"]})
        finite = values.dropna().to_numpy(dtype=float)
        row = dict(zip(group_columns, key if isinstance(key, tuple) else (key,)))
        mean = float(np.mean(finite)) if len(finite) else np.nan
        std = float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan
        positive = int((finite > 1e-12).sum())
        negative = int((finite < -1e-12).sum())
        zero = int(len(finite) - positive - negative)
        # Concordance is measured against the direction of the mean. For an
        # exactly-zero mean, only exact-zero observations are concordant.
        concordant = positive if mean > 1e-12 else negative if mean < -1e-12 else zero
        verdicts = np.where(finite > 0.01, "positive", np.where(finite < -0.01, "negative", "tie"))
        verdict_counts = {v: int((verdicts == v).sum()) for v in ("positive", "tie", "negative")}
        verdict_consistent_n = max(verdict_counts.values(), default=0)
        row.update({
            "metric": metric,
            "n_seeds": len(seeds),
            "seeds": ",".join(map(str, seeds)),
            **{f"seed_{seed}": float(part.loc[pd.to_numeric(part["seed"], errors="coerce") == seed, metric].iloc[0])
               for seed in seeds},
            "mean": mean,
            "sample_std": std,
            "std": std,
            "standard_error_n3_estimate": std / np.sqrt(len(finite)) if len(finite) > 1 else np.nan,
            "min": float(np.min(finite)) if len(finite) else np.nan,
            "max": float(np.max(finite)) if len(finite) else np.nan,
            "range": float(np.ptp(finite)) if len(finite) else np.nan,
            "sign_consistency": sign_verdict(finite),
            "positive_n": positive,
            "negative_n": negative,
            "zero_n": zero,
            "sign_concordant_n": concordant,
            "sign_concordance": f"{concordant}/{len(finite)}",
            "verdict_margin_pp": 1.0,
            "verdict_consistent_n": verdict_consistent_n,
            "verdict_consistency": f"{verdict_consistent_n}/{len(finite)}",
        })
        rows.append(row)
    return pd.DataFrame(rows)


def paired_seed_effect(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    variant_column: str,
    left: str,
    right: str,
    metric: str,
) -> pd.DataFrame:
    """Compute ``left-right`` inside each seed before any aggregation."""
    required = [*group_columns, "seed", variant_column, metric]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"missing columns for paired seed effect: {missing}")
    keys = [*group_columns, "seed"]
    subset = frame[frame[variant_column].isin([left, right])]
    counts = subset.groupby(keys + [variant_column], dropna=False).size()
    if (counts > 1).any():
        raise ValueError("paired effect inputs contain duplicate seed/cell/variant rows")
    wide = subset.pivot(index=keys, columns=variant_column, values=metric)
    if left not in wide or right not in wide:
        raise ValueError(f"paired effect requires both {left!r} and {right!r} in every cell")
    if wide[[left, right]].isna().any().any():
        raise ValueError("paired effect has an unpaired seed/cell")
    out = wide.reset_index()
    out["comparison"] = f"{left}-{right}"
    out["delta"] = pd.to_numeric(out[left], errors="raise") - pd.to_numeric(out[right], errors="raise")
    return out[[*keys, "comparison", left, right, "delta"]]


def causal_attribution(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    arm_column: str = "arm",
    metric: str,
    atol: float = 1e-12,
) -> pd.DataFrame:
    """Build extra-pass/mechanism/total effects and validate their identity."""
    keys = [*group_columns, "seed"]
    subset = frame[frame[arm_column].isin(["global", "proxy_plain", "proxy"])]
    counts = subset.groupby(keys + [arm_column], dropna=False).size()
    if (counts > 1).any():
        raise ValueError("attribution inputs contain duplicate seed/cell/arm rows")
    wide = subset.pivot(index=keys, columns=arm_column, values=metric)
    required = ["global", "proxy_plain", "proxy"]
    if any(c not in wide for c in required) or wide[required].isna().any().any():
        raise ValueError("attribution requires paired global/proxy_plain/proxy rows")
    out = wide.reset_index()
    out["extra_pass"] = out["proxy_plain"] - out["global"]
    out["mechanism"] = out["proxy"] - out["proxy_plain"]
    out["total"] = out["proxy"] - out["global"]
    residual = out["total"] - out["extra_pass"] - out["mechanism"]
    if not np.allclose(residual, 0.0, atol=atol, rtol=0.0):
        raise AssertionError("total != extra_pass + mechanism")
    out["identity_residual"] = residual
    return out
