"""Validate a completed baseline and export its blocks without training models."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from article1.analysis import load_results
from article1.experiments import FOCAL_REGIMES, T8_BLOCKS


def export_blocks(out: Path) -> list[Path]:
    out = Path(out)
    # Require all 540 rows, matching provenance, CRN, fixed budget and SR revision.
    load_results(out, stage="baseline")
    source = pd.read_csv(out / "results_baseline.csv", dtype=str, keep_default_na=False)
    exports = {
        out / filename: source.loc[source.method.isin(methods)].copy()
        for block, (filename, methods) in T8_BLOCKS.items()
        if block != "baseline"
    }
    exports[out / "results_expert_logit_focal.csv"] = source.loc[
        source.method.eq("expert_logit")
        & source.dataset.eq("cifar")
        & source.regime.isin(FOCAL_REGIMES)
    ].copy()

    for path, rows in exports.items():
        if not path.exists():
            rows.to_csv(path, index=False, mode="x")
    return list(exports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("OUTPUTS/article1_v3"))
    args = parser.parse_args()
    for path in export_blocks(args.output_root):
        print(path)


if __name__ == "__main__":
    main()
