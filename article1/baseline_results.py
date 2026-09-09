"""Validate a completed baseline and export its blocks without training models."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from article1.analysis import load_results
from article1.experiments import FOCAL_REGIMES, T8_BLOCKS


def export_blocks(out: Path, *, design: str = 'auto') -> list[Path]:
    out = Path(out)
    # Require every identity of the explicit/inferred six- or ten-method design.
    load_results(out, stage="baseline", design=design)
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

    exports = {path: rows for path, rows in exports.items() if not rows.empty}
    def canonical(frame):
        return frame.sort_values('run_id').sort_index(axis=1).reset_index(drop=True)
    for path, rows in exports.items():
        if path.exists():
            existing = pd.read_csv(path, dtype=str, keep_default_na=False)
            if 'run_id' not in existing or not canonical(existing).equals(canonical(rows)):
                raise ValueError(f'existing block differs from baseline; inspect manually: {path}')
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
