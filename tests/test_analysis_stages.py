from itertools import product

import pandas as pd
import pytest

from article1 import DATASETS, PROTOCOL_VERSION, REGIMES, SEEDS, THRESHOLDS
from article1.analysis import (
    CRN,
    METRICS,
    ROUTING,
    SUPPORT_MASS,
    comparisons,
    load_results,
)
from article1.experiments import ANALYSIS_BLOCKS, T8_BLOCKS


@pytest.mark.parametrize(
    "stage,count",
    [("rq1", 108), ("aggregation", 216), ("expertise", 270), ("support", 324)],
)
def test_analysis_loads_only_complete_requested_blocks(tmp_path, stage, count):
    conditions = []
    for dataset, regime, seed in product(DATASETS, REGIMES, SEEDS):
        conditions.append(
            {
                "dataset": dataset,
                "regime": regime,
                "seed": seed,
                "protocol_version": PROTOCOL_VERSION,
                "proxy_sha256": "shared",
                "expertise_threshold": THRESHOLDS[dataset],
                "M_density": 1,
                "experts_per_class_mean": 10,
            }
        )
    pd.DataFrame(conditions).to_csv(tmp_path / "conditions.csv", index=False)
    for block in ANALYSIS_BLOCKS[stage]:
        filename, methods = T8_BLOCKS[block]
        rows = []
        for condition, method in product(conditions, methods):
            row = {field: "shared" for field in CRN}
            row.update({field: 0.0 for field in METRICS + ROUTING})
            row.update(
                {
                    k: condition[k]
                    for k in ("dataset", "regime", "seed", "protocol_version")
                }
            )
            row.update(
                method=method,
                temperature=8,
                updates=1200,
                run_id=f"{condition['dataset']}-{condition['regime']}-{condition['seed']}-{method}",
            )
            row[SUPPORT_MASS] = 0.0
            row["target_revision"] = 2 if method == "expert_prob_sr" else 1
            rows.append(row)
        pd.DataFrame(rows).to_csv(tmp_path / filename, index=False)
    # A partial later experiment must not prevent analysis of the selected phase.
    (tmp_path / "results_rq2_temperature.csv").write_text("unfinished\n")
    context = load_results(tmp_path, stage=stage)
    assert len(context["t8"]) == count
    effects = comparisons(context)
    assert ("feddf_pooling" in effects) == (stage != "rq1")
    assert ("support" in effects) == (stage == "support")
    final_file = tmp_path / T8_BLOCKS[ANALYSIS_BLOCKS[stage][-1]][0]
    pd.read_csv(final_file).iloc[:-1].to_csv(final_file, index=False)
    with pytest.raises(ValueError, match="Incomplete T=8 grid"):
        load_results(tmp_path, stage=stage)
