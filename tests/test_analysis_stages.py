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
    [
        ("rq1", 108),
        ("aggregation", 216),
        ("expertise", 270),
        ("support", 324),
        ("baseline", 324),
    ],
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
    assert ("support" in effects) == (stage in {"support", "baseline"})
    if stage == "baseline":
        from article1.baseline_results import export_blocks

        master = tmp_path / "results_baseline.csv"
        before = master.read_bytes()
        paths = export_blocks(tmp_path)
        assert len(paths) == 4
        assert export_blocks(tmp_path) == paths
        assert master.read_bytes() == before
        assert not (tmp_path / "results_expert_logit_focal.csv").exists()
        assert not (tmp_path / "results_controls.csv").exists()
        for block in ("rq1", "aggregation", "expertise", "support"):
            load_results(tmp_path, stage=block)
        # Existing conflicting destinations are not overwritten.
        conflict = paths[0]
        conflict.write_text("invalid\n")
        with pytest.raises(ValueError, match="existing block differs"):
            export_blocks(tmp_path)
        assert conflict.read_text() == "invalid\n"
        conflict.unlink()
        export_blocks(tmp_path)
        # Revision 1 on other methods is legitimate; SR revision 1 is not.
        rows = pd.read_csv(master)
        rows.loc[rows.method.eq("expert_prob_sr"), "target_revision"] = 1
        rows.to_csv(master, index=False)
        with pytest.raises(ValueError, match="stable target revision 2"):
            load_results(tmp_path, stage="baseline")
        master.write_bytes(before)
    final_file = tmp_path / T8_BLOCKS[ANALYSIS_BLOCKS[stage][-1]][0]
    pd.read_csv(final_file).iloc[:-1].to_csv(final_file, index=False)
    with pytest.raises(ValueError, match="Incomplete T=8 grid"):
        load_results(tmp_path, stage=stage)


def write_design(root, design):
    from article1.analysis import BASELINE_DESIGNS
    conditions = [dict(dataset=d,regime=r,seed=s,protocol_version=PROTOCOL_VERSION,
                       proxy_sha256='shared',expertise_threshold=THRESHOLDS[d],M_density=1,experts_per_class_mean=10)
                  for d,r,s in product(DATASETS,REGIMES,SEEDS)]
    pd.DataFrame(conditions).to_csv(root/'conditions.csv',index=False)
    rows=[]
    for c,method in product(conditions,BASELINE_DESIGNS[design]):
        row={field:'shared' for field in CRN}
        row.update({field:0. for field in METRICS+ROUTING})
        row.update(c, method=method,temperature=8,updates=1200,
                   run_id=f"{c['dataset']}-{c['regime']}-{c['seed']}-{method}",
                   target_revision=2 if method=='expert_prob_sr' else 1)
        row[SUPPORT_MASS]=0.
        rows.append(row)
    frame=pd.DataFrame(rows)
    frame.to_csv(root/'results_baseline.csv',index=False)
    return frame


@pytest.mark.parametrize('design,count',[('six',324),('ten',540)])
def test_explicit_and_inferred_designs_are_strict(tmp_path,design,count):
    frame=write_design(tmp_path,design)
    assert len(load_results(tmp_path,'baseline')['t8'])==count
    assert len(load_results(tmp_path,'baseline',design=design)['t8'])==count
    other='ten' if design=='six' else 'six'
    with pytest.raises(ValueError): load_results(tmp_path,'baseline',design=other)
    frame.iloc[:-1].to_csv(tmp_path/'results_baseline.csv',index=False)
    with pytest.raises(ValueError,match='Incomplete T=8 grid'): load_results(tmp_path,'baseline',design=design)


def test_definitive_crn_includes_consumed_batches(tmp_path):
    frame=write_design(tmp_path,'six')
    frame.loc[0,'consumed_batches_sha256']='different'
    frame.to_csv(tmp_path/'results_baseline.csv',index=False)
    with pytest.raises(ValueError,match='CRN mismatch'): load_results(tmp_path,'baseline')


def test_historical_export_does_not_lose_controls(tmp_path):
    from article1.baseline_results import export_blocks
    write_design(tmp_path,'ten')
    assert len(export_blocks(tmp_path))==6
    assert len(pd.read_csv(tmp_path/'results_expert_logit_focal.csv'))==9
