"""Editorial statistics use individual seeds; the historical report is not input."""
from pathlib import Path
import json
import pandas as pd
import pytest
from article1.editorial import summary
from article1.progress import compatible_pairs

ROOT = Path(__file__).resolve().parents[1]


def test_paired_statistics_keep_covariance_and_percentage_points():
    rows = []
    for seed, a, b in [(42, .91, .90), (43, .61, .60), (44, .81, .80)]:
        for method, score in [('expert_prob', a), ('feddf_prob', b)]:
            rows.append(dict(dataset='cifar', regime='iid', seed=seed, method=method,
                             student_test_accuracy=score, trace='same', run_id=f'{method}{seed}'))
    pairs, _ = compatible_pairs(pd.DataFrame(rows), 'expert_prob','feddf_prob',
                                fields=['trace'], metrics=['student_test_accuracy'])
    s = summary(pairs, ['dataset','regime'], ['delta_student_test_accuracy']).iloc[0]
    assert s['mean'] == pytest.approx(1)
    assert s.sd == pytest.approx(0, abs=1e-12)
    assert s.n == 3 and s.seeds == '42,43,44'


def test_single_seed_does_not_invent_sd_or_fill_missing_values():
    f = pd.DataFrame([dict(dataset='cifar', seed=42, student_test_accuracy=.8)])
    s = summary(f,['dataset'],['student_test_accuracy']).iloc[0]
    assert s['mean']==80 and s.n==1 and pd.isna(s.sd)


def test_reused_seed_cannot_be_counted_as_another_replica():
    f = pd.DataFrame([dict(dataset='cifar',seed=42,student_test_accuracy=.8)]*2)
    with pytest.raises(ValueError,match='Duplicate experimental seed'):
        summary(f,['dataset'],['student_test_accuracy'])


def test_historical_support_summary_retains_its_original_evidence():
    f = pd.read_csv(ROOT/'docs/article1_closure/tables/main_contrast_summary.csv')
    s = f[f.contrast.eq('support')]
    assert len(s)==90 and s.n.eq(3).all() and s.seeds.eq('42,43,44').all()
