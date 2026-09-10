"""Editorial evidence gates must reject incomplete or altered paired results."""

import shutil
from pathlib import Path

import pandas as pd
import pytest

from article1.paper import load_evidence

SOURCE = Path(__file__).resolve().parents[1] / "docs/article1_closure"


def test_published_evidence_is_complete():
    data = load_evidence(SOURCE)
    assert len(data["proxy_curve_paired"]) == 45
    iid = data["proxy_curve_paired"].query('regime == "iid" and proxy_size == 500')
    assert 100 * iid.delta_student_test_accuracy.mean() == pytest.approx(10.88)


@pytest.mark.parametrize("damage", ["missing", "wrong_delta"])
def test_incomplete_or_inconsistent_curve_is_rejected(tmp_path, damage):
    shutil.copytree(SOURCE / "tables", tmp_path / "tables")
    shutil.copyfile(SOURCE / "manifest.json", tmp_path / "manifest.json")
    path = tmp_path / "tables/proxy_curve_paired.csv"
    rows = pd.read_csv(path)
    if damage == "missing":
        rows = rows.iloc[:-1]
    else:
        rows.loc[0, "delta_student_test_accuracy"] += 0.1
    rows.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_evidence(tmp_path)


def test_absolute_mean_paths_and_ce_deduplication():
    from article1.paper import absolute_means, supervised_curve_pairs

    data = load_evidence(SOURCE)
    means = absolute_means(data)
    assert len(means) == 3 * 6 * 7 * 2
    ce = supervised_curve_pairs(data["proxy_curve_paired"])
    assert len(ce) == 15
    assert ce.query(
        "proxy_size == 10000"
    ).student_test_accuracy_right.mean() == pytest.approx(0.7901333333333334)
    data["main_contrast_summary"].loc[
        lambda f: f.contrast.eq("selection_logit"), "mean"
    ] += 0.01
    with pytest.raises(ValueError, match="paths disagree"):
        absolute_means(data)


def test_conflicting_ce_reuse_is_not_silently_deduplicated():
    from article1.paper import supervised_curve_pairs

    data = load_evidence(SOURCE)
    curve = data["proxy_curve_paired"].copy()
    curve.loc[0, "student_test_accuracy_right"] += 0.01
    with pytest.raises(ValueError, match="Conflicting CE"):
        supervised_curve_pairs(curve)
