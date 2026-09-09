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
