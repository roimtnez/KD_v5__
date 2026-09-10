"""The self-contained editorial notebook must run from published evidence alone."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_notebook_executes_without_private_results(tmp_path, monkeypatch):
    notebook = json.loads((ROOT / "notebooks/article1_paper.ipynb").read_text())
    namespace = {"display": lambda *args: None}
    monkeypatch.chdir(ROOT)
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        # Redirect figure outputs only; inputs remain the real published CSVs.
        source = source.replace(
            "FIGURES = ROOT / 'OUTPUTS/article1_paper'",
            f"FIGURES = Path({str(tmp_path)!r})",
        )
        exec(compile(source, "article1_paper.ipynb", "exec"), namespace)  # noqa: S102 - trusted repository notebook
    support = namespace["support"]
    cifar = support[support.dataset.eq("cifar")]
    assert (cifar[cifar.metric.eq("target_nll")]["mean"] < 0).all()
    acc = cifar[cifar.metric.eq("student_test_accuracy")].set_index("regime")["mean"]
    assert (acc < 0).sum() == 5
    assert acc["iid"] == pytest.approx(-0.0653333333333333)
    assert (tmp_path / "target_vs_student.png").is_file()
    plt.close("all")


def test_support_summary_has_explicit_paired_seed_evidence():
    table = pd.read_csv(ROOT / "docs/article1_closure/tables/main_contrast_summary.csv")
    support = table[table.contrast.eq("support")]
    assert len(support) == 90
    assert support.n.eq(3).all() and support.seeds.eq("42,43,44").all()
