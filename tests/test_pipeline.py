import sys

import pytest

import run_article1_pipeline as pipeline


def test_default_is_sequence_plan(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run executed a command")

    monkeypatch.setattr(pipeline.subprocess, "run", forbidden)
    monkeypatch.setattr(sys, "argv", ["run_article1_pipeline.py"])
    pipeline.main()
    plan = capsys.readouterr().out
    assert "--phase partitions" in plan
    assert "article1.runner supervised" not in plan
    assert "--stage teachers" not in plan


@pytest.mark.parametrize(
    "stage,methods",
    [
        ("rq1", list(pipeline.T8_BLOCKS["rq1"][1])),
        ("aggregation", ["feddf_prob", "oracle_prob"]),
        ("expertise", ["expert_prob"]),
        ("support", ["expert_prob_sr"]),
    ],
)
def test_execution_is_limited_to_selected_kd_stage(monkeypatch, stage, methods):
    commands = []
    monkeypatch.setattr(
        pipeline.subprocess, "run", lambda cmd, **kw: commands.append(cmd)
    )
    monkeypatch.setattr(pipeline, "check_all_partitions", lambda: None)
    monkeypatch.setattr(pipeline, "check_sources", lambda: None)
    monkeypatch.setattr(pipeline, "check_pilot", lambda: None)
    monkeypatch.setattr(pipeline, "execute_notebook", lambda *a, **kw: None)
    monkeypatch.setattr(
        sys, "argv", ["pipeline", "--phase", stage, "--execute", "--device", "cpu"]
    )
    pipeline.main()
    grids = [c for c in commands if c[1] == "run_article1_grid.py"]
    assert len(grids) == 1
    assert grids[0][grids[0].index("--methods") + 1 :][: len(methods)] == methods
    assert "teachers" not in grids[0]
    assert all("supervised" not in c for c in commands)


def test_baseline_executes_probability_methods_before_logit_methods(monkeypatch):
    commands = []
    monkeypatch.setattr(
        pipeline.subprocess, "run", lambda cmd, **kw: commands.append(cmd)
    )
    monkeypatch.setattr(pipeline, "check_sources", lambda: None)
    monkeypatch.setattr(pipeline, "execute_notebook", lambda *a, **kw: None)
    monkeypatch.setattr(
        sys, "argv", ["pipeline", "--phase", "baseline", "--execute", "--device", "cpu"]
    )
    pipeline.main()
    grid = next(command for command in commands if command[1] == "run_article1_grid.py")
    methods = grid[grid.index("--methods") + 1 : grid.index("--results")]
    first_logit = next(index for index, method in enumerate(methods) if method.endswith("_logit"))
    assert all(not method.endswith("_logit") for method in methods[:first_logit])
    assert all(method.endswith("_logit") for method in methods[first_logit:])


def test_partitions_only_never_reaches_training(monkeypatch):
    commands, checks, notebooks = [], [], []
    monkeypatch.setattr(
        pipeline.subprocess, "run", lambda cmd, **kw: commands.append(cmd)
    )
    monkeypatch.setattr(pipeline, "check_all_partitions", lambda: checks.append(True))
    monkeypatch.setattr(
        pipeline, "execute_notebook", lambda *a, **kw: notebooks.append(kw)
    )
    monkeypatch.setattr(sys, "argv", ["pipeline", "--execute", "--phase", "partitions"])
    pipeline.main()
    assert checks == [True]
    assert len(notebooks) == 9
    assert len(commands) == 3
    assert commands[-1][-2:] == ["--stage", "partition"]


def test_curve_plan_includes_full_ce_anchors_and_only_focal_cells(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["pipeline", "--phase", "proxy-curve"])
    pipeline.main()
    plan = capsys.readouterr().out
    assert plan.count("article1.runner supervised") == 15
    assert plan.count("article1.runner distill") == 36
    assert "--stage teachers" not in plan


def test_execute_requires_explicit_phase(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pipeline", "--execute"])
    with pytest.raises(SystemExit, match="2"):
        pipeline.main()


def test_minimal_sequence_excludes_optional_full_grids(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["pipeline"])
    pipeline.main()
    plan = capsys.readouterr().out
    assert "--phase expertise --execute" in plan
    assert "--phase controls --execute" not in plan
    assert "--phase temperature --execute" not in plan


def test_presence_is_separate_and_never_retrains_teachers(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['pipeline', '--phase', 'presence'])
    monkeypatch.setattr(pipeline.subprocess, 'run', lambda *a, **k: pytest.fail('Plan must not execute'))
    pipeline.main()
    plan = capsys.readouterr().out
    assert plan.count('article1.runner distill') == 54
    assert plan.count('--method presence_prob') == 54
    assert 'results_presence.csv' in plan and 'article1.presence' in plan
    assert '--stage teachers' not in plan
    assert '--method expert_prob_sr' not in plan
