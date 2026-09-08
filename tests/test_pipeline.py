import sys

import run_article1_pipeline as pipeline


def test_default_is_plan_and_proxy_curve_is_opt_in(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run executed a command")

    monkeypatch.setattr(pipeline.subprocess, "run", forbidden)
    monkeypatch.setattr(sys, "argv", ["run_article1_pipeline.py"])
    pipeline.main()
    plan = capsys.readouterr().out
    assert plan.count("article1.runner supervised") == 9
    assert "--proxy-size" not in plan
    monkeypatch.setattr(sys, "argv", ["run_article1_pipeline.py", "--with-proxy-curve"])
    pipeline.main()
    plan = capsys.readouterr().out
    assert plan.count("article1.runner supervised") == 21
    assert plan.count("article1.runner distill") == 36


def test_partitions_only_never_reaches_training(monkeypatch):
    commands, checks, notebooks = [], [], []
    monkeypatch.setattr(
        pipeline.subprocess, "run", lambda cmd, **kw: commands.append(cmd)
    )
    monkeypatch.setattr(pipeline, "check_all_partitions", lambda: checks.append(True))
    monkeypatch.setattr(
        pipeline, "execute_notebook", lambda *a, **kw: notebooks.append(kw)
    )
    monkeypatch.setattr(
        sys, "argv", ["run_article1_pipeline.py", "--execute", "--partitions-only"]
    )
    pipeline.main()
    assert checks == [True]
    assert len(notebooks) == 9
    assert len(commands) == 3  # compile, tests, partition stage
    assert commands[-1][-2:] == ["--stage", "partition"]
