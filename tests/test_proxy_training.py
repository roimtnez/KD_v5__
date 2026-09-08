"""Fast synthetic checks; no dataset downloads or research executions."""

import json

import numpy as np
import pytest

from article1.hashes import file_sha256
from article1.proxy import proxy_positions


def test_proxy_subsets_are_nested_balanced_and_keep_full_order():
    indices = np.arange(100, 200, dtype=np.int64)[::-1]
    labels = indices % 10
    previous = set()
    for size in (1, 10, 23, 50, 100):
        rows = proxy_positions(indices, labels, size, 42)
        assert previous <= set(rows)
        counts = np.bincount(labels[rows], minlength=10)
        assert counts.max() - counts.min() <= 1
        assert len(rows) == size
        np.testing.assert_array_equal(rows, proxy_positions(indices, labels, size, 42))
        previous = set(rows)
    np.testing.assert_array_equal(rows, np.arange(100))
    assert not np.array_equal(
        proxy_positions(indices, labels, 50, 42),
        proxy_positions(indices, labels, 50, 43),
    )


@pytest.mark.parametrize("size", [0, -1, 21])
def test_invalid_proxy_size_rejected(size):
    with pytest.raises(ValueError):
        proxy_positions(np.arange(20), np.arange(20) % 10, size, 42)


def test_selection_membership_is_independent_of_cache_row_order():
    indices = np.arange(40)
    labels = indices % 10
    perm = np.random.default_rng(2).permutation(40)
    left = proxy_positions(indices, labels, 20, 42)
    right = proxy_positions(indices[perm], labels[perm], 20, 42)
    assert set(indices[left]) == set(indices[perm][right])
    with pytest.raises(ValueError):
        proxy_positions(np.array([1, 1]), np.array([0, 1]), 1, 42)


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from article1 import runner

    torch.set_num_threads(1)

    class Toy(torch.utils.data.Dataset):
        targets = np.arange(20) % 2

        def __len__(self):
            return 20

        def __getitem__(self, i):
            return torch.tensor([float(i) / 20, 1.0]), int(self.targets[i])

    toy = Toy()
    monkeypatch.setattr(runner, "datasets_for", lambda *args: (toy, toy, toy))
    monkeypatch.setattr(
        runner,
        "build_model",
        lambda _: torch.nn.Sequential(
            torch.nn.Linear(2, 4), torch.nn.Dropout(0.2), torch.nn.Linear(4, 2)
        ),
    )
    cache = tmp_path / "teacher_cache.npz"
    indices = np.arange(10, dtype=np.int64)
    labels = toy.targets[indices]
    np.savez(
        cache,
        proxy_idx=indices,
        labels=labels,
        logits=np.random.default_rng(1).normal(size=(10, 2, 2)).astype(np.float32),
        M=np.ones((2, 2), dtype=np.uint8),
    )
    metadata = {
        "protocol": "article1-v3",
        "dataset": "mnist",
        "seed": 42,
        "regime": "iid",
        "cache_sha256": file_sha256(cache),
    }
    meta = tmp_path / "metadata.json"
    meta.write_text(json.dumps(metadata))
    common = {
        "cache": cache,
        "data_dir": tmp_path,
        "results": tmp_path / "study.csv",
        "dataset": "mnist",
        "seed": 42,
        "batch_size": 4,
        "device": "cpu",
    }
    return runner, common, meta


def test_ce_kd_pairing_partial_epoch_and_subset(runtime):
    runner, common, _ = runtime
    ce = runner.supervised_proxy(**common, updates=3, proxy_size=6)
    kd = runner.distill(**common, method="expert_logit", updates=3, proxy_size=6)
    for key in (
        "student_init_sha256",
        "proxy_sha256",
        "proxy_labels_sha256",
        "batch_order_sha256",
        "consumed_batches_sha256",
        "examples_seen",
        "updates",
        "batch_size",
        "optimizer",
        "learning_rate",
        "weight_decay",
    ):
        assert ce[key] == kd[key], key
    assert ce["epochs_completed"] == 1
    assert ce["epochs_started"] == 2
    assert ce["examples_seen"] == 10
    assert ce["proxy_size"] == 6
    repeat = runner.supervised_proxy(**common, updates=3, proxy_size=6)
    assert repeat["student_final_sha256"] == ce["student_final_sha256"]
    changed = runner.supervised_proxy(**common, updates=2, proxy_size=6)
    assert changed["run_id"] != ce["run_id"]
    # Same generated permutations, but only actually consumed batches count.
    one = runner.supervised_proxy(**common, updates=1, proxy_size=6)
    assert one["batch_order_sha256"] == changed["batch_order_sha256"]
    assert one["consumed_batches_sha256"] != changed["consumed_batches_sha256"]


def test_full_proxy_epoch_and_update_budgets_match(runtime):
    runner, common, _ = runtime
    old = runner.distill(**common, method="expert_logit", epochs=2)
    new = runner.distill(**common, method="expert_logit", updates=6, proxy_size=10)
    for key in (
        "student_init_sha256",
        "student_final_sha256",
        "batch_order_sha256",
        "consumed_batches_sha256",
        "student_test_accuracy",
        "student_test_nll",
    ):
        assert old[key] == new[key], key
    assert (
        old["run_id"] != new["run_id"]
    )  # Explicit budget study has a distinct recipe.


def test_cache_integrity_and_label_alignment(runtime):
    runner, common, meta = runtime
    cache = common["cache"]
    with cache.open("ab") as out:
        out.write(b"corruption")
    with pytest.raises(ValueError, match="hash"):
        runner.supervised_proxy(**common, updates=1)
    metadata = json.loads(meta.read_text())
    with np.load(cache) as source:
        arrays = dict(source)
    arrays["labels"] = 1 - arrays["labels"]
    np.savez(cache, **arrays)
    metadata["cache_sha256"] = file_sha256(cache)
    meta.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="labels"):
        runner.supervised_proxy(**common, updates=1)


def test_supervised_identity_reusable_across_regimes(runtime):
    runner, common, meta = runtime
    first = runner.supervised_proxy(**common, updates=1)
    metadata = json.loads(meta.read_text())
    metadata["regime"] = "single"
    meta.write_text(json.dumps(metadata))
    second = runner.supervised_proxy(**common, updates=1)
    assert first["run_id"] == second["run_id"]
    assert first["student_final_sha256"] == second["student_final_sha256"]
    with pytest.raises(ValueError, match="requires"):
        runner.distill(**common, method="expert_logit", proxy_size=5)
    with pytest.raises(ValueError, match="separate CSV"):
        runner.supervised_proxy(
            **{**common, "results": common["results"].with_name("results.csv")}
        )
