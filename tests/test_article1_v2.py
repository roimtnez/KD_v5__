import numpy as np
import pytest
import csv
import hashlib
import json
import os
from pathlib import Path

from article1.distillation import METHODS, authority_from_holdout, build_target, metadata_identity
from article1.partitioning import make_partitions, save_partitions, validate_splits
from article1.audit import audit
from article1.analysis import paired_effects
from article1.backfill import backfill
from article1 import audit as audit_component
from article1 import conditions as conditions_component
from article1.hashes import array_sha256
from article1.local_training import configure_determinism


def test_holdout_authority_requires_observations():
    M = authority_from_holdout(np.array([[.99, .99]]), np.array([[2, 0]]), .9)
    assert M.tolist() == [[1, 0]]


def test_M_is_reconstructible_from_holdout_only():
    accuracy = np.array([[.9, .1], [.2, .95]])
    counts = np.array([[4, 4], [4, 4]])
    cached = authority_from_holdout(accuracy, counts, .8)
    np.testing.assert_array_equal(cached, authority_from_holdout(accuracy.copy(), counts.copy(), .8))


def test_new_partitions_record_protocol_and_creation_commit(tmp_path):
    proxy = np.array([0, 1]); clients = [{"train_idx": np.array([2]), "holdout_idx": np.array([3]), "test_idx": np.array([4])}]
    save_partitions(tmp_path / "partition", proxy_idx=proxy, clients=clients, metadata={"dataset": "mnist"})
    metadata = json.loads((tmp_path / "partition" / "metadata.json").read_text())
    assert metadata["protocol_version"] == "article1-v2"
    assert metadata["creation_commit"] != "unknown"


def test_deterministic_execution_contract_is_enabled():
    import torch

    configure_determinism()
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.backends.cudnn.benchmark
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cuda.matmul.allow_tf32
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_splits_reserve_proxy_and_are_disjoint():
    labels = np.repeat(np.arange(10), 40)
    proxy, clients = make_partitions(labels, regime="alpha0p5", seed=42, clients=10, proxy_size=100)
    validate_splits(proxy, clients)
    assert len(proxy) == 100


def test_single_and_multi_partitions_are_flat_and_disjoint():
    labels = np.repeat(np.arange(10), 40)
    for regime in ("single", "multi"):
        proxy, clients = make_partitions(labels, regime=regime, seed=42, clients=10, proxy_size=100)
        validate_splits(proxy, clients)
        for client in clients:
            for indices in client.values():
                assert indices.ndim == 1


def test_logit_methods_are_finite_normalized_and_common_full_mask_identity():
    rng = np.random.default_rng(2); z = rng.normal(size=(9, 3, 4)); y = rng.integers(0, 4, 9); M = np.ones((3, 4), dtype=np.uint8)
    feddf = build_target(z, y, M, method="feddf_logit", temperature=2)
    expert = build_target(z, y, M, method="expert_logit", temperature=2)
    np.testing.assert_allclose(feddf.probabilities, expert.probabilities)
    for method in METHODS:
        q = build_target(z, y, M, method=method, temperature=2).probabilities
        assert np.isfinite(q).all(); np.testing.assert_allclose(q.sum(1), 1.0)


def test_expert_variants_share_routing_and_sr_is_zero_outside_support():
    z = np.array([[[5., 0., 0.], [0., 2., 4.]], [[0., 4., 0.], [0., 0., 5.]]])
    y = np.array([0, 2]); M = np.array([[1, 1, 0], [0, 0, 1]], dtype=np.uint8)
    targets = [build_target(z, y, M, method=name, temperature=1) for name in ("expert_logit", "expert_prob", "expert_prob_sr")]
    assert np.array_equal(targets[0].selected, targets[1].selected)
    assert np.array_equal(targets[1].selected, targets[2].selected)
    # First sample selects teacher 0, which cannot emit class 2 under SR.
    assert targets[2].probabilities[0, 2] == 0
    assert targets[1].metrics["pre_restriction_outside_support_mass"] is not None
    assert targets[2].metrics["pre_restriction_outside_support_mass"] is not None
    assert all(np.allclose(target.weights.sum(axis=1), 1.0) for target in targets)


def test_effective_teachers_reports_uniform_and_concentrated_weighting():
    z = np.array([[[8., 0.], [0., 0.]]]); y = np.array([0]); M = np.ones((2, 2), dtype=np.uint8)
    uniform = build_target(z, y, M, method="feddf_logit")
    confidence = build_target(z, y, M, method="confidence_logit")
    assert uniform.metrics["effective_teachers"] == 2.0
    assert confidence.metrics["effective_teachers"] < 2.0


def test_oracle_and_consensus_have_the_shared_fallback():
    z = np.array([[[5., 0.], [4., 0.]]])
    y = np.array([1]); M = np.zeros((2, 2), dtype=np.uint8)
    fallback = build_target(z, y, M, method="feddf_logit", temperature=2).probabilities
    for method in ("oracle_logit", "oracle_prob", "expert_logit"):
        out = build_target(z, y, M, method=method, temperature=2)
        assert out.fallback.tolist() == [True]; np.testing.assert_allclose(out.probabilities, fallback)
    # Soft vote can choose a class that no teacher hard-predicts.
    soft_winner_without_hard_support = np.log(np.array([[[.49, .51, .0001], [.49, .0001, .51]]]))
    consensus = build_target(soft_winner_without_hard_support, np.array([0]), np.zeros((2, 3), dtype=np.uint8), method="consensus_logit")
    assert consensus.fallback.tolist() == [True]


def test_oracle_uses_correct_teacher_outputs_not_artificial_one_hot():
    z = np.array([[[4., 1.], [3., 0.]]]); y = np.array([0])
    q = build_target(z, y, np.ones((2, 2), dtype=np.uint8), method="oracle_logit", temperature=1).probabilities
    assert 0 < q[0, 1] < 1


def test_target_identity_changes_with_temperature_and_recipe():
    common = dict(method="feddf_logit", source_hash="source", proxy_hash="proxy", mask_hash="mask")
    first = metadata_identity(temperature=8, config={"epochs": 30}, **common)
    assert first != metadata_identity(temperature=2, config={"epochs": 30}, **common)
    assert first != metadata_identity(temperature=8, config={"epochs": 31}, **common)
    assert first != metadata_identity(temperature=8, config={"epochs": 30}, **{**common, "source_hash": "other"})
    assert first != metadata_identity(temperature=8, config={"epochs": 30}, **{**common, "mask_hash": "other"})


def test_proxy_hash_is_canonical_across_conditions_runner_and_audit():
    proxy = np.array([10, 2, 7], dtype=np.int64)
    expected = array_sha256(proxy)
    # Each component imports this one canonical implementation rather than
    # maintaining a subtly different dtype/shape-aware variant.
    assert "from article1.hashes import array_sha256" in Path("article1/runner.py").read_text()
    assert expected == conditions_component.array_sha256(proxy)
    assert expected == audit_component.array_sha256(proxy)


def test_analysis_and_audit_keep_temperature_observations_separate(tmp_path):
    rows = []
    for temperature, expert_accuracy in ((4.0, .70), (8.0, .90)):
        for method, accuracy in (("feddf_logit", .60), ("expert_logit", expert_accuracy)):
            rows.append({"dataset": "mnist", "regime": "iid", "seed": "42", "method": method,
                         "temperature": str(temperature), "student_test_accuracy": str(accuracy)})
    results = tmp_path / "results.csv"
    with results.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    effects = paired_effects(results, temperature=8)
    assert len(effects) == 1
    assert effects[0]["temperature"] == 8.0
    assert effects[0]["paired_seeds"] == 1
    assert effects[0]["mean_accuracy_delta"] == pytest.approx(.3)


def test_audit_accepts_consistent_single_condition(tmp_path):
    source = tmp_path / "sources" / "mnist-seed42-iid"; source.mkdir(parents=True)
    mask = np.ones((1, 2), dtype=np.uint8); proxy = np.array([4, 5]); logits = np.zeros((2, 1, 2), dtype=np.float32)
    np.savez_compressed(source / "teacher_cache.npz", proxy_idx=proxy, labels=np.array([0, 1]), logits=logits, M=mask,
                        holdout_accuracy=np.ones((1, 2)), holdout_counts=np.ones((1, 2)))
    cache_hash = hashlib.sha256((source / "teacher_cache.npz").read_bytes()).hexdigest()
    (source / "metadata.json").write_text(json.dumps({"dataset": "mnist", "seed": 42, "regime": "iid", "cache_sha256": cache_hash}))
    row = {"dataset": "mnist", "seed": "42", "regime": "iid", "method": "feddf_logit", "run_id": "run",
           "cache_sha256": cache_hash, "M_sha256": hashlib.sha256(mask.tobytes()).hexdigest(), "proxy_sha256": hashlib.sha256(proxy.tobytes()).hexdigest(),
           "student_init_sha256": "init", "batch_order_sha256": "order", "updates": "1", "temperature": "8"}
    results = tmp_path / "results.csv"
    with results.open("w", newline="") as handle: writer = csv.DictWriter(handle, fieldnames=row); writer.writeheader(); writer.writerow(row)
    report = audit(results, source_root=tmp_path / "sources", datasets=("mnist",), seeds=(42,), regimes=("iid",), methods=("feddf_logit",))
    assert report["ok"]


def test_backfill_only_changes_derived_target_columns_and_supports_dry_run(tmp_path):
    source = tmp_path / "sources" / "mnist-seed42-iid"; source.mkdir(parents=True)
    logits = np.array([[[2., 0.], [0., 2.]]], dtype=np.float32)
    np.savez_compressed(source / "teacher_cache.npz", logits=logits, labels=np.array([0]), M=np.array([[1, 0], [0, 1]], dtype=np.uint8))
    results = tmp_path / "results.csv"
    original = {"run_id": "immutable", "dataset": "mnist", "seed": "42", "regime": "iid", "method": "expert_prob",
                "temperature": "8", "student_test_accuracy": ".5", "student_test_nll": "1.2", "cache_sha256": "cache",
                "proxy_sha256": "proxy", "M_sha256": "mask", "mean_outside_support_mass": ".1"}
    with results.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=original); writer.writeheader(); writer.writerow(original)
    before = results.read_bytes()
    dry = backfill(results, tmp_path / "sources", dry_run=True)
    assert dry["dry_run"] and results.read_bytes() == before
    report = backfill(results, tmp_path / "sources")
    assert report["changed_rows"] == 1
    with results.open(newline="") as handle: migrated = next(csv.DictReader(handle))
    for field in ("run_id", "student_test_accuracy", "student_test_nll", "cache_sha256", "proxy_sha256", "M_sha256", "temperature"):
        assert migrated[field] == original[field]
    assert "mean_outside_support_mass" not in migrated
    assert migrated["effective_teachers"] and migrated["pre_restriction_outside_support_mass"]
