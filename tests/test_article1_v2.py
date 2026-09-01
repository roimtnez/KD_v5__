import numpy as np
import csv
import hashlib
import json

from article1.distillation import METHODS, authority_from_holdout, build_target, metadata_identity
from article1.partitioning import make_partitions, validate_splits
from article1.audit import audit


def test_holdout_authority_requires_observations():
    M = authority_from_holdout(np.array([[.99, .99]]), np.array([[2, 0]]), .9)
    assert M.tolist() == [[1, 0]]


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
