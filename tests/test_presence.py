import numpy as np
import pytest

from article1.distillation import build_target, softmax
from article1.presence import presence_from_training


def test_presence_uses_only_training_labels():
    clients = [
        {
            "train_idx": np.array([0, 1]),
            "validation_idx": np.array([2]),
            "expertise_idx": np.array([3]),
        }
    ]
    mask, counts = presence_from_training(clients, [0, 0, 1, 2], classes=3)
    np.testing.assert_array_equal(mask, [[1, 0, 0]])
    np.testing.assert_array_equal(counts, [[2, 0, 0]])
    changed, _ = presence_from_training(clients, [0, 0, 2, 1], classes=3)
    np.testing.assert_array_equal(mask, changed)


def test_presence_keeps_full_vectors_and_common_fallback():
    z = np.array(
        [[[4.0, 1.0, 0.0], [0.0, 3.0, 2.0]], [[2.0, 0.0, 1.0], [0.0, 1.0, 3.0]]]
    )
    mask = np.array([[1, 0, 0], [0, 1, 0]])
    q = build_target(z, np.array([0, 2]), mask, method="presence_prob")
    np.testing.assert_allclose(q.probabilities[0], softmax(z[0, 0], 8))
    assert q.probabilities[0, 2] > 0  # no support restriction
    np.testing.assert_allclose(q.probabilities[1], softmax(z[1].mean(0), 8))
    assert q.fallback.tolist() == [False, True]


def test_equal_masks_equal_targets_but_competence_can_change_routing():
    z = np.array([[[4.0, 0.0], [0.0, 3.0]]])
    y = np.array([0])
    presence = np.ones((2, 2), dtype=np.uint8)
    p = build_target(z, y, presence, method="presence_prob")
    e = build_target(z, y, presence, method="expert_prob")
    np.testing.assert_array_equal(p.probabilities, e.probabilities)
    expertise = np.array([[1, 0], [0, 1]])
    e = build_target(z, y, expertise, method="expert_prob")
    assert not np.allclose(e.probabilities, p.probabilities)


def test_sr_true_label_probability_not_smaller():
    rng = np.random.default_rng(42)
    z = rng.normal(size=(20, 4, 3)) * 5
    y = np.arange(20) % 3
    mask = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 0, 0]])
    full = build_target(z, y, mask, method="expert_prob")
    sr = build_target(z, y, mask, method="expert_prob_sr")
    assert np.all(
        sr.probabilities[np.arange(20), y]
        >= full.probabilities[np.arange(20), y] - 1e-7
    )
    assert sr.metrics["pre_restriction_outside_support_mass"] == pytest.approx(
        full.metrics["pre_restriction_outside_support_mass"]
    )


def test_presence_reconstruction_binds_original_partitions(tmp_path):
    from article1.hashes import file_sha256
    from article1.partitioning import make_partitions, save_partitions
    from article1.presence import load_presence

    labels = np.repeat(np.arange(10), 100)
    proxy, clients = make_partitions(labels, regime="single", seed=42, proxy_size=100)
    path = tmp_path / "partitions/mnist-seed42-single"
    meta = {"dataset": "mnist", "regime": "single", "seed": 42}
    save_partitions(
        path, proxy_idx=proxy, clients=clients, labels=labels, metadata=meta
    )
    cache = tmp_path / "sources/mnist-seed42-single/teacher_cache.npz"
    cache.parent.mkdir(parents=True)
    np.savez(cache, proxy_idx=proxy)
    meta.update(partition_metadata_sha256=file_sha256(path / "metadata.json"))
    mask, _ = load_presence(cache, meta, labels)
    np.testing.assert_array_equal(mask, np.eye(10, dtype=np.uint8))
    with pytest.raises(ValueError, match="labels"):
        load_presence(cache, meta, (labels + 1) % 10)
    meta["partition_metadata_sha256"] = "wrong"
    with pytest.raises(ValueError, match="provenance"):
        load_presence(cache, meta, labels)


def test_presence_pair_rejects_unpaired_training_and_mask_provenance():
    import json

    import pandas as pd

    from article1.distillation import metadata_identity
    from article1.presence import compare_presence
    from article1.progress import KD

    common = {key: "same" for key in KD}
    common.update(
        dataset="mnist",
        regime="iid",
        seed=42,
        proxy_size=10000,
        temperature=8.0,
        training_recipe_json=json.dumps({"epochs": 30}),
        student_test_accuracy=0.9,
        student_test_nll=0.3,
        target_accuracy=0.8,
        target_nll=0.4,
        target_entropy=0.5,
        fallback_rate=0.0,
        mean_selected_teachers=5.0,
    )
    expert = dict(common, method="expert_prob", M_sha256="expert-mask")
    presence = dict(
        common,
        method="presence_prob",
        M_sha256="presence-mask",
        expertise_M_sha256="expert-mask",
        routing_mask_source="private_train_class_presence",
    )
    for row in (expert, presence):
        row["run_id"] = metadata_identity(
            method=row["method"],
            temperature=8.0,
            config={"epochs": 30},
            source_hash=row["cache_sha256"],
            proxy_hash=row["proxy_sha256"],
            mask_hash=row["M_sha256"],
        )
    left, right = pd.DataFrame([presence]), pd.DataFrame([expert])
    assert compare_presence(left, right).delta_student_test_accuracy.iloc[0] == 0
    left.loc[0, "consumed_batches_sha256"] = "different"
    with pytest.raises(ValueError, match="consumed_batches"):
        compare_presence(left, right)
    left.loc[0, "consumed_batches_sha256"] = "same"
    left.loc[0, "expertise_M_sha256"] = "wrong"
    with pytest.raises(ValueError, match="expertise mask"):
        compare_presence(left, right)
