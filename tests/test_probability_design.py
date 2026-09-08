import hashlib
import json

import numpy as np
import pytest

from article1.distillation import build_target, metadata_identity, softmax


def test_feddf_probability_is_arithmetic_not_geometric_pooling():
    z = np.array([[[4.0, 0.0, 1.0], [0.0, 2.0, 1.0]]])
    y = np.array([0])
    actual = build_target(z, y, None, method="feddf_prob", temperature=1)
    np.testing.assert_allclose(actual.probabilities, softmax(z, 1).mean(1))
    other = build_target(z, y, None, method="feddf_logit", temperature=1)
    assert not np.allclose(actual.probabilities, other.probabilities)


@pytest.mark.parametrize("temperature", [1.0, 4.0, 8.0])
def test_sr_keeps_all_selected_teachers_with_tiny_full_support_mass(temperature):
    z = np.array([[[-1000.0, -999.0, 0.0], [0.0, 0.0, 0.0]]])
    mask = np.array([[1, 1, 0], [1, 0, 1]])
    target = build_target(
        z, np.array([0]), mask, method="expert_prob_sr", temperature=temperature
    )
    expected = (
        np.r_[softmax(np.array([-1000.0, -999.0]), temperature), 0] + [0.5, 0, 0.5]
    ) / 2
    np.testing.assert_allclose(target.probabilities[0], expected, rtol=1e-6)
    np.testing.assert_allclose(target.weights, [[0.5, 0.5]])
    assert target.metrics["target_revision"] == 2


def test_sr_single_supported_class_survives_probability_underflow():
    target = build_target(
        np.array([[[-1000.0, 0.0]]]),
        np.array([0]),
        np.array([[1, 0]]),
        method="expert_prob_sr",
        temperature=1,
    )
    np.testing.assert_array_equal(target.probabilities, [[1.0, 0.0]])


def test_sr_empty_masks_use_identical_feddf_logit_fallback():
    z = np.array([[[1.0, 2.0], [3.0, 0.0]]])
    sr = build_target(z, np.array([0]), np.zeros((2, 2)), method="expert_prob_sr")
    baseline = build_target(z, np.array([0]), None, method="feddf_logit")
    np.testing.assert_array_equal(sr.probabilities, baseline.probabilities)


@pytest.mark.parametrize("method", ["expert_prob_sr", "expert_prob", "expert_logit"])
def test_only_corrected_sr_changes_legacy_identity(method):
    args = {
        "method": method,
        "temperature": 8.0,
        "config": {},
        "source_hash": "a",
        "proxy_hash": "b",
        "mask_hash": "c",
    }
    old = hashlib.sha256(
        json.dumps(args, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    assert (metadata_identity(**args) != old) == (method == "expert_prob_sr")


def test_fractional_expertise_mask_is_rejected_before_integer_cast():
    with pytest.raises(ValueError, match="binary"):
        build_target(
            np.zeros((1, 1, 2)),
            np.array([0]),
            np.array([[0.5, 1.0]]),
            method="expert_prob",
        )
