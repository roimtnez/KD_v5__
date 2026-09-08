"""Protocol v3: no overlap and no expertise feedback into checkpoint selection."""

import json

import numpy as np
import pytest

from article1 import PROTOCOL_VERSION, REGIMES
from article1.partitioning import (
    ROLES,
    _split_client,
    load_partitions,
    make_partitions,
    save_partitions,
    validate_splits,
)


def test_stratification_integer_rounding_and_rare_classes():
    for n in range(1, 101):
        labels = np.zeros(n, dtype=np.int64)
        split = _split_client(np.arange(n), labels, seed=42)
        counts = np.array([len(split[r + "_idx"]) for r in ROLES])
        assert counts.sum() == n
        assert counts[0] >= 1
        assert np.all(np.abs(counts - n * np.array([0.7, 0.1, 0.2])) < 1)
        assert set(np.concatenate(list(split.values()))) == set(range(n))
    one = _split_client(np.array([0]), np.array([0]), seed=42)
    assert one["train_idx"].tolist() == [0]
    assert one["expertise_idx"].size == one["validation_idx"].size == 0


@pytest.mark.parametrize("regime", REGIMES)
def test_all_regimes_cover_data_and_keep_assigned_support(regime):
    labels = np.repeat(np.arange(10), 1000)
    proxy, clients = make_partitions(labels, regime=regime, seed=42, proxy_size=1000)
    validate_splits(proxy, clients, total_examples=len(labels))
    repeat_proxy, repeat_clients = make_partitions(
        labels, regime=regime, seed=42, proxy_size=1000
    )
    np.testing.assert_array_equal(proxy, repeat_proxy)
    for cid, client in enumerate(clients):
        for role in ROLES:
            np.testing.assert_array_equal(
                client[role + "_idx"], repeat_clients[cid][role + "_idx"]
            )
        combined = np.concatenate(list(client.values()))
        for c in np.unique(labels[combined]):
            n = (labels[combined] == c).sum()
            actual = np.array([(labels[client[r + "_idx"]] == c).sum() for r in ROLES])
            assert np.all(np.abs(actual - n * np.array([0.7, 0.1, 0.2])) < 1)
        if regime in ("single", "multi"):
            assert len(np.unique(labels[combined])) == (1 if regime == "single" else 2)


def test_manifest_and_protocol_prevent_accidental_reuse(tmp_path):
    labels = np.repeat(np.arange(10), 100)
    proxy, clients = make_partitions(labels, regime="iid", seed=42, proxy_size=100)
    save_partitions(
        tmp_path,
        proxy_idx=proxy,
        clients=clients,
        labels=labels,
        metadata={"dataset": "mnist", "seed": 42, "regime": "iid"},
    )
    load_partitions(tmp_path, labels)
    with pytest.raises(FileExistsError):
        save_partitions(
            tmp_path, proxy_idx=proxy, clients=clients, labels=labels, metadata={}
        )
    with pytest.raises(ValueError, match="labels"):
        load_partitions(tmp_path, 9 - labels)
    file = tmp_path / "client_000.npz"
    with file.open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="manifest"):
        load_partitions(tmp_path, labels)
    meta = tmp_path / "metadata.json"
    payload = json.loads(meta.read_text())
    payload["protocol_version"] = "article1-v2"
    meta.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="incompatible"):
        load_partitions(tmp_path)


def test_missing_empty_and_overlapping_splits_fail():
    clients = [
        {
            "train_idx": np.array([1]),
            "validation_idx": np.array([2]),
            "expertise_idx": np.array([3]),
        }
    ]
    with pytest.raises(ValueError, match="cover"):
        validate_splits(np.array([0]), clients, total_examples=5)
    clients[0]["expertise_idx"] = np.array([1])
    with pytest.raises(ValueError, match="overlap"):
        validate_splits(np.array([0]), clients)
    clients[0]["expertise_idx"] = np.array([], dtype=np.int64)
    with pytest.raises(ValueError, match="empty"):
        validate_splits(np.array([0]), clients)


def test_teacher_expertise_never_selects_or_updates_checkpoint(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from article1 import local_training as local

    torch.set_num_threads(1)
    labels = np.repeat(np.arange(10), 20)
    proxy, clients = make_partitions(labels, regime="single", seed=42, proxy_size=100)
    partitions = tmp_path / "partitions"
    save_partitions(
        partitions,
        proxy_idx=proxy,
        clients=clients,
        labels=labels,
        metadata={"dataset": "mnist", "seed": 42, "regime": "single"},
    )

    class Toy(torch.utils.data.Dataset):
        targets = labels

        def __len__(self):
            return len(labels)

        def __getitem__(self, i):
            return torch.tensor([float(i) / len(labels), 1.0]), int(labels[i])

    toy = Toy()
    monkeypatch.setattr(local, "datasets_for", lambda *args: (toy, toy, None))
    monkeypatch.setattr(local, "build_model", lambda _: torch.nn.Linear(2, 10))
    roles = {tuple(proxy.tolist()): "proxy"}
    for c in clients:
        for role in ROLES:
            roles[tuple(c[role + "_idx"].tolist())] = role
    phase = {"flip": False}
    calls = []
    original = local.logits_for

    def observed(model, loader, device):
        role = roles[tuple(loader.dataset.indices)]
        logits, y = original(model, loader, device)
        calls.append(role)
        if role == "expertise":
            # Contradictory competence evidence must not affect model selection.
            logits = np.zeros_like(logits)
            logits[np.arange(len(y)), (y + int(phase["flip"])) % 10] = 10
        return logits, y

    monkeypatch.setattr(local, "logits_for", observed)
    args = {
        "dataset": "mnist",
        "data_dir": tmp_path,
        "partition_dir": partitions,
        "seed": 42,
        "regime": "single",
        "epochs": 2,
        "batch_size": 8,
        "device": "cpu",
    }
    first = local.train_and_cache(**args, output_dir=tmp_path / "first")
    phase["flip"] = True
    second = local.train_and_cache(**args, output_dir=tmp_path / "second")
    a = json.loads(first.with_name("metadata.json").read_text())
    b = json.loads(second.with_name("metadata.json").read_text())
    assert a["teacher_state_sha256"] == b["teacher_state_sha256"]
    assert a["selection_records"] == b["selection_records"]
    assert calls == ["validation", "validation", "expertise", "proxy"] * 20
    with np.load(first) as x, np.load(second) as y:
        np.testing.assert_array_equal(x["logits"], y["logits"])
        assert x["M"].sum() == 10 and y["M"].sum() == 0
        assert not {"holdout_accuracy", "test_accuracy", "test_counts"} & set(x.files)
    assert a["protocol_version"] == PROTOCOL_VERSION
