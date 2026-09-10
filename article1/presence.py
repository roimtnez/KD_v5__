"""Class-presence control: private TRAIN labels only, never expertise or test."""

from pathlib import Path

import numpy as np

from article1.hashes import array_sha256, file_sha256
from article1.partitioning import load_partitions


def presence_from_training(clients, labels, classes=10):
    """A[k,c] = 1 iff teacher k received class c for gradient updates."""
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.stack(
        [
            np.bincount(labels[client["train_idx"]], minlength=classes)
            for client in clients
        ]
    )
    if counts.shape != (len(clients), classes):
        raise ValueError("Training labels outside the class range")
    return (counts > 0).astype(np.uint8), counts


def load_presence(cache, metadata, labels):
    """Bind reconstructed presence to the exact partitions used by the teachers."""
    cache = Path(cache)
    # Prefer the relocatable standard layout, then the recorded original path.
    path = cache.parent.parent.parent / "partitions" / cache.parent.name
    if not path.is_dir():
        path = Path(metadata["proxy_source"]).parent
    if file_sha256(path / "metadata.json") != metadata["partition_metadata_sha256"]:
        raise ValueError("Presence partitions differ from teacher provenance")
    proxy, clients, meta = load_partitions(path, labels)
    if any(meta[key] != metadata[key] for key in ("dataset", "seed", "regime")):
        raise ValueError("Presence partition condition differs from teachers")
    with np.load(cache, allow_pickle=False) as data:
        if not np.array_equal(proxy, data["proxy_idx"]):
            raise ValueError("Presence proxy differs from teachers")
    mask, counts = presence_from_training(clients, labels)
    return mask, {
        "routing_mask_source": "private_train_class_presence",
        "presence_counts_sha256": array_sha256(counts),
        "partition_metadata_sha256": metadata["partition_metadata_sha256"],
    }


def compare_presence(presence, expert):
    """Paired EXPERT minus presence; masks differ intentionally, recipe must not."""
    import json

    import pandas as pd

    from article1.distillation import metadata_identity
    from article1.progress import KD

    keys = ["dataset", "regime", "seed"]
    left = expert[expert.method == "expert_prob"].copy()
    right = presence[presence.method == "presence_prob"].copy()
    for frame in (left, right):
        if not frame.empty and (
            not frame.temperature.eq(8).all() or not frame.proxy_size.eq(10000).all()
        ):
            raise ValueError("Presence comparison requires T=8 and N=10000")
        for _, source in frame.iterrows():
            expected = metadata_identity(
                method=source.method,
                temperature=float(source.temperature),
                config=json.loads(source.training_recipe_json),
                source_hash=source.cache_sha256,
                proxy_hash=source.proxy_sha256,
                mask_hash=source.M_sha256,
            )
            if source.run_id != expected:
                raise ValueError("Presence comparison run identity mismatch")
        if frame.duplicated(keys).any():
            raise ValueError("Duplicate condition in presence comparison")
        if "valid" in frame and not frame.valid.eq(True).all():
            raise ValueError("Invalid rows in presence comparison")
    joined = left.merge(
        right, on=keys, suffixes=("_expert", "_presence"), validate="one_to_one"
    )
    if len(joined) != len(right):
        raise ValueError("Presence result lacks EXPERT reference")
    metrics = [
        "student_test_accuracy",
        "student_test_nll",
        "target_accuracy",
        "target_nll",
        "target_entropy",
        "fallback_rate",
        "mean_selected_teachers",
    ]
    records = []
    for _, row in joined.iterrows():
        for field in KD:
            if field == "M_sha256":
                continue
            a, b = row[field + "_expert"], row[field + "_presence"]
            if field == "training_recipe_json":
                a, b = json.loads(a), json.loads(b)
            if a != b or (not isinstance(a, dict) and pd.isna(a)):
                raise ValueError(f"Unpaired presence experiment: {field}")
        if (
            row.get("expertise_M_sha256_presence", row.get("expertise_M_sha256"))
            != row.M_sha256_expert
        ):
            raise ValueError("Presence source expertise mask differs from EXPERT")
        if (
            row.get("routing_mask_source_presence", row.get("routing_mask_source"))
            != "private_train_class_presence"
        ):
            raise ValueError("Unknown presence-mask definition")
        record = {key: row[key] for key in keys}
        for metric in metrics:
            a, b = float(row[metric + "_expert"]), float(row[metric + "_presence"])
            if not np.isfinite([a, b]).all():
                raise ValueError(f"Nonfinite metric: {metric}")
            record["delta_" + metric] = a - b
        records.append(record)
    return pd.DataFrame(records)


def check_expert_references(frame, source_root):
    """Fail before spending compute if the full-proxy reference is incompatible."""
    import json
    from itertools import product

    from article1 import DATASETS, REGIMES, SEEDS
    from article1.distillation import kd_config

    expert = frame[frame.method.eq("expert_prob")]
    keys = ["dataset", "regime", "seed"]
    if expert.duplicated(keys).any() or set(map(tuple, expert[keys].to_numpy())) != set(
        product(DATASETS, REGIMES, SEEDS)
    ):
        raise ValueError("Presence requires all 54 unique EXPERT references")
    for _, row in expert.iterrows():
        if (
            row.temperature != 8
            or row.proxy_size != 10000
            or row.updates != 1200
            or json.loads(row.training_recipe_json) != kd_config(30, 256)
        ):
            raise ValueError("EXPERT reference differs from planned presence recipe")
        cache = (
            Path(source_root)
            / f"{row.dataset}-seed{row.seed}-{row.regime}"
            / "teacher_cache.npz"
        )
        if file_sha256(cache) != row.cache_sha256:
            raise ValueError("EXPERT reference uses different teacher cache")
        with np.load(cache, allow_pickle=False) as data:
            if array_sha256(data["M"]) != row.M_sha256:
                raise ValueError("EXPERT reference uses different expertise mask")


def main():
    """Analyze completed controls without changing source CSVs or training."""
    import argparse

    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--expert", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() in {args.results.resolve(), args.expert.resolve()}:
        raise ValueError("Analysis output must differ from source results")
    pairs = compare_presence(pd.read_csv(args.results), pd.read_csv(args.expert))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.output, index=False)
    print(
        f"EXPERT minus presence: {len(pairs)}/54 paired conditions; partial results are provisional"
    )


if __name__ == "__main__":
    main()
