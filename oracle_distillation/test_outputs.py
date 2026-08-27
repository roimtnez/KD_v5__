"""Compact, reproducible test predictions retained instead of terminal models."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from oracle_distillation.utils import collect_logits


def save_test_outputs(
    path: Path,
    *,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    dataset: str,
    method: str,
    seed: int,
    expected_accuracy: float | None = None,
) -> Path:
    """Atomically save test logits, predictions, labels and confusion matrix.

    Logits stay float32 so the archived surface exactly reproduces model argmax
    metrics.  The optional accuracy assertion prevents a mismatched loader from
    being silently attached to an otherwise valid result.
    """
    logits, labels = collect_logits(model, loader, device)
    predictions = logits.argmax(axis=1).astype(np.int64)
    accuracy = float(np.mean(predictions == labels))
    if expected_accuracy is not None and not np.isclose(
        accuracy, float(expected_accuracy), rtol=0.0, atol=1e-12,
    ):
        raise AssertionError(
            f"test output accuracy {accuracy} != evaluated accuracy {expected_accuracy}"
        )
    num_classes = int(logits.shape[1])
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(
            handle,
            logits=logits.astype(np.float32, copy=False),
            predictions=predictions,
            labels=labels,
            confusion_matrix=confusion,
            accuracy=np.array(accuracy, dtype=np.float64),
            dataset=np.array(str(dataset)),
            method=np.array(str(method)),
            seed=np.array(int(seed), dtype=np.int64),
        )
    tmp.replace(path)
    return path
