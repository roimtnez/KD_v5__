"""Content-addressed, write-once storage helpers for Article-1 artifacts."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def canonical_hash(payload: dict) -> str:
    raw = json.dumps(jsonable(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def write_json_once(path: Path, payload: dict) -> None:
    """Write atomically; byte-identical scientific artifacts may be resumed."""
    path = Path(path)
    serialized = json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n"
    if path.exists():
        existing_payload = json.loads(path.read_text(encoding="utf-8"))
        candidate = jsonable(payload)
        if "created_at" in existing_payload and "created_at" in candidate:
            candidate["created_at"] = existing_payload["created_at"]
        if path.read_text(encoding="utf-8") != json.dumps(candidate, indent=2, sort_keys=True) + "\n":
            raise FileExistsError(f"refusing to overwrite non-identical artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(serialized)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_npz_once(path: Path, **arrays) -> None:
    path = Path(path)
    if path.exists():
        with np.load(path, allow_pickle=False) as existing:
            if set(existing.files) != set(arrays):
                raise FileExistsError(f"existing artifact schema differs: {path}")
            for key, value in arrays.items():
                persisted = np.asarray(existing[key])
                candidate = np.asarray(value)
                # ``equal_nan`` dispatches ``isnan`` and is therefore invalid
                # for provenance strings such as SHA-256 scalar arrays.
                supports_nan = (
                    np.issubdtype(persisted.dtype, np.inexact)
                    and np.issubdtype(candidate.dtype, np.inexact)
                )
                same = np.array_equal(
                    persisted, candidate, equal_nan=True,
                ) if supports_nan else np.array_equal(persisted, candidate)
                if not same:
                    raise FileExistsError(f"existing artifact differs at {key}: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".npz.tmp")
    os.close(fd)
    try:
        with open(temporary, "wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
