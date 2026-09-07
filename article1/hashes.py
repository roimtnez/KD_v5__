"""Canonical, stable hashes for Article-1 artifacts."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import numpy as np


def file_sha256(path: Path) -> str:
    """Hash a file's exact bytes without loading it wholly into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    """Hash the canonical contiguous byte representation of an index array.

    Proxy indices are always persisted as ``int64``.  Keeping this hash to the
    persisted bytes preserves compatibility with pre-provenance v2 caches,
    while giving every component one definition to use.
    """
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def git_commit(root: Path = Path(".")) -> str:
    """Return the creating source revision, never a guessed replacement."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def artifact_commit(metadata: dict, field: str = "cache_creation_commit") -> str:
    """Read declared provenance while explicitly labelling unverifiable v2 data."""
    value = metadata.get(field)
    return str(value) if value else "legacy_unverified"
