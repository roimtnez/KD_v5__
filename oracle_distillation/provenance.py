"""Persist portable execution provenance for resumable experiment commands."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip() or None
    except Exception:
        return None


def _portable(value: Any, root: Path) -> Any:
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        path = Path(value)
        if path.is_absolute():
            try:
                return str(path.resolve().relative_to(root.resolve()))
            except ValueError:
                return value
        return value
    if isinstance(value, (list, tuple)):
        return [_portable(v, root) for v in value]
    if isinstance(value, dict):
        return {str(k): _portable(v, root) for k, v in value.items()}
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)


def write_execution_provenance(
    config_dir: Path,
    *,
    name: str,
    experiment_id: str,
    seed: int,
    args: Namespace | Mapping[str, Any],
    input_paths: Mapping[str, Any],
    output_paths: Mapping[str, Any],
    repo_root: Path | None = None,
) -> Path:
    """Atomically write one deterministic config file for an execution scope."""
    root = (repo_root or Path.cwd()).resolve()
    config_dir = Path(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    values = vars(args) if isinstance(args, Namespace) else dict(args)
    payload = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "seed": int(seed),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(root),
        "arguments": _portable(values, root),
        "inputs": _portable(dict(input_paths), root),
        "outputs": _portable(dict(output_paths), root),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    path = config_dir / f"{name}.json"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
    return path
