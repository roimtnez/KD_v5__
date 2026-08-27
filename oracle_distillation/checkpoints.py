"""Small, metadata-complete model checkpoints with legacy-load compatibility."""
from __future__ import annotations

import hashlib
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import torch


SCHEMA_VERSION = 1


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor values and metadata without persisting a checkpoint.

    This is used by paired experimental arms to prove identical initial/final
    students while keeping checkpoint-free terminal-model policy intact.
    """
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def checkpoint_sha256(path: Path) -> str:
    """Return the content hash recorded by manifests for a persisted checkpoint."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip() or None
    except Exception:
        return None


def _portable_config(config: Any) -> Any:
    if is_dataclass(config):
        config = asdict(config)
    if isinstance(config, Mapping):
        return {str(k): _portable_config(v) for k, v in config.items()}
    if isinstance(config, (list, tuple)):
        return [_portable_config(v) for v in config]
    if isinstance(config, Path):
        return str(config)
    if config is None or isinstance(config, (str, bool, int, float)):
        return config
    return str(config)


def save_model_checkpoint(
    path: Path,
    model: torch.nn.Module,
    *,
    architecture: str,
    dataset: str,
    regime: str,
    method: str,
    seed: int,
    config: Any = None,
    repo_root: Path | None = None,
) -> Path:
    """Persist only weights and reproducibility metadata, atomically.

    Optimizer/scheduler state, gradients, datasets and Python model objects are
    deliberately excluded. Weights remain in their original precision.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    root = (repo_root or Path(__file__).resolve().parents[1]).resolve()
    payload = {
        "checkpoint_schema_version": SCHEMA_VERSION,
        "state_dict": model.state_dict(),
        "architecture": str(architecture),
        "dataset": str(dataset),
        "regime": str(regime),
        "method": str(method),
        "seed": int(seed),
        "config": _portable_config(config or {}),
        "git_commit": _git_commit(root),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def normalise_state_dict(obj: Any) -> dict[str, torch.Tensor]:
    """Extract weights from the new envelope or historical state_dict files."""
    if isinstance(obj, Mapping):
        for key in ("state_dict", "model_state_dict", "model", "student"):
            nested = obj.get(key)
            if isinstance(nested, Mapping):
                obj = nested
                break
    if not isinstance(obj, Mapping):
        raise TypeError("checkpoint is not a state_dict or a known checkpoint wrapper")
    state = {str(k): v for k, v in obj.items() if torch.is_tensor(v)}
    if not state:
        raise TypeError("checkpoint wrapper did not contain tensor weights")
    for prefix in ("module.", "model.", "student."):
        if all(key.startswith(prefix) for key in state):
            state = {key[len(prefix):]: value for key, value in state.items()}
            break
    return state


def load_checkpoint_state(
    path: Path, *, map_location: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    try:
        obj = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location=map_location)
    return normalise_state_dict(obj)
