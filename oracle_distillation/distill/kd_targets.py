"""KdTargets — the KD learning goal, decoupled from how it was produced.

A KD run learns a ``[N_proxy, C]`` target-logits tensor over the proxy. Where it
came from (K global teachers collapsed by ProxyAnalysisBuilder, or a per-client
personal target) is irrelevant to training. This module owns loading that target
from ``proxy_analysis.npz`` and the small proxy-index IO the runner needs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from oracle_distillation.analysis.target_builders import GLOBAL_METHOD_TO_KEY


def load_proxy_indices(npz_path: Path) -> np.ndarray:
    """Read the proxy split's global dataset indices from an NPZ."""
    arr = np.load(npz_path)
    for k in ("proxy_idx", "indices"):
        if k in arr.files:
            return arr[k].astype(np.int64)
    raise KeyError(f"{npz_path}: no 'proxy_idx' or 'indices' key")


def resolve_analysis_npz(analysis_dir: Path) -> Path:
    """Locate proxy_analysis.npz under analysis_dir (or its proxy_analysis/ subdir)."""
    if analysis_dir.is_dir():
        cand = analysis_dir / "proxy_analysis.npz"
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"No proxy_analysis.npz under {analysis_dir}. "
        "Run ProxyAnalysisBuilder (analysis.proxy_analysis_builder) first."
    )


@dataclass
class KdTargets:
    """The goal a student learns: target logits + alignment + diagnostics labels."""
    logits: torch.Tensor                    # [N_proxy, C] on device
    proxy_idx: np.ndarray                    # [N_proxy] global dataset indices
    y_true: Optional[np.ndarray] = None      # [N_proxy] int64 (diagnostics)
    fine_to_coarse: Optional[np.ndarray] = None  # int64[C] for CIFAR-100, else None

    @classmethod
    def from_analysis_npz(
        cls, analysis_dir: Path, method: str, *, dataset: str, device: torch.device,
    ) -> "KdTargets":
        """Load a registered Phase-A target, including confidence/energy bases."""
        npz_path = resolve_analysis_npz(analysis_dir)
        key = GLOBAL_METHOD_TO_KEY.get(method)
        if key is None:
            raise ValueError(f"Unknown method '{method}'. Supported: {list(GLOBAL_METHOD_TO_KEY)}")

        data = np.load(npz_path)
        if key not in data.files:
            raise KeyError(f"{npz_path} does not contain '{key}'. Rebuild proxy_analysis.")
        if "dataset" in data.files:
            stored = str(data["dataset"][0])
            if stored != dataset:
                raise ValueError(
                    f"proxy_analysis.npz was built for dataset='{stored}' but this run "
                    f"uses '{dataset}'. Rebuild the analysis."
                )

        arr = np.asarray(data[key], dtype=np.float32)
        proxy_idx = data["proxy_idx"].astype(np.int64)
        if arr.shape[0] != proxy_idx.shape[0]:
            raise ValueError(f"{key} rows ({arr.shape[0]}) != proxy_idx rows ({proxy_idx.shape[0]})")

        y_true = data["y_true_proxy"].astype(np.int64) if "y_true_proxy" in data.files else None
        f2c = data["fine_to_coarse"].astype(np.int64) if "fine_to_coarse" in data.files else None
        return cls(torch.from_numpy(arr).to(device), proxy_idx, y_true, f2c)
