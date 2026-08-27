from __future__ import annotations

from pathlib import Path
from typing import List

import torch

from data.dataset_config import get_dataset_config
from oracle_distillation.models import build_model, dataset_default_arch
from oracle_distillation.checkpoints import load_checkpoint_state


def list_teachers(teachers_root: Path) -> List[Path]:
    tdir = teachers_root / "teachers" if (teachers_root / "teachers").is_dir() else teachers_root
    files = sorted(tdir.glob("cid_*.pt"))
    if not files:
        raise FileNotFoundError(f"No teacher checkpoints found in: {tdir}")
    return files


def load_teachers(teacher_paths: List[Path], dataset: str, device: torch.device) -> List[torch.nn.Module]:
    ds_cfg = get_dataset_config(dataset)
    num_classes = ds_cfg.num_classes
    arch = ds_cfg.arch
    teachers: List[torch.nn.Module] = []
    for p in teacher_paths:
        m = build_model(arch, num_classes=num_classes).to(device)
        sd = load_checkpoint_state(p, map_location=device)
        m.load_state_dict(sd, strict=True)
        m.eval()
        teachers.append(m)
    return teachers
