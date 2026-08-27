"""Shared argparse wiring for the KD CLIs (global + personal)."""
from __future__ import annotations

import argparse


def add_training_args(ap: argparse.ArgumentParser, *, val_frac_default: float) -> None:
    """Register the training/optimization flags shared by both KD CLIs.

    ``val_frac_default`` is the one knob that legitimately differs: global KD
    runs without validation (0.0) by default, personal KD enables early
    stopping on the KD val loss (0.15).
    """
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=8.0)
    ap.add_argument("--student_arch", type=str, default=None,
                    help="Student architecture. Defaults to 'mnist_net' for mnist, 'resnet9' otherwise.")
    ap.add_argument("--arch", type=str, default=None,
                    help="Override the dataset's backbone for this run (teachers + students). "
                         "E.g. for cifar100: 'resnet18' (scratch, default) or "
                         "'resnet18_pretrained' (ImageNet body + CIFAR stem surgery, opt-in).")
    ap.add_argument("--val_frac", type=float, default=val_frac_default,
                    help="Fraction of proxy used as validation set during KD training. "
                         "0.0 = no validation, no early stopping.")
    ap.add_argument("--patience", type=int, default=5,
                    help="Early-stopping patience (epochs without improvement). "
                         "Only active when val_frac > 0.")
    ap.add_argument(
        "--save_models",
        action="store_true",
        help="Persist minimal metadata + state_dict checkpoints. Disabled by default; metrics and logits caches remain saved.",
    )
