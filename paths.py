"""Portable repository path constants kept for legacy import compatibility.

Older modules import these names from a root-level ``paths`` module.  Keeping
the constants relative to this file makes the checkout relocatable and avoids
reviving the historical absolute KD_v4 paths embedded in old run configs.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"

PROXY_CIFAR_PATH = SPLITS_DIR / "cifar10_train_proxy_10000_seed_42.npz"
PROXY_MNIST_PATH = SPLITS_DIR / "proxy_mnist_10k.npz"
PROXY_FMNIST_PATH = SPLITS_DIR / "proxy_fmnist_10k.npz"
PROXY_CINIC_PATH = SPLITS_DIR / "proxy_cinic.npz"
PROXY_CIFAR100_PATH = SPLITS_DIR / "proxy_cifar100_10k.npz"
