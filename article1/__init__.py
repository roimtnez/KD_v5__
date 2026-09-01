"""Small, reproducible pipeline for Article 1.

The package has one active target implementation: :mod:`article1.distillation`.
It deliberately does not read historical Article-1 artifacts.
"""

PROTOCOL_VERSION = "article1-v2"
DATASETS = ("mnist", "fmnist", "cifar")
REGIMES = ("iid", "alpha1p0", "alpha0p5", "alpha0p1", "multi", "single")
SEEDS = (42, 43, 44)
THRESHOLDS = {"mnist": 0.9, "fmnist": 0.8, "cifar": 0.7}
