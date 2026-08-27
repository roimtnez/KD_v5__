"""Frozen protocol constants for the first Article-1 pilot."""
from __future__ import annotations

from dataclasses import asdict, dataclass

PILOT_REGIMES = ("iid", "alpha0p1", "multi", "single")
PILOT_METHODS = (
    "feddf",
    "expert_full",
    "expert_support",
    "oracle_full",
    "oracle_maskgated_full",
    "oracle_maskgated_support",
)
SUPERVISED_METHODS = ("supervised_proxy_standard", "supervised_proxy_matched")
PROXY_BUDGETS = (100, 250, 500, 1000, 2500, 5000, 10000)


@dataclass(frozen=True)
class Article1PilotConfig:
    article: str = "article_1_server_expertise"
    experiment_version: str = "article1_support_v1"
    seed: int = 42
    dataset: str = "cifar"
    temperature: float = 8.0
    student_arch: str = "resnet9"
    epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    mask_semantics: str = "legacy_mask"
    mask_source: str = "local_test_acc_per_class>=0.7"
    proxy_size: int = 10_000
    labeled_proxy_size: int = 10_000
    student_init_seed: int = 42
    batch_order_seed: int = 42
    training_recipe: str = "adamw_30ep_bs256_lr1e-3_wd1e-4_no_scheduler_no_es"

    @property
    def kd_updates(self) -> int:
        # ceil(10_000 / 256) == 40; 30 epochs == 1,200 updates.
        return self.epochs * ((self.proxy_size + self.batch_size - 1) // self.batch_size)

    def to_dict(self) -> dict:
        return {**asdict(self), "kd_updates": self.kd_updates}
