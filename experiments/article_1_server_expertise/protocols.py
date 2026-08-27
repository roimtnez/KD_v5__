"""Immutable, Article-1-only protocol specifications.

The three quantities below are intentionally separate: how many public
examples exist, their class composition, and the source of teacher expertise.
In particular, ``proxy_size`` never means "number of labels revealed".
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


Composition = Literal["balanced", "uniform", "long_tail", "reduced_coverage"]
OptimizationBudget = Literal["fixed_epochs", "fixed_updates"]
AuthoritySource = Literal["known", "estimated_soft", "estimated_hard"]


@dataclass(frozen=True)
class ProxyDesignSpec:
    """A deterministic subset of a fixed public-proxy pool.

    ``reduced_coverage`` is an intentionally adversarial representativeness
    arm; it removes whole classes.  ``long_tail`` keeps all classes represented
    while changing their proportions, making coverage distinguishable from
    class balance at an equal total size.
    """

    size: int
    composition: Composition = "balanced"
    seed: int = 42
    namespace: str = "article1_proxy_v1"
    dropped_classes: tuple[int, ...] = ()
    long_tail_ratio: float = 8.0

    def validate(self, num_classes: int, available: int) -> None:
        if not 1 <= self.size <= available:
            raise ValueError(f"proxy size must be in [1,{available}]")
        if self.composition not in {"balanced", "uniform", "long_tail", "reduced_coverage"}:
            raise ValueError(f"unknown proxy composition {self.composition!r}")
        if self.long_tail_ratio < 1.0:
            raise ValueError("long_tail_ratio must be at least one")
        dropped = tuple(sorted(set(self.dropped_classes)))
        if dropped != self.dropped_classes:
            raise ValueError("dropped_classes must be sorted and unique")
        if any(cls < 0 or cls >= num_classes for cls in dropped):
            raise ValueError("dropped class is outside the label range")
        if self.composition == "reduced_coverage" and not dropped:
            raise ValueError("reduced_coverage requires at least one dropped class")
        if self.composition != "reduced_coverage" and dropped:
            raise ValueError("dropped_classes is only valid for reduced_coverage")
        if len(dropped) >= num_classes:
            raise ValueError("at least one class must remain covered")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TrainingBudgetSpec:
    """Optimization budget, reported alongside every proxy-size comparison."""

    mode: OptimizationBudget = "fixed_epochs"
    epochs: int = 30
    updates: int | None = None

    def planned_updates(self, proxy_size: int, batch_size: int) -> int:
        if self.mode == "fixed_epochs":
            if self.epochs < 1:
                raise ValueError("epochs must be positive")
            return self.epochs * ((proxy_size + batch_size - 1) // batch_size)
        if self.mode == "fixed_updates":
            if self.updates is None or self.updates < 1:
                raise ValueError("fixed_updates requires a positive updates value")
            return self.updates
        raise ValueError(f"unknown optimization budget {self.mode!r}")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ExpertiseEstimateSpec:
    """Estimator settings for Article 1, never a label-light protocol.

    Calibration labels are required and must originate from a pre-declared set
    disjoint from the public proxy and all evaluation sets.  The runner records
    this declaration as provenance instead of assuming it from file names.
    """

    source: AuthoritySource = "known"
    threshold: float = 0.7
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    min_class_examples: int = 5
    calibration_role: str = "external_disjoint_competence_calibration"

    def validate(self) -> None:
        if self.source not in {"known", "estimated_soft", "estimated_hard"}:
            raise ValueError(f"unknown authority source {self.source!r}")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0,1]")
        if self.prior_alpha <= 0 or self.prior_beta <= 0:
            raise ValueError("Beta prior parameters must be positive")
        if self.min_class_examples < 1:
            raise ValueError("min_class_examples must be positive")
        if self.source != "known" and self.calibration_role != "external_disjoint_competence_calibration":
            raise ValueError(
                "estimated expertise is valid only with an explicitly external, "
                "disjoint competence-calibration role"
            )

    def to_dict(self) -> dict:
        return asdict(self)
