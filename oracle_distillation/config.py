"""Central execution configuration for KD runs.

This module separates *what* to run (execution spec / method policy) from the
training engine and CLI parsing. It centralizes run-wide knobs such as dataset,
selected method, warm-start policy, and thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class OptimizationSpec:
    epochs: int = 50
    batch_size: int = 256
    optimizer: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    temperature: float = 8.0
    patience: int = 5
    min_delta: float = 0.0
    val_frac: float = 0.0
    min_val: int = 0


@dataclass(frozen=True)
class MethodPolicy:
    name: str
    is_personal: bool = False
    # Whether this method warm-starts its student from a global checkpoint. The
    # checkpoint itself is whichever model is selected via global_source — there is
    # no separate "warm-start model" choice; warm-start and the class_mask anchor
    # always point at the SAME selected method (see build_execution_spec).
    warm_start: bool = False
    global_source: str = "expert"


METHOD_POLICIES: dict[str, MethodPolicy] = {
    "feddf": MethodPolicy("feddf"),
    "consensus": MethodPolicy("consensus"),
    "oracle": MethodPolicy("oracle"),
    "expert": MethodPolicy("expert"),
    # mask-free one-shot bases (global, server needs no competence mask)
    "confidence": MethodPolicy("confidence"),
    "energy": MethodPolicy("energy"),
    "personal_expert": MethodPolicy("personal_expert", is_personal=True),
    "personal_oracle": MethodPolicy("personal_oracle", is_personal=True),
    "personal_class_mask": MethodPolicy(
        "personal_class_mask",
        is_personal=True,
        warm_start=True,
        global_source="expert",
    ),
}


@dataclass(frozen=True)
class ExecutionSpec:
    dataset: str
    proxy_dataset: str
    method: str
    seed: int = 42
    device: str = "auto"
    student_arch: Optional[str] = None
    is_personal: bool = False
    warm_start_method: Optional[str] = None
    global_source: str = "expert"
    save_models: bool = False
    optimization: OptimizationSpec = field(default_factory=OptimizationSpec)

    @property
    def personal_method(self) -> Optional[str]:
        return self.method if self.is_personal else None

    @property
    def epochs(self) -> int:
        return self.optimization.epochs

    @property
    def batch_size(self) -> int:
        return self.optimization.batch_size

    @property
    def optimizer(self) -> str:
        return self.optimization.optimizer

    @property
    def lr(self) -> float:
        return self.optimization.lr

    @property
    def weight_decay(self) -> float:
        return self.optimization.weight_decay

    @property
    def temperature(self) -> float:
        return self.optimization.temperature

    @property
    def patience(self) -> int:
        return self.optimization.patience

    @property
    def min_delta(self) -> float:
        return self.optimization.min_delta

    @property
    def val_frac(self) -> float:
        return self.optimization.val_frac

    @property
    def min_val(self) -> int:
        return self.optimization.min_val


def build_execution_spec(
    *,
    args,
    dataset: str,
    method: str,
    is_personal: Optional[bool] = None,
    global_source: Optional[str] = None,
    min_delta: float = 0.0,
) -> ExecutionSpec:
    """Build the immutable ExecutionSpec for one KD method from CLI args.

    Shared by the global and personal CLIs. ``is_personal`` defaults from the
    policy; ``global_source`` defaults from the policy but the personal CLI
    overrides it via ``--global_source``. ``warm_start_method`` is derived from
    the resolved ``global_source`` when the policy warm-starts at all — warm-start
    and the class_mask anchor always point at the SAME selected method (e.g.
    ``--global_source oracle`` both warm-starts from and anchors on oracle).
    """
    policy = METHOD_POLICIES.get(method.lower())
    resolved_global_source = (global_source if global_source is not None
                               else (policy.global_source if policy else "expert"))
    return ExecutionSpec(
        dataset=dataset,
        proxy_dataset=dataset,
        method=method.lower(),
        seed=args.seed,
        device=args.device,
        student_arch=args.student_arch,
        is_personal=is_personal if is_personal is not None
                    else (bool(policy.is_personal) if policy else False),
        warm_start_method=resolved_global_source if (policy and policy.warm_start) else None,
        global_source=resolved_global_source,
        save_models=bool(getattr(args, "save_models", False)),
        optimization=OptimizationSpec(
            epochs=args.epochs,
            batch_size=args.batch_size,
            optimizer="adamw",
            lr=args.lr,
            weight_decay=args.weight_decay,
            temperature=args.temperature,
            patience=args.patience,
            min_delta=min_delta,
            val_frac=args.val_frac,
            min_val=0,
        ),
    )

