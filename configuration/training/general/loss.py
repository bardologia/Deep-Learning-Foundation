from __future__ import annotations

import copy

from dataclasses import dataclass, field, fields
from enum        import Enum


class ParamMatching(str, Enum):
    HUNGARIAN = "hungarian"
    SORTED_GT = "sorted_gt"


@dataclass
class LossConfig:
    use_mse_curve    : bool  = False
    weight_mse_curve : float = 1.0

    use_l1_curve    : bool  = False
    weight_l1_curve : float = 0.0

    use_huber_curve    : bool  = False
    weight_huber_curve : float = 0.0
    huber_delta        : float = 1.0

    use_charbonnier_curve    : bool  = False
    weight_charbonnier_curve : float = 0.0
    charbonnier_eps          : float = 1e-3

    use_cosine_curve    : bool  = False
    weight_cosine_curve : float = 0.0

    use_param_l1    : bool  = False
    weight_param_l1 : float = 0.1

    use_param_huber    : bool  = False
    weight_param_huber : float = 0.0
    param_huber_delta  : float = 0.5

    use_param_mse    : bool  = False
    weight_param_mse : float = 0.0

    use_param_legacy    : bool  = False
    weight_param_legacy : float = 1.0
    legacy_bounds_min   : tuple = (1e-05, -15.0, 0.01, 1e-05, 1.0, 1.0)
    legacy_bounds_max   : tuple = (10.0, 5.0, 5.0, 10.0, 40.0, 20.0)

    param_weights : tuple = (1.0, 1.0, 1.0)

    param_matching : ParamMatching = ParamMatching.HUNGARIAN

    use_active_normalization : bool  = False
    presence_balance         : bool  = False
    active_weight            : float = 1.0
    inactive_weight          : float = 1.0
    amp_focal_gamma          : float = 0.0
    amp_focal_delta          : float = 0.5

    amp_zero_thr : float = 1e-4

    use_smoothness_tv    : bool  = False
    weight_smoothness_tv : float = 1e-4

    use_total_power    : bool  = False
    weight_total_power : float = 0.0

    use_moments     : bool  = False
    weight_moments  : float = 0.0
    moments_weights : tuple = (1.0, 1.0, 1.0)

    use_coherence_resyn    : bool  = False
    weight_coherence_resyn : float = 0.0

    use_covariance_match    : bool  = False
    weight_covariance_match : float = 0.0

    use_capon_cycle    : bool  = False
    weight_capon_cycle : float = 0.0
    capon_loading      : float = 1e-2

    physics_floor : float = 1e-3


@dataclass
class LossCurriculumConfig:
    enabled    : bool = False
    swap_epoch : int  = 0
    inherit    : bool = True

    warmup   : LossConfig = field(default_factory=LossConfig)
    complete : LossConfig = field(default_factory=LossConfig)

    reset_lr        : bool = False
    reset_warmup    : bool = False
    reset_optimizer : bool = False

    @property
    def initial_stage(self) -> LossConfig:
        return self.warmup if self.enabled else self.complete

    def active_stages(self) -> list[LossConfig]:
        return [self.warmup, self.complete] if self.enabled else [self.complete]


class CurriculumInheritance:
    def __init__(self, curriculum: LossCurriculumConfig, defaults: LossCurriculumConfig, overrides: dict, prefix: str = "curriculum") -> None:
        self.curriculum = curriculum
        self.defaults   = defaults
        self.overrides  = overrides
        self.prefix     = prefix

    def _inheritable(self, name: str) -> bool:
        if f"{self.prefix}.warmup.{name}" in self.overrides:
            return False

        if f"{self.prefix}.complete.{name}" not in self.overrides:
            return False

        return getattr(self.defaults.warmup, name) == getattr(self.defaults.complete, name)

    def apply(self) -> dict:
        if not self.curriculum.inherit:
            return {}

        inherited = {}
        for spec in fields(LossConfig):
            if not self._inheritable(spec.name):
                continue

            value = copy.deepcopy(getattr(self.curriculum.complete, spec.name))
            setattr(self.curriculum.warmup, spec.name, value)
            inherited[spec.name] = value

        return inherited
