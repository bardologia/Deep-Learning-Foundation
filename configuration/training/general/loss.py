"""Loss term toggles, the two-stage curriculum and its inheritance rule.

A LossConfig is a flat set of toggle-and-weight pairs, one per loss term. The
curriculum holds two such stages and swaps between them at a fixed epoch; the
inheritance helper propagates complete-stage overrides back into the warmup
stage whenever the user did not set the warmup value explicitly.
"""

from __future__ import annotations

import copy

from dataclasses import dataclass, field, fields
from enum        import Enum


class ParamMatching(str, Enum):
    """Rule pairing predicted Gaussian slots with ground-truth components."""

    HUNGARIAN = "hungarian"
    SORTED_GT = "sorted_gt"


@dataclass
class LossConfig:
    """Enabled loss terms and their weights for one curriculum stage.

    Curve terms compare the reconstructed elevation profile against the target
    profile; parameter terms compare matched Gaussian slots; physics terms
    compare quantities resynthesised into the measurement domain.

    Attributes:
        use_mse_curve            : Whether the squared-error curve term is active.
        weight_mse_curve         : Weight of that term.
        use_l1_curve             : Whether the absolute-error curve term is active.
        weight_l1_curve          : Weight of that term.
        use_huber_curve          : Whether the Huber curve term is active.
        weight_huber_curve       : Weight of that term.
        huber_delta              : Transition point of the Huber curve term.
        use_charbonnier_curve    : Whether the Charbonnier curve term is active.
        weight_charbonnier_curve : Weight of that term.
        charbonnier_eps          : Smoothing constant of the Charbonnier curve term.
        use_cosine_curve         : Whether the cosine-similarity curve term is active.
        weight_cosine_curve      : Weight of that term.
        use_log_energy_curve     : Whether the log-energy curve term is active.
        weight_log_energy_curve  : Weight of that term.
        use_param_l1             : Whether the absolute-error parameter term is active.
        weight_param_l1          : Weight of that term.
        use_param_huber          : Whether the Huber parameter term is active.
        weight_param_huber       : Weight of that term.
        param_huber_delta        : Transition point of the Huber parameter term.
        use_param_mse            : Whether the squared-error parameter term is active.
        weight_param_mse         : Weight of that term.
        param_weights            : Relative weights of amplitude, mean and sigma inside the parameter terms.
        param_matching           : Rule pairing predicted slots with ground-truth components.
        use_active_normalization : Whether parameter errors are averaged over active slots only.
        presence_balance         : Whether the per-slot presence loss is class-balanced.
        active_weight            : Weight given to slots holding a scatterer.
        inactive_weight          : Weight given to empty slots.
        amp_focal_gamma          : Focusing exponent of the amplitude focal weighting; 0 disables it.
        amp_focal_delta          : Amplitude scale of that focal weighting.
        amp_zero_thr             : Amplitude below which a ground-truth slot counts as empty.
        use_smoothness_tv        : Whether the spatial total-variation regulariser is active.
        weight_smoothness_tv     : Weight of that regulariser.
        use_total_power          : Whether the total-power matching term is active.
        weight_total_power       : Weight of that term.
        use_moments              : Whether the profile moment matching term is active.
        weight_moments           : Weight of that term.
        moments_weights          : Relative weights of the matched moments.
        use_coherence_resyn      : Whether the resynthesised coherence term is active.
        weight_coherence_resyn   : Weight of that term.
        use_covariance_match     : Whether the covariance matching term is active.
        weight_covariance_match  : Weight of that term.
        use_capon_cycle          : Whether the Capon re-beamforming cycle term is active.
        weight_capon_cycle       : Weight of that term.
        capon_loading            : Diagonal loading added before the Capon inversion.
        physics_floor            : Numerical floor added to physics denominators.
    """

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

    use_log_energy_curve    : bool  = False
    weight_log_energy_curve : float = 0.0

    use_param_l1    : bool  = False
    weight_param_l1 : float = 0.1

    use_param_huber    : bool  = False
    weight_param_huber : float = 0.0
    param_huber_delta  : float = 0.5

    use_param_mse    : bool  = False
    weight_param_mse : float = 0.0

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

    power_floor : float = 1e-3


@dataclass
class LossCurriculumConfig:
    """Two-stage loss schedule swapping at a fixed epoch.

    When disabled, only the complete stage is ever used, so a single-stage run
    is expressed by leaving `enabled` False and filling `complete`.

    Attributes:
        enabled         : Whether the warmup stage precedes the complete stage.
        swap_epoch      : Epoch at which training swaps to the complete stage.
        inherit         : Whether unset warmup fields inherit their complete-stage overrides.
        warmup          : Loss terms active before the swap.
        complete        : Loss terms active after the swap.
        reset_lr        : Whether the learning rate is reset at the swap.
        reset_warmup    : Whether the learning-rate warmup restarts at the swap.
        reset_optimizer : Whether the optimizer state is cleared at the swap.
    """

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
        """Loss configuration training starts from."""
        return self.warmup if self.enabled else self.complete

    def active_stages(self) -> list[LossConfig]:
        """Returns the stages the run will actually visit, in order."""
        return [self.warmup, self.complete] if self.enabled else [self.complete]


class CurriculumInheritance:
    """Propagates complete-stage loss overrides into an untouched warmup stage.

    A field is inherited only when the user overrode it on the complete stage,
    left the warmup stage alone, and both stage defaults agreed to begin with,
    so an intentionally differing default is never silently overwritten.

    Attributes:
        curriculum : Curriculum being mutated in place.
        defaults   : Curriculum defaults the overrides were applied on top of.
        overrides  : Dotted override keys the user supplied.
        prefix     : Dotted prefix addressing the curriculum in those keys.
    """

    def __init__(self, curriculum: LossCurriculumConfig, defaults: LossCurriculumConfig, overrides: dict, prefix: str = "curriculum") -> None:
        """Stores the curriculum, its defaults and the user overrides.

        Args:
            curriculum : Curriculum whose warmup stage may be updated in place.
            defaults   : Curriculum defaults before the overrides were applied.
            overrides  : Dotted override keys the user supplied.
            prefix     : Dotted prefix addressing the curriculum in those keys.
        """
        self.curriculum = curriculum
        self.defaults   = defaults
        self.overrides  = overrides
        self.prefix     = prefix

    def _inheritable(self, name: str) -> bool:
        """Returns whether one loss field may be copied from complete to warmup."""
        if f"{self.prefix}.warmup.{name}" in self.overrides:
            return False

        if f"{self.prefix}.complete.{name}" not in self.overrides:
            return False

        return getattr(self.defaults.warmup, name) == getattr(self.defaults.complete, name)

    def apply(self) -> dict:
        """Copies the inheritable complete-stage values into the warmup stage.

        Returns:
            The field names that were inherited, mapped to the copied values; empty when inheritance is disabled.
        """
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
