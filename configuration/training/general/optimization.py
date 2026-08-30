"""Optimizer, scheduler, warmup, early-stopping and gradient-clipping settings.

Holds the optimization sub-configs shared by every training entry point, together
with the enums that name the supported scheduler, warmup and clipping strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum        import Enum


class SchedulerType(Enum):
    """Learning-rate schedule families selectable for a training run."""

    COSINE_ANNEALING = "cosine_annealing"
    CONSTANT         = "constant"
    LINEAR           = "linear"
    POLYNOMIAL       = "polynomial"
    EXPONENTIAL      = "exponential"
    STEP             = "step"


class WarmupMode(Enum):
    """Shapes of the learning-rate ramp applied over the warmup steps."""

    LINEAR      = "linear"
    COSINE      = "cosine"
    EXPONENTIAL = "exponential"
    POLYNOMIAL  = "polynomial"


class ClipMode(Enum):
    """Gradient-norm clipping strategies, from disabled to running-statistics based."""

    DISABLED            = "disabled"
    FIXED               = "fixed"
    ADAPTIVE_PERCENTILE = "adaptive_percentile"
    ADAPTIVE_MEAN_STD   = "adaptive_mean_std"


@dataclass
class OptimizerConfig:
    """AdamW hyper-parameters for the trained parameter groups.

    Attributes:
        betas        : Adam exponential decay rates for the first and second moments.
        eps          : Numerical stabiliser added to the denominator of the Adam update.
        weight_decay : Decoupled weight decay coefficient.
        lr_scale     : Multiplier applied to the base learning rate for this group.
    """

    betas        : tuple = (0.9, 0.999)
    eps          : float = 1e-8
    weight_decay : float = 0.1
    lr_scale     : float = 1.0


@dataclass
class SchedulerConfig:
    """Learning-rate schedule settings applied after warmup.

    Attributes:
        type      : Value of a `SchedulerType` naming the schedule family.
        epochs    : Number of epochs the schedule is stretched over.
        eta_min   : Floor learning rate reached at the end of a cosine schedule.
        step_size : Epoch interval between decays for the step schedule.
        gamma     : Multiplicative decay factor for the step and exponential schedules.
        power     : Exponent of the polynomial schedule.
    """

    type      : str   = "cosine_annealing"
    epochs    : int   = 100
    eta_min   : float = 1e-6
    step_size : int   = 30
    gamma     : float = 0.1
    power     : float = 1.0


@dataclass
class WarmupConfig:
    """Learning-rate warmup settings applied over the first optimizer steps.

    Attributes:
        warmup_fraction     : Fraction of the run's total optimizer steps spent ramping the learning rate.
        warmup_start_factor : Fraction of the base learning rate used at step zero.
        warmup_enabled      : Whether the warmup ramp runs at all.
        warmup_mode         : Value of a `WarmupMode` naming the ramp shape.
        warmup_poly_power   : Exponent used by the polynomial ramp.
    """

    warmup_fraction     : float = 0.03
    warmup_start_factor : float = 0.1
    warmup_enabled      : bool  = True
    warmup_mode         : str   = "linear"
    warmup_poly_power   : float = 2.0


@dataclass
class EarlyStoppingConfig:
    """Validation-loss early-stopping settings.

    Attributes:
        patience     : Validation rounds without improvement tolerated before stopping.
        min_delta    : Smallest validation-loss drop that counts as an improvement.
        restore_best : Whether the best checkpoint is reloaded once training stops.
    """

    patience     : int   = 15
    min_delta    : float = 0.0
    restore_best : bool  = True


@dataclass
class GradientClipperConfig:
    """Gradient-norm clipping settings, fixed or driven by running statistics.

    Attributes:
        clip_mode           : Value of a `ClipMode` naming the clipping strategy.
        max_grad_norm       : Norm threshold used by the fixed strategy.
        adaptive_window     : Number of recent gradient norms kept for the adaptive modes.
        adaptive_percentile : Percentile of the window used as the adaptive threshold.
        adaptive_mean_std_k : Standard deviations above the window mean used as threshold.
        clip_epsilon        : Stabiliser added to the norm before the scaling ratio is formed.
        log_histogram_freq  : Optimizer-step interval between gradient-norm histograms.
    """

    clip_mode           : str   = "fixed"
    max_grad_norm       : float = 1.0
    adaptive_window     : int   = 200
    adaptive_percentile : float = 95.0
    adaptive_mean_std_k : float = 2.0
    clip_epsilon        : float = 1e-6
    log_histogram_freq  : int   = 100
