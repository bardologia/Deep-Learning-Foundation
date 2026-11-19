
from __future__ import annotations

from dataclasses import dataclass
from enum        import Enum


class OptimizerType(Enum):

    ADAMW   = "adamw"
    ADAM    = "adam"
    SGD     = "sgd"
    RMSPROP = "rmsprop"


class SchedulerType(Enum):

    COSINE_ANNEALING = "cosine_annealing"
    CONSTANT         = "constant"
    LINEAR           = "linear"
    POLYNOMIAL       = "polynomial"
    EXPONENTIAL      = "exponential"
    STEP             = "step"


class WarmupMode(Enum):

    LINEAR      = "linear"
    COSINE      = "cosine"
    EXPONENTIAL = "exponential"
    POLYNOMIAL  = "polynomial"


class ClipMode(Enum):

    DISABLED            = "disabled"
    FIXED               = "fixed"
    ADAPTIVE_PERCENTILE = "adaptive_percentile"
    ADAPTIVE_MEAN_STD   = "adaptive_mean_std"


@dataclass
class OptimizerConfig:

    type         : str   = "adamw"
    betas        : tuple = (0.9, 0.999)
    eps          : float = 1e-8
    weight_decay : float = 0.1
    momentum     : float = 0.9
    nesterov     : bool  = True
    lr_scale     : float = 1.0


@dataclass
class SchedulerConfig:

    type      : str   = "cosine_annealing"
    epochs    : int   = 100
    eta_min   : float = 1e-6
    step_size : int   = 30
    gamma     : float = 0.1
    power     : float = 1.0


@dataclass
class WarmupConfig:

    warmup_fraction     : float = 0.03
    warmup_start_factor : float = 0.1
    warmup_enabled      : bool  = True
    warmup_mode         : str   = "linear"
    warmup_poly_power   : float = 2.0


@dataclass
class EarlyStoppingConfig:

    patience     : int   = 15
    min_delta    : float = 0.0
    restore_best : bool  = True


@dataclass
class GradientClipperConfig:

    clip_mode           : str   = "fixed"
    max_grad_norm       : float = 1.0
    adaptive_window     : int   = 200
    adaptive_percentile : float = 95.0
    adaptive_mean_std_k : float = 2.0
    clip_epsilon        : float = 1e-6
    log_histogram_freq  : int   = 100
