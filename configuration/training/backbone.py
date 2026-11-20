
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

from configuration.training.general.run          import TrainingPathsConfig, TrainingQueueConfig, standard_seeds
from configuration.dataset                       import AugmentationConfig, InputConfig
from configuration.inference.general             import InferenceConfig
from configuration.dataset.gaussian           import GaussianConfig
from configuration.normalization.general         import NormalizationConfig
from configuration.training.general.loss         import LossConfig, ParamMatching
from configuration.training.general.optimization import EarlyStoppingConfig, GradientClipperConfig, OptimizerConfig, SchedulerConfig, WarmupConfig
from configuration.resources                     import ResourceConfig
from configuration.training.general.runtime      import IOConfig, MemoryConfig, TrainingLoopConfig
from configuration.training.general.trials       import TrialCatalog


class FullModel:

    PARAM_MATCH   = ParamMatching.HUNGARIAN
    COSINE_WEIGHT = 0.5


def default_loss() -> LossConfig:
    return LossConfig(
        use_param_l1             = True,
        weight_param_l1          = 1.0,
        param_matching           = FullModel.PARAM_MATCH,
        use_active_normalization = True,
        presence_balance         = True,
        use_cosine_curve         = True,
        weight_cosine_curve      = FullModel.COSINE_WEIGHT,
    )


@dataclass
class PatchTrialsConfig:

    minimum               : tuple[int, int] = (0, 0)
    maximum               : tuple[int, int] = (96, 48)
    step                  : int             = 0
    stride_ratio          : float           = 0.5
    constant_pixel_budget : bool            = True


@dataclass
class PairTrialsConfig:

    base_component   : str         = "param_l1"
    base_weight      : float       = 1.0
    components       : list[str]   = field(default_factory=lambda: ["cosine_curve", "total_power_relerr", "moments"])
    weights          : list[float] = field(default_factory=lambda: [0.01, 0.05, 0.25])
    include_baseline : bool        = True


@dataclass
class LogEnergyTrialsConfig:

    weights            : list[float] = field(default_factory=lambda: [0.01, 0.05, 0.25, 0.5, 1.0])
    replace_covariance : bool        = False
    include_baseline   : bool        = True


@dataclass
class ReachTrialsConfig:

    rungs           : list            = field(default_factory=TrialCatalog.reach_rungs)
    patch_size      : tuple[int, int] = (32, 32)
    patch_stride    : tuple[int, int] = (16, 16)
    in_channels     : int             = 9
    match_tolerance : float           = 0.01


@dataclass
class SecondaryTrialsConfig:

    strategy      : str             = "consecutive"
    n_secondaries : int             = 4
    fixed         : tuple[str, ...] = ()
    n_trials      : int             = 8
    mean          : float | None    = None
    sigma         : float | None    = None
    block_step    : int             = 1
    spacing       : int             = 2
    seed          : int             = 0


@dataclass
class HeadMatchingTrialsConfig:

    heads     : list[str] = field(default_factory=lambda: ["conv", "set_pred"])
    matchings : list[str] = field(default_factory=lambda: ["sorted_gt", "hungarian"])


@dataclass
class AugmentationTrialsConfig:

    flip_probability  : float       = 0.5
    noise_probability : float       = 0.5
    noise_stds        : list[float] = field(default_factory=lambda: [0.01, 0.03, 0.1])


@dataclass
class ClampTrialsConfig:

    leaky_slopes       : list[float] = field(default_factory=lambda: [0.1])
    param_leaky_slopes : list[float] = field(default_factory=lambda: [0.1])


@dataclass
class StabilityTrialsConfig:

    warmup_fractions : list[float] = field(default_factory=lambda: [0.03])
    warmup_mode      : str         = "linear"
    clip_mode        : str         = "fixed"
    max_grad_norms   : list[float] = field(default_factory=lambda: [1.0])


@dataclass
class EmaTrialsConfig:

    decays : list[float] = field(default_factory=lambda: [0.999])


@dataclass
class BackboneTrainerConfig:

    gaussian         : GaussianConfig
    early_stopping   : EarlyStoppingConfig   = field(default_factory=EarlyStoppingConfig)
    warmup           : WarmupConfig          = field(default_factory=WarmupConfig)
    scheduler        : SchedulerConfig       = field(default_factory=SchedulerConfig)
    io               : IOConfig              = field(default_factory=IOConfig)
    optimizer        : OptimizerConfig       = field(default_factory=OptimizerConfig)
    training         : TrainingLoopConfig    = field(default_factory=TrainingLoopConfig)
    loss             : LossConfig            = field(default_factory=LossConfig)
    resources        : ResourceConfig        = field(default_factory=ResourceConfig)
    memory           : MemoryConfig          = field(default_factory=MemoryConfig)
    gradient_clipper : GradientClipperConfig = field(default_factory=GradientClipperConfig)


@dataclass
class BackboneEntryConfig:

    run_name        : str | None = None
    resume          : bool       = True
    resume_from     : str        = ""
    backbone_name   : str        = "resunet"
    backbone_head   : str        = "set_pred"
    gpu             : int        = 0
    seed            : int        = 0
    seeds           : list[int]  = field(default_factory=standard_seeds)
    logdir          : Path       = Path("runs/backbone")
    model_overrides : dict       = field(default_factory=dict)

    paths         : TrainingPathsConfig  = field(default_factory=TrainingPathsConfig)
    training      : TrainingQueueConfig  = field(default_factory=TrainingQueueConfig)
    loss          : LossConfig           = field(default_factory=default_loss)
    input         : InputConfig          = field(default_factory=InputConfig.full_stack)
    normalization : NormalizationConfig  = field(default_factory=NormalizationConfig)
    augmentation  : AugmentationConfig   = field(default_factory=AugmentationConfig)


    prior_bias_init : bool = False

    infer_after  : bool            = False
    infer_at_end : bool            = False
    inference    : InferenceConfig = field(default_factory=lambda: InferenceConfig.for_run_directory())

    trials_enabled       : bool                      = False
    trials_mode          : str                       = "presence"
    presence_trials      : dict                      = field(default_factory=TrialCatalog.presence_trials)
    pair_trials          : PairTrialsConfig          = field(default_factory=PairTrialsConfig)
    log_energy_trials    : LogEnergyTrialsConfig     = field(default_factory=LogEnergyTrialsConfig)
    secondary_trials     : SecondaryTrialsConfig     = field(default_factory=SecondaryTrialsConfig)
    patch_trials         : PatchTrialsConfig         = field(default_factory=PatchTrialsConfig)
    input_trials         : dict                      = field(default_factory=TrialCatalog.input_trials)
    context_trials       : list                      = field(default_factory=TrialCatalog.context_trials)
    reach_trials         : ReachTrialsConfig         = field(default_factory=ReachTrialsConfig)
    head_trials          : HeadMatchingTrialsConfig  = field(default_factory=HeadMatchingTrialsConfig)
    augmentation_trials  : AugmentationTrialsConfig  = field(default_factory=AugmentationTrialsConfig)
    clamp_trials         : ClampTrialsConfig         = field(default_factory=ClampTrialsConfig)
    stability_trials     : StabilityTrialsConfig     = field(default_factory=StabilityTrialsConfig)
    ema_trials           : EmaTrialsConfig           = field(default_factory=EmaTrialsConfig)

    gpus            : list[int] = field(default_factory=lambda: [0])
    gpus_file       : str       = ""
    poll_interval_s : float     = 5.0
