
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

from configuration.training.general.run          import TrainingPathsConfig, TrainingQueueConfig, standard_seeds
from configuration.dataset                       import AugmentationConfig
from configuration.inference.general             import InferenceConfig
from configuration.normalization.general         import NormalizationConfig
from configuration.training.general.loss         import LossConfig
from configuration.training.general.optimization import EarlyStoppingConfig, GradientClipperConfig, OptimizerConfig, SchedulerConfig, WarmupConfig
from configuration.resources                     import ResourceConfig
from configuration.training.general.runtime      import IOConfig, MemoryConfig, TrainingLoopConfig
from configuration.training.general.trials       import TrialCatalog


@dataclass
class PatchTrialsConfig:

    minimum               : tuple[int, int] = (0, 0)
    maximum               : tuple[int, int] = (96, 48)
    step                  : int             = 0
    stride_ratio          : float           = 0.5
    constant_pixel_budget : bool            = True


@dataclass
class ReachTrialsConfig:

    rungs           : list            = field(default_factory=TrialCatalog.reach_rungs)
    patch_size      : tuple[int, int] = (32, 32)
    patch_stride    : tuple[int, int] = (16, 16)
    in_channels     : int             = 4
    match_tolerance : float           = 0.01


@dataclass
class AugmentationTrialsConfig:

    flip_probability  : float       = 0.5
    noise_probability : float       = 0.5
    noise_stds        : list[float] = field(default_factory=lambda: [0.01, 0.03, 0.1])


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
    backbone_head   : str        = "conv"
    gpu             : int        = 0
    seed            : int        = 0
    seeds           : list[int]  = field(default_factory=standard_seeds)
    logdir          : Path       = Path("runs/backbone")
    model_overrides : dict       = field(default_factory=dict)

    paths         : TrainingPathsConfig  = field(default_factory=TrainingPathsConfig)
    training      : TrainingQueueConfig  = field(default_factory=TrainingQueueConfig)
    loss          : LossConfig           = field(default_factory=LossConfig)
    normalization : NormalizationConfig  = field(default_factory=NormalizationConfig)
    augmentation  : AugmentationConfig   = field(default_factory=AugmentationConfig)

    infer_after  : bool            = False
    infer_at_end : bool            = False
    inference    : InferenceConfig = field(default_factory=lambda: InferenceConfig.for_run_directory())

    trials_enabled      : bool                     = False
    trials_mode         : str                      = "augmentation"
    patch_trials        : PatchTrialsConfig        = field(default_factory=PatchTrialsConfig)
    context_trials      : list                     = field(default_factory=TrialCatalog.context_trials)
    reach_trials        : ReachTrialsConfig        = field(default_factory=ReachTrialsConfig)
    augmentation_trials : AugmentationTrialsConfig = field(default_factory=AugmentationTrialsConfig)
    stability_trials    : StabilityTrialsConfig    = field(default_factory=StabilityTrialsConfig)
    ema_trials          : EmaTrialsConfig          = field(default_factory=EmaTrialsConfig)

    gpus            : list[int] = field(default_factory=lambda: [0])
    gpus_file       : str       = ""
    poll_interval_s : float     = 5.0
