from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

from configuration.dataset                       import AugmentationConfig
from configuration.normalization.general         import NormalizationConfig
from configuration.training.general.optimization import EarlyStoppingConfig, GradientClipperConfig, OptimizerConfig, SchedulerConfig, WarmupConfig
from configuration.training.general.pretraining  import PretrainConfig
from configuration.training.general.run          import TrainingPathsConfig, standard_seeds
from configuration.training.general.runtime      import IOConfig, MemoryConfig, OverfitCheckConfig, ResourceConfig, TrainingLoopConfig
from configuration.sar.geometry_config           import GeometryConfig


@dataclass
class UnrolledTrainingConfig:
    epochs               : int        = 60
    scheduler_epochs     : int | None = None
    validation_frequency : int        = 1
    eta_min              : float      = 1e-6
    early_stop_patience  : int        = 30
    early_stop_min_delta : float      = 0.0
    max_grad_norm        : float      = 1.0

    resume                  : bool = False
    snapshot_every_n_epochs : int  = 0
    log_debug               : bool = False

    warmup_enabled : bool = True
    warmup_steps   : int  = 200

    use_ema   : bool  = False
    ema_decay : float = 0.999

    reserve_vram      : bool  = False
    vram_keep_free_gb : float = 1.0

    batch_size                  : int  = 256
    num_workers                 : int  = 4
    prefetch_factor             : int  = 2
    gradient_accumulation_steps : int  = 1
    abort_on_nonfinite_loss     : bool = True

    scale_lr_with_batch     : bool = True
    lr_reference_batch_size : int  = 256

    patch_size            : tuple[int, int] = (64, 32)
    patch_stride          : tuple[int, int] = (32, 16)
    use_symmetric_padding : bool            = True
    train_azimuth         : tuple[int, int] = (1000, 13000)
    val_azimuth           : tuple[int, int] = (13064, 14500)
    test_azimuth          : tuple[int, int] = (14564, 16000)


@dataclass
class UnrolledTrainerConfig:
    params_per_gaussian   : int   = 3
    curve_loss            : str   = "l1"
    measurement_noise_std : float = 0.0
    power_floor           : float = 1e-6

    early_stopping   : EarlyStoppingConfig   = field(default_factory=EarlyStoppingConfig)
    warmup           : WarmupConfig          = field(default_factory=WarmupConfig)
    scheduler        : SchedulerConfig       = field(default_factory=SchedulerConfig)
    io               : IOConfig              = field(default_factory=IOConfig)
    optimizer        : OptimizerConfig       = field(default_factory=OptimizerConfig)
    training         : TrainingLoopConfig    = field(default_factory=TrainingLoopConfig)
    resources        : ResourceConfig        = field(default_factory=ResourceConfig)
    memory           : MemoryConfig          = field(default_factory=MemoryConfig)
    gradient_clipper : GradientClipperConfig = field(default_factory=GradientClipperConfig)


@dataclass
class UnrolledEntryConfig:
    run_name        : str | None = None
    resume          : bool       = True
    model_name      : str        = "gamma_net"
    gpu             : int        = 0
    gpus            : list[int]  = field(default_factory=lambda: [0, 1, 3])
    gpus_file       : str        = ""
    poll_interval_s : float      = 5.0
    seed            : int        = 0
    seeds           : list[int]  = field(default_factory=standard_seeds)
    logdir          : Path       = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/unrolled")
    model_overrides : dict       = field(default_factory=dict)

    paths         : TrainingPathsConfig    = field(default_factory=TrainingPathsConfig)
    training      : UnrolledTrainingConfig = field(default_factory=UnrolledTrainingConfig)
    pretrain      : PretrainConfig         = field(default_factory=PretrainConfig)
    geometry      : GeometryConfig         = field(default_factory=GeometryConfig)
    normalization : NormalizationConfig    = field(default_factory=NormalizationConfig)
    augmentation  : AugmentationConfig     = field(default_factory=AugmentationConfig)
    overfit_check : OverfitCheckConfig     = field(default_factory=OverfitCheckConfig)

    curve_loss            : str   = "l1"
    measurement_noise_std : float = 0.0
    power_floor           : float = 1e-6
