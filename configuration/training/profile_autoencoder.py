from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

from configuration.dataset.profile_autoencoder       import ProfileAugmentationConfig
from configuration.training.general.run              import TrainingPathsConfig, TrainingQueueConfig, standard_seeds
from configuration.architectures.profile_autoencoder import ProfileAutoencoderBaseConfig, MlpAutoencoderConfig
from configuration.sar.geometry_config               import GeometryConfig
from configuration.training.general.optimization     import EarlyStoppingConfig, GradientClipperConfig, OptimizerConfig, SchedulerConfig, WarmupConfig
from configuration.training.general.runtime          import IOConfig, MemoryConfig, OverfitCheckConfig, ResourceConfig, TrainingLoopConfig
from configuration.training.general.pretraining      import PretrainConfig
from configuration.training.general.trainer          import SharedSubConfigInheritance


@dataclass
class ProfileAeLossConfig:
    curve_kind      : str   = "mse"
    huber_delta     : float = 1.0
    charbonnier_eps : float = 1e-3


@dataclass
class ProfileAeTrainerConfig(SharedSubConfigInheritance):
    gaussian         : object
    autoencoder      : ProfileAutoencoderBaseConfig = field(default_factory=MlpAutoencoderConfig)
    ae_loss          : ProfileAeLossConfig          = field(default_factory=ProfileAeLossConfig)
    geometry         : GeometryConfig               = field(default_factory=GeometryConfig)
    early_stopping   : EarlyStoppingConfig          = field(default_factory=EarlyStoppingConfig)
    warmup           : WarmupConfig                 = field(default_factory=WarmupConfig)
    scheduler        : SchedulerConfig              = field(default_factory=SchedulerConfig)
    io               : IOConfig                     = field(default_factory=IOConfig)
    optimizer        : OptimizerConfig              = field(default_factory=OptimizerConfig)
    training         : TrainingLoopConfig           = field(default_factory=TrainingLoopConfig)
    resources        : ResourceConfig               = field(default_factory=ResourceConfig)
    memory           : MemoryConfig                 = field(default_factory=MemoryConfig)
    gradient_clipper : GradientClipperConfig        = field(default_factory=GradientClipperConfig)


@dataclass
class ProfileAeEntryConfig:
    run_name        : str | None = None
    resume          : bool       = True
    gpu             : int        = 0
    gpus            : list[int]  = field(default_factory=lambda: [0, 1, 3])
    gpus_file       : str        = ""
    poll_interval_s : float      = 5.0
    seed            : int        = 0
    seeds           : list[int]  = field(default_factory=standard_seeds)
    logdir          : Path       = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/profile_autoencoder")

    pixel_subsample : float = 1.0
    keep_empty_frac : float = 0.05

    profile_augmentation : ProfileAugmentationConfig = field(default_factory=ProfileAugmentationConfig)

    ae_model_name   : str                 = "mlp_ae"
    model_overrides : dict                = field(default_factory=dict)
    ae_loss         : ProfileAeLossConfig = field(default_factory=ProfileAeLossConfig)
    geometry        : GeometryConfig      = field(default_factory=GeometryConfig)

    paths         : TrainingPathsConfig = field(default_factory=TrainingPathsConfig)
    training      : TrainingQueueConfig = field(default_factory=lambda: TrainingQueueConfig(batch_size=1024, num_workers=32, prefetch_factor=2))
    pretrain      : PretrainConfig      = field(default_factory=PretrainConfig)
    overfit_check : OverfitCheckConfig  = field(default_factory=OverfitCheckConfig)
