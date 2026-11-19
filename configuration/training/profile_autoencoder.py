
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

from configuration.dataset.profile_autoencoder       import ProfileAugmentationConfig, ProfileDatasetConfig
from configuration.training.general.run              import TrainingPathsConfig, TrainingQueueConfig, standard_seeds
from configuration.training.general.trials           import ProfileAeTrialCatalog
from configuration.architectures.profile_autoencoder import ProfileAutoencoderBaseConfig, MlpAutoencoderConfig
from configuration.training.backbone                 import EmaTrialsConfig, StabilityTrialsConfig
from configuration.training.general.optimization     import EarlyStoppingConfig, GradientClipperConfig, OptimizerConfig, SchedulerConfig, WarmupConfig
from configuration.resources                         import ResourceConfig
from configuration.training.general.runtime          import IOConfig, MemoryConfig, OverfitCheckConfig, TrainingLoopConfig
from configuration.training.general.pretraining      import PretrainConfig
from configuration.training.general.trainer          import SharedSubConfigInheritance


@dataclass
class ProfileAeLossConfig:

    use_mse_curve    : bool  = True
    weight_mse_curve : float = 1.0

    use_l1_curve    : bool  = False
    weight_l1_curve : float = 1.0

    use_huber_curve    : bool  = False
    weight_huber_curve : float = 1.0
    huber_delta        : float = 1.0

    use_charbonnier_curve    : bool  = False
    weight_charbonnier_curve : float = 1.0
    charbonnier_eps          : float = 1e-3

    use_cosine_curve    : bool  = False
    weight_cosine_curve : float = 1.0

    use_log_energy_curve    : bool  = False
    weight_log_energy_curve : float = 1.0

    use_tv_curve    : bool  = False
    weight_tv_curve : float = 1.0

    use_sobolev    : bool  = False
    weight_sobolev : float = 1.0

    use_latent_noise    : bool  = False
    latent_noise_std    : float = 0.05
    weight_latent_noise : float = 1.0


@dataclass
class EmbeddingDimTrialsConfig:

    dims : list[int] = field(default_factory=lambda: [8, 16, 24, 32, 48, 64])


@dataclass
class CapacityTrialsConfig:

    widths : list[int] = field(default_factory=list)
    depths : list[int] = field(default_factory=list)


@dataclass
class SmoothnessTrialsConfig:

    sobolev_weights   : list[float] = field(default_factory=lambda: [0.1, 0.5, 1.0])
    latent_noise_stds : list[float] = field(default_factory=lambda: [0.02, 0.05, 0.1])
    combine           : bool        = False
    include_baseline  : bool        = True


@dataclass
class ProfileAugmentationTrialsConfig:

    flip_probability      : float       = 0.5
    shift_probability     : float       = 0.5
    max_shift             : int         = 4
    amp_scale_probability : float       = 0.5
    noise_probability     : float       = 0.5
    noise_stds            : list[float] = field(default_factory=lambda: [0.005, 0.01, 0.03])


@dataclass
class SamplingTrialsConfig:

    subsamples  : list[float] = field(default_factory=lambda: [0.25, 0.5, 1.0])
    empty_fracs : list[float] = field(default_factory=lambda: [0.0, 0.05, 0.2])


@dataclass
class ProfileAeTrainerConfig(SharedSubConfigInheritance):

    gaussian         : object
    autoencoder      : ProfileAutoencoderBaseConfig = field(default_factory=MlpAutoencoderConfig)
    ae_loss          : ProfileAeLossConfig          = field(default_factory=ProfileAeLossConfig)
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
    resume_from     : str        = ""
    gpu             : int        = 0
    gpus            : list[int]  = field(default_factory=lambda: [0])
    ddp             : bool       = False
    gpus_file       : str        = ""
    poll_interval_s : float      = 5.0
    seed            : int        = 0
    seeds           : list[int]  = field(default_factory=standard_seeds)
    logdir          : Path       = Path("runs/profile_autoencoder")

    pixel_subsample   : float = 1.0
    keep_empty_frac   : float = 0.05
    stats_max_samples : int   = ProfileDatasetConfig.stats_max_samples

    profile_augmentation : ProfileAugmentationConfig = field(default_factory=ProfileAugmentationConfig)

    ae_model_name   : str                 = "mlp_ae"
    model_overrides : dict                = field(default_factory=dict)
    ae_loss         : ProfileAeLossConfig = field(default_factory=ProfileAeLossConfig)

    encoder_lr : float = ProfileAutoencoderBaseConfig.encoder_lr
    decoder_lr : float = ProfileAutoencoderBaseConfig.decoder_lr
    encoder_wd : float = ProfileAutoencoderBaseConfig.encoder_wd
    decoder_wd : float = ProfileAutoencoderBaseConfig.decoder_wd

    paths         : TrainingPathsConfig = field(default_factory=TrainingPathsConfig)
    training      : TrainingQueueConfig = field(default_factory=lambda: TrainingQueueConfig(batch_size=8192, num_workers=14, prefetch_factor=4))
    pretrain      : PretrainConfig      = field(default_factory=PretrainConfig)
    overfit_check : OverfitCheckConfig  = field(default_factory=OverfitCheckConfig)

    trials_enabled : bool = False
    trials_mode    : str  = "arch"

    arch_trials         : list                            = field(default_factory=ProfileAeTrialCatalog.arch_trials)
    embedding_trials    : EmbeddingDimTrialsConfig        = field(default_factory=EmbeddingDimTrialsConfig)
    capacity_trials     : CapacityTrialsConfig            = field(default_factory=CapacityTrialsConfig)
    loss_trials         : dict                            = field(default_factory=ProfileAeTrialCatalog.loss_trials)
    smoothness_trials   : SmoothnessTrialsConfig          = field(default_factory=SmoothnessTrialsConfig)
    augmentation_trials : ProfileAugmentationTrialsConfig = field(default_factory=ProfileAugmentationTrialsConfig)
    sampling_trials     : SamplingTrialsConfig            = field(default_factory=SamplingTrialsConfig)
    ema_trials          : EmaTrialsConfig                 = field(default_factory=EmaTrialsConfig)
    stability_trials    : StabilityTrialsConfig           = field(default_factory=StabilityTrialsConfig)
