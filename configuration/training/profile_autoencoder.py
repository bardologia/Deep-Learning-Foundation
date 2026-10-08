"""Configuration for training the 1D autoencoder over per-pixel z profiles."""

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
    """Summable reconstruction, shape, smoothness and latent-noise terms on the z profile.

    Each ``use_*`` flag enables one term and its ``weight_*`` scales that term in
    the total, mirroring the backbone loss shape so several penalties combine.

    Attributes:
        use_mse_curve            : Whether the squared-error reconstruction term is active.
        weight_mse_curve         : Weight of that term.
        use_l1_curve             : Whether the absolute-error reconstruction term is active.
        weight_l1_curve          : Weight of that term.
        use_huber_curve          : Whether the Huber reconstruction term is active.
        weight_huber_curve       : Weight of that term.
        huber_delta              : Transition point of the Huber penalty.
        use_charbonnier_curve    : Whether the Charbonnier reconstruction term is active.
        weight_charbonnier_curve : Weight of that term.
        charbonnier_eps          : Smoothing constant of the Charbonnier penalty.
        use_cosine_curve         : Whether the cosine shape term over the z axis is active.
        weight_cosine_curve      : Weight of that term.
        use_log_energy_curve     : Whether the log energy-ratio term is active.
        weight_log_energy_curve  : Weight of that term.
        use_tv_curve             : Whether the first-difference total variation of the residual is penalised.
        weight_tv_curve          : Weight of that term.
        use_sobolev              : Whether the second difference of the residual is penalised.
        weight_sobolev           : Weight of that term.
        use_latent_noise         : Whether Gaussian noise is injected into the latent code.
        latent_noise_std         : Standard deviation of that latent noise.
        weight_latent_noise      : Weight of the noisy-latent reconstruction term.
    """

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
    """Latent-width ladder swept by the embedding trial mode.

    Attributes:
        dims : Embedding dimensions trained, one run each.
    """

    dims : list[int] = field(default_factory=lambda: [8, 16, 24, 32, 48, 64])


@dataclass
class CapacityTrialsConfig:
    """Width-by-depth capacity grid swept within the configured autoencoder family.

    Attributes:
        widths : Width values of the family's width field, empty to walk the family's tunable choices.
        depths : Depth values, empty to walk the family's tunable choices.
    """

    widths : list[int] = field(default_factory=list)
    depths : list[int] = field(default_factory=list)


@dataclass
class SmoothnessTrialsConfig:
    """Sobolev and latent-noise smoothness arms swept on top of the configured reconstruction loss.

    Attributes:
        sobolev_weights   : Weights of the residual-curvature penalty, one arm each.
        latent_noise_stds : Standard deviations of the injected latent noise, one arm each.
        combine           : Whether the full Sobolev-by-noise grid is added on top of the one-at-a-time arms.
        include_baseline  : Whether a run with both levers off is added.
    """

    sobolev_weights   : list[float] = field(default_factory=lambda: [0.1, 0.5, 1.0])
    latent_noise_stds : list[float] = field(default_factory=lambda: [0.02, 0.05, 0.1])
    combine           : bool        = False
    include_baseline  : bool        = True


@dataclass
class ProfileAugmentationTrialsConfig:
    """Cumulative profile-augmentation ladder from no augmentation to flips, shifts, scaling and noise.

    Attributes:
        flip_probability      : Flip probability applied from the first augmented rung on.
        shift_probability     : Shift probability applied from the shift rung on.
        max_shift             : Largest absolute bin shift of the shift augmentation.
        amp_scale_probability : Amplitude-scaling probability applied from the scale rung on.
        noise_probability     : Noise probability applied on the noise rungs.
        noise_stds            : Noise standard deviations, one final rung each.
    """

    flip_probability      : float       = 0.5
    shift_probability     : float       = 0.5
    max_shift             : int         = 4
    amp_scale_probability : float       = 0.5
    noise_probability     : float       = 0.5
    noise_stds            : list[float] = field(default_factory=lambda: [0.005, 0.01, 0.03])


@dataclass
class SamplingTrialsConfig:
    """Training-data mix grid over the pixel subsample and the kept empty-profile fraction.

    Attributes:
        subsamples  : Fractions of scene pixels kept as training profiles.
        empty_fracs : Fractions of empty profiles kept in the sample.
    """

    subsamples  : list[float] = field(default_factory=lambda: [0.25, 0.5, 1.0])
    empty_fracs : list[float] = field(default_factory=lambda: [0.0, 0.05, 0.2])


@dataclass
class ProfileAeTrainerConfig(SharedSubConfigInheritance):
    """Everything the profile-autoencoder trainer needs, assembled from sub-configs.

    Attributes:
        gaussian         : Gaussian label configuration describing the z grid.
        autoencoder      : Architecture config of the encoder/decoder pair.
        ae_loss          : Reconstruction, smoothness and latent-noise settings.
        geometry         : Acquisition geometry.
        early_stopping   : Validation-loss stopping rule.
        warmup           : Learning-rate warmup ramp.
        scheduler        : Learning-rate schedule.
        io               : Output root for checkpoints and logs.
        optimizer        : AdamW hyper-parameters.
        training         : Epoch loop, precision and resume behaviour.
        resources        : Resource-monitor cadence and warning levels.
        memory           : VRAM reservation settings.
        gradient_clipper : Gradient-norm clipping strategy.
    """

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
    """Command-line surface of the profile-autoencoder training entry point.

    Attributes:
        run_name             : Explicit run folder name, or None to derive one from the settings.
        resume               : Whether an interrupted run continues from its last checkpoint.
        resume_from          : Path of a run directory, a seed sweep root or a trial sweep root to resume, or any run inside them; the configuration it recorded is loaded, the command-line overrides are applied on top, complete units are reused and the rest continue or restart in their own directories. Empty starts a fresh run.
        gpu                  : CUDA device index used for a single-seed run.
        gpus                 : CUDA device indices the seed sweep fans out over, and the rank devices of a DDP run.
        ddp                  : Whether a single-seed run trains data-parallel across the `gpus` devices, one synchronised rank per device sharing the configured global batch.
        gpus_file            : File listing device indices, re-read to resize the pool live.
        poll_interval_s      : Seconds between scheduler polls of the device pool.
        seed                 : Seed of a single-seed run.
        seeds                : Seeds trained by the sweep.
        logdir               : Root directory the run folders are created under.
        pixel_subsample      : Fraction of scene pixels kept as training profiles.
        keep_empty_frac      : Fraction of profiles with no component kept in the sample.
        stats_max_samples    : Profiles drawn when estimating normalization statistics.
        profile_augmentation : Augmentation applied to the sampled profiles.
        ae_model_name        : Registry key of the profile-autoencoder architecture.
        model_overrides      : Architecture fields overridden on top of that default.
        ae_loss              : Reconstruction, smoothness and latent-noise settings.
        geometry             : Acquisition geometry.
        encoder_lr           : Learning rate of the encoder parameter group.
        decoder_lr           : Learning rate of the decoder parameter group.
        encoder_wd           : Weight decay of the encoder parameter group.
        decoder_wd           : Weight decay of the decoder parameter group.
        paths                : Dataset and parameter-label locations.
        training             : Training schedule expanded into the trainer sub-configs.
        pretrain             : Batch-size and loader probes run before training.
        overfit_check        : Pre-training overfit sanity gate.
        trials_enabled       : Whether the multi-trial GPU sweep replaces the single training run.
        trials_mode          : Name of the trial sweep executed.
        arch_trials          : Family rungs of the architecture sweep.
        embedding_trials     : Latent-width ladder of the embedding sweep.
        capacity_trials      : Width-by-depth grid of the capacity sweep.
        loss_trials          : Reconstruction-loss arms of the loss sweep, keyed by run label.
        smoothness_trials    : Sobolev and latent-noise arms of the smoothness sweep.
        augmentation_trials  : Cumulative ladder of the profile-augmentation sweep.
        sampling_trials      : Data-mix grid of the sampling sweep.
        ema_trials           : Weight-EMA ladder of the ema sweep; the amp mode compares mixed-precision off against on and needs no settings of its own.
        stability_trials     : Warmup and gradient-clip arms of the stability sweep.
    """

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
