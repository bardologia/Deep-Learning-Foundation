"""Configuration for training the 2D image autoencoder over amplitude patches."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

from configuration.training.general.run             import TrainingPathsConfig, TrainingQueueConfig, standard_seeds
from configuration.architectures.image_autoencoder  import Conv2dImageAutoencoderConfig, ImageAutoencoderBaseConfig
from configuration.dataset                          import AugmentationConfig
from configuration.normalization.general            import NormalizationConfig
from configuration.training.general.optimization    import EarlyStoppingConfig, GradientClipperConfig, OptimizerConfig, SchedulerConfig, WarmupConfig
from configuration.training.general.runtime         import IOConfig, MemoryConfig, OverfitCheckConfig, ResourceConfig, TrainingLoopConfig
from configuration.training.general.pretraining     import PretrainConfig
from configuration.training.general.trainer         import SharedSubConfigInheritance


@dataclass
class ImageAeLossConfig:
    """Reconstruction loss applied between the input and reconstructed image patch.

    Attributes:
        recon_kind      : Reconstruction penalty family, one of mse, l1, huber or charbonnier.
        huber_delta     : Transition point of the Huber penalty.
        charbonnier_eps : Smoothing constant of the Charbonnier penalty.
    """

    recon_kind      : str   = "mse"
    huber_delta     : float = 1.0
    charbonnier_eps : float = 1e-3


@dataclass
class ImageAeTrainerConfig(SharedSubConfigInheritance):
    """Everything the image-autoencoder trainer needs, assembled from sub-configs.

    Attributes:
        gaussian          : Gaussian label configuration describing the z grid.
        image_autoencoder : Architecture config of the encoder/decoder pair.
        ae_loss           : Reconstruction loss settings.
        geometry          : Acquisition geometry.
        early_stopping    : Validation-loss stopping rule.
        warmup            : Learning-rate warmup ramp.
        scheduler         : Learning-rate schedule.
        io                : Output root for checkpoints and logs.
        optimizer         : AdamW hyper-parameters.
        training          : Epoch loop, precision and resume behaviour.
        resources         : Resource-monitor cadence and warning levels.
        memory            : VRAM reservation settings.
        gradient_clipper  : Gradient-norm clipping strategy.
    """

    gaussian          : object
    image_autoencoder : ImageAutoencoderBaseConfig = field(default_factory=Conv2dImageAutoencoderConfig)
    ae_loss           : ImageAeLossConfig          = field(default_factory=ImageAeLossConfig)
    early_stopping    : EarlyStoppingConfig        = field(default_factory=EarlyStoppingConfig)
    warmup            : WarmupConfig               = field(default_factory=WarmupConfig)
    scheduler         : SchedulerConfig            = field(default_factory=SchedulerConfig)
    io                : IOConfig                   = field(default_factory=IOConfig)
    optimizer         : OptimizerConfig            = field(default_factory=OptimizerConfig)
    training          : TrainingLoopConfig         = field(default_factory=TrainingLoopConfig)
    resources         : ResourceConfig             = field(default_factory=ResourceConfig)
    memory            : MemoryConfig               = field(default_factory=MemoryConfig)
    gradient_clipper  : GradientClipperConfig      = field(default_factory=GradientClipperConfig)


@dataclass
class ImageAeEntryConfig:
    """Command-line surface of the image-autoencoder training entry point.

    Attributes:
        run_name        : Explicit run folder name, or None to derive one from the settings.
        resume          : Whether an interrupted run continues from its last checkpoint.
        resume_from     : Run directory or sweep folder whose state is resumed.
        gpu             : CUDA device index used for a single-seed run.
        gpus            : CUDA device indices the seed sweep fans out over, and the rank devices of a DDP run.
        ddp             : Whether a single-seed run trains data-parallel across the `gpus` devices, one synchronised rank per device sharing the configured global batch.
        gpus_file       : File listing device indices, re-read to resize the pool live.
        poll_interval_s : Seconds between scheduler polls of the device pool.
        seed            : Seed of a single-seed run.
        seeds           : Seeds trained by the sweep.
        logdir          : Root directory the run folders are created under.
        ae_model_name   : Registry key of the image-autoencoder architecture.
        model_overrides : Architecture fields overridden on top of that default.
        ae_loss         : Reconstruction loss settings.
        geometry        : Acquisition geometry.
        paths           : Dataset and parameter-label locations.
        training        : Training schedule expanded into the trainer sub-configs.
        pretrain        : Batch-size and loader probes run before training.
        normalization   : Input normalization strategy and statistics.
        augmentation    : Patch augmentation settings.
        overfit_check   : Pre-training overfit sanity gate.
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
    logdir          : Path       = Path("runs/image_autoencoder")

    ae_model_name   : str               = "conv2d_ae"
    model_overrides : dict              = field(default_factory=dict)
    ae_loss         : ImageAeLossConfig = field(default_factory=ImageAeLossConfig)

    paths         : TrainingPathsConfig = field(default_factory=TrainingPathsConfig)
    training      : TrainingQueueConfig = field(default_factory=lambda: TrainingQueueConfig(batch_size=512, num_workers=16, prefetch_factor=2))
    pretrain      : PretrainConfig      = field(default_factory=PretrainConfig)
    normalization : NormalizationConfig = field(default_factory=NormalizationConfig)
    augmentation  : AugmentationConfig  = field(default_factory=AugmentationConfig)
    overfit_check : OverfitCheckConfig  = field(default_factory=OverfitCheckConfig)
