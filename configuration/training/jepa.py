from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

from configuration.training.general.run               import TrainingPathsConfig, TrainingQueueConfig, standard_seeds
from configuration.architectures.profile_autoencoder  import ProfileAutoencoderBaseConfig
from configuration.dataset                            import AugmentationConfig
from configuration.normalization.general              import NormalizationConfig
from configuration.architectures.image_autoencoder    import ImageAutoencoderBaseConfig
from configuration.inference.general                  import InferenceConfig
from configuration.sar.geometry_config                import GeometryConfig
from configuration.training.general.loss              import LossConfig, ParamMatching
from configuration.training.general.optimization      import EarlyStoppingConfig, GradientClipperConfig, OptimizerConfig, SchedulerConfig, WarmupConfig
from configuration.training.general.runtime           import IOConfig, MemoryConfig, OverfitCheckConfig, ResourceConfig, TrainingLoopConfig
from configuration.training.general.pretraining       import PretrainConfig
from configuration.training.general.trainer           import SharedSubConfigInheritance


def default_param_loss() -> LossConfig:
    return LossConfig(
        use_param_l1             = True,
        weight_param_l1          = 1.0,
        param_matching           = ParamMatching.SORTED_GT,
        use_active_normalization = True,
        presence_balance         = False,
    )


@dataclass
class EmbeddingLossConfig:
    use_embedding_mse    : bool  = True
    weight_embedding_mse : float = 1.0

    use_embedding_cosine    : bool  = False
    weight_embedding_cosine : float = 0.0

    use_embedding_smoothl1    : bool  = False
    weight_embedding_smoothl1 : float = 0.0
    smoothl1_beta             : float = 1.0

    use_curve_recon    : bool  = True
    weight_curve_recon : float = 1.0
    curve_kind         : str   = "mse"
    huber_delta        : float = 1.0
    charbonnier_eps    : float = 1e-3


@dataclass
class JepaTrainerConfig(SharedSubConfigInheritance):
    gaussian       : object
    autoencoder    : ProfileAutoencoderBaseConfig | None = None
    embedding_loss : EmbeddingLossConfig                 = field(default_factory=EmbeddingLossConfig)

    profile_autoencoder_mode       : str        = "frozen"
    target_provider                : str        = "stopgrad"
    profile_autoencoder_checkpoint : str | None = None

    ae_finetune_lr : float = 3e-5
    ae_finetune_wd : float = 1e-4

    image_autoencoder            : ImageAutoencoderBaseConfig | None = None
    image_autoencoder_mode       : str                               = "frozen"
    image_autoencoder_checkpoint : str | None                        = None

    image_ae_finetune_lr : float = 3e-5
    image_ae_finetune_wd : float = 1e-4

    param_loss : LossConfig = field(default_factory=default_param_loss)

    geometry         : GeometryConfig        = field(default_factory=GeometryConfig)
    early_stopping   : EarlyStoppingConfig   = field(default_factory=EarlyStoppingConfig)
    warmup           : WarmupConfig          = field(default_factory=WarmupConfig)
    scheduler        : SchedulerConfig       = field(default_factory=SchedulerConfig)
    io               : IOConfig              = field(default_factory=IOConfig)
    optimizer        : OptimizerConfig       = field(default_factory=OptimizerConfig)
    training         : TrainingLoopConfig    = field(default_factory=TrainingLoopConfig)
    resources        : ResourceConfig        = field(default_factory=ResourceConfig)
    memory           : MemoryConfig          = field(default_factory=MemoryConfig)
    gradient_clipper : GradientClipperConfig = field(default_factory=GradientClipperConfig)


class JepaDefaults:
    @staticmethod
    def inference() -> InferenceConfig:
        return InferenceConfig(
            run_directory   = Path("."),
            save_cubes      = True,
            cpu_workers     = 16,
            gif_axes        = ["elevation", "range", "azimuth"],
            compute_reduced = False,
        )


@dataclass
class JepaEntryConfig:
    run_name        : str | None = None
    resume          : bool       = True
    backbone_name   : str        = "resunet"
    backbone_head   : str        = "conv"
    gpu             : int        = 0
    gpus            : list[int]  = field(default_factory=lambda: [0, 1, 3])
    gpus_file       : str        = ""
    poll_interval_s : float      = 5.0
    seed            : int        = 0
    seeds           : list[int]  = field(default_factory=standard_seeds)
    logdir          : Path       = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/jepa")
    model_overrides : dict       = field(default_factory=dict)

    profile_autoencoder_logdir : Path       = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/profile_autoencoder")
    profile_autoencoder_run    : str | None = None
    profile_autoencoder_mode   : str        = "frozen"
    target_provider            : str        = "stopgrad"
    ae_finetune_lr             : float      = 3e-5
    ae_finetune_wd             : float      = 1e-4

    image_autoencoder_logdir : Path       = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/image_autoencoder")
    image_autoencoder_run    : str | None = None
    image_autoencoder_mode   : str        = "frozen"
    image_ae_finetune_lr     : float      = 3e-5
    image_ae_finetune_wd     : float      = 1e-4

    embedding_loss : EmbeddingLossConfig = field(default_factory=EmbeddingLossConfig)
    param_loss     : LossConfig          = field(default_factory=default_param_loss)
    geometry       : GeometryConfig      = field(default_factory=GeometryConfig)

    paths         : TrainingPathsConfig = field(default_factory=TrainingPathsConfig)
    training      : TrainingQueueConfig = field(default_factory=TrainingQueueConfig)
    pretrain      : PretrainConfig      = field(default_factory=PretrainConfig)
    normalization : NormalizationConfig = field(default_factory=NormalizationConfig)
    augmentation  : AugmentationConfig  = field(default_factory=AugmentationConfig)
    overfit_check : OverfitCheckConfig  = field(default_factory=OverfitCheckConfig)

    infer_after : bool            = False
    inference   : InferenceConfig = field(default_factory=JepaDefaults.inference)
