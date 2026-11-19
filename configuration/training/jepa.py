
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib     import Path

from configuration.training.general.run               import TrainingPathsConfig, TrainingQueueConfig, standard_seeds
from configuration.training.general.trials             import JepaTrialCatalog, TrialCatalog
from configuration.training.backbone                  import AugmentationTrialsConfig, EmaTrialsConfig, LossScaleProbeConfig, StabilityTrialsConfig
from configuration.architectures.profile_autoencoder  import ProfileAutoencoderBaseConfig
from configuration.dataset                            import AugmentationConfig, InputConfig
from configuration.normalization.general              import NormalizationConfig
from configuration.architectures.image_autoencoder    import ImageAutoencoderBaseConfig
from configuration.inference.general                  import InferenceConfig
from configuration.training.general.loss              import LossConfig, ParamMatching
from configuration.training.general.optimization      import EarlyStoppingConfig, GradientClipperConfig, OptimizerConfig, SchedulerConfig, WarmupConfig
from configuration.resources                          import ResourceConfig
from configuration.training.general.runtime           import IOConfig, MemoryConfig, OverfitCheckConfig, TrainingLoopConfig
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

    use_embedding_huber    : bool  = False
    weight_embedding_huber : float = 0.0
    embedding_huber_delta  : float = 1.0

    use_embedding_charbonnier    : bool  = False
    weight_embedding_charbonnier : float = 0.0
    embedding_charbonnier_eps    : float = 1e-3

    use_embedding_variance    : bool  = False
    weight_embedding_variance : float = 0.0
    variance_gamma            : float = 1.0

    use_embedding_covariance    : bool  = False
    weight_embedding_covariance : float = 0.0

    use_curve_recon    : bool  = True
    weight_curve_recon : float = 1.0
    curve_kind         : str   = "mse"
    huber_delta        : float = 1.0
    charbonnier_eps    : float = 1e-3

    use_curve_sobolev    : bool  = False
    weight_curve_sobolev : float = 1.0


@dataclass
class JepaMaskingConfig:

    enabled        : bool  = False
    strategy       : str   = "block"
    mask_ratio     : float = 0.5
    n_blocks       : int   = 4
    fill           : str   = "zeros"
    fill_noise_std : float = 1.0
    loss_on        : str   = "masked"


@dataclass
class JepaCurriculumConfig:

    enabled    : bool = False
    swap_epoch : int  = 15

    warmup_embedding : EmbeddingLossConfig = field(default_factory=EmbeddingLossConfig)
    warmup_param     : LossConfig          = field(default_factory=default_param_loss)

    reset_lr        : bool = False
    reset_warmup    : bool = False
    reset_optimizer : bool = False


@dataclass
class CurveTrialsConfig:

    kinds            : list[str]   = field(default_factory=lambda: ["mse", "huber", "charbonnier"])
    weights          : list[float] = field(default_factory=lambda: [0.25, 1.0])
    sobolev_weights  : list[float] = field(default_factory=lambda: [0.1, 1.0])
    include_baseline : bool        = True


@dataclass
class MaskingTrialsConfig:

    strategies       : list[str]   = field(default_factory=lambda: ["block", "random"])
    ratios           : list[float] = field(default_factory=lambda: [0.25, 0.5, 0.75])
    fill             : str         = "zeros"
    loss_on          : str         = "masked"
    include_baseline : bool        = True


@dataclass
class JepaCurriculumTrialsConfig:

    swap_epochs      : list[int] = field(default_factory=lambda: [5, 15, 30])
    resets           : list[str] = field(default_factory=lambda: ["none", "all"])
    include_baseline : bool      = True


@dataclass
class CouplingTrialsConfig:

    modes            : list[str]   = field(default_factory=lambda: ["frozen", "finetune"])
    finetune_lrs     : list[float] = field(default_factory=lambda: [1e-5, 3e-5, 1e-4])
    target_providers : list[str]   = field(default_factory=lambda: ["stopgrad"])


@dataclass
class AeRunTrialsConfig:

    runs : list[str] = field(default_factory=list)


@dataclass
class JepaCouplingConfig:

    profile_autoencoder_logdir : Path       = Path("runs/profile_autoencoder")
    profile_autoencoder_run    : str | None = None
    profile_autoencoder_mode   : str        = "frozen"
    target_provider            : str        = "stopgrad"
    ae_finetune_lr             : float      = 3e-5
    ae_finetune_wd             : float      = 1e-4

    image_autoencoder_logdir : Path       = Path("runs/image_autoencoder")
    image_autoencoder_run    : str | None = None
    image_autoencoder_mode   : str        = "frozen"
    image_ae_finetune_lr     : float      = 3e-5
    image_ae_finetune_wd     : float      = 1e-4

    embedding_loss : EmbeddingLossConfig  = field(default_factory=EmbeddingLossConfig)
    param_loss     : LossConfig           = field(default_factory=default_param_loss)
    curriculum     : JepaCurriculumConfig = field(default_factory=JepaCurriculumConfig)
    masking        : JepaMaskingConfig    = field(default_factory=JepaMaskingConfig)

    def entry_kwargs(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


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

    param_loss : LossConfig           = field(default_factory=default_param_loss)
    curriculum : JepaCurriculumConfig = field(default_factory=JepaCurriculumConfig)
    masking    : JepaMaskingConfig    = field(default_factory=JepaMaskingConfig)

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
        return InferenceConfig.for_run_directory()


@dataclass
class JepaEntryConfig:

    run_name        : str | None = None
    resume          : bool       = True
    resume_from     : str        = ""
    backbone_name   : str        = "resunet"
    backbone_head   : str        = "conv"
    gpu             : int        = 0
    gpus            : list[int]  = field(default_factory=lambda: [0])
    ddp             : bool       = False
    gpus_file       : str        = ""
    poll_interval_s : float      = 5.0
    seed            : int        = 0
    seeds           : list[int]  = field(default_factory=standard_seeds)
    logdir          : Path       = Path("runs/jepa")
    model_overrides : dict       = field(default_factory=dict)

    profile_autoencoder_logdir : Path       = Path("runs/profile_autoencoder")
    profile_autoencoder_run    : str | None = None
    profile_autoencoder_mode   : str        = "frozen"
    target_provider            : str        = "stopgrad"
    ae_finetune_lr             : float      = 3e-5
    ae_finetune_wd             : float      = 1e-4

    image_autoencoder_logdir : Path       = Path("runs/image_autoencoder")
    image_autoencoder_run    : str | None = None
    image_autoencoder_mode   : str        = "frozen"
    image_ae_finetune_lr     : float      = 3e-5
    image_ae_finetune_wd     : float      = 1e-4

    embedding_loss : EmbeddingLossConfig  = field(default_factory=EmbeddingLossConfig)
    param_loss     : LossConfig           = field(default_factory=default_param_loss)
    curriculum     : JepaCurriculumConfig = field(default_factory=JepaCurriculumConfig)
    masking        : JepaMaskingConfig    = field(default_factory=JepaMaskingConfig)
    probe          : LossScaleProbeConfig = field(default_factory=LossScaleProbeConfig)

    input         : InputConfig         = field(default_factory=InputConfig.full_stack)
    paths         : TrainingPathsConfig = field(default_factory=TrainingPathsConfig)
    training      : TrainingQueueConfig = field(default_factory=TrainingQueueConfig)
    pretrain      : PretrainConfig      = field(default_factory=PretrainConfig)
    normalization : NormalizationConfig = field(default_factory=NormalizationConfig)
    augmentation  : AugmentationConfig  = field(default_factory=AugmentationConfig)
    overfit_check : OverfitCheckConfig  = field(default_factory=OverfitCheckConfig)

    infer_after  : bool            = False
    infer_at_end : bool            = False
    inference    : InferenceConfig = field(default_factory=JepaDefaults.inference)

    trials_enabled : bool = False
    trials_mode    : str  = "embedding"

    embedding_trials     : dict                       = field(default_factory=JepaTrialCatalog.embedding_losses)
    curve_trials         : CurveTrialsConfig          = field(default_factory=CurveTrialsConfig)
    masking_trials       : MaskingTrialsConfig        = field(default_factory=MaskingTrialsConfig)
    curriculum_trials    : JepaCurriculumTrialsConfig = field(default_factory=JepaCurriculumTrialsConfig)
    coupling_trials      : CouplingTrialsConfig       = field(default_factory=CouplingTrialsConfig)
    ae_run_trials        : AeRunTrialsConfig          = field(default_factory=AeRunTrialsConfig)
    context_trials       : list                       = field(default_factory=TrialCatalog.context_trials)
    input_trials         : dict                       = field(default_factory=TrialCatalog.input_trials)
    augmentation_trials  : AugmentationTrialsConfig   = field(default_factory=AugmentationTrialsConfig)
    ema_trials           : EmaTrialsConfig            = field(default_factory=EmaTrialsConfig)
    stability_trials     : StabilityTrialsConfig      = field(default_factory=StabilityTrialsConfig)
