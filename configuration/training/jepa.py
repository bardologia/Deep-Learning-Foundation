"""Configuration for JEPA training, where a backbone predicts autoencoder embeddings.

Couples a frozen or fine-tuned profile autoencoder (the target encoder for the
elevation profiles) and an optional image autoencoder to the backbone that has to
predict their latent codes from the SAR stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib     import Path

from configuration.training.general.run               import TrainingPathsConfig, TrainingQueueConfig, standard_seeds
from configuration.training.general.trials             import JepaTrialCatalog, TrialCatalog
from configuration.training.backbone                  import AugmentationTrialsConfig, EmaTrialsConfig, LossScaleProbeConfig, NormalizationTrialsConfig, StabilityTrialsConfig
from configuration.architectures.profile_autoencoder  import ProfileAutoencoderBaseConfig
from configuration.dataset                            import AugmentationConfig, InputConfig
from configuration.normalization.general              import NormalizationConfig
from configuration.architectures.image_autoencoder    import ImageAutoencoderBaseConfig
from configuration.inference.general                  import InferenceConfig
from configuration.training.general.loss              import LossConfig, ParamMatching
from configuration.training.general.optimization      import EarlyStoppingConfig, GradientClipperConfig, OptimizerConfig, SchedulerConfig, WarmupConfig
from configuration.training.general.runtime           import IOConfig, MemoryConfig, OverfitCheckConfig, ResourceConfig, TrainingLoopConfig
from configuration.training.general.pretraining       import PretrainConfig
from configuration.training.general.trainer           import SharedSubConfigInheritance


def default_param_loss() -> LossConfig:
    """Returns the param-L1 loss on GT-sorted slots used to read out Gaussian parameters."""
    return LossConfig(
        use_param_l1             = True,
        weight_param_l1          = 1.0,
        param_matching           = ParamMatching.SORTED_GT,
        use_active_normalization = True,
        presence_balance         = False,
    )


@dataclass
class EmbeddingLossConfig:
    """Losses tying the predicted embedding to the target embedding and its decoding.

    Attributes:
        use_embedding_mse            : Whether a squared error on the latent code is applied.
        weight_embedding_mse         : Weight of that squared error.
        use_embedding_cosine         : Whether a cosine penalty on the latent code is applied.
        weight_embedding_cosine      : Weight of that cosine penalty.
        use_embedding_smoothl1       : Whether a smooth-L1 penalty on the latent code is applied.
        weight_embedding_smoothl1    : Weight of that smooth-L1 penalty.
        smoothl1_beta                : Transition point of the smooth-L1 penalty.
        use_embedding_huber          : Whether a Huber penalty on the latent code is applied.
        weight_embedding_huber       : Weight of that Huber penalty.
        embedding_huber_delta        : Transition point of the Huber embedding penalty.
        use_embedding_charbonnier    : Whether a Charbonnier penalty on the latent code is applied.
        weight_embedding_charbonnier : Weight of that Charbonnier penalty.
        embedding_charbonnier_eps    : Smoothing constant of the Charbonnier embedding penalty.
        use_embedding_variance       : Whether the per-dimension spread of the predicted embedding is held above a floor, preventing dimensional collapse.
        weight_embedding_variance    : Weight of that variance hinge.
        variance_gamma               : Standard-deviation floor of the variance hinge.
        use_embedding_covariance     : Whether the off-diagonal covariance of the predicted embedding is penalised, decorrelating its dimensions.
        weight_embedding_covariance  : Weight of that covariance penalty.
        use_curve_recon              : Whether the decoded elevation profile is compared to ground truth.
        weight_curve_recon           : Weight of that decoded-profile penalty.
        curve_kind                   : Penalty family used on the decoded profile.
        huber_delta                  : Transition point of the Huber profile penalty.
        charbonnier_eps              : Smoothing constant of the Charbonnier profile penalty.
        use_curve_sobolev            : Whether the derivative of the decoded profile is penalised.
        weight_curve_sobolev         : Weight of that derivative penalty.
    """

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
    """Context masking of the input patch for the JEPA embedding regime.

    When enabled, part of each input patch is hidden before the backbone
    predicts the profile embeddings, and the embedding loss can be restricted
    to the hidden pixels, turning the run into context-to-target prediction.

    Attributes:
        enabled        : Whether input masking is applied during training.
        strategy       : How masked pixels are chosen: ``block`` hides rectangular regions, ``random`` hides independent pixels.
        mask_ratio     : Target fraction of the patch pixels hidden per sample.
        n_blocks       : Number of rectangles the block strategy hides per sample.
        fill           : Value written into hidden pixels: ``zeros`` or Gaussian ``noise``.
        fill_noise_std : Standard deviation of the noise fill in normalized units.
        loss_on        : Pixels the embedding loss is evaluated on: ``masked`` or ``all``.
    """

    enabled        : bool  = False
    strategy       : str   = "block"
    mask_ratio     : float = 0.5
    n_blocks       : int   = 4
    fill           : str   = "zeros"
    fill_noise_std : float = 1.0
    loss_on        : str   = "masked"


@dataclass
class JepaCurriculumConfig:
    """Two-stage JEPA loss schedule swapping to the main loss at a fixed epoch.

    When enabled, training starts on the warmup stage of the active regime
    (``warmup_embedding`` with a coupled profile autoencoder, ``warmup_param``
    otherwise) and swaps to the run's main ``embedding_loss`` or ``param_loss``
    at the swap epoch, so the run name still encodes the final stage.

    Attributes:
        enabled          : Whether the warmup stage precedes the main loss.
        swap_epoch       : Epoch at which training swaps to the main loss.
        warmup_embedding : Embedding-loss stage active before the swap when a profile autoencoder is coupled.
        warmup_param     : Parameter-loss stage active before the swap when no profile autoencoder is coupled.
        reset_lr         : Whether the learning rate is reset at the swap.
        reset_warmup     : Whether the learning-rate warmup restarts at the swap.
        reset_optimizer  : Whether the optimizer state is cleared at the swap.
    """

    enabled    : bool = False
    swap_epoch : int  = 15

    warmup_embedding : EmbeddingLossConfig = field(default_factory=EmbeddingLossConfig)
    warmup_param     : LossConfig          = field(default_factory=default_param_loss)

    reset_lr        : bool = False
    reset_warmup    : bool = False
    reset_optimizer : bool = False


@dataclass
class CurveTrialsConfig:
    """Decoded-curve loss arms swept on top of the configured latent terms.

    Attributes:
        kinds            : Penalty families of the decoded-profile term, one arm per kind and weight.
        weights          : Weights of the decoded-profile term.
        sobolev_weights  : Weights of the decoded-profile curvature penalty, one arm each on top of an mse curve term.
        include_baseline : Whether a run with the decoded-curve terms off is added.
    """

    kinds            : list[str]   = field(default_factory=lambda: ["mse", "huber", "charbonnier"])
    weights          : list[float] = field(default_factory=lambda: [0.25, 1.0])
    sobolev_weights  : list[float] = field(default_factory=lambda: [0.1, 1.0])
    include_baseline : bool        = True


@dataclass
class MaskingTrialsConfig:
    """Context-masking grid of the JEPA embedding regime.

    Attributes:
        strategies       : Masking strategies swept, ``block`` or ``random``.
        ratios           : Hidden-pixel fractions swept per strategy.
        fill             : Value written into hidden pixels on every arm: ``zeros`` or ``noise``.
        loss_on          : Pixels the embedding loss is evaluated on for every arm: ``masked`` or ``all``.
        include_baseline : Whether a run without masking is added.
    """

    strategies       : list[str]   = field(default_factory=lambda: ["block", "random"])
    ratios           : list[float] = field(default_factory=lambda: [0.25, 0.5, 0.75])
    fill             : str         = "zeros"
    loss_on          : str         = "masked"
    include_baseline : bool        = True


@dataclass
class JepaCurriculumTrialsConfig:
    """Curriculum-schedule grid over the swap epoch and the state resets at the swap.

    Attributes:
        swap_epochs      : Epochs at which training swaps to the main loss.
        resets           : State resets applied at the swap: ``none``, ``lr``, ``warmup``, ``optimizer`` or ``all``.
        include_baseline : Whether a curriculum-free run is added.
    """

    swap_epochs      : list[int] = field(default_factory=lambda: [5, 15, 30])
    resets           : list[str] = field(default_factory=lambda: ["none", "all"])
    include_baseline : bool      = True


@dataclass
class CouplingTrialsConfig:
    """Autoencoder-coupling arms over the freeze mode, the fine-tune rate and the target provider.

    Attributes:
        modes            : Profile-encoder modes swept, ``frozen`` or ``finetune``.
        finetune_lrs     : Learning rates of the fine-tuned encoder, one arm each when ``finetune`` is swept.
        target_providers : Target providers swept, ``stopgrad`` or ``live``.
    """

    modes            : list[str]   = field(default_factory=lambda: ["frozen", "finetune"])
    finetune_lrs     : list[float] = field(default_factory=lambda: [1e-5, 3e-5, 1e-4])
    target_providers : list[str]   = field(default_factory=lambda: ["stopgrad"])


@dataclass
class AeRunTrialsConfig:
    """Pretrained profile-autoencoder runs compared by coupling each into its own JEPA training.

    Attributes:
        runs : Run folder names under the profile-autoencoder log root, one arm each.
    """

    runs : list[str] = field(default_factory=list)


@dataclass
class JepaCouplingConfig:
    """Which pretrained autoencoders the JEPA backbone is coupled to, and how.

    Attributes:
        profile_autoencoder_logdir : Root holding the profile-autoencoder runs.
        profile_autoencoder_run    : Run folder supplying the profile encoder weights.
        profile_autoencoder_mode   : Whether that encoder stays frozen or is fine-tuned.
        target_provider            : How the target embedding is produced, e.g. stop-gradient.
        ae_finetune_lr             : Learning rate applied to the profile autoencoder when fine-tuned.
        ae_finetune_wd             : Weight decay applied to it when fine-tuned.
        image_autoencoder_logdir   : Root holding the image-autoencoder runs.
        image_autoencoder_run      : Run folder supplying the image encoder weights.
        image_autoencoder_mode     : Whether that encoder stays frozen or is fine-tuned.
        image_ae_finetune_lr       : Learning rate applied to the image autoencoder when fine-tuned.
        image_ae_finetune_wd       : Weight decay applied to it when fine-tuned.
        embedding_loss             : Latent and decoded-profile loss settings.
        param_loss                 : Loss used on Gaussian parameters read out of the prediction.
        curriculum                 : Two-stage warmup schedule swapping to the main loss.
        masking                    : Context masking of the input patch in the embedding regime.
    """

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
        """Returns every coupling field as a keyword-argument mapping for the entry config."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass
class JepaTrainerConfig(SharedSubConfigInheritance):
    """Everything the JEPA trainer needs, with the coupled autoencoders resolved.

    Attributes:
        gaussian                       : Gaussian label configuration describing the elevation grid.
        autoencoder                    : Architecture config of the profile autoencoder, or None.
        embedding_loss                 : Latent and decoded-profile loss settings.
        profile_autoencoder_mode       : Whether the profile encoder is frozen or fine-tuned.
        target_provider                : How the target embedding is produced, e.g. stop-gradient.
        profile_autoencoder_checkpoint : Checkpoint file the profile weights load from.
        ae_finetune_lr                 : Learning rate for the profile autoencoder when fine-tuned.
        ae_finetune_wd                 : Weight decay for it when fine-tuned.
        image_autoencoder              : Architecture config of the image autoencoder, or None.
        image_autoencoder_mode         : Whether the image encoder is frozen or fine-tuned.
        image_autoencoder_checkpoint   : Checkpoint file the image weights load from.
        image_ae_finetune_lr           : Learning rate for the image autoencoder when fine-tuned.
        image_ae_finetune_wd           : Weight decay for it when fine-tuned.
        param_loss                     : Loss on Gaussian parameters read out of the prediction.
        curriculum                     : Two-stage warmup schedule swapping to the main loss.
        masking                        : Context masking of the input patch in the embedding regime.
        early_stopping                 : Validation-loss stopping rule.
        warmup                         : Learning-rate warmup ramp.
        scheduler                      : Learning-rate schedule.
        io                             : Output root for checkpoints and logs.
        optimizer                      : AdamW hyper-parameters.
        training                       : Epoch loop, precision and resume behaviour.
        resources                      : Resource-monitor cadence and warning levels.
        memory                         : VRAM reservation settings.
        gradient_clipper               : Gradient-norm clipping strategy.
    """

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
    """Factories for the JEPA entry-point defaults that are not plain literals."""

    @staticmethod
    def inference() -> InferenceConfig:
        """Returns the run-directory inference config."""
        return InferenceConfig.for_run_directory()


@dataclass
class JepaEntryConfig:
    """Command-line surface of the JEPA training entry point.

    Attributes:
        run_name                   : Explicit run folder name, or None to derive one from the settings.
        resume                     : Whether an interrupted run continues from its last checkpoint.
        resume_from                : Run directory or sweep folder whose state is resumed.
        backbone_name              : Registry key of the backbone predicting the embeddings.
        backbone_head              : Output head attached to that backbone.
        gpu                        : CUDA device index used for a single-seed run.
        gpus                       : CUDA device indices the seed sweep fans out over, and the rank devices of a DDP run.
        ddp                        : Whether a single-seed run trains data-parallel across the `gpus` devices, one synchronised rank per device sharing the configured global batch.
        gpus_file                  : File listing device indices, re-read to resize the pool live.
        poll_interval_s            : Seconds between scheduler polls of the device pool.
        seed                       : Seed of a single-seed run.
        seeds                      : Seeds trained by the sweep.
        logdir                     : Root directory the run folders are created under.
        model_overrides            : Backbone fields overridden on top of the registry default.
        profile_autoencoder_logdir : Root holding the profile-autoencoder runs.
        profile_autoencoder_run    : Run folder supplying the profile encoder weights.
        profile_autoencoder_mode   : Whether that encoder stays frozen or is fine-tuned.
        target_provider            : How the target embedding is produced, e.g. stop-gradient.
        ae_finetune_lr             : Learning rate for the profile autoencoder when fine-tuned.
        ae_finetune_wd             : Weight decay for it when fine-tuned.
        image_autoencoder_logdir   : Root holding the image-autoencoder runs.
        image_autoencoder_run      : Run folder supplying the image encoder weights.
        image_autoencoder_mode     : Whether that encoder stays frozen or is fine-tuned.
        image_ae_finetune_lr       : Learning rate for the image autoencoder when fine-tuned.
        image_ae_finetune_wd       : Weight decay for it when fine-tuned.
        embedding_loss             : Latent and decoded-profile loss settings.
        param_loss                 : Loss on Gaussian parameters read out of the prediction.
        curriculum                 : Two-stage warmup schedule swapping to the main loss.
        masking                    : Context masking of the input patch in the embedding regime.
        probe                      : Pre-training probe measuring the raw magnitude of each loss term.
        input                      : Input channel selection of the SAR stack.
        paths                      : Dataset and parameter-label locations.
        training                   : Training schedule expanded into the trainer sub-configs.
        pretrain                   : Batch-size and loader probes run before training.
        normalization              : Input normalization strategy and statistics.
        augmentation               : Patch augmentation settings.
        overfit_check              : Pre-training overfit sanity gate.
        infer_after                : Whether inference runs after each seed finishes.
        infer_at_end               : Whether inference runs once after the whole sweep finishes.
        inference                  : Inference settings used by those passes.
        trials_enabled             : Whether the multi-trial GPU sweep replaces the single training run.
        trials_mode                : Name of the trial sweep executed.
        embedding_trials           : Latent-loss arms of the embedding sweep, keyed by run label.
        curve_trials               : Decoded-curve arms of the curve sweep.
        masking_trials             : Context-masking grid of the masking sweep.
        curriculum_trials          : Swap-epoch and reset grid of the curriculum sweep.
        coupling_trials            : Freeze-mode, fine-tune-rate and target-provider arms of the coupling sweep.
        ae_run_trials              : Pretrained autoencoder runs compared by the ae_run sweep.
        context_trials             : Constant-depth backbone rungs of the context sweep.
        input_trials               : Input-channel arms of the input sweep, keyed by run label.
        augmentation_trials        : Cumulative ladder of the patch-augmentation sweep.
        normalization_trials       : Initial and final presets of the normalization ladder.
        ema_trials                 : Weight-EMA ladder of the ema sweep; the amp mode compares mixed-precision off against on and needs no settings of its own.
        stability_trials           : Warmup and gradient-clip arms of the stability sweep.
    """

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
    normalization_trials : NormalizationTrialsConfig  = field(default_factory=NormalizationTrialsConfig)
    ema_trials           : EmaTrialsConfig            = field(default_factory=EmaTrialsConfig)
    stability_trials     : StabilityTrialsConfig      = field(default_factory=StabilityTrialsConfig)
