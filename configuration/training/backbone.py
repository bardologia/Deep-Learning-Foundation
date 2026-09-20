"""Training configuration for the single-trunk backbone family.

Defines the standard two-stage loss curriculum, the trial catalogues driving
the ablation and sweep modes, and the entry config the `train_backbone` entry
point consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

from configuration.training.general.run          import TrainingPathsConfig, TrainingQueueConfig, standard_seeds
from configuration.dataset                       import AugmentationConfig, InputConfig
from configuration.inference.general             import InferenceConfig
from configuration.dataset.gaussian           import GaussianConfig
from configuration.normalization.general         import NormalizationConfig
from configuration.training.general.ablation     import AblationCatalog
from configuration.training.general.loss         import LossConfig, LossCurriculumConfig
from configuration.training.general.optimization import EarlyStoppingConfig, GradientClipperConfig, OptimizerConfig, SchedulerConfig, WarmupConfig
from configuration.training.general.runtime      import IOConfig, MemoryConfig, OverfitCheckConfig, ResourceConfig, TrainingLoopConfig
from configuration.training.general.pretraining  import PretrainConfig
from configuration.training.general.trials       import TrialCatalog


def default_curriculum() -> LossCurriculumConfig:
    """Builds the standard curriculum: parameter L1 plus cosine, then covariance.

    The warmup stage optimises Hungarian-matched parameter L1 with the cosine
    curve term; the complete stage adds the covariance-matching physics term.

    Returns:
        The two-stage LossCurriculumConfig used as the default for backbone runs.
    """
    warmup = LossConfig(
        use_param_l1             = True,
        weight_param_l1          = 1.0,
        param_matching           = AblationCatalog.PARAM_MATCH_FULL,
        use_active_normalization = True,
        presence_balance         = False,
        use_cosine_curve         = True,
        weight_cosine_curve      = AblationCatalog.COSINE_WEIGHT,
    )

    complete = LossConfig(
        use_param_l1             = True,
        weight_param_l1          = 1.0,
        param_matching           = AblationCatalog.PARAM_MATCH_FULL,
        use_active_normalization = True,
        presence_balance         = True,
        use_cosine_curve         = True,
        weight_cosine_curve      = AblationCatalog.COSINE_WEIGHT,
    )

    return LossCurriculumConfig(
        enabled    = True,
        swap_epoch = AblationCatalog.CURRICULUM_SWAP_EPOCH,
        warmup     = warmup,
        complete   = complete,
    )


@dataclass
class PatchTrialsConfig:
    """Trial settings sweeping the training patch shape over a rectangular grid.

    Attributes:
        minimum               : Smallest patch shape included, as (rows, cols) pixels; 0 starts at one step.
        maximum               : Largest patch shape included, as (rows, cols) pixels.
        step                  : Spacing between consecutive patch lengths in pixels, or 0 to derive it from the backbone's encoder pyramid.
        stride_ratio          : Patch stride as a fraction of the patch length, per axis.
        constant_pixel_budget : Whether each trial's batch size is rescaled so every patch shape sees the same pixel count per optimizer step.
    """

    minimum               : tuple[int, int] = (0, 0)
    maximum               : tuple[int, int] = (96, 48)
    step                  : int             = 0
    stride_ratio          : float           = 0.5
    constant_pixel_budget : bool            = True


@dataclass
class PairTrialsConfig:
    """Trial settings pairing a base loss term with one auxiliary term.

    Attributes:
        base_component   : Loss term kept enabled in every trial.
        base_weight      : Weight of that base term.
        components       : Auxiliary loss terms paired with the base, one trial each.
        weights          : Loss weights swept for each auxiliary term.
        include_baseline : Whether a base-only run is added.
    """

    base_component   : str         = "param_l1"
    base_weight      : float       = 1.0
    components       : list[str]   = field(default_factory=lambda: ["cosine_curve", "total_power_relerr", "moments"])
    weights          : list[float] = field(default_factory=lambda: [0.01, 0.05, 0.25])
    include_baseline : bool        = True


@dataclass
class LogEnergyTrialsConfig:
    """Trial settings sweeping the log-energy curve term.

    Attributes:
        weights            : Loss weights swept for the log-energy term.
        replace_covariance : Whether the term replaces the covariance term instead of joining it.
        include_baseline   : Whether a run without the term is added.
    """

    weights            : list[float] = field(default_factory=lambda: [0.01, 0.05, 0.25, 0.5, 1.0])
    replace_covariance : bool        = False
    include_baseline   : bool        = True


@dataclass
class ReachTrialsConfig:
    """Trial settings comparing architectures at matched parameter count.

    Attributes:
        rungs           : Per-rung architecture specifications from the trial catalogue.
        patch_size      : Training patch shape as (range, row) pixels.
        patch_stride    : Patch stride as (range, row) pixels.
        in_channels     : Number of input channels the rungs are sized for.
        match_tolerance : Allowed relative deviation between rung parameter counts.
    """

    rungs           : list            = field(default_factory=TrialCatalog.reach_rungs)
    patch_size      : tuple[int, int] = (32, 32)
    patch_stride    : tuple[int, int] = (16, 16)
    in_channels     : int             = 9
    match_tolerance : float           = 0.01


@dataclass
class SecondaryTrialsConfig:
    """Trial settings sweeping which secondary tracks enter the stack.

    Attributes:
        strategy      : Rule choosing the secondaries, e.g. "consecutive".
        n_secondaries : Number of secondary tracks per trial.
        n_trials      : Number of track subsets drawn.
        mean          : Mean of the sampled baseline distribution in metres, or None when the strategy is deterministic.
        sigma         : Standard deviation of that distribution in metres, or None when the strategy is deterministic.
        block_step    : Step between consecutive blocks of secondaries.
        spacing       : Track index spacing inside one subset.
        seed          : Seed for the subset draw.
    """

    strategy      : str          = "consecutive"
    n_secondaries : int          = 4
    n_trials      : int          = 8
    mean          : float | None = None
    sigma         : float | None = None
    block_step    : int          = 1
    spacing       : int          = 2
    seed          : int          = 0


@dataclass
class HeadMatchingTrialsConfig:
    """Trial settings crossing output heads with slot-matching rules.

    Attributes:
        backbone  : Backbone family held fixed across the trials.
        heads     : Output heads swept, e.g. "conv" and "set_pred".
        matchings : Slot-matching rules swept, e.g. "sorted_gt" and "hungarian".
    """

    backbone  : str       = "unet"
    heads     : list[str] = field(default_factory=lambda: ["conv", "set_pred"])
    matchings : list[str] = field(default_factory=lambda: ["sorted_gt", "hungarian"])


@dataclass
class AugmentationTrialsConfig:
    """Trial settings for the cumulative augmentation ladder.

    The ladder starts with every augmentation off, then adds horizontal flips,
    then vertical flips, and finally additive Gaussian input noise at each of
    the configured strengths on top of both flips.

    Attributes:
        flip_probability  : Probability applied to each enabled flip rung.
        noise_probability : Probability of injecting noise on the noise rungs.
        noise_stds        : Noise standard deviations in normalised input units, one noise rung each.
    """

    flip_probability  : float       = 0.5
    noise_probability : float       = 0.5
    noise_stds        : list[float] = field(default_factory=lambda: [0.01, 0.03, 0.1])


@dataclass
class ClampTrialsConfig:
    """Trial settings for the output-clamp ladder.

    The ladder compares no clamp, a hard clamp and one leaky clamp per
    configured slope pair on otherwise identical runs; the clamp bounds
    themselves come from the run's normalization config so the axis isolates
    the clamp shape alone.

    Attributes:
        leaky_slopes       : Leak slopes outside the clamp bounds for curve outputs, one leaky rung each.
        param_leaky_slopes : Leak slopes outside the clamp bounds for Gaussian parameter outputs, paired with `leaky_slopes` by position.
    """

    leaky_slopes       : list[float] = field(default_factory=lambda: [0.1])
    param_leaky_slopes : list[float] = field(default_factory=lambda: [0.1])


@dataclass
class StabilityTrialsConfig:
    """Trial settings for the training-stability arms.

    The ladder starts with neither learning-rate warmup nor gradient clipping,
    then plans one warmup-only rung per configured fraction of the total
    optimizer steps, one clipping-only rung per configured norm threshold, and
    one combined rung pairing the first fraction with each threshold.

    Attributes:
        warmup_fractions : Fractions of the total optimizer steps the warmup ramp spans, one warmup rung each.
        warmup_mode      : Shape of that ramp, a `WarmupMode` value.
        clip_mode        : Clipping strategy of the clipping rungs, a non-disabled `ClipMode` value.
        max_grad_norms   : Norm thresholds of the fixed clipping strategy, one clipping rung each.
    """

    warmup_fractions : list[float] = field(default_factory=lambda: [0.03])
    warmup_mode      : str         = "linear"
    clip_mode        : str         = "fixed"
    max_grad_norms   : list[float] = field(default_factory=lambda: [1.0])


@dataclass
class EmaTrialsConfig:
    """Trial settings for the weight-EMA on/off ladder.

    The ladder compares the raw weights against an exponential moving average
    of the weights on otherwise identical runs: one rung trains with EMA off,
    then per configured decay one rung trains with the plain decay and one with
    the decay warmup ramp.

    Attributes:
        decays : EMA decay rates, one EMA-on rung each.
    """

    decays : list[float] = field(default_factory=lambda: [0.999])


@dataclass
class LossScaleProbeConfig:
    """Pre-training probe measuring the magnitude of each loss term.

    Attributes:
        enabled        : Whether the probe runs before training.
        n_batches      : Batches accumulated to estimate the term scales.
        reference      : Loss term the others are reported relative to, or None for absolute scales.
        exit_after     : Whether the process stops once the probe has reported.
        enabled_losses : Loss toggles overriding the curriculum during the probe.
    """

    enabled        : bool       = False
    n_batches      : int        = 1000
    reference      : str | None = "param_l1"
    exit_after     : bool       = True
    enabled_losses : dict       = field(default_factory=dict)


@dataclass
class BackboneTrainerConfig:
    """Resolved trainer settings persisted with a backbone run.

    Attributes:
        gaussian         : Layout of the predicted Gaussian mixture.
        early_stopping   : Patience and monitored metric for early stopping.
        warmup           : Learning-rate warmup schedule.
        scheduler        : Learning-rate scheduler settings.
        io               : Checkpoint and logging IO settings.
        optimizer        : Optimizer type, learning rate and weight decay.
        training         : Epoch count, batch size and loop-level toggles.
        curriculum       : Two-stage loss curriculum.
        resources        : Device and worker allocation.
        memory           : Mixed precision and memory-saving toggles.
        gradient_clipper : Gradient clipping settings.
    """

    gaussian         : GaussianConfig
    early_stopping   : EarlyStoppingConfig   = field(default_factory=EarlyStoppingConfig)
    warmup           : WarmupConfig          = field(default_factory=WarmupConfig)
    scheduler        : SchedulerConfig       = field(default_factory=SchedulerConfig)
    io               : IOConfig              = field(default_factory=IOConfig)
    optimizer        : OptimizerConfig       = field(default_factory=OptimizerConfig)
    training         : TrainingLoopConfig    = field(default_factory=TrainingLoopConfig)
    curriculum       : LossCurriculumConfig  = field(default_factory=LossCurriculumConfig)
    resources        : ResourceConfig        = field(default_factory=ResourceConfig)
    memory           : MemoryConfig          = field(default_factory=MemoryConfig)
    gradient_clipper : GradientClipperConfig = field(default_factory=GradientClipperConfig)


@dataclass
class BackboneEntryConfig:
    """Entry-point settings for training one or many backbone runs.

    Beyond the single-run fields, this config carries every trial catalogue the
    sweep modes read and the ablation feature list.

    Attributes:
        run_name              : Explicit run directory name, or None to derive one.
        resume                : Whether an interrupted run continues from its checkpoint.
        resume_from           : Run directory or sweep folder whose state is resumed.
        backbone_name         : Backbone family trained.
        backbone_head         : Output head attached to the backbone.
        gpu                   : GPU device index for a single-run launch.
        seed                  : Seed used for a single-run launch.
        seeds                 : Seeds trained when the run is fanned out over a seed sweep.
        logdir                : Root directory receiving the run directories.
        model_overrides       : Architecture keyword overrides passed to the backbone.
        paths                 : Dataset and parameter paths of the training data.
        training              : Queue-level training settings.
        pretrain              : Batch-size and dataloader pretuning settings.
        curriculum            : Two-stage loss curriculum.
        geometry              : Acquisition geometry used by the physics loss terms.
        input                 : Input channel selection fed to the backbone.
        normalization         : Input and output normalization choices.
        augmentation          : Training-time augmentation probabilities.
        probe                 : Loss-scale probe settings run before training.
        overfit_check         : Single-batch overfitting sanity gate run before training.
        infer_after           : Whether inference runs after each seed finishes.
        infer_at_end          : Whether inference runs once after the whole sweep.
        inference             : Inference settings used by those automatic passes.
        trials_enabled        : Whether the entry point fans out into a trial sweep.
        trials_mode           : Name of the trial sweep executed.
        warmup_losses         : Loss toggle sets swept for the warmup stage.
        complete_losses       : Loss toggle sets swept for the complete stage.
        presence_trials       : Slot-presence toggle sets swept.
        pair_trials           : Base-plus-auxiliary loss pair sweep settings.
        log_energy_trials     : Log-energy term sweep settings.
        secondary_trials      : Secondary track selection sweep settings.
        patch_trials          : Rectangular patch-shape grid sweep settings.
        input_trials          : Input channel selections swept.
        context_trials        : Context ladder rungs swept.
        reach_trials          : Matched-capacity architecture sweep settings.
        head_trials           : Output head and matching rule sweep settings.
        augmentation_trials   : Cumulative augmentation ladder settings.
        clamp_trials          : Output-clamp ladder settings.
        stability_trials      : Warmup and gradient-clipping ladder settings.
        ema_trials            : Weight-EMA on/off ladder settings; the amp mode compares mixed-precision off against on and needs no settings of its own.
        ablation_features     : Ordered feature list of the leave-one-out ablation.
        ablation_include_full : Whether the full-model reference run is included.
        gpus                  : GPU device indices available to the sweep workers, and the rank devices of a DDP run.
        ddp                   : Whether a single-seed run trains data-parallel across the `gpus` devices, one synchronised rank per device sharing the configured global batch.
        gpus_file             : Path to a file whose contents override the GPU pool at runtime.
        poll_interval_s       : Seconds between scheduler polls of the worker pool.
    """

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
    pretrain      : PretrainConfig       = field(default_factory=PretrainConfig)
    curriculum    : LossCurriculumConfig = field(default_factory=default_curriculum)
    input         : InputConfig          = field(default_factory=InputConfig.full_stack)
    normalization : NormalizationConfig  = field(default_factory=NormalizationConfig)
    augmentation  : AugmentationConfig   = field(default_factory=AugmentationConfig)

    probe         : LossScaleProbeConfig = field(default_factory=LossScaleProbeConfig)
    overfit_check : OverfitCheckConfig   = field(default_factory=OverfitCheckConfig)

    infer_after  : bool            = False
    infer_at_end : bool            = False
    inference    : InferenceConfig = field(default_factory=lambda: InferenceConfig.for_run_directory())

    trials_enabled       : bool                      = False
    trials_mode          : str                       = "curriculum"
    warmup_losses        : dict                      = field(default_factory=TrialCatalog.warmup_losses)
    complete_losses      : dict                      = field(default_factory=TrialCatalog.complete_losses)
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

    ablation_features     : list = field(default_factory=AblationCatalog.default_features)
    ablation_include_full : bool = True

    gpus            : list[int] = field(default_factory=lambda: [0])
    ddp             : bool      = False
    gpus_file       : str       = ""
    poll_interval_s : float     = 5.0
