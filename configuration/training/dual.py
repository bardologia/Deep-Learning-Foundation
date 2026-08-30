"""Training configuration for the dual-trunk family.

A dual model routes the input channels into a parameter trunk and an existence
trunk. This module supplies the trunk-capacity and routing trial catalogues on
top of the shared backbone trial configs, plus the `train_dual` entry config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

from configuration.training.general.run         import TrainingPathsConfig, TrainingQueueConfig, standard_seeds
from configuration.dataset                      import AugmentationConfig, InputConfig
from configuration.inference.general            import InferenceConfig
from configuration.normalization.general        import NormalizationConfig
from configuration.training.backbone            import AugmentationTrialsConfig, ClampTrialsConfig, EmaTrialsConfig, HeadMatchingTrialsConfig, LogEnergyTrialsConfig, LossScaleProbeConfig, NormalizationTrialsConfig, PairTrialsConfig, PatchTrialsConfig, ReachTrialsConfig, SecondaryTrialsConfig, StabilityTrialsConfig, default_curriculum
from configuration.training.general.trials      import TrialCatalog
from configuration.training.general.ablation    import AblationCatalog
from configuration.training.general.loss        import LossCurriculumConfig
from configuration.training.general.runtime     import OverfitCheckConfig
from configuration.training.general.pretraining import PretrainConfig


dual_curriculum = default_curriculum


def _parity_resunet_features() -> list[int]:
    """Returns the per-stage channel widths giving each trunk half the capacity."""
    return [48, 96, 184, 352]


def _default_dual_routing_trials() -> dict:
    """Builds the routing trials crossing the trunk-input assignments with the aux channel.

    Returns:
        Mapping from trial label to the inputs of each trunk, every assignment
        duplicated as a no-aux variant and a variant appending the aux group to
        both trunks.
    """
    groups = {"full": ["chan", "derived"], "chan": ["chan"], "derived": ["derived"]}

    assignments = [
        ("full",    "full"),
        ("chan",    "full"),
        ("full",    "chan"),
        ("derived", "full"),
        ("full",    "derived"),
        ("chan",    "derived"),
        ("derived", "chan"),
    ]

    trials = {}
    for params, existence in assignments:
        trials[f"{params}-{existence}-noaux"] = {"params": list(groups[params]), "existence": list(groups[existence])}
        trials[f"{params}-{existence}-aux"]   = {"params": groups[params] + ["aux"], "existence": groups[existence] + ["aux"]}

    return trials


def _default_dual_ratio_trials() -> dict:
    """Returns the capacity-split trials, mapping a ratio label to trunk widths."""
    return {
        "50-50" : {"params": _parity_resunet_features(), "existence": _parity_resunet_features()},
        "60-40" : {"params": [60, 108, 204, 384],        "existence": [56, 92, 168, 312]},
        "70-30" : {"params": [48, 116, 220, 416],        "existence": [36, 76, 144, 272]},
        "80-20" : {"params": [60, 124, 236, 444],        "existence": [28, 52, 116, 224]},
        "90-10" : {"params": [64, 128, 248, 472],        "existence": [24, 48, 84, 156]},
    }


def _default_dual_head_trials() -> HeadMatchingTrialsConfig:
    """Returns the head trials comparing slot matching rules on the set-prediction head."""
    return HeadMatchingTrialsConfig(backbone="unet_skip", heads=["set_pred"], matchings=["sorted_gt", "hungarian"])


def _default_dual_reach_rungs() -> list:
    """Returns the reach rungs contrasting a fixed-context CNN against a UNet trunk."""
    return [
        {"label" : "cnn33", "backbone" : "local_cnn", "features" : [478] * 8, "dropout" : 0.15, "trunk_wd" : 1e-4},
        {"label" : "unet",  "backbone" : "unet",      "dropout" : 0.15},
    ]


def _default_dual_reach_trials() -> ReachTrialsConfig:
    """Returns the reach trial settings carrying the dual rungs."""
    return ReachTrialsConfig(rungs=_default_dual_reach_rungs())


@dataclass
class DualRoutingTrialsConfig:
    """Trial settings sweeping which input groups each trunk receives.

    Attributes:
        params_features    : Per-stage channel widths of the parameter trunk.
        existence_features : Per-stage channel widths of the existence trunk.
        trials             : Mapping from trial label to the input groups of each trunk.
    """

    params_features    : list[int] = field(default_factory=_parity_resunet_features)
    existence_features : list[int] = field(default_factory=_parity_resunet_features)
    trials             : dict      = field(default_factory=_default_dual_routing_trials)


@dataclass
class DualRatioTrialsConfig:
    """Trial settings sweeping the capacity split between the two trunks.

    Attributes:
        trials          : Mapping from ratio label to the per-stage widths of each trunk.
        match_tolerance : Allowed relative deviation between trial parameter counts.
    """

    trials          : dict  = field(default_factory=_default_dual_ratio_trials)
    match_tolerance : float = 0.01


@dataclass
class DualEntryConfig:
    """Entry-point settings for training one or many dual-trunk runs.

    Attributes:
        run_name              : Explicit run directory name, or None to derive one.
        resume                : Whether an interrupted run continues from its checkpoint.
        resume_from           : Run directory or sweep folder whose state is resumed.
        model_name            : Dual model family trained.
        gpu                   : GPU device index for a single-run launch.
        seed                  : Seed used for a single-run launch.
        seeds                 : Seeds trained when the run is fanned out over a seed sweep.
        logdir                : Root directory receiving the run directories.
        model_overrides       : Architecture keyword overrides passed to the dual model.
        params_backbone       : Backbone family used as the parameter trunk.
        existence_backbone    : Backbone family used as the existence trunk.
        params_input          : Input groups routed into the parameter trunk.
        existence_input       : Input groups routed into the existence trunk.
        params_overrides      : Architecture keyword overrides for the parameter trunk.
        existence_overrides   : Architecture keyword overrides for the existence trunk.
        paths                 : Dataset and parameter paths of the training data.
        training              : Queue-level training settings.
        pretrain              : Batch-size and dataloader pretuning settings.
        curriculum            : Two-stage loss curriculum.
        input                 : Input channel selection fed to the trunks.
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
        normalization_trials  : Initial and final normalization preset pair.
        routing_trials        : Input routing sweep settings.
        ratio_trials          : Trunk capacity split sweep settings.
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
    model_name      : str        = "dual_resunet"
    gpu             : int        = 0
    seed            : int        = 0
    seeds           : list[int]  = field(default_factory=standard_seeds)
    logdir          : Path       = Path("runs/dual")
    model_overrides : dict       = field(default_factory=dict)

    params_backbone     : str       = "unet_skip"
    existence_backbone  : str       = "unet_skip"
    params_input        : list[str] = field(default_factory=lambda: ["chan", "derived", "aux"])
    existence_input     : list[str] = field(default_factory=lambda: ["chan", "derived", "aux"])
    params_overrides    : dict      = field(default_factory=dict)
    existence_overrides : dict      = field(default_factory=dict)

    paths         : TrainingPathsConfig  = field(default_factory=TrainingPathsConfig)
    training      : TrainingQueueConfig  = field(default_factory=TrainingQueueConfig)
    pretrain      : PretrainConfig       = field(default_factory=PretrainConfig)
    curriculum    : LossCurriculumConfig = field(default_factory=dual_curriculum)
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
    reach_trials         : ReachTrialsConfig         = field(default_factory=_default_dual_reach_trials)
    head_trials          : HeadMatchingTrialsConfig  = field(default_factory=_default_dual_head_trials)
    augmentation_trials  : AugmentationTrialsConfig  = field(default_factory=AugmentationTrialsConfig)
    clamp_trials         : ClampTrialsConfig         = field(default_factory=ClampTrialsConfig)
    stability_trials     : StabilityTrialsConfig     = field(default_factory=StabilityTrialsConfig)
    ema_trials           : EmaTrialsConfig           = field(default_factory=EmaTrialsConfig)
    normalization_trials : NormalizationTrialsConfig = field(default_factory=NormalizationTrialsConfig)
    routing_trials       : DualRoutingTrialsConfig   = field(default_factory=DualRoutingTrialsConfig)
    ratio_trials         : DualRatioTrialsConfig     = field(default_factory=DualRatioTrialsConfig)

    ablation_features     : list = field(default_factory=AblationCatalog.dual_default_features)
    ablation_include_full : bool = True

    gpus            : list[int] = field(default_factory=lambda: [0])
    ddp             : bool      = False
    gpus_file       : str       = ""
    poll_interval_s : float     = 5.0
