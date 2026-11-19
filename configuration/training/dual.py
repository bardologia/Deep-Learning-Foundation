
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

from configuration.training.general.run         import TrainingPathsConfig, TrainingQueueConfig, standard_seeds
from configuration.dataset                      import AugmentationConfig, InputConfig
from configuration.inference.general            import InferenceConfig
from configuration.normalization.general        import NormalizationConfig
from configuration.training.backbone            import AugmentationTrialsConfig, ClampTrialsConfig, EmaTrialsConfig, HeadMatchingTrialsConfig, LogEnergyTrialsConfig, LossScaleProbeConfig, PairTrialsConfig, PatchTrialsConfig, ReachTrialsConfig, SecondaryTrialsConfig, StabilityTrialsConfig, default_curriculum
from configuration.training.general.trials      import TrialCatalog
from configuration.training.general.ablation    import AblationCatalog
from configuration.training.general.loss        import LossCurriculumConfig
from configuration.training.general.runtime     import OverfitCheckConfig
from configuration.training.general.pretraining import PretrainConfig


dual_curriculum = default_curriculum


def _parity_resunet_features() -> list[int]:
    return [48, 96, 184, 352]


def _default_dual_routing_trials() -> dict:
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
    return {
        "50-50" : {"params": _parity_resunet_features(), "existence": _parity_resunet_features()},
        "60-40" : {"params": [60, 108, 204, 384],        "existence": [56, 92, 168, 312]},
        "70-30" : {"params": [48, 116, 220, 416],        "existence": [36, 76, 144, 272]},
        "80-20" : {"params": [60, 124, 236, 444],        "existence": [28, 52, 116, 224]},
        "90-10" : {"params": [64, 128, 248, 472],        "existence": [24, 48, 84, 156]},
    }


def _default_dual_head_trials() -> HeadMatchingTrialsConfig:
    return HeadMatchingTrialsConfig(heads=["set_pred"], matchings=["sorted_gt", "hungarian"])


def _default_dual_reach_rungs() -> list:
    return [
        {"label" : "cnn33", "backbone" : "local_cnn", "features" : [478] * 8, "dropout" : 0.15, "trunk_wd" : 1e-4},
        {"label" : "unet",  "backbone" : "unet",      "dropout" : 0.15},
    ]


def _default_dual_reach_trials() -> ReachTrialsConfig:
    return ReachTrialsConfig(rungs=_default_dual_reach_rungs())


@dataclass
class DualRoutingTrialsConfig:

    params_features    : list[int] = field(default_factory=_parity_resunet_features)
    existence_features : list[int] = field(default_factory=_parity_resunet_features)
    trials             : dict      = field(default_factory=_default_dual_routing_trials)


@dataclass
class DualRatioTrialsConfig:

    trials          : dict  = field(default_factory=_default_dual_ratio_trials)
    match_tolerance : float = 0.01


@dataclass
class DualEntryConfig:

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
    routing_trials       : DualRoutingTrialsConfig   = field(default_factory=DualRoutingTrialsConfig)
    ratio_trials         : DualRatioTrialsConfig     = field(default_factory=DualRatioTrialsConfig)

    ablation_features     : list = field(default_factory=AblationCatalog.dual_default_features)
    ablation_include_full : bool = True

    gpus            : list[int] = field(default_factory=lambda: [0])
    ddp             : bool      = False
    gpus_file       : str       = ""
    poll_interval_s : float     = 5.0
