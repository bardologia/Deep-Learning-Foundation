from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

from configuration.training.general.run          import TrainingPathsConfig, TrainingQueueConfig, standard_seeds
from configuration.dataset                       import AugmentationConfig, InputConfig
from configuration.inference.general             import InferenceConfig
from configuration.sar.gaussian_config           import GaussianConfig
from configuration.sar.geometry_config           import GeometryConfig
from configuration.normalization.general         import NormalizationConfig
from configuration.training.general.ablation     import AblationCatalog
from configuration.training.general.loss         import LossConfig, LossCurriculumConfig
from configuration.training.general.optimization import EarlyStoppingConfig, GradientClipperConfig, OptimizerConfig, SchedulerConfig, WarmupConfig
from configuration.training.general.runtime      import IOConfig, MemoryConfig, OverfitCheckConfig, ResourceConfig, TrainingLoopConfig
from configuration.training.general.pretraining  import PretrainConfig


def default_curriculum() -> LossCurriculumConfig:
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
        presence_balance         = False,
        use_cosine_curve         = True,
        weight_cosine_curve      = AblationCatalog.COSINE_WEIGHT,
        use_coherence_resyn      = True,
        weight_coherence_resyn   = AblationCatalog.PHYSICS_WEIGHT,
        use_covariance_match     = True,
        weight_covariance_match  = AblationCatalog.PHYSICS_WEIGHT,
    )

    return LossCurriculumConfig(
        enabled    = True,
        swap_epoch = AblationCatalog.CURRICULUM_SWAP_EPOCH,
        warmup     = warmup,
        complete   = complete,
    )


def _default_inference() -> InferenceConfig:
    return InferenceConfig(
        run_directory   = Path("."),
        save_cubes      = True,
        cpu_workers     = 16,
        gif_axes        = ["elevation", "range", "azimuth"],
        compute_reduced = False,
    )


def _default_warmup_losses() -> dict:
    return {
        "pL11" : {"use_param_l1": True, "weight_param_l1": 1.0},
    }


def _default_complete_losses() -> dict:
    curve_terms = {
        "mse"   : ("use_mse_curve",          "weight_mse_curve"),
        "l1"    : ("use_l1_curve",           "weight_l1_curve"),
        "huber" : ("use_huber_curve",        "weight_huber_curve"),
        "charb" : ("use_charbonnier_curve",  "weight_charbonnier_curve"),
        "cos"   : ("use_cosine_curve",       "weight_cosine_curve"),
    }
    weights = [0.01, 0.05, 0.02]

    losses = {}
    for label, (use_key, weight_key) in curve_terms.items():
        for weight in weights:
            losses[f"pL11-{label}{weight:g}"] = {
                "use_param_l1"    : True,
                "weight_param_l1" : 1.0,
                use_key           : True,
                weight_key        : weight,
            }

    return losses


def _default_presence_trials() -> dict:
    return {
        "none" : {"use_active_normalization": False, "presence_balance": False},
        "A"    : {"use_active_normalization": True,  "presence_balance": False},
        "B"    : {"use_active_normalization": False, "presence_balance": True},
        "AB"   : {"use_active_normalization": True,  "presence_balance": True},
    }


def _default_context_trials() -> list:
    return [
        {"label" : "cnn01", "backbone" : "local_cnn", "features" : [1277] * 10, "block_kernels" : [1] * 10},
        {"label" : "cnn09", "backbone" : "local_cnn", "features" : [848]  * 10, "block_kernels" : [3] * 2 + [1] * 8},
        {"label" : "cnn29", "backbone" : "local_cnn", "features" : [502]  * 10, "block_kernels" : [3] * 7 + [1] * 3},
        {"label" : "cnn41", "backbone" : "local_cnn", "features" : [426]  * 10, "block_kernels" : [3] * 10},
    ]


def _default_reach_rungs() -> list:
    return [
        {"label" : "cnn33", "backbone" : "local_cnn", "features" : [479] * 8, "dropout" : 0.15, "trunk_wd" : 1e-4},
        {"label" : "unet",  "backbone" : "unet",      "dropout" : 0.15},
    ]


def _default_augmentation_trials() -> dict:
    return {"on": True, "off": False}


def _default_input_trials() -> dict:
    return {
        "amp-allsec-noifg"  : {"tracks": "all",     "use_primary": True,  "use_secondaries": True,  "use_interferograms": False},
        "noamp-allsec-ifg"  : {"tracks": "all",     "use_primary": False, "use_secondaries": False, "use_interferograms": True},
        "amp-allsec-ifg"    : {"tracks": "all",     "use_primary": True,  "use_secondaries": True,  "use_interferograms": True},
        "amp-redsec-ifg"    : {"tracks": "reduced", "use_primary": True,  "use_secondaries": True,  "use_interferograms": True},
        "amp-redsec-noifg"  : {"tracks": "reduced", "use_primary": True,  "use_secondaries": True,  "use_interferograms": False},
        "noamp-redsec-ifg"  : {"tracks": "reduced", "use_primary": False, "use_secondaries": False, "use_interferograms": True},
    }


@dataclass
class PatchTrialsConfig:
    sizes          : list[int] = field(default_factory=lambda: [32, 48, 64, 96, 128])
    stride_ratio   : float     = 0.5
    find_max_batch : bool      = True
    scale_lr       : bool      = True


@dataclass
class PhysicsTrialsConfig:
    components       : list[str]   = field(default_factory=lambda: ["coherence_resyn", "covariance_match"])
    weights          : list[float] = field(default_factory=lambda: [0.01, AblationCatalog.PHYSICS_WEIGHT, 0.25])
    include_baseline : bool        = True


@dataclass
class PairTrialsConfig:
    base_component   : str         = "param_l1"
    base_weight      : float       = 1.0
    components       : list[str]   = field(default_factory=lambda: ["cosine_curve", "coherence_resyn", "covariance_match"])
    weights          : list[float] = field(default_factory=lambda: [0.01, AblationCatalog.PHYSICS_WEIGHT, 0.25])
    include_baseline : bool        = True


@dataclass
class ReachTrialsConfig:
    rungs           : list            = field(default_factory=_default_reach_rungs)
    patch_size      : tuple[int, int] = (32, 32)
    patch_stride    : tuple[int, int] = (16, 16)
    in_channels     : int             = 9
    match_tolerance : float           = 0.01


@dataclass
class SecondaryTrialsConfig:
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
    backbone  : str       = "unet"
    heads     : list[str] = field(default_factory=lambda: ["conv", "set_pred"])
    matchings : list[str] = field(default_factory=lambda: ["sorted_gt", "hungarian"])


@dataclass
class NormalizationTrialsConfig:
    initial_pass_mag  : str = "zscore_log1p"
    initial_ifg_phase : str = "min_max"
    initial_out_amp   : str = "zscore"
    initial_out_sigma : str = "zscore"

    final_pass_mag  : str = "robust_iqr_log1p"
    final_ifg_phase : str = "fixed_div_pi"
    final_out_amp   : str = "robust_iqr_log1p"
    final_out_sigma : str = "robust_iqr_log1p"


@dataclass
class LossScaleProbeConfig:
    enabled        : bool       = True
    n_batches      : int        = 10
    reference      : str | None = None
    exit_after     : bool       = True
    enabled_losses : dict       = field(default_factory=dict)


@dataclass
class BackboneTrainerConfig:
    gaussian         : GaussianConfig
    geometry         : GeometryConfig        = field(default_factory=GeometryConfig)
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
    run_name        : str | None = None
    resume          : bool       = True
    backbone_name   : str        = "resunet"
    backbone_head   : str        = "set_pred"
    gpu             : int        = 0
    seed            : int        = 0
    seeds           : list[int]  = field(default_factory=standard_seeds)
    logdir          : Path       = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/backbone")
    model_overrides : dict       = field(default_factory=dict)

    paths         : TrainingPathsConfig  = field(default_factory=TrainingPathsConfig)
    training      : TrainingQueueConfig  = field(default_factory=TrainingQueueConfig)
    pretrain      : PretrainConfig       = field(default_factory=PretrainConfig)
    curriculum    : LossCurriculumConfig = field(default_factory=default_curriculum)
    geometry      : GeometryConfig       = field(default_factory=GeometryConfig)
    input         : InputConfig          = field(default_factory=InputConfig.full_stack)
    normalization : NormalizationConfig  = field(default_factory=NormalizationConfig)
    augmentation  : AugmentationConfig   = field(default_factory=AugmentationConfig)

    probe_enabled        : bool = False
    probe_n_batches      : int  = 1000
    probe_reference      : str  = "param_l1"
    probe_exit_after     : bool = True
    probe_enabled_losses : dict = field(default_factory=dict)

    overfit_check : OverfitCheckConfig = field(default_factory=OverfitCheckConfig)

    infer_after  : bool            = False
    infer_at_end : bool            = False
    inference    : InferenceConfig = field(default_factory=_default_inference)

    trials_enabled       : bool                      = False
    trials_mode          : str                       = "curriculum"
    warmup_losses        : dict                      = field(default_factory=_default_warmup_losses)
    complete_losses      : dict                      = field(default_factory=_default_complete_losses)
    presence_trials      : dict                      = field(default_factory=_default_presence_trials)
    physics_trials       : PhysicsTrialsConfig       = field(default_factory=PhysicsTrialsConfig)
    pair_trials          : PairTrialsConfig          = field(default_factory=PairTrialsConfig)
    secondary_trials     : SecondaryTrialsConfig     = field(default_factory=SecondaryTrialsConfig)
    patch_trials         : PatchTrialsConfig         = field(default_factory=PatchTrialsConfig)
    input_trials         : dict                      = field(default_factory=_default_input_trials)
    context_trials       : list                      = field(default_factory=_default_context_trials)
    reach_trials         : ReachTrialsConfig         = field(default_factory=ReachTrialsConfig)
    head_trials          : HeadMatchingTrialsConfig  = field(default_factory=HeadMatchingTrialsConfig)
    augmentation_trials  : dict                      = field(default_factory=_default_augmentation_trials)
    normalization_trials : NormalizationTrialsConfig = field(default_factory=NormalizationTrialsConfig)

    ablation_features     : list = field(default_factory=AblationCatalog.default_features)
    ablation_include_full : bool = True

    gpus            : list[int] = field(default_factory=lambda: [0, 1, 3])
    gpus_file       : str       = ""
    poll_interval_s : float     = 5.0
