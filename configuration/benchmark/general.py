from __future__ import annotations

from dataclasses import dataclass, field

from configuration.benchmark.jepa               import JepaBenchConfig
from configuration.dataset                      import AugmentationConfig, InputConfig
from configuration.normalization.general        import NormalizationConfig
from configuration.sar.geometry_config          import GeometryConfig
from configuration.training.backbone            import default_curriculum
from configuration.training.general.loss        import LossConfig
from configuration.training.general.pretraining import PretrainConfig
from configuration.training.general.run         import RunPathsConfig, TrainingQueueConfig, standard_seeds
from configuration.training.general.runtime     import OverfitCheckConfig
from configuration.training.profile_autoencoder import ProfileAeLossConfig


@dataclass
class MaxBatchConfig:
    vram_budget_gb : float = PretrainConfig.vram_budget_gb
    max_batch      : int   = PretrainConfig.max_batch
    measure_steps  : int   = PretrainConfig.measure_steps
    seed           : int   = PretrainConfig.seed


@dataclass
class SizeMatchConfig:
    reference_model : str             = "unet"
    tolerance       : float           = 0.05
    max_iterations  : int             = 100
    scale_low       : float           = 0.05
    scale_high      : float           = 8.0
    in_channels     : int             = 9
    locked_params   : tuple[str, ...] = ("embedding_dim", "embedding_dims")


@dataclass
class InferenceQueueConfig:
    split           : str        = "test"
    checkpoint_name : str        = "best_model.pt"
    batch_size      : int | None = None
    num_workers     : int        = 4
    cpu_workers     : int        = 16

    save_plots      : bool = False
    save_animations : bool = False
    save_cubes      : bool = False
    stitch_window   : str  = "hann"

    compute_reduced: bool = True

    n_best_profiles   : int = 12
    n_worst_profiles  : int = 12
    n_random_profiles : int = 12

    n_range_slices     : int = 5
    n_azimuth_slices   : int = 5
    n_elevation_slices : int = 5

    gif_axes       : list[str] = field(default_factory=lambda: ["elevation"])
    gif_fps        : int       = 12
    gif_max_frames : int       = 150


@dataclass
class ComparisonReportConfig:
    embed_images: bool = False


def _default_ae_loss():
    return ProfileAeLossConfig()


def _default_base_loss():
    return default_curriculum().complete


@dataclass
class BenchmarkConfig:
    training_type: str = "backbone"

    paths      : RunPathsConfig         = field(default_factory=RunPathsConfig)
    max_batch  : MaxBatchConfig         = field(default_factory=MaxBatchConfig)
    size_match : SizeMatchConfig        = field(default_factory=SizeMatchConfig)
    training   : TrainingQueueConfig    = field(default_factory=TrainingQueueConfig)
    inference  : InferenceQueueConfig   = field(default_factory=InferenceQueueConfig)
    comparison : ComparisonReportConfig = field(default_factory=ComparisonReportConfig)

    input         : InputConfig         = field(default_factory=InputConfig.full_stack)
    geometry      : GeometryConfig      = field(default_factory=GeometryConfig)
    normalization : NormalizationConfig = field(default_factory=NormalizationConfig)
    augmentation  : AugmentationConfig  = field(default_factory=AugmentationConfig)
    loss          : LossConfig          = field(default_factory=_default_base_loss)
    overfit_check : OverfitCheckConfig  = field(default_factory=OverfitCheckConfig)

    ae_loss         : object          = field(default_factory=_default_ae_loss)
    jepa            : JepaBenchConfig = field(default_factory=JepaBenchConfig)
    pixel_subsample : float           = 1.0
    keep_empty_frac : float           = 0.05

    gpus            : list[int]  = field(default_factory=lambda: [2, 3])
    gpus_file       : str        = ""
    heads           : list[str]  = field(default_factory=lambda: ["conv"])
    skip_models     : list[str]  = field(default_factory=list)
    run_tag         : str | None = None
    resume          : bool       = True
    infer_after     : bool       = True
    seed            : int        = 0
    seeds           : list[int]  = field(default_factory=standard_seeds)
    poll_interval_s : float      = 5.0

    sweep_loss_components: list[str] = field(default_factory=lambda: ["param_l1"])

    def runs_size_match(self) -> bool:
        return self.training_type == "backbone"

    def runs_max_batch(self) -> bool:
        return self.training_type == "backbone"

    def runs_inference(self) -> bool:
        return self.infer_after and self.training_type in ("backbone", "jepa")
