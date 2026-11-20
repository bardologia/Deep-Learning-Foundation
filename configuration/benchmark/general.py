
from __future__ import annotations

from dataclasses import dataclass, field

from configuration.dataset                      import AugmentationConfig, InputConfig
from configuration.normalization.general        import NormalizationConfig
from configuration.training.backbone            import default_loss
from configuration.training.general.loss        import LossConfig
from configuration.training.general.run         import RunPathsConfig, TrainingQueueConfig, standard_seeds


@dataclass
class InferenceQueueConfig:

    split           : str        = "test"
    checkpoint_name : str        = "best_model.pt"
    batch_size      : int | None = None
    num_workers     : int        = 4
    cpu_workers     : int        = 16
    gif_workers     : int        = 40
    seed            : int        = 0

    save_plots       : bool  = False
    save_animations  : bool  = False
    save_cubes       : bool  = False
    stitch_window    : str   = "hann"
    cube_dtype       : str   = "float32"
    render_amp_floor : float = 0.0

    compute_cube_ssim : bool = False

    compute_flip_consistency : bool = False
    compute_stratified       : bool = True

    n_best_profiles   : int = 12
    n_worst_profiles  : int = 12
    n_random_profiles : int = 12
    profile_seed      : int = 0

    n_col_slices : int = 5
    n_row_slices : int = 5
    n_z_slices   : int = 5

    gif_axes            : list[str]    = field(default_factory=lambda: ["z"])
    gif_fps             : int          = 12
    gif_duration        : float | None = None
    gif_max_frames      : int          = 150
    gif_dpi             : int          = 110
    gif_separate_panels : bool         = False

    cmap_intensity      : str  = "jet"
    cmap_error          : str  = "magma"
    normalize_intensity : bool = True
    fig_dpi             : int  = 150
    save_dpi            : int  = 300
    figure_style        : str  = "report"


@dataclass
class ComparisonReportConfig:

    embed_images: bool = False


@dataclass
class BenchmarkConfig:

    stage : str = "full"

    paths      : RunPathsConfig         = field(default_factory=RunPathsConfig)
    training   : TrainingQueueConfig    = field(default_factory=TrainingQueueConfig)
    inference  : InferenceQueueConfig   = field(default_factory=InferenceQueueConfig)
    comparison : ComparisonReportConfig = field(default_factory=ComparisonReportConfig)

    input         : InputConfig          = field(default_factory=InputConfig.full_stack)
    normalization : NormalizationConfig  = field(default_factory=NormalizationConfig)
    augmentation  : AugmentationConfig   = field(default_factory=AugmentationConfig)
    loss          : LossConfig           = field(default_factory=default_loss)

    gpus            : list[int]  = field(default_factory=lambda: [0])
    gpus_file       : str        = ""
    heads           : list[str]  = field(default_factory=lambda: ["conv"])
    skip_models     : list[str]  = field(default_factory=list)
    run_tag         : str | None = None
    resume          : bool       = True
    resume_from     : str        = ""
    infer_after     : bool       = True
    seed            : int        = 0
    seeds           : list[int]  = field(default_factory=standard_seeds)
    poll_interval_s : float      = 5.0

    sweep_loss_components: list[str] = field(default_factory=lambda: ["param_l1"])

    def runs_inference(self) -> bool:
        return self.infer_after
