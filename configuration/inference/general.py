
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path
from typing      import List, Optional

from configuration.reference_layout import ReferenceLayout


@dataclass
class InferencePaths:

    figures_subdir    : str = "figures"
    animations_subdir : str = "animations"
    logs_subdir       : str = "logs"
    cubes_subdir      : str = "cubes"
    metrics_filename  : str = "metrics.json"
    report_filename   : str = "report.md"


@dataclass
class TargetDatasetConfig:

    enabled          : bool      = False
    dataset_path     : Path      = ReferenceLayout.DATASET_DIR
    parameters_path  : Path      = ReferenceLayout.PARAMETERS
    secondary_labels : List[str] = field(default_factory=list)
    row_start        : int       = 0
    row_end          : int       = 0
    col_start        : int       = 0
    col_end          : int       = 0


@dataclass
class InferenceConfig:

    run_directory : Path
    output_subdir : Optional[str] = None
    device        : str           = "cuda"
    seed          : int           = 0
    log_level     : str           = "INFO"

    split           : str                 = "test"
    region_override : Optional[List[int]] = None
    checkpoint_name : str                 = "best_model.pt"
    batch_size      : Optional[int]       = None
    num_workers     : int                 = 4
    cpu_workers     : int                 = 80
    gif_workers     : int                 = 40

    save_plots       : bool  = True
    save_animations  : bool  = True
    save_cubes       : bool  = True
    stitch_window    : str   = "hann"
    cube_dtype       : str   = "float32"
    render_amp_floor : float = 0.0

    compute_cube_ssim : bool = False

    compute_flip_consistency : bool  = False
    compute_stratified       : bool  = True

    extract_curve_params    : bool  = True
    extract_prominence_frac : float = 0.05

    n_best_profiles   : int = 12
    n_worst_profiles  : int = 12
    n_random_profiles : int = 12
    profile_seed      : int = 0

    n_col_slices : int = 5
    n_row_slices : int = 5
    n_z_slices   : int = 5

    gif_axes            : List[str]       = field(default_factory=lambda: ["z"])
    gif_fps             : int             = 12
    gif_duration        : Optional[float] = None
    gif_max_frames      : int             = 150
    gif_dpi             : int             = 110
    gif_separate_panels : bool            = False

    cmap_intensity      : str  = "jet"
    cmap_error          : str  = "magma"
    normalize_intensity : bool = True
    fig_dpi             : int  = 150
    save_dpi            : int  = 300
    figure_style        : str  = "report"

    paths               : InferencePaths      = field(default_factory=InferencePaths)
    target              : TargetDatasetConfig = field(default_factory=TargetDatasetConfig)

    @classmethod
    def for_run_directory(cls) -> "InferenceConfig":
        return cls(
            run_directory = Path("."),
            save_cubes    = True,
            cpu_workers   = 16,
            gif_axes      = ["z", "col", "row"],
        )
