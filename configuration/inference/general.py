from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path
from typing      import List, Optional


@dataclass
class InferencePaths:
    figures_subdir    : str = "figures"
    animations_subdir : str = "animations"
    logs_subdir       : str = "logs"
    cubes_subdir      : str = "cubes"
    metrics_filename  : str = "metrics.json"
    report_filename   : str = "report.md"


@dataclass
class InferenceConfig:
    run_directory : Path
    output_subdir : Optional[str] = None
    device        : str           = "cuda"
    seed          : int           = 0
    log_level     : str           = "INFO"

    split           : str           = "test"
    checkpoint_name : str           = "best_model.pt"
    batch_size      : Optional[int] = None
    num_workers     : int           = 4
    cpu_workers     : int           = 80
    gif_workers     : int           = 40

    save_plots      : bool = True
    save_animations : bool = True
    save_cubes      : bool = True
    stitch_window   : str  = "hann"
    cube_dtype      : str  = "float32"
    render_amp_floor : float = 0.0

    compute_reduced      : bool           = True
    reduced_effort       : str            = "high"
    reduced_cache_subdir : str            = "data"
    reduced_env_name     : str            = "stetools"
    reduced_pyrat_dir    : Optional[Path] = None

    compute_data_consistency : bool  = True
    physics_floor            : float = 1e-3
    phase_multilook          : int   = 9

    compute_flip_consistency : bool  = False
    compute_stratified       : bool  = True

    n_best_profiles   : int = 12
    n_worst_profiles  : int = 12
    n_random_profiles : int = 12
    profile_seed      : int = 0

    n_range_slices     : int = 5
    n_azimuth_slices   : int = 5
    n_elevation_slices : int = 5

    gif_axes       : List[str] = field(default_factory=lambda: ["elevation"])
    gif_fps        : int       = 12
    gif_max_frames : int       = 150
    gif_dpi        : int       = 110

    cmap_intensity      : str  = "jet"
    cmap_error          : str  = "magma"
    normalize_intensity : bool = True
    fig_dpi             : int  = 150
    save_dpi            : int  = 300
    figure_style        : str  = "report"

    paths               : InferencePaths = field(default_factory=InferencePaths)
