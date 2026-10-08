"""Configuration for backbone and dual volumetric inference runs.

Holds the output layout, the analysis toggles (flip consistency, stratified
metrics, curve-parameter fitting) and the plotting and animation settings
applied to every inference run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path
from typing      import List, Optional


@dataclass
class InferencePaths:
    """Subdirectory and file names written inside an inference output directory.

    Attributes:
        figures_subdir    : Directory holding saved figures.
        animations_subdir : Directory holding rendered GIF animations.
        logs_subdir       : Directory holding inference log files.
        cubes_subdir      : Directory holding saved prediction cubes.
        metrics_filename  : JSON file collecting the computed metrics.
        report_filename   : Markdown report file summarising the run.
    """

    figures_subdir    : str = "figures"
    animations_subdir : str = "animations"
    logs_subdir       : str = "logs"
    cubes_subdir      : str = "cubes"
    metrics_filename  : str = "metrics.json"
    report_filename   : str = "report.md"


@dataclass
class InferenceConfig:
    """Settings for one volumetric inference pass over a trained run.

    Attributes:
        run_directory            : Training run directory holding the checkpoint and config.
        output_subdir            : Name of the inference output directory inside the run.
        device                   : Torch device string used for the forward passes.
        seed                     : Seed used for the random profile and slice selections.
        log_level                : Logging verbosity for the inference logger.
        split                    : Dataset split evaluated, typically "train", "val" or "test".
        region_override          : Crop region as [row_start, row_end, col_start, col_end] overriding the split region.
        checkpoint_name          : Checkpoint file name loaded from the run directory.
        batch_size               : Patch batch size, or None to reuse the training batch size.
        num_workers              : Dataloader worker processes.
        cpu_workers              : Process-pool size for CPU-side analyses.
        gif_workers              : Process-pool size for animation frame rendering.
        save_plots               : Whether figures are written to disk.
        save_animations          : Whether GIF animations are rendered.
        save_cubes               : Whether stitched prediction cubes are written.
        stitch_window            : Window applied when blending overlapping patches.
        cube_dtype               : NumPy dtype name used when writing cubes.
        render_amp_floor         : Amplitude below which components are hidden in renders.
        compute_flip_consistency : Whether flip-equivariance metrics run.
        compute_stratified       : Whether metrics are stratified by scene strata.
        extract_curve_params     : Whether Gaussians are refitted to predicted curves.
        extract_prominence_frac  : Peak prominence fraction used in curve fitting.
        n_best_profiles          : Number of best-scoring profiles plotted.
        n_worst_profiles         : Number of worst-scoring profiles plotted.
        n_random_profiles        : Number of randomly drawn profiles plotted.
        profile_seed             : Seed for the random profile draw.
        n_col_slices             : Number of col slices plotted.
        n_row_slices             : Number of row slices plotted.
        n_z_slices               : Number of z slices plotted.
        gif_axes                 : Cube axes animated, from "z", "col" and "row".
        gif_fps                  : Animation frame rate, used when gif_duration is None.
        gif_duration             : Total playback length of each animation in seconds, or None to let gif_fps govern.
        gif_max_frames           : Upper bound on frames per animation.
        gif_dpi                  : Rendering resolution of animation frames.
        gif_separate_panels      : Whether each cube (prediction, GT, raw cube) is also written as its own single-panel GIF sized to the slice aspect ratio.
        cmap_intensity           : Colormap used for intensity imagery.
        cmap_error               : Colormap used for error imagery.
        normalize_intensity      : Whether intensity maps are normalized before display.
        fig_dpi                  : Figure resolution used on screen.
        save_dpi                 : Figure resolution used when saving.
        figure_style             : Named matplotlib style applied to the figures.
        paths                    : Output subdirectory and file-name layout.
    """

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

    paths               : InferencePaths = field(default_factory=InferencePaths)

    @classmethod
    def for_run_directory(cls) -> "InferenceConfig":
        """Builds the default nested inference config used by entry configs.

        The run directory is a placeholder that the launcher replaces with the
        actual run being inferred.

        Returns:
            An InferenceConfig saving cubes and animating all three cube axes.
        """
        return cls(
            run_directory = Path("."),
            save_cubes    = True,
            cpu_workers   = 16,
            gif_axes      = ["z", "col", "row"],
        )
