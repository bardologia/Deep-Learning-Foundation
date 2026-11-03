"""Configuration for backbone and dual volumetric inference runs.

Holds the output layout, the analysis toggles (SSIM against the raw cube, flip
consistency, stratified metrics, curve-parameter fitting), the plotting and
animation settings applied to every inference run, and the target dataset a run
can be scored on instead of the dataset it was trained on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path
from typing      import List, Optional

from configuration.reference_layout import ReferenceLayout


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
class TargetDatasetConfig:
    """Dataset a trained run is scored on in place of the split it was trained with.

    The run keeps its checkpoint, patch geometry, input channel layout and training
    normalization statistics; only the scene, its labels, the secondary channels fed
    to the model and the scored crop change. The target scene must share the run's z
    axis, Gaussian count K and number of secondary channels. An axis whose start and
    end are both 0 spans the target's whole global crop, so the default scores the full scene.

    Attributes:
        enabled          : Whether the run is scored on this dataset instead of its own split.
        dataset_path     : Dataset directory of the target scene.
        parameters_path  : Gaussian label cube (parameters.npy) of a parameter run inside the target dataset, the reference every metric is scored against.
        secondary_labels : Secondary channel labels of the target scene fed to the model, as many as the run was trained with; empty feeds every secondary channel of the target stack.
        row_start        : First row of the scored crop of the target scene, inclusive; 0 together with row_end 0 takes the whole row extent.
        row_end          : Last row of the scored crop of the target scene, exclusive.
        col_start        : First col of the scored crop of the target scene, inclusive; 0 together with col_end 0 takes the whole col extent.
        col_end          : Last col of the scored crop of the target scene, exclusive.
    """

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
        compute_cube_ssim        : Whether the standard SSIM sweep of the prediction against the raw cube runs, in raw intensity and unit area.
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
        target                   : Foreign dataset the run is scored on instead of its own split, off by default.
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
