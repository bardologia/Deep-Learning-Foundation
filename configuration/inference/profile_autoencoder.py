"""Configuration for profile-autoencoder inference runs and their entry point."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path
from typing      import List, Optional

_RUNS_ROOT = "runs"


@dataclass
class ProfileAeInferencePaths:
    """Subdirectory and file names written inside a profile-AE inference directory.

    Attributes:
        figures_subdir   : Directory holding saved figures.
        logs_subdir      : Directory holding inference log files.
        metrics_filename : JSON file collecting the computed metrics.
        report_filename  : Markdown report file summarising the run.
    """

    figures_subdir   : str = "figures"
    logs_subdir      : str = "logs"
    metrics_filename : str = "metrics.json"
    report_filename  : str = "report.md"


@dataclass
class ProfileAeInferenceConfig:
    """Settings for one profile-autoencoder reconstruction pass over a trained run.

    Attributes:
        run_directory    : Training run directory holding the checkpoint and config.
        output_subdir    : Name of the inference output directory inside the run.
        device           : Torch device string used for the forward passes.
        seed             : Seed used for the random curve selection.
        log_level        : Logging verbosity for the inference logger.
        split            : Dataset split evaluated, typically "train", "val" or "test".
        checkpoint_name  : Checkpoint file name loaded from the run directory.
        batch_size       : Number of z profiles per forward pass.
        num_workers      : Dataloader worker processes.
        pixel_subsample  : Fraction of pixels retained from the evaluated split.
        keep_empty_frac  : Fraction of empty (no-component) profiles retained.
        save_plots       : Whether figures are written to disk.
        n_best_curves    : Number of best-reconstructed profiles plotted.
        n_worst_curves   : Number of worst-reconstructed profiles plotted.
        n_random_curves  : Number of randomly drawn profiles plotted.
        n_scatter_points : Points subsampled into the prediction-versus-truth scatter.
        curve_seed       : Seed for the random profile draw.
        fig_dpi          : Figure resolution used on screen.
        save_dpi         : Figure resolution used when saving.
        figure_style     : Named matplotlib style applied to the figures.
        paths            : Output subdirectory and file-name layout.
    """

    run_directory   : Path
    output_subdir   : Optional[str] = None
    device          : str           = "cuda"
    seed            : int           = 0
    log_level       : str           = "INFO"

    split           : str           = "test"
    checkpoint_name : str           = "best_model.pt"
    batch_size      : Optional[int] = 4096
    num_workers     : int           = 4

    pixel_subsample : float = 1.0
    keep_empty_frac : float = 0.05

    save_plots       : bool = True
    n_best_curves    : int  = 12
    n_worst_curves   : int  = 12
    n_random_curves  : int  = 12
    n_scatter_points : int  = 20000
    curve_seed       : int  = 0

    fig_dpi      : int = 150
    save_dpi     : int = 300
    figure_style : str = "report"

    paths : ProfileAeInferencePaths = field(default_factory=ProfileAeInferencePaths)


@dataclass
class ProfileAeInferenceEntryConfig:
    """Fan-out settings for inferring a batch of profile-autoencoder runs.

    Attributes:
        runs_dir          : Root directory scanned for profile-autoencoder run directories.
        run_filter        : Substrings a run directory name must contain to be selected; empty selects all.
        gpus              : GPU device indices available to the inference workers.
        gpus_file         : Path to a file whose contents override the GPU pool at runtime.
        poll_interval_s   : Seconds between scheduler polls of the worker pool.
        profile_inference : Per-run inference settings applied to every selected run.
    """

    runs_dir        : str       = f"{_RUNS_ROOT}/profile_autoencoder"
    run_filter      : List[str] = field(default_factory=list)
    gpus            : List[int] = field(default_factory=lambda: [0])
    gpus_file       : str       = ""
    poll_interval_s : float     = 5.0

    profile_inference : ProfileAeInferenceConfig = field(default_factory=lambda: ProfileAeInferenceConfig(run_directory=Path(".")))
