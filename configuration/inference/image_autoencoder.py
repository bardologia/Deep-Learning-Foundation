"""Configuration for image-autoencoder inference runs and their entry point."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path
from typing      import List, Optional

_RUNS_ROOT = "runs"


@dataclass
class ImageAeInferencePaths:
    """Subdirectory and file names written inside an image-AE inference directory.

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
class ImageAeInferenceConfig:
    """Settings for one image-autoencoder reconstruction pass over a trained run.

    Attributes:
        run_directory    : Training run directory holding the checkpoint and config.
        output_subdir    : Name of the inference output directory inside the run.
        device           : Torch device string used for the forward passes.
        log_level        : Logging verbosity for the inference logger.
        split            : Dataset split evaluated, typically "train", "val" or "test".
        checkpoint_name  : Checkpoint file name loaded from the run directory.
        batch_size       : Patch batch size, or None to reuse the training batch size.
        num_workers      : Dataloader worker processes.
        save_plots       : Whether figures are written to disk.
        n_best_patches   : Number of best-reconstructed patches plotted.
        n_worst_patches  : Number of worst-reconstructed patches plotted.
        n_random_patches : Number of randomly drawn patches plotted.
        n_scatter_points : Points subsampled into the prediction-versus-truth scatter.
        patch_seed       : Seed for the random patch draw.
        fig_dpi          : Figure resolution used on screen.
        save_dpi         : Figure resolution used when saving.
        figure_style     : Named matplotlib style applied to the figures.
        paths            : Output subdirectory and file-name layout.
    """

    run_directory   : Path
    output_subdir   : Optional[str] = None
    device          : str           = "cuda"
    log_level       : str           = "INFO"

    split           : str           = "test"
    checkpoint_name : str           = "best_model.pt"
    batch_size      : Optional[int] = None
    num_workers     : int           = 4

    save_plots       : bool = True
    n_best_patches   : int  = 12
    n_worst_patches  : int  = 12
    n_random_patches : int  = 12
    n_scatter_points : int  = 20000
    patch_seed       : int  = 0

    fig_dpi      : int = 150
    save_dpi     : int = 300
    figure_style : str = "report"

    paths : ImageAeInferencePaths = field(default_factory=ImageAeInferencePaths)


@dataclass
class ImageAeInferenceEntryConfig:
    """Fan-out settings for inferring a batch of image-autoencoder runs.

    Attributes:
        runs_dir        : Root directory scanned for image-autoencoder run directories.
        run_filter      : Substrings a run directory name must contain to be selected; empty selects all.
        gpus            : GPU device indices available to the inference workers.
        gpus_file       : Path to a file whose contents override the GPU pool at runtime.
        poll_interval_s : Seconds between scheduler polls of the worker pool.
        image_inference : Per-run inference settings applied to every selected run.
    """

    runs_dir        : str       = f"{_RUNS_ROOT}/image_autoencoder"
    run_filter      : List[str] = field(default_factory=list)
    gpus            : List[int] = field(default_factory=lambda: [0])
    gpus_file       : str       = ""
    poll_interval_s : float     = 5.0

    image_inference : ImageAeInferenceConfig = field(default_factory=lambda: ImageAeInferenceConfig(run_directory=Path(".")))
