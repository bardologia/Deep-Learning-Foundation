from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path
from typing      import List, Optional

from configuration.reference_layout import ReferenceLayout


@dataclass
class InferencePaths:

    figures_subdir   : str = "figures"
    logs_subdir      : str = "logs"
    maps_subdir      : str = "maps"
    metrics_filename : str = "metrics.json"
    report_filename  : str = "report.md"


@dataclass
class TargetDatasetConfig:

    enabled      : bool = False
    dataset_path : Path = ReferenceLayout.DATASET_DIR


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

    stitch_window : str  = "hann"
    save_maps     : bool = True
    save_plots    : bool = True

    cmap         : str = "viridis"
    cmap_error   : str = "magma"
    fig_dpi      : int = 150
    save_dpi     : int = 300
    figure_style : str = "report"

    paths  : InferencePaths      = field(default_factory=InferencePaths)
    target : TargetDatasetConfig = field(default_factory=TargetDatasetConfig)

    @classmethod
    def for_run_directory(cls) -> "InferenceConfig":
        return cls(run_directory=Path("."))
