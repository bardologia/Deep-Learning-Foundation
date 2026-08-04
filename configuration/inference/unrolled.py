from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path
from typing      import List, Optional

_RUNS_ROOT = "/ste/rnd/User/vice_vi/DLR-TomoSAR/runs"


@dataclass
class UnrolledInferencePaths:
    figures_subdir   : str = "figures"
    logs_subdir      : str = "logs"
    metrics_filename : str = "metrics.json"
    report_filename  : str = "report.md"


@dataclass
class UnrolledInferenceConfig:
    run_directory   : Path
    output_subdir   : Optional[str] = None
    device          : str           = "cuda"
    seed            : int           = 0
    log_level       : str           = "INFO"

    split           : str = "test"
    checkpoint_name : str = "best_model.pt"

    measurement_noise_std : Optional[float] = None
    chunk_cells           : int             = 4_000_000

    save_plots         : bool = True
    n_example_profiles : int  = 3
    save_profile_cube  : bool = False

    fig_dpi      : int = 150
    save_dpi     : int = 300
    figure_style : str = "report"

    paths : UnrolledInferencePaths = field(default_factory=UnrolledInferencePaths)


@dataclass
class UnrolledInferenceEntryConfig:
    runs_dir        : str       = f"{_RUNS_ROOT}/unrolled"
    run_filter      : List[str] = field(default_factory=list)
    gpus            : List[int] = field(default_factory=lambda: [0])
    gpus_file       : str       = ""
    poll_interval_s : float     = 5.0

    unrolled_inference : UnrolledInferenceConfig = field(default_factory=lambda: UnrolledInferenceConfig(run_directory=Path(".")))
