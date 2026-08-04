from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path
from typing      import List

from configuration.inference.general import InferenceConfig

_RUNS_ROOT = "/ste/rnd/User/vice_vi/DLR-TomoSAR/runs"


@dataclass
class DualInferenceEntryConfig:
    runs_dir        : str       = f"{_RUNS_ROOT}/dual"
    run_filter      : List[str] = field(default_factory=list)
    gpus            : List[int] = field(default_factory=lambda: [0])
    gpus_file       : str       = ""
    poll_interval_s : float     = 5.0

    inference : InferenceConfig = field(default_factory=lambda: InferenceConfig(
        run_directory = Path("."),
        save_cubes    = True,
        cpu_workers   = 16,
        gif_axes      = ["elevation", "range", "azimuth"],
    ))
