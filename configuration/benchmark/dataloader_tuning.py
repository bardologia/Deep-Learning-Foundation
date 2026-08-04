from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

from configuration.training.general.run import RunPathsConfig


@dataclass
class DataLoaderTuningEntryConfig:
    mode        : str  = "profile_autoencoder"
    model_name  : str  = ""
    gpu         : int  = 0
    seed        : int  = 0
    use_amp     : bool = False

    pixel_subsample : float = 0.2
    keep_empty_frac : float = 0.05

    batch_sizes      : list = field(default_factory=lambda: [256, 512, 1024, 2048, 4096])
    worker_counts    : list = field(default_factory=lambda: [0, 2, 4, 6, 8])
    prefetch_factors : list = field(default_factory=lambda: [2, 4, 8, 16])

    reference_prefetch : int = 4
    warmup_batches     : int = 8
    timed_batches      : int = 60
    cpu_threads        : int = 8

    data_wait_target: float = 0.05

    refine       : bool = True
    save_figures : bool = True

    synthetic_samples : int = 200_000
    synthetic_length  : int = 96

    paths      : RunPathsConfig = field(default_factory=RunPathsConfig)
    output_dir : Path           = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/dataloader_tuning")
