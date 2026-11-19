
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PretrainConfig:

    find_batch_size : bool = False
    tune_loader     : bool = False
    find_lr         : bool = False

    vram_budget_gb : float = 40.0
    max_batch      : int   = 512
    measure_steps  : int   = 3

    worker_counts    : tuple[int, ...] = (0, 2, 4, 6, 8)
    prefetch_factors : tuple[int, ...] = (2, 4, 8)
    warmup_batches   : int             = 8
    timed_batches    : int             = 60
    data_wait_target : float           = 0.05

    lr_scale_min         : float = 1e-2
    lr_scale_max         : float = 1e4
    lr_rungs             : int   = 60
    lr_rung_steps        : int   = 8
    lr_probe_batches     : int   = 4
    lr_divergence_factor : float = 4.0
    lr_safety_factor     : float = 10.0
    lr_min_improvement   : float = 0.02
    lr_noise_sigmas      : float = 3.0

    seed : int = 42
