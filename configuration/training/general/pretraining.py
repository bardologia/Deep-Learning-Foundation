from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PretrainConfig:
    find_batch_size : bool = False
    tune_loader     : bool = False

    vram_budget_gb : float = 40.0
    max_batch      : int   = 512
    measure_steps  : int   = 3

    worker_counts    : tuple[int, ...] = (0, 2, 4, 6, 8)
    prefetch_factors : tuple[int, ...] = (2, 4, 8)
    warmup_batches   : int             = 8
    timed_batches    : int             = 60
    data_wait_target : float           = 0.05

    seed : int = 42
