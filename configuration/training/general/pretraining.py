"""Settings for the pre-training probes that size the batch, the data loader and the learning rate."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PretrainConfig:
    """Controls the batch-size, loader-tuning and learning-rate passes run before training.

    Attributes:
        find_batch_size      : Whether the batch-size probe runs before training.
        tune_loader          : Whether the worker/prefetch grid search runs before training.
        find_lr              : Whether the learning-rate range test runs before training.
        vram_budget_gb       : VRAM ceiling in gigabytes the batch-size probe must respect.
        max_batch            : Largest batch size the probe is allowed to try.
        measure_steps        : Training steps timed per candidate batch size.
        worker_counts        : Candidate `num_workers` values for the loader search.
        prefetch_factors     : Candidate `prefetch_factor` values for the loader search.
        warmup_batches       : Batches consumed untimed before each loader measurement.
        timed_batches        : Batches timed per loader configuration.
        data_wait_target     : Target fraction of step time spent waiting on the loader.
        lr_scale_min         : Smallest learning-rate multiplier the range test starts from.
        lr_scale_max         : Largest learning-rate multiplier the range test ramps to.
        lr_rungs             : Multiplier rungs of the ramp, each measured from the same rewound trainer state.
        lr_rung_steps        : Optimizer steps taken at each rung, mirroring the real step including accumulation and clipping.
        lr_probe_batches     : Held-out batches measured at every rung, the same batches throughout the ramp.
        lr_divergence_factor : Multiple of the best rung loss that ends the ramp.
        lr_safety_factor     : Divisor applied to the minimum-loss multiplier.
        lr_min_improvement   : Smallest relative drop below the first rung that counts as a response.
        lr_noise_sigmas      : Standard errors of the rung measurement the drop must also clear.
        seed                 : Seed fixing the probe batches so measurements are comparable.
    """

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
