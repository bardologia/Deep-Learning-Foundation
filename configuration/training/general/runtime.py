"""Runtime sub-configs: output location, overfit gate, loop, VRAM and monitoring."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IOConfig:
    """Output root the trainer writes checkpoints, logs and reports under.

    Attributes:
        logdir : Directory holding one folder per run.
    """

    logdir : str = "runs"


@dataclass
class OverfitCheckConfig:
    """Pre-training sanity gate that drives the model onto a handful of examples.

    Attributes:
        enabled         : Whether the gate runs before the real training loop.
        n_examples      : Number of examples the model must memorise.
        max_steps       : Optimizer-step budget for the gate.
        steps_per_epoch : Steps grouped into one reported epoch of the gate.
        pass_loss_ratio : Fraction of the initial loss that counts as a pass.
        stop_threshold  : Absolute loss below which the gate stops early.
    """

    enabled         : bool  = False
    n_examples      : int   = 2
    max_steps       : int   = 300
    steps_per_epoch : int   = 25
    pass_loss_ratio : float = 0.05
    stop_threshold  : float = 1e-6


@dataclass
class TrainingLoopConfig:
    """Epoch loop behaviour, precision, EMA and checkpoint-resume settings.

    Attributes:
        epochs                      : Number of epochs the loop runs.
        validation_frequency        : Epoch interval between validation passes.
        use_amp                     : Whether mixed-precision autocast wraps the forward pass.
        gradient_accumulation_steps : Batches accumulated before an optimizer step.
        log_debug                   : Whether per-step debug logging is emitted.
        log_all_losses              : Whether every loss term is logged, not just the total.
        abort_on_nonfinite_loss     : Whether a NaN or inf loss aborts the run.
        use_ema                     : Whether an exponential moving average of the weights is tracked.
        ema_decay                   : Target decay rate of that moving average.
        ema_warmup                  : Whether the EMA decay ramps as min(decay, (1+t)/(10+t)) over the update count t.
        resume                      : Whether training continues from the run's last checkpoint.
        snapshot_every_n_epochs     : Epoch interval between extra weight snapshots.
    """

    epochs                      : int   = 3
    validation_frequency        : int   = 5
    use_amp                     : bool  = False
    gradient_accumulation_steps : int   = 1
    log_debug                   : bool  = False
    log_all_losses              : bool  = False
    abort_on_nonfinite_loss     : bool  = True
    use_ema                     : bool  = False
    ema_decay                   : float = 0.999
    ema_warmup                  : bool  = True
    resume                      : bool  = False
    snapshot_every_n_epochs     : int   = 0


@dataclass
class MemoryConfig:
    """Up-front VRAM reservation used to fence a run against neighbouring jobs.

    Attributes:
        reserve_vram      : Whether spare VRAM is claimed before training starts.
        vram_keep_free_gb : Gigabytes deliberately left unreserved on the device.
    """

    reserve_vram      : bool  = False
    vram_keep_free_gb : float = 1.0


@dataclass
class ResourceConfig:
    """Resource monitoring cadence and the utilisation levels that raise a warning.

    Attributes:
        enabled            : Whether the resource monitor runs alongside training.
        poll_interval_sec  : Seconds between resource samples.
        log_to_tensorboard : Whether samples are also written as TensorBoard scalars.
        warn_ram_pct       : Host RAM utilisation percentage that triggers a warning.
        warn_vram_pct      : Device VRAM utilisation percentage that triggers a warning.
        warn_shm_pct       : Shared-memory utilisation percentage that triggers a warning.
        warn_cooldown_sec  : Seconds silenced after a warning before it may repeat.
    """

    enabled            : bool  = True
    poll_interval_sec  : float = 5.0
    log_to_tensorboard : bool  = True
    warn_ram_pct       : float = 90.0
    warn_vram_pct      : float = 90.0
    warn_shm_pct       : float = 80.0
    warn_cooldown_sec  : float = 30.0
