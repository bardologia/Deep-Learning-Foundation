"""Runtime sub-configs: output location, overfit gate, loop and VRAM reservation."""

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
        compile_model               : Whether the model forward is compiled with torch.compile.
        compile_mode                : torch.compile mode: "default", "reduce-overhead" or "max-autotune".
        channels_last               : Whether the model and 4-D batches use the channels_last memory format.
        activation_checkpointing    : fnmatch patterns of submodule names wrapped in gradient checkpointing; empty disables it.
        gradient_accumulation_steps : Batches accumulated before an optimizer step.
        log_debug                   : Whether per-step debug logging is emitted.
        log_all_losses              : Whether every loss term is logged, not just the total.
        abort_on_nonfinite_loss     : Whether a NaN or inf loss aborts the run.
        use_ema                     : Whether an exponential moving average of the weights is tracked.
        ema_decay                   : Target decay rate of that moving average.
        ema_warmup                  : Whether the EMA decay ramps as min(decay, (1+t)/(10+t)) over the update count t.
        use_swa                     : Whether an equal-weight stochastic average of the weights is tracked from the SWA start epoch.
        swa_start_fraction          : Fraction of the epoch budget after which each finished epoch joins the stochastic weight average.
        resume                      : Whether training continues from the run's last checkpoint.
        snapshot_every_n_epochs     : Epoch interval between extra weight snapshots.
    """

    epochs                      : int   = 3
    validation_frequency        : int   = 5
    use_amp                     : bool  = False
    compile_model               : bool  = False
    compile_mode                : str   = "default"
    channels_last               : bool  = False
    activation_checkpointing    : tuple = ()
    gradient_accumulation_steps : int   = 1
    log_debug                   : bool  = False
    log_all_losses              : bool  = False
    abort_on_nonfinite_loss     : bool  = True
    use_ema                     : bool  = False
    ema_decay                   : float = 0.999
    ema_warmup                  : bool  = True
    use_swa                     : bool  = False
    swa_start_fraction          : float = 0.75
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
