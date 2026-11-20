
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IOConfig:

    logdir : str = "runs"


@dataclass
class TrainingLoopConfig:

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

    reserve_vram      : bool  = False
    vram_keep_free_gb : float = 1.0
