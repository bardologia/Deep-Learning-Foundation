from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IOConfig:
    logdir : str = "/ste/rnd/User/vice_vi/DLR-TomoSAR/runs"
    writer = None


@dataclass
class OverfitCheckConfig:
    enabled         : bool  = False
    n_examples      : int   = 2
    max_steps       : int   = 300
    steps_per_epoch : int   = 25
    pass_loss_ratio : float = 0.05
    stop_threshold  : float = 1e-6


@dataclass
class TrainingLoopConfig:
    epochs                      : int   = 3
    validation_frequency        : int   = 5
    use_amp                     : bool  = False
    gradient_accumulation_steps : int   = 1
    log_debug                   : bool  = False
    log_all_losses              : bool  = False
    abort_on_nonfinite_loss     : bool  = True
    use_ema                     : bool  = False
    ema_decay                   : float = 0.999
    resume                      : bool  = False
    snapshot_every_n_epochs     : int   = 0


@dataclass
class MemoryConfig:
    reserve_vram      : bool  = False
    vram_keep_free_gb : float = 1.0


@dataclass
class ResourceConfig:
    enabled            : bool  = True
    poll_interval_sec  : float = 5.0
    log_to_tensorboard : bool  = True
    warn_ram_pct       : float = 90.0
    warn_vram_pct      : float = 90.0
    warn_swap_pct      : float = 50.0
    warn_shm_pct       : float = 80.0
    warn_cooldown_sec  : float = 30.0
