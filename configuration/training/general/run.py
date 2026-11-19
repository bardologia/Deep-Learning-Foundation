
from __future__ import annotations

from dataclasses import dataclass
from pathlib     import Path

from configuration.reference_layout              import ReferenceLayout
from configuration.training.general.optimization import ClipMode, OptimizerType, SchedulerType, WarmupMode


DEFAULT_DATASET_DIR = ReferenceLayout.DATASET_DIR
DEFAULT_PARAMS_RUN  = ReferenceLayout.PARAMS_RUN


def standard_seeds() -> list[int]:
    return [0, 1]


@dataclass
class TrainingPathsConfig:

    dataset_path        : Path  = DEFAULT_DATASET_DIR
    parameters_path     : Path  = DEFAULT_DATASET_DIR / "params" / DEFAULT_PARAMS_RUN / "parameters.npy"
    secondary_labels    : tuple = ("CH01", "CH02", "CH03", "CH04")
    fingerprint_dataset : bool  = True


@dataclass
class RunPathsConfig(TrainingPathsConfig):

    log_base_dir : Path = Path("runs/benchmark")


@dataclass
class TrainingQueueConfig:

    epochs               : int        = 60
    scheduler_epochs     : int | None = None
    validation_frequency : int        = 1
    batch_size           : int        = 64
    num_workers          : int        = 4
    prefetch_factor      : int        = 2
    eta_min              : float      = 1e-6
    early_stop_patience  : int        = 30
    early_stop_min_delta : float      = 0.0
    log_all_losses       : bool       = False
    log_debug            : bool       = False

    use_amp                     : bool  = False
    compile_model               : bool  = False
    compile_mode                : str   = "default"
    channels_last               : bool  = False
    activation_checkpointing    : tuple = ()
    gradient_accumulation_steps : int   = 1
    abort_on_nonfinite_loss     : bool  = True
    use_ema                     : bool  = False
    ema_decay                   : float = 0.999
    ema_warmup                  : bool  = True
    use_swa                     : bool  = False
    swa_start_fraction          : float = 0.75
    resume                      : bool  = False
    snapshot_every_n_epochs     : int   = 0

    reserve_vram      : bool  = False
    vram_keep_free_gb : float = 1.0

    optimizer_type : OptimizerType = OptimizerType.ADAMW
    momentum       : float         = 0.9
    nesterov       : bool          = True

    scale_lr_with_batch     : bool  = True
    lr_reference_batch_size : int   = 64
    lr_multiplier           : float = 1.0

    scheduler_type      : SchedulerType = SchedulerType.COSINE_ANNEALING
    scheduler_step_size : int           = 30
    scheduler_gamma     : float         = 0.1
    scheduler_power     : float         = 1.0

    warmup_enabled    : bool       = True
    warmup_fraction   : float      = 0.03
    warmup_mode       : WarmupMode = WarmupMode.LINEAR
    warmup_poly_power : float      = 2.0

    clip_mode                : ClipMode = ClipMode.FIXED
    max_grad_norm            : float    = 1.0
    clip_adaptive_window     : int      = 200
    clip_adaptive_percentile : float    = 95.0
    clip_adaptive_mean_std_k : float    = 2.0

    patch_size            : tuple[int, int] = (64, 32)
    patch_stride          : tuple[int, int] = (32, 16)
    use_symmetric_padding : bool            = True

    train_rows : tuple[int, int] = (1000, 1358)
    val_rows   : tuple[int, int] = (1358, 1434)
    test_rows  : tuple[int, int] = (1434, 1512)

    def lr_scale(self, batch_size: int | None = None) -> float:
        batch = self.batch_size if batch_size is None else batch_size

        return (batch / self.lr_reference_batch_size if self.scale_lr_with_batch else 1.0) * self.lr_multiplier
