"""Dataset paths, run output locations and the queued training schedule.

Defines the dataset directory and parameter-file defaults every training entry
point starts from, plus the flat queue config the launchers expand into the
optimizer, scheduler, warmup and clipping sub-configs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib     import Path

from configuration.training.general.optimization import ClipMode, SchedulerType, WarmupMode


DEFAULT_DATASET_DIR = Path("datasets/synthetic_small")
DEFAULT_PARAMS_RUN  = "params_k2"


def standard_seeds() -> list[int]:
    """Returns the seeds every multi-seed sweep is run over."""
    return [0, 1]


@dataclass
class TrainingPathsConfig:
    """Locations of the preprocessed dataset and its Gaussian parameter labels.

    Attributes:
        dataset_path     : Directory holding the preprocessed channel stack and metadata.
        parameters_path  : Parameter array pairing Gaussian labels with that dataset.
        secondary_labels : Track labels of the secondary channels fed to the model.
    """

    dataset_path     : Path  = DEFAULT_DATASET_DIR
    parameters_path  : Path  = DEFAULT_DATASET_DIR / "params" / DEFAULT_PARAMS_RUN / "parameters.npy"
    secondary_labels : tuple = ("CH01", "CH02", "CH03", "CH04")


@dataclass
class RunPathsConfig(TrainingPathsConfig):
    """Dataset paths extended with the root directory run outputs are written under.

    Attributes:
        log_base_dir : Directory below which each run creates its own output folder.
    """

    log_base_dir : Path = Path("runs/benchmark")


@dataclass
class TrainingQueueConfig:
    """Flat training schedule a launcher expands into the trainer sub-configs.

    Groups the loop length, loader sizing, optimizer schedule, gradient clipping,
    patching and the row ranges that split the scene into train, validation
    and test regions.

    Attributes:
        epochs                      : Number of epochs the training loop runs.
        scheduler_epochs            : Epochs the schedule spans when it differs from `epochs`.
        validation_frequency        : Epoch interval between validation passes.
        batch_size                  : Patches per optimizer step.
        num_workers                 : Data-loader worker processes.
        prefetch_factor             : Batches each worker prefetches.
        eta_min                     : Floor learning rate of the cosine schedule.
        early_stop_patience         : Validation rounds without improvement before stopping.
        early_stop_min_delta        : Smallest validation drop counted as an improvement.
        log_all_losses              : Whether every loss term is logged, not just the total.
        log_debug                   : Whether per-step debug logging is emitted.
        use_amp                     : Whether mixed-precision autocast wraps the forward pass.
        gradient_accumulation_steps : Batches accumulated before an optimizer step.
        abort_on_nonfinite_loss     : Whether a NaN or inf loss aborts the run.
        use_ema                     : Whether an exponential moving average of the weights is tracked.
        ema_decay                   : Target decay rate of that moving average.
        ema_warmup                  : Whether the EMA decay ramps as min(decay, (1+t)/(10+t)) over the update count t.
        resume                      : Whether training continues from the run's last checkpoint.
        snapshot_every_n_epochs     : Epoch interval between extra weight snapshots.
        reserve_vram                : Whether spare VRAM is claimed up front to fence other jobs.
        vram_keep_free_gb           : Gigabytes deliberately left unreserved.
        scale_lr_with_batch         : Whether the learning rate is scaled by batch ratio.
        lr_reference_batch_size     : Batch size the configured learning rate refers to.
        lr_multiplier               : Factor applied to every group learning rate on top of the batch-size rule, overwritten by the pre-training LR range test.
        scheduler_type              : Learning-rate schedule family.
        scheduler_step_size         : Epoch interval between step-schedule decays.
        scheduler_gamma             : Decay factor of the step and exponential schedules.
        scheduler_power             : Exponent of the polynomial schedule.
        warmup_enabled              : Whether the learning rate ramps at the start of training.
        warmup_fraction             : Fraction of the run's total optimizer steps spent on that ramp.
        warmup_mode                 : Shape of the ramp.
        warmup_poly_power           : Exponent of the polynomial ramp.
        clip_mode                   : Gradient-clipping strategy.
        max_grad_norm               : Norm threshold of the fixed clipping strategy.
        clip_adaptive_window        : Recent gradient norms kept for the adaptive modes.
        clip_adaptive_percentile    : Percentile of that window used as threshold.
        clip_adaptive_mean_std_k    : Standard deviations above the window mean used.
        patch_size                  : Patch extent in pixels as (rows, cols).
        patch_stride                : Step in pixels between neighbouring patches, (rows, cols).
        use_symmetric_padding       : Whether the scene is padded symmetrically before patching.
        train_rows                  : Row span in pixels assigned to training.
        val_rows                    : Row span in pixels assigned to validation.
        test_rows                   : Row span in pixels assigned to testing.
    """

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
    gradient_accumulation_steps : int   = 1
    abort_on_nonfinite_loss     : bool  = True
    use_ema                     : bool  = False
    ema_decay                   : float = 0.999
    ema_warmup                  : bool  = True
    resume                      : bool  = False
    snapshot_every_n_epochs     : int   = 0

    reserve_vram      : bool  = False
    vram_keep_free_gb : float = 1.0

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
