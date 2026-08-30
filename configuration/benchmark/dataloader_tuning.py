"""Entry configuration for the data loader throughput tuning experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

from configuration.training.general.run import RunPathsConfig


@dataclass
class DataLoaderTuningEntryConfig:
    """Configuration of a data loader tuning sweep over batch, worker and prefetch settings.

    Attributes:
        mode               : Training pipeline whose loader is being tuned.
        model_name         : Model family used to build the probe workload.
        gpu                : Index of the GPU the timing runs execute on.
        seed               : Random seed for sampling and model initialisation.
        use_amp            : Whether the probe forward and backward pass uses mixed precision.
        pixel_subsample    : Fraction of pixels kept when building the probe dataset.
        keep_empty_frac    : Fraction of empty-profile pixels retained.
        batch_sizes        : Candidate batch sizes to time.
        worker_counts      : Candidate data loader worker counts to time.
        prefetch_factors   : Candidate per-worker prefetch factors to time.
        reference_prefetch : Prefetch factor held fixed while sweeping other axes.
        warmup_batches     : Batches consumed before timing starts.
        timed_batches      : Batches consumed while timing.
        cpu_threads        : Torch intra-op thread count during the probe.
        data_wait_target   : Target fraction of step time spent waiting on data.
        refine             : Whether a second refinement pass runs around the best setting.
        save_figures       : Whether timing figures are written to the output directory.
        synthetic_samples  : Number of profiles in the synthetic fallback dataset.
        synthetic_length   : Elevation samples per synthetic profile.
        paths              : Dataset and run path configuration.
        output_dir         : Directory receiving the tuning report and figures.
    """

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
    output_dir : Path           = Path("runs/dataloader_tuning")


@dataclass
class TrainingProfileEntryConfig:
    """Configuration of a torch.profiler capture over full training steps.

    Attributes:
        mode            : Training family whose feed is profiled.
        model_name      : Model registry key used to build the workload; empty picks the family default.
        gpu             : Index of the GPU the profiled steps execute on.
        seed            : Random seed for sampling and model initialisation.
        use_amp         : Whether the profiled forward runs under bfloat16 autocast.
        batch_size      : Samples per profiled step.
        num_workers     : Data loader worker processes during profiling.
        prefetch_factor : Batches prefetched per worker.
        warmup_steps    : Steps the profiler observes without recording.
        active_steps    : Steps recorded into the trace.
        record_shapes   : Whether operator input shapes are recorded.
        profile_memory  : Whether allocator events are recorded.
        with_stack      : Whether Python stacks are recorded; inflates the trace considerably.
        pixel_subsample : Fraction of pixels kept when building the probe dataset.
        keep_empty_frac : Fraction of empty-profile pixels retained.
        synthetic_samples : Number of profiles in the synthetic fallback dataset.
        synthetic_length  : Z samples per synthetic profile.
        paths           : Dataset and run path configuration.
        output_dir      : Directory receiving one timestamped profile per run.
    """

    mode       : str  = "backbone"
    model_name : str  = ""
    gpu        : int  = 0
    seed       : int  = 0
    use_amp    : bool = False

    batch_size      : int = 64
    num_workers     : int = 4
    prefetch_factor : int = 4

    warmup_steps   : int  = 8
    active_steps   : int  = 24
    record_shapes  : bool = True
    profile_memory : bool = True
    with_stack     : bool = False

    pixel_subsample : float = 0.2
    keep_empty_frac : float = 0.05

    synthetic_samples : int = 200_000
    synthetic_length  : int = 96

    paths      : RunPathsConfig = field(default_factory=RunPathsConfig)
    output_dir : Path           = Path("runs/profiles")
