"""Entry configuration for the backbone inference entry point."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing      import List

from configuration.inference.general import InferenceConfig

_RUNS_ROOT = "runs"


@dataclass
class BackboneInferenceEntryConfig:
    """Fan-out settings for inferring a batch of backbone training runs.

    Attributes:
        runs_dir        : Any path holding runs: one run directory, a seed sweep, a trial sweep, an experiment workspace or a whole runs tree; every backbone/JEPA run at any depth below it is scored.
        run_filter      : Names selecting runs under runs_dir: a run's path relative to runs_dir, a directory whose runs are all taken, or a run's own name; empty selects every run found.
        skip_completed  : Whether runs already holding a completed inference for the configured split are left alone, so a relaunch over an experiment scores only the runs still missing it.
        gpus            : GPU device indices available to the inference workers.
        gpus_file       : Path to a file whose contents override the GPU pool at runtime.
        poll_interval_s : Seconds between scheduler polls of the worker pool.
        inference       : Per-run inference settings applied to every selected run.
    """

    runs_dir        : str       = _RUNS_ROOT
    run_filter      : List[str] = field(default_factory=list)
    gpus            : List[int] = field(default_factory=lambda: [0])
    gpus_file       : str       = ""
    poll_interval_s : float     = 5.0
    skip_completed  : bool      = True

    inference : InferenceConfig = field(default_factory=lambda: InferenceConfig.for_run_directory())
