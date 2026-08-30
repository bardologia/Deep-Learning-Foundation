"""Entry configuration for the dual-trunk inference entry point."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing      import List

from configuration.inference.general import InferenceConfig

_RUNS_ROOT = "runs"


@dataclass
class DualInferenceEntryConfig:
    """Fan-out settings for inferring a batch of dual-trunk training runs.

    Attributes:
        runs_dir        : Root directory scanned for dual run directories.
        run_filter      : Substrings a run directory name must contain to be selected; empty selects all.
        gpus            : GPU device indices available to the inference workers.
        gpus_file       : Path to a file whose contents override the GPU pool at runtime.
        poll_interval_s : Seconds between scheduler polls of the worker pool.
        inference       : Per-run inference settings applied to every selected run.
    """

    runs_dir        : str       = f"{_RUNS_ROOT}/dual"
    run_filter      : List[str] = field(default_factory=list)
    gpus            : List[int] = field(default_factory=lambda: [0])
    gpus_file       : str       = ""
    poll_interval_s : float     = 5.0

    inference : InferenceConfig = field(default_factory=lambda: InferenceConfig.for_run_directory())
