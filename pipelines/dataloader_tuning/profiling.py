"""Pipeline assembling a feed target and profiling its training steps.

Reuses the dataloader-tuning feed adapters to build a real (dataset, model,
loss) triple for the configured family, then hands it to the step profiler,
which records full forward-backward-optimizer steps and writes the chrome
trace, operator table and summary under a timestamped output directory.
"""

from __future__ import annotations

from pathlib import Path

import torch

from pipelines.dataloader_tuning.adapters import FeedTargetFactory
from tools.benchmarking.step_profiler     import StepProfiler
from tools.runtime.run_tag                import RunTag


class TrainingProfilePipeline:
    """Builds the configured feed target and profiles its training steps.

    Attributes:
        entry  : Profiling entry configuration.
        logger : Logger receiving progress and artifact lines.
    """

    def __init__(self, entry, logger) -> None:
        """Stores the entry configuration and logger."""
        self.entry  = entry
        self.logger = logger

    def _device(self) -> torch.device:
        """Returns the CUDA device when visible, the CPU otherwise."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self) -> dict:
        """Assembles the feed and profiles the configured number of steps.

        Returns:
            The profiler summary payload.
        """
        output_dir = Path(self.entry.output_dir) / RunTag.now()
        feed       = FeedTargetFactory.build(self.entry, output_dir / "feed", self.logger)

        profiler = StepProfiler(
            feed            = feed,
            device          = self._device(),
            batch_size      = self.entry.batch_size,
            num_workers     = self.entry.num_workers,
            prefetch_factor = self.entry.prefetch_factor,
            use_amp         = self.entry.use_amp,
            warmup_steps    = self.entry.warmup_steps,
            active_steps    = self.entry.active_steps,
            record_shapes   = self.entry.record_shapes,
            profile_memory  = self.entry.profile_memory,
            with_stack      = self.entry.with_stack,
            logger          = self.logger,
        )

        return profiler.run(output_dir)
