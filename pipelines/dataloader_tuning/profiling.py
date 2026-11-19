
from __future__ import annotations

from pathlib import Path

import torch

from pipelines.dataloader_tuning.adapters import FeedAdapter
from tools.benchmarking.step_profiler     import StepProfiler
from tools.runtime.run_tag                import RunTag


class TrainingProfilePipeline:

    def __init__(self, entry, logger) -> None:
        self.entry  = entry
        self.logger = logger

    def _device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self) -> dict:
        output_dir = Path(self.entry.output_dir) / RunTag.now()
        feed       = FeedAdapter(self.entry, output_dir / "feed", self.logger).build()

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
