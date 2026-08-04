from __future__ import annotations

import os
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from tools.benchmarking      import LoaderSpec, GpuFeedBenchmark, DataLoaderSweep, SweepReport
from tools.monitoring.logger import Logger


class LoaderTuner:
    def __init__(
        self,
        dataset            : Dataset,
        model              : nn.Module,
        to_model_input     : Callable,
        forward_loss       : Callable,
        device             : torch.device,
        logger             : Logger,
        use_amp            : bool          = False,
        seed               : int           = 0,
        warmup_batches     : int           = 8,
        timed_batches      : int           = 60,
        gpu_index          : Optional[int] = None,
        cpu_threads        : Optional[int] = None,
        worker_counts      : tuple         = (0, 2, 4, 6, 8),
        prefetch_factors   : tuple         = (2, 4, 8),
        reference_prefetch : int           = 4,
        data_wait_target   : float         = 0.05,
    ) -> None:
        self.dataset            = dataset
        self.model              = model
        self.to_model_input     = to_model_input
        self.forward_loss       = forward_loss
        self.device             = device
        self.logger             = logger
        self.use_amp            = bool(use_amp)
        self.seed               = int(seed)
        self.warmup_batches     = int(warmup_batches)
        self.timed_batches      = int(timed_batches)
        self.gpu_index          = gpu_index
        self.cpu_threads        = cpu_threads
        self.worker_counts      = tuple(worker_counts)
        self.prefetch_factors   = tuple(prefetch_factors)
        self.reference_prefetch = int(reference_prefetch)
        self.data_wait_target   = float(data_wait_target)

    def _benchmark(self) -> GpuFeedBenchmark:
        return GpuFeedBenchmark(
            dataset        = self.dataset,
            model          = self.model,
            to_model_input = self.to_model_input,
            forward_loss   = self.forward_loss,
            device         = self.device,
            use_amp        = self.use_amp,
            seed           = self.seed,
            warmup_batches = self.warmup_batches,
            timed_batches  = self.timed_batches,
            gpu_index      = self.gpu_index,
            cpu_threads    = self.cpu_threads,
        )

    def _worker_counts(self) -> list[int]:
        cores   = os.cpu_count() or 8
        kept    = [workers for workers in self.worker_counts if workers <= cores]
        dropped = [workers for workers in self.worker_counts if workers > cores]

        if dropped:
            self.logger.subsection(f"Skipping worker counts above {cores} cores: {dropped}")

        return kept or [min(self.worker_counts) if self.worker_counts else 0]

    def _worker_specs(self, batch_size: int) -> list[LoaderSpec]:
        return [
            LoaderSpec(batch_size=batch_size, num_workers=workers, prefetch_factor=self.reference_prefetch, pin_memory=True, persistent_workers=True)
            for workers in self._worker_counts()
        ]

    def _prefetch_specs(self, batch_size: int, num_workers: int) -> list[LoaderSpec]:
        workers = max(1, num_workers)

        return [
            LoaderSpec(batch_size=batch_size, num_workers=workers, prefetch_factor=prefetch, pin_memory=pin_memory, persistent_workers=True)
            for prefetch in self.prefetch_factors
            for pin_memory in (True, False)
        ]

    def _refine(self, benchmark, batch_size: int, num_workers: int) -> dict:
        refine_results = DataLoaderSweep(benchmark, self._prefetch_specs(batch_size, num_workers)).run()
        refine_report  = SweepReport(refine_results, wait_threshold=self.data_wait_target)

        prefetch_factor = self.reference_prefetch
        pin_memory      = True

        if not refine_report.ok_frame.empty:
            best            = refine_report.ok_frame.sort_values("end_to_end_samples_per_s", ascending=False).iloc[0]
            prefetch_factor = int(best["prefetch_factor"])
            pin_memory      = bool(best["pin_memory"])

        return {"prefetch_factor": prefetch_factor, "pin_memory": pin_memory}

    def run(self, batch_size: int) -> Optional[dict]:
        benchmark = self._benchmark()

        main_results   = DataLoaderSweep(benchmark, self._worker_specs(batch_size)).run()
        recommendation = SweepReport(main_results, wait_threshold=self.data_wait_target).recommendation

        if not recommendation.get("found"):
            self.logger.subsection("Loader tuner found no successful configuration; keeping current loader settings")
            return None

        num_workers = int(recommendation["num_workers"])
        refined     = self._refine(benchmark, batch_size, num_workers)

        choice = {"num_workers": num_workers, "prefetch_factor": refined["prefetch_factor"], "pin_memory": refined["pin_memory"]}

        self.logger.subsection(f"Loader tuner -> workers={choice['num_workers']} prefetch={choice['prefetch_factor']} pin={int(choice['pin_memory'])} (cpu_bound={recommendation['cpu_bound']})")

        return choice
