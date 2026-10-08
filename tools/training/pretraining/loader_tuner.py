"""Selection of data-loader worker count and prefetch factor by timed feed benchmarks."""

from __future__ import annotations

import os
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from tools.benchmarking      import LoaderSpec, GpuFeedBenchmark, DataLoaderSweep, SweepReport
from tools.monitoring.logger import Logger


class LoaderTuner:
    """Sweeps data-loader worker counts, then prefetch factors, on the real feed path.

    The first sweep varies worker count at a reference prefetch factor and picks
    the recommendation of the sweep report; the second sweep refines the prefetch
    factor at that worker count by end-to-end throughput. Every measured
    configuration is logged as it finishes, an out-of-memory one as a warning,
    and kept in ``records`` for the caller to save.

    Attributes:
        dataset            : Dataset the benchmark loads from.
        model              : Model the timed forward and backward passes run on.
        to_model_input     : Callable moving a batch to the device for the model.
        forward_loss       : Callable returning the loss of a batch.
        device             : Device the benchmark runs on.
        logger             : Logger for the sweep decisions.
        use_amp            : Whether the benchmark runs under autocast.
        seed               : Seed of the benchmark.
        warmup_batches     : Untimed batches before each measurement.
        timed_batches      : Timed batches per configuration.
        gpu_index          : GPU index reported by the benchmark, or None.
        cpu_threads        : Torch thread count for the benchmark, or None.
        worker_counts      : Worker counts probed in the first sweep.
        prefetch_factors   : Prefetch factors probed in the refinement sweep.
        reference_prefetch : Prefetch factor held fixed during the worker sweep.
        data_wait_target   : Data-wait fraction above which a configuration counts as input-bound.
        records            : Every measured configuration of the last run, in order, each stamped with the sweep phase that ran it.
    """

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
        """Stores the benchmark inputs and the sweep grids.

        Args:
            dataset            : Dataset the benchmark loads from.
            model              : Model used for the timed passes.
            to_model_input     : Callable moving a batch to the device.
            forward_loss       : Callable returning the loss of a batch.
            device             : Device the benchmark runs on.
            logger             : Logger for the sweep decisions.
            use_amp            : Whether to run the timed passes under autocast.
            seed               : Benchmark seed.
            warmup_batches     : Untimed batches before each measurement.
            timed_batches      : Timed batches per configuration.
            gpu_index          : GPU index reported by the benchmark, or None.
            cpu_threads        : Torch thread count for the benchmark, or None.
            worker_counts      : Worker counts to probe.
            prefetch_factors   : Prefetch factors to probe in the refinement.
            reference_prefetch : Prefetch factor held fixed in the worker sweep.
            data_wait_target   : Data-wait fraction marking an input-bound feed.
        """
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
        self.records            = []

    def _benchmark(self) -> GpuFeedBenchmark:
        """Returns the feed benchmark configured from the tuner settings."""
        return GpuFeedBenchmark(
            dataset        = self.dataset,
            model          = self.model,
            to_model_input = self.to_model_input,
            forward_loss   = self.forward_loss,
            device         = self.device,
            logger         = self.logger,
            use_amp        = self.use_amp,
            seed           = self.seed,
            warmup_batches = self.warmup_batches,
            timed_batches  = self.timed_batches,
            gpu_index      = self.gpu_index,
            cpu_threads    = self.cpu_threads,
        )

    def _worker_counts(self) -> list[int]:
        """Returns the configured worker counts that fit the machine's core count."""
        cores   = os.cpu_count() or 8
        kept    = [workers for workers in self.worker_counts if workers <= cores]
        dropped = [workers for workers in self.worker_counts if workers > cores]

        if dropped:
            self.logger.subsection(f"Skipping worker counts above {cores} cores: {dropped}")

        return kept or [min(self.worker_counts) if self.worker_counts else 0]

    def _worker_specs(self, batch_size: int) -> list[LoaderSpec]:
        """Returns one pinned-memory loader spec per probed worker count at the reference prefetch."""
        return [
            LoaderSpec(batch_size=batch_size, num_workers=workers, prefetch_factor=self.reference_prefetch, pin_memory=True, persistent_workers=True)
            for workers in self._worker_counts()
        ]

    def _prefetch_specs(self, batch_size: int, num_workers: int) -> list[LoaderSpec]:
        """Returns one loader spec per probed prefetch factor at a fixed worker count.

        Args:
            batch_size  : Batch size of the probe loaders.
            num_workers : Worker count chosen by the first sweep; raised to one so the prefetch factor has an effect.

        Returns:
            Loader specifications covering the configured prefetch factors.
        """
        workers = max(1, num_workers)

        return [
            LoaderSpec(batch_size=batch_size, num_workers=workers, prefetch_factor=prefetch, pin_memory=True, persistent_workers=True)
            for prefetch in self.prefetch_factors
        ]

    def _report(self, phase: str, record: dict) -> None:
        """Stamps, keeps and logs one finished configuration of a sweep.

        Args:
            phase  : Sweep that measured the configuration, 'workers' or 'prefetch'.
            record : Measurement record of the configuration, its status 'ok' or 'oom'.
        """
        record["phase"] = phase
        self.records.append(record)

        spec = f"{phase} sweep: workers={record['num_workers']} prefetch={record['prefetch_factor']}"

        if record["status"] != "ok":
            self.logger.warning(f"{spec} ran out of memory; configuration skipped")
            return

        self.logger.subsection(f"{spec} -> {record['end_to_end_samples_per_s']:.0f} samples/s end to end, data wait {100.0 * record['data_wait_fraction']:.1f}%, feed ratio {record['feed_ratio']:.2f}")

    def _sweep(self, benchmark, specs: list[LoaderSpec], phase: str) -> list[dict]:
        """Measures a list of loader configurations, announcing the sweep and reporting each result as it finishes.

        Args:
            benchmark : Feed benchmark measuring every configuration.
            specs     : Configurations to measure, in order.
            phase     : Name of the sweep, 'workers' or 'prefetch'.

        Returns:
            One measurement record per configuration.
        """
        self.logger.subsection(f"{phase} sweep: {len(specs)} configuration(s), each timed over {self.warmup_batches} warmup + {self.timed_batches} batches in three passes (loader only, compute ceiling, end to end)")

        return DataLoaderSweep(benchmark, specs, on_result=lambda record: self._report(phase, record)).run()

    def _refine(self, benchmark, batch_size: int, num_workers: int) -> Optional[dict]:
        """Returns the measurement of the prefetch factor with the best end-to-end throughput at a worker count.

        Args:
            benchmark   : Feed benchmark reused from the first sweep.
            batch_size  : Batch size of the probe loaders.
            num_workers : Worker count chosen by the first sweep.

        Returns:
            Record of the fastest successful refinement configuration, or None
            when every one ran out of memory, after a warning that the
            reference prefetch factor of the worker sweep is kept untuned.
        """
        refine_results = self._sweep(benchmark, self._prefetch_specs(batch_size, num_workers), "prefetch")
        refine_report  = SweepReport(refine_results, wait_threshold=self.data_wait_target)

        if refine_report.ok_frame.empty:
            self.logger.warning(f"every prefetch refinement configuration at workers={max(1, num_workers)} ran out of memory; the reference prefetch factor {self.reference_prefetch} of the worker sweep is kept, untuned")
            return None

        return refine_report.ok_frame.sort_values("end_to_end_samples_per_s", ascending=False).iloc[0].to_dict()

    def run(self, batch_size: int) -> Optional[dict]:
        """Runs the worker sweep and the prefetch refinement for a batch size.

        Args:
            batch_size : Batch size the loader settings are tuned for.

        Returns:
            Dictionary with "num_workers" and "prefetch_factor", or None when no
            configuration succeeded and the current settings should be kept.
        """
        self.records = []

        benchmark = self._benchmark()

        main_results   = self._sweep(benchmark, self._worker_specs(batch_size), "workers")
        recommendation = SweepReport(main_results, wait_threshold=self.data_wait_target).recommendation

        if not recommendation["found"]:
            self.logger.warning(f"Loader tuner found no successful configuration among {len(main_results)} measured ({recommendation['reason']}); keeping the current loader settings")
            return None

        num_workers = int(recommendation["num_workers"])
        refined     = self._refine(benchmark, batch_size, num_workers)
        measured    = refined if refined is not None else recommendation
        prefetch    = int(refined["prefetch_factor"]) if refined is not None else self.reference_prefetch

        choice = {"num_workers": num_workers, "prefetch_factor": prefetch}

        self.logger.subsection(f"Loader tuner -> workers={choice['num_workers']} prefetch={choice['prefetch_factor']}: {measured['end_to_end_samples_per_s']:.0f} samples/s end to end, data wait {100.0 * measured['data_wait_fraction']:.1f}% (target {100.0 * self.data_wait_target:.1f}%), feed ratio {measured['feed_ratio']:.2f} (pinned memory throughout, matching the training loader; cpu_bound={recommendation['cpu_bound']})")

        return choice
