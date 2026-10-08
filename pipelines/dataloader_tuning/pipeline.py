"""Sweep that tunes dataloader settings against measured GPU feed throughput.

Sweeps batch size against worker count, optionally refines prefetch factor and
memory pinning around the winner, and writes the recommended loader
configuration alongside the raw measurements.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from tools.benchmarking                   import LoaderSpec, GpuFeedBenchmark, GpuUtilizationSampler, DataLoaderSweep, SweepReport
from tools.monitoring.logger              import Logger
from tools.runtime.event_stream           import JsonEventStream
from tools.runtime.reproducibility        import Reproducibility
from pipelines.dataloader_tuning.adapters import FeedTargetFactory


class DataLoaderTuningPipeline:
    """Runs the dataloader tuning sweep and reports the recommended settings.

    Attributes:
        config     : Dataloader tuning entry configuration.
        output_dir : Mode-specific directory holding results, logs, and figures.
        work_dir   : Scratch directory the dataset pipelines write into.
        figure_dir : Directory the sweep figures are written to.
        gpu        : Requested index, torch name and NVML index, name and UUID of the measured device, set once the benchmark is built.
        logger     : Logger used for progress reporting.
        events     : Stream emitting machine-readable progress events.
    """

    def __init__(self, config) -> None:
        """Prepares the output tree, logging, and global seeding.

        Args:
            config : Dataloader tuning entry configuration.
        """
        self.config     = config
        self.output_dir = Path(config.output_dir) / config.mode
        self.work_dir   = self.output_dir / "work"
        self.figure_dir = self.output_dir / "figures"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gpu    = None
        self.logger = Logger(log_dir=str(self.output_dir / "logs"), name="dataloader_tuning", level="INFO")
        self.events = JsonEventStream("@TUNE")

        Reproducibility.seed_everything(config.seed)

    def _build_target(self):
        """Logs the sweep configuration and builds the feed target to benchmark."""
        self.logger.section("[DataLoader Tuning]")
        self.logger.kv_table({
            "Mode"          : self.config.mode,
            "Device"        : "cuda" if torch.cuda.is_available() else "cpu",
            "Batch sizes"   : self.config.batch_sizes,
            "Worker counts" : self.config.worker_counts,
            "Timed batches" : self.config.timed_batches,
            "AMP"           : self.config.use_amp,
        })

        return FeedTargetFactory.build(self.config, self.work_dir, self.logger)

    def _gpu(self) -> dict:
        """Returns the device the sweep measures on: the requested index, the name torch reports, and the NVML index, name and UUID the utilisation is sampled from; None fields on a CPU run."""
        if not torch.cuda.is_available():
            return {"index": self.config.gpu, "name": None, "nvml_index": None, "nvml_name": None, "uuid": None}

        nvml = GpuUtilizationSampler.nvml_device()

        return {"index": self.config.gpu, "name": torch.cuda.get_device_name(0), "nvml_index": nvml["index"], "nvml_name": nvml["name"], "uuid": nvml["uuid"]}

    def _build_benchmark(self, target):
        """Builds the GPU feed benchmark for a target.

        Args:
            target : Feed target holding the dataset, model, and forward loss.

        Returns:
            A tuple (benchmark, device); the device is CUDA when available.
        """
        device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu = self._gpu()

        if self.gpu["uuid"] is not None:
            self.logger.info(f"GPU utilisation is sampled from NVML device {self.gpu['nvml_index']} ({self.gpu['nvml_name']}, {self.gpu['uuid']}), the device torch computes on as cuda:0 ({self.gpu['name']}) for requested index {self.config.gpu}")

        return GpuFeedBenchmark(
            dataset        = target.dataset,
            model          = target.model,
            to_model_input = target.to_model_input,
            forward_loss   = target.forward_loss,
            device         = device,
            logger         = self.logger,
            use_amp        = self.config.use_amp,
            seed           = self.config.seed,
            warmup_batches = self.config.warmup_batches,
            timed_batches  = self.config.timed_batches,
            gpu_index      = None,
            cpu_threads    = self.config.cpu_threads,
        ), device

    def _worker_counts(self) -> list[int]:
        """Returns the worker counts to sweep, capped at the available CPU cores.

        Returns:
            The configured counts that fit within the core count, or a
            single-entry list with the core count when none do.

        Raises:
            ValueError : If no worker count was configured.
        """
        if not self.config.worker_counts:
            raise ValueError("worker_counts is empty; select at least one dataloader worker count to benchmark")

        cores   = os.cpu_count() or 8
        kept    = [workers for workers in self.config.worker_counts if workers <= cores]
        dropped = [workers for workers in self.config.worker_counts if workers > cores]

        if dropped:
            self.logger.subsection(f"Skipping worker counts above {cores} cores: {dropped}")

        if kept:
            return kept

        self.logger.warning(f"Every configured worker count exceeds the {cores} available cores; benchmarking {cores} workers instead of {self.config.worker_counts}")

        return [cores]

    def _main_specs(self) -> list[LoaderSpec]:
        """Returns the loader specs of the main batch-size by worker-count grid."""
        worker_counts = self._worker_counts()

        return [
            LoaderSpec(batch_size=batch_size, num_workers=workers, prefetch_factor=self.config.reference_prefetch, pin_memory=True, persistent_workers=True)
            for batch_size in self.config.batch_sizes
            for workers in worker_counts
        ]

    def _run_sweep(self, benchmark, specs, phase: str) -> list[dict]:
        """Benchmarks every loader spec, logging and emitting each result.

        Args:
            benchmark : GPU feed benchmark to run.
            specs     : Loader specifications to time.
            phase     : Sweep phase name stamped onto each result, 'main' or 'refine'.

        Returns:
            One measurement record per spec.
        """
        def _on_result(record):
            record["phase"] = phase
            self.events.emit("result", record)
            if record.get("status") == "ok":
                self.logger.subsection(
                    f"bs={record['batch_size']:>5} workers={record['num_workers']} "
                    f"pf={record['prefetch_factor']} pin={int(record['pin_memory'])} "
                    f"-> {record['end_to_end_samples_per_s']:.0f} samp/s, "
                    f"wait {100.0 * record['data_wait_fraction']:.1f}%, "
                    f"gpu {record['gpu_util_mean']:.0f}%, feed {record['feed_ratio']:.2f}"
                )
            else:
                self.logger.subsection(f"bs={record['batch_size']} workers={record['num_workers']} -> {record['status']}")

        return DataLoaderSweep(benchmark, specs, on_result=_on_result).run()

    def _refine_specs(self, recommendation: dict) -> list[LoaderSpec]:
        """Returns the prefetch-factor by pinning grid around the recommended point."""
        batch_size  = recommendation["batch_size"]
        num_workers = max(1, recommendation["num_workers"])

        return [
            LoaderSpec(batch_size=batch_size, num_workers=num_workers, prefetch_factor=prefetch, pin_memory=pin_memory, persistent_workers=True)
            for prefetch in self.config.prefetch_factors
            for pin_memory in (True, False)
        ]

    def _final_config(self, recommendation: dict, refine_report) -> dict:
        """Merges the main recommendation with the refinement winner.

        Args:
            recommendation : Best batch size and worker count from the main sweep.
            refine_report  : Report of the refinement phase, or None when skipped.

        Returns:
            The loader configuration to adopt, including prefetch factor and
            pinning taken from the fastest refinement measurement when present.
        """
        final = {
            "batch_size"         : recommendation["batch_size"],
            "num_workers"        : recommendation["num_workers"],
            "prefetch_factor"    : self.config.reference_prefetch,
            "pin_memory"         : True,
            "persistent_workers" : True,
            "cpu_bound"          : recommendation["cpu_bound"],
        }

        if refine_report is not None and not refine_report.ok_frame.empty:
            best = refine_report.ok_frame.sort_values("end_to_end_samples_per_s", ascending=False).iloc[0]
            final["prefetch_factor"] = int(best["prefetch_factor"])
            final["pin_memory"]      = bool(best["pin_memory"])

        return final

    def _write(self, target, device, main_results, refine_results, recommendation, final, report) -> Path:
        """Saves the sweep figures and the results payload.

        Args:
            target         : Benchmarked feed target, described in the payload.
            device         : Device the benchmark ran on.
            main_results   : Measurements of the main sweep.
            refine_results : Measurements of the refinement sweep.
            recommendation : Best point of the main sweep.
            final          : Loader configuration to adopt.
            report         : Report of the main sweep, used to render the figures.

        Returns:
            Path of the written `results.json`.
        """
        if self.config.save_figures and not report.ok_frame.empty:
            report.save_all(self.figure_dir)

        payload = {
            "mode"           : self.config.mode,
            "device"         : str(device),
            "gpu"            : self.gpu,
            "model_name"     : target.model_name,
            "sample"         : target.sample_text,
            "item_source"    : target.item_source,
            "config_hint"    : target.config_hint,
            "main"           : main_results,
            "refine"         : refine_results,
            "recommendation" : recommendation,
            "final"          : final,
        }

        results_path = self.output_dir / "results.json"
        results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        return results_path

    def run(self):
        """Runs the main sweep, the optional refinement, and writes the results.

        Returns:
            The recommended loader configuration.

        Raises:
            SystemExit : If no loader configuration completed successfully.
        """
        target            = self._build_target()
        benchmark, device = self._build_benchmark(target)

        parameter_count = sum(parameter.numel() for parameter in target.model.parameters())
        main_specs      = self._main_specs()

        self.events.emit("meta", {
            "mode"        : self.config.mode,
            "device"      : str(device),
            "model_name"  : target.model_name,
            "parameters"  : int(parameter_count),
            "sample"      : target.sample_text,
            "item_source" : target.item_source,
            "config_hint" : target.config_hint,
            "n_specs"     : len(main_specs),
            "wait_target" : self.config.data_wait_target,
        })

        main_results = self._run_sweep(benchmark, main_specs, phase="main")

        report         = SweepReport(main_results, wait_threshold=self.config.data_wait_target)
        recommendation = report.recommendation

        if not recommendation["found"]:
            self.events.emit("recommendation", {"recommendation": recommendation, "final": None})
            self.logger.error(f"No configuration succeeded: {recommendation['reason']}")
            self.logger.close()
            raise SystemExit(f"dataloader tuning found no working configuration: {recommendation['reason']}")

        refine_results = []
        refine_report  = None
        if self.config.refine and recommendation.get("found"):
            refine_results = self._run_sweep(benchmark, self._refine_specs(recommendation), phase="refine")
            refine_report  = SweepReport(refine_results, wait_threshold=self.config.data_wait_target)

        final = self._final_config(recommendation, refine_report)

        self.events.emit("recommendation", {"recommendation": recommendation, "final": final})

        results_path = self._write(target, device, main_results, refine_results, recommendation, final, report)

        self.logger.section("[Recommended configuration]")
        self.logger.kv_table({**final, "results": str(results_path)})

        self.events.emit("done", {"results_path": str(results_path), "figure_dir": str(self.figure_dir)})
        self.logger.close()

        return final
