from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from tools.benchmarking                   import LoaderSpec, GpuFeedBenchmark, DataLoaderSweep, SweepReport
from tools.monitoring.logger              import Logger
from tools.runtime.event_stream           import JsonEventStream
from tools.runtime.reproducibility        import Reproducibility
from pipelines.dataloader_tuning.adapters import build_feed_target


class DataLoaderTuningPipeline:
    def __init__(self, config) -> None:
        self.config     = config
        self.output_dir = Path(config.output_dir) / config.mode
        self.work_dir   = self.output_dir / "work"
        self.figure_dir = self.output_dir / "figures"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = Logger(log_dir=str(self.output_dir / "logs"), name="dataloader_tuning", level="INFO")
        self.events = JsonEventStream("@TUNE")

        Reproducibility.seed_everything(config.seed)

    def _build_target(self):
        self.logger.section("[DataLoader Tuning]")
        self.logger.kv_table({
            "Mode"          : self.config.mode,
            "Device"        : "cuda" if torch.cuda.is_available() else "cpu",
            "Batch sizes"   : self.config.batch_sizes,
            "Worker counts" : self.config.worker_counts,
            "Timed batches" : self.config.timed_batches,
            "AMP"           : self.config.use_amp,
        })

        return build_feed_target(self.config, self.work_dir, self.logger)

    def _build_benchmark(self, target):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        return GpuFeedBenchmark(
            dataset        = target.dataset,
            model          = target.model,
            to_model_input = target.to_model_input,
            forward_loss   = target.forward_loss,
            device         = device,
            use_amp        = self.config.use_amp,
            seed           = self.config.seed,
            warmup_batches = self.config.warmup_batches,
            timed_batches  = self.config.timed_batches,
            gpu_index      = self.config.gpu,
            cpu_threads    = self.config.cpu_threads,
        ), device

    def _worker_counts(self) -> list[int]:
        cores   = os.cpu_count() or 8
        kept    = [workers for workers in self.config.worker_counts if workers <= cores]
        dropped = [workers for workers in self.config.worker_counts if workers > cores]

        if dropped:
            self.logger.subsection(f"Skipping worker counts above {cores} cores: {dropped}")

        return kept or [min(self.config.worker_counts) if self.config.worker_counts else 0]

    def _main_specs(self) -> list[LoaderSpec]:
        worker_counts = self._worker_counts()

        return [
            LoaderSpec(batch_size=batch_size, num_workers=workers, prefetch_factor=self.config.reference_prefetch, pin_memory=True, persistent_workers=True)
            for batch_size in self.config.batch_sizes
            for workers in worker_counts
        ]

    def _run_sweep(self, benchmark, specs, phase: str) -> list[dict]:
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
        batch_size  = recommendation["batch_size"]
        num_workers = max(1, recommendation["num_workers"])

        return [
            LoaderSpec(batch_size=batch_size, num_workers=num_workers, prefetch_factor=prefetch, pin_memory=pin_memory, persistent_workers=True)
            for prefetch in self.config.prefetch_factors
            for pin_memory in (True, False)
        ]

    def _final_config(self, recommendation: dict, refine_report) -> dict:
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
        if self.config.save_figures and not report.ok_frame.empty:
            report.save_all(self.figure_dir)

        payload = {
            "mode"           : self.config.mode,
            "device"         : str(device),
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
