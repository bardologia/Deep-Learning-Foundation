
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib  import Path

import torch

from tools.benchmarking                   import LoaderSpec, GpuFeedBenchmark, GpuUtilizationSampler, DataLoaderSweep, SweepReport
from tools.monitoring.logger              import Logger
from tools.runtime.config_cli             import ConfigCli
from tools.runtime.event_stream           import JsonEventStream
from tools.runtime.git_state              import RunProvenance
from tools.runtime.reproducibility        import Reproducibility
from tools.runtime.run_tag                import RunTag
from pipelines.dataloader_tuning.adapters import FeedAdapter


class DataLoaderTuningPipeline:

    def __init__(self, config) -> None:
        self.config     = config
        self.model_name = config.model_name or FeedAdapter.DEFAULT_MODEL
        self.output_dir = Path(config.output_dir) / f"{self.model_name}_{RunTag.now()}"
        self.work_dir   = Path(config.output_dir) / "work"
        self.figure_dir = self.output_dir / "figures"

        self.output_dir.mkdir(parents=True)
        ConfigCli.save_resolved(config, self.output_dir / "resolved_config.json")

        self.started_at = datetime.now()
        self.gpu        = None
        self.logger     = Logger(log_dir=str(self.output_dir / "logs"), name="dataloader_tuning", level="INFO")
        self.events     = JsonEventStream("@TUNE")

        Reproducibility.seed_everything(config.seed)

    def _build_target(self):
        self.logger.section("[DataLoader Tuning]")
        self.logger.kv_table({
            "Device"        : "cuda" if torch.cuda.is_available() else "cpu",
            "Batch sizes"   : self.config.batch_sizes,
            "Worker counts" : self.config.worker_counts,
            "Timed batches" : self.config.timed_batches,
            "AMP"           : self.config.use_amp,
            "Output dir"    : str(self.output_dir),
        })

        return FeedAdapter(self.config, self.work_dir, self.logger).build()

    def _gpu(self) -> dict:
        if not torch.cuda.is_available():
            return {"index": self.config.gpu, "name": None, "nvml_index": None, "nvml_name": None, "uuid": None}

        nvml = GpuUtilizationSampler.nvml_device()

        return {"index": self.config.gpu, "name": torch.cuda.get_device_name(0), "nvml_index": nvml["index"], "nvml_name": nvml["name"], "uuid": nvml["uuid"]}

    def _build_benchmark(self, target):
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

    def _log_feed_target(self, target, device) -> int:
        parameters = int(sum(parameter.numel() for parameter in target.model.parameters()))
        model      = target.model_name if self.config.model_name else f"{target.model_name} (default)"
        gpu        = "none (CPU)" if self.gpu["name"] is None else f"{self.gpu['name']}, requested index {self.gpu['index']}, NVML {self.gpu['nvml_index']}"

        self.logger.kv_table({
            "Model"          : model,
            "Parameters"     : f"{parameters:,}",
            "Sample"         : target.sample_text,
            "Item source"    : target.item_source,
            "Device"         : str(device),
            "GPU"            : gpu,
            "Torch threads"  : torch.get_num_threads(),
            "Warm-up/timed"  : f"{self.config.warmup_batches}/{self.config.timed_batches} batches",
            "Wait target"    : f"{self.config.data_wait_target:.0%}",
            "Refine"         : self.config.refine,
            "Output dir"     : str(self.output_dir),
        }, title="Feed target")

        if device.type != "cuda":
            self.logger.warning("No CUDA device is visible: the sweep runs on the CPU, and its recommendation says nothing about feeding a GPU")

        return parameters

    def _worker_counts(self) -> list[int]:
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
        worker_counts = self._worker_counts()

        return [
            LoaderSpec(batch_size=batch_size, num_workers=workers, prefetch_factor=self.config.reference_prefetch, pin_memory=True, persistent_workers=True)
            for batch_size in self.config.batch_sizes
            for workers in worker_counts
        ]

    def _run_sweep(self, benchmark, specs, phase: str) -> list[dict]:
        total    = len(specs)
        finished = 0

        def _on_start(spec):
            self.logger.debug(f"[{phase} {finished + 1}/{total}] measuring {spec.label}")

        def _on_result(record):
            nonlocal finished
            finished       += 1
            record["phase"] = phase
            self.events.emit("result", record)

            position = f"[{phase} {finished}/{total}]"
            if record["status"] == "ok":
                self.logger.subsection(f"{position} bs={record['batch_size']:>5} workers={record['num_workers']} pf={record['prefetch_factor']} pin={int(record['pin_memory'])} -> {record['end_to_end_samples_per_s']:.0f} samp/s, wait {100.0 * record['data_wait_fraction']:.1f}%, gpu {record['gpu_util_mean']:.0f}%, feed {record['feed_ratio']:.2f}")
            else:
                self.logger.subsection(f"{position} bs={record['batch_size']} workers={record['num_workers']} -> {record['status']}")

        return DataLoaderSweep(benchmark, specs, on_result=_on_result, on_start=_on_start).run()

    def _refine_specs(self, recommendation: dict) -> list[LoaderSpec]:
        return [
            LoaderSpec(batch_size=recommendation["batch_size"], num_workers=recommendation["num_workers"], prefetch_factor=prefetch, pin_memory=pin_memory, persistent_workers=True)
            for prefetch in self.config.prefetch_factors
            for pin_memory in (True, False)
        ]

    def _refine(self, benchmark, recommendation: dict) -> tuple[list[dict], SweepReport | None]:
        if not self.config.refine:
            return [], None

        if recommendation["num_workers"] == 0:
            self.logger.warning("Refinement skipped: the main sweep recommends 0 loader workers, and the refinement varies the per-worker prefetch factor, which a 0-worker loader has none of; pin_memory stays as the main sweep measured it")
            return [], None

        results = self._run_sweep(benchmark, self._refine_specs(recommendation), phase="refine")

        return results, SweepReport(results, wait_threshold=self.config.data_wait_target)

    def _final_config(self, recommendation: dict, refine_report) -> dict:
        workers = recommendation["num_workers"]
        final   = {
            "batch_size"         : recommendation["batch_size"],
            "num_workers"        : workers,
            "prefetch_factor"    : self.config.reference_prefetch if workers > 0 else 0,
            "pin_memory"         : True,
            "persistent_workers" : workers > 0,
            "cpu_bound"          : recommendation["cpu_bound"],
        }

        if refine_report is None:
            return final

        if refine_report.ok_frame.empty:
            self.logger.warning(f"No refinement spec succeeded; keeping prefetch {final['prefetch_factor']} and pin_memory True as the main sweep measured them")
            return final

        best = refine_report.ok_frame.sort_values("end_to_end_samples_per_s", ascending=False).iloc[0]
        final["prefetch_factor"] = int(best["prefetch_factor"])
        final["pin_memory"]      = bool(best["pin_memory"])

        self.logger.info(f"Refinement winner prefetch {final['prefetch_factor']}, pin_memory {final['pin_memory']}: {best['end_to_end_samples_per_s']:.0f} samp/s against {recommendation['end_to_end_samples_per_s']:.0f} samp/s at the reference prefetch {self.config.reference_prefetch} with pinning")

        return final

    @staticmethod
    def _tally(results: list[dict]) -> str:
        ok = sum(1 for record in results if record["status"] == "ok")

        return f"{ok} ok, {len(results) - ok} out of memory"

    def _log_sweep_summary(self, main_results: list[dict], refine_results: list[dict], report: SweepReport, recommendation: dict) -> None:
        self.logger.kv_table({
            "Main sweep" : self._tally(main_results),
            "Refinement" : self._tally(refine_results) if refine_results else "not run",
            "Saturated"  : f"{len(report.saturated())} of {len(report.ok_frame)} working specs",
        }, title="Sweep summary")

        if recommendation["cpu_bound"]:
            self.logger.warning(f"No configuration reached the {self.config.data_wait_target:.0%} data-wait target or fed the GPU as fast as it computes; recommending the fastest unsaturated setting, so training will wait on data")

    def _write(self, target, device, main_results, refine_results, recommendation, final, status: str) -> Path:
        payload = {
            "status"         : status,
            "device"         : str(device),
            "gpu"            : self.gpu,
            "model_name"     : target.model_name,
            "sample"         : target.sample_text,
            "item_source"    : target.item_source,
            "config_hint"    : target.config_hint,
            "config"         : ConfigCli.to_mapping(self.config),
            "provenance"     : RunProvenance.collect(),
            "started_at"     : self.started_at.isoformat(timespec="seconds"),
            "finished_at"    : datetime.now().isoformat(timespec="seconds") if status != "partial" else None,
            "main"           : main_results,
            "refine"         : refine_results,
            "recommendation" : recommendation,
            "final"          : final,
        }

        results_path = self.output_dir / "results.json"
        results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        return results_path

    def run(self):
        with self.logger:
            target            = self._build_target()
            benchmark, device = self._build_benchmark(target)
            parameter_count   = self._log_feed_target(target, device)
            main_specs        = self._main_specs()

            self.events.emit("meta", {
                "device"      : str(device),
                "model_name"  : target.model_name,
                "parameters"  : parameter_count,
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
                self._write(target, device, main_results, [], recommendation, None, "failed")
                self.events.emit("recommendation", {"recommendation": recommendation, "final": None})
                self.logger.error(f"No configuration succeeded: {recommendation['reason']}")
                raise SystemExit(f"dataloader tuning found no working configuration: {recommendation['reason']}")

            self._write(target, device, main_results, [], recommendation, None, "partial")

            refine_results, refine_report = self._refine(benchmark, recommendation)

            final = self._final_config(recommendation, refine_report)
            self._log_sweep_summary(main_results, refine_results, report, recommendation)

            self.events.emit("recommendation", {"recommendation": recommendation, "final": final})

            if self.config.save_figures and not report.ok_frame.empty:
                report.save_all(self.figure_dir)

            results_path = self._write(target, device, main_results, refine_results, recommendation, final, "complete")

            self.logger.section("[Recommended configuration]")
            self.logger.kv_table({**final, "results": str(results_path)})

            self.events.emit("done", {"results_path": str(results_path), "figure_dir": str(self.figure_dir)})

            return final
