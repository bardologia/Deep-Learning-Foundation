from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib     import Path

from configuration.benchmark                       import BenchmarkConfig
from configuration.sar.gaussian_config             import GaussianConfig
from pipelines.backbone.training.loss_terms        import LossComponentCatalog
from pipelines.benchmark.results                   import BenchmarkSeedCollector
from pipelines.shared.comparison.comparison_report import ComparisonReport
from pipelines.benchmark.sizing                    import SizeMatcher, SizeMatchResult
from pipelines.shared.training.run_naming          import RunNaming
from pipelines.shared.training.seed_sweep          import SeedSet
from tools.orchestration                           import ExperimentStage, GpuJob, QueuedInferenceStage, QueuedTrainingStage
from tools.data.io                                 import FileIO
from tools.monitoring.logger                       import Logger
from tools.runtime.run_tag                         import RunTag


class SeedExpandedStage:
    @classmethod
    def components(cls, config: BenchmarkConfig) -> list[str | None]:
        if config.training_type != "backbone":
            return [None]

        components = list(config.sweep_loss_components)
        if not components:
            raise SystemExit("sweep_loss_components is empty; select at least one loss component to sweep")

        valid   = set(LossComponentCatalog.names())
        unknown = [component for component in components if component not in valid]
        if unknown:
            raise SystemExit(f"unknown loss component(s) {unknown}; valid: {', '.join(sorted(valid))}")

        return components

    @staticmethod
    def n_gaussians(config: BenchmarkConfig) -> int:
        return GaussianConfig.from_dataset(config.paths.dataset_path, config.paths.parameters_path).n_default_gaussians

    @classmethod
    def unit_base(cls, config: BenchmarkConfig, model: str, component: str | None) -> str:
        if config.training_type == "backbone":
            return RunNaming.benchmark_unit(model, component, config.loss, cls.n_gaussians(config), config.augmentation)
        if config.training_type == "jepa":
            return RunNaming.benchmark_unit(model, None, config.jepa.param_loss, cls.n_gaussians(config), config.augmentation)

        return model if component is None else f"{model}__{component}"

    def _expand(self, config: BenchmarkConfig, models: list[str]) -> list[str]:
        components = self.components(config)
        pairs      = [(model, component) for model in models for component in components]
        self._pair = {self.unit_base(config, model, component): (model, component) for model, component in pairs}

        units      = SeedSet.units(list(self._pair.keys()), config.seeds)
        self._unit = {name: (base, seed) for base, seed, name in units}

        return [name for _, _, name in units]

    def _seed_command(self, action: str, name: str) -> list[str]:
        base, seed       = self._unit[name]
        model, component = self._pair[base]

        loss_args = ["--loss-component", component] if component is not None else []

        return [sys.executable, str(self.entry_script), "--worker", action, "--model", model, *loss_args, *SeedSet.cli_args(seed), "--run-tag", self.run_tag, "--run-dir", str(self.run_dir)]


class MaxBatchStage(ExperimentStage):
    def __init__(self, config: BenchmarkConfig, entry_script: Path, run_tag: str, models: list[str], logger: Logger) -> None:
        super().__init__(config=config, run_tag=run_tag, logger=logger, entry_script=entry_script)
        self.models       = models
        self.stage_dir    = self.run_dir / "max_batch"
        self.records_path = self.run_dir / "pipeline" / "max_batch.json"
        self.report_path  = self.run_dir / "pipeline" / "max_batch_report.md"

    def _result_path(self, model_name: str) -> Path:
        return self.stage_dir / model_name / "max_batch_result.json"

    def _has_result(self, model_name: str) -> bool:
        path = self._result_path(model_name)

        if not self.config.resume or not path.exists():
            return False

        return FileIO.load_json(path)["status"] == "PASS"

    def _job(self, model_name: str) -> GpuJob:
        return GpuJob(
            name     = model_name,
            command  = [sys.executable, str(self.entry_script), "--worker", "maxbatch", "--model", model_name, "--run-tag", self.run_tag, "--run-dir", str(self.run_dir)],
            log_path = self.stage_dir / model_name / "worker.log",
        )

    def _load_result(self, model_name: str) -> dict:
        path = self._result_path(model_name)

        if not path.exists():
            return {
                "model"      : model_name,
                "status"     : "FAIL",
                "batch_size" : None,
                "peak_gb"    : None,
                "budget_gb"  : self.config.max_batch.vram_budget_gb,
                "ceiling"    : self.config.max_batch.max_batch,
                "context_gb" : None,
                "trials"     : [],
                "error"      : f"missing result file: {path}",
            }

        return FileIO.load_json(path)

    def _write_report(self, records: dict) -> None:
        budget  = self.config.max_batch.vram_budget_gb
        ceiling = self.config.max_batch.max_batch

        lines = [
            "# Maximum Batch Size Report",
            f"\n_Generated {RunTag.timestamp()}  —  run tag `{self.run_tag}`_\n",
            f"Largest power-of-two batch size whose peak training memory stays under **{budget:g} GB** and below **{ceiling}**.",
            f"Measured at patch size {tuple(self.config.training.patch_size)} with {self.config.size_match.in_channels} input channels.\n",
            "## Results\n",
            "| Model | Max batch | Peak (GB) | Status | Error |",
            "| --- | --- | --- | --- | --- |",
        ]

        for model_name in self.models:
            record     = records[model_name]
            batch_size = record["batch_size"] if record["batch_size"] is not None else "—"
            peak_gb    = f"{record['peak_gb']:.2f}" if record.get("peak_gb") is not None else "—"
            error      = (record.get("error") or "—").strip().splitlines()[-1]
            lines.append(f"| `{model_name}` | {batch_size} | {peak_gb} | {record['status']} | {error} |")

        lines.append("")

        self.report_path.write_text("\n".join(lines), encoding="utf-8")
        self.logger.info(f"Report written to: {self.report_path}")

    def _log_summary(self, records: dict) -> None:
        passed = [m for m in records if records[m]["status"] == "PASS"]
        failed = [m for m in records if records[m]["status"] != "PASS"]

        self.logger.subsection("Max batch summary")
        self.logger.kv_table({
            "Total"  : len(records),
            "Passed" : len(passed),
            "Failed" : len(failed),
        }, title=f"{len(passed)}/{len(records)} measured")

        for model_name in passed:
            self.logger.info(f"{model_name:<18} batch {records[model_name]['batch_size']:>4}  peak {records[model_name]['peak_gb']:.2f} GB")

        for model_name in failed:
            self.logger.error(f"FAILED  {model_name}")

    def run(self) -> dict:
        self.logger.section("Maximum batch size")
        self.logger.kv_table({
            "Models"        : len(self.models),
            "VRAM budget"   : f"{self.config.max_batch.vram_budget_gb:g} GB",
            "Ceiling"       : self.config.max_batch.max_batch,
            "Measure steps" : self.config.max_batch.measure_steps,
            "GPUs"          : self.config.gpus,
        }, title="Configuration")

        cached  = [m for m in self.models if self._has_result(m)]
        pending = [m for m in self.models if m not in cached]

        for model_name in cached:
            self.logger.info(f"{model_name}: cached result reused")

        if self.config.resume:
            for model_name in pending:
                if self._result_path(model_name).exists():
                    self.logger.warning(f"{model_name}: previous result was not PASS, re-probing")

        if pending:
            self._run_queue([self._job(m) for m in pending])

        records = {m: self._load_result(m) for m in self.models}

        self._write_results(records, self.records_path)
        self._write_report(records)
        self._log_summary(records)

        return records


class SizeMatchStage(ExperimentStage):
    def __init__(self, config: BenchmarkConfig, run_tag: str, models: list[str], logger: Logger) -> None:
        super().__init__(config=config, run_tag=run_tag, logger=logger)
        self.models       = models
        self.records_path = self.run_dir / "pipeline" / "size_match.json"
        self.report_path  = self.run_dir / "pipeline" / "size_match_report.md"

    def _validate_reference(self, reference: str) -> None:
        if reference not in self.models:
            raise SystemExit(f"size_match.reference_model '{reference}' is not one of the swept model keys {self.models}; give the full key including the head (e.g. 'unet-set_pred') so the reference is capacity-matched against its own architecture and not bisected against a different head")

    def _load_cached(self) -> dict | None:
        if not self.config.resume or not self.records_path.exists():
            return None

        records = FileIO.load_json(self.records_path)

        if all(m in records for m in self.models):
            return records

        return None

    def _reference_record(self, reference: str, target: int) -> dict:
        return asdict(SizeMatchResult(
            model         = reference,
            scale         = 1.0,
            overrides     = {},
            parameters    = target,
            target        = target,
            deviation_pct = 0.0,
            iterations    = 0,
        ))

    def _write_report(self, records: dict, target: int, out_channels: int) -> None:
        lines = [
            "# Capacity Matching Report",
            f"\n_Generated {RunTag.timestamp()}  —  run tag `{self.run_tag}`_\n",
            f"Reference model `{self.config.size_match.reference_model}` at **{target:,}** parameters.",
            f"Counting performed with {self.config.size_match.in_channels} input channels, {out_channels} output channels, patch size {tuple(self.config.training.patch_size)}.\n",
            "## Matched Widths\n",
            "| Model | Scale | Scaled attributes | Parameters | Δ vs reference | Iterations | Flags |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]

        for model_name in sorted(records.keys()):
            record    = records[model_name]
            overrides = ", ".join(f"`{k}={v}`" for k, v in record["overrides"].items()) or "_(defaults)_"
            flags     = "; ".join(record["flags"]) or "—"
            lines.append(f"| `{model_name}` | {record['scale']:.4f} | {overrides} | {record['parameters']:,} | {record['deviation_pct']:+.3f} % | {record['iterations']} | {flags} |")

        lines.append("")

        self.report_path.write_text("\n".join(lines), encoding="utf-8")
        self.logger.info(f"Report written to: {self.report_path}")

    def run(self) -> dict:
        self._validate_reference(self.config.size_match.reference_model)

        matcher = SizeMatcher(config=self.config, logger=self.logger)

        self.logger.section("Capacity matching")
        self.logger.kv_table({
            "Reference model" : self.config.size_match.reference_model,
            "Tolerance"       : f"{100.0 * self.config.size_match.tolerance:.2f} %",
            "Max iterations"  : self.config.size_match.max_iterations,
            "In channels"     : self.config.size_match.in_channels,
            "Out channels"    : matcher.out_channels,
        }, title="Configuration")

        cached = self._load_cached()
        if cached is not None:
            self.logger.info(f"Cached size match reused from: {self.records_path}")
            return cached

        reference = self.config.size_match.reference_model
        target    = matcher.reference_count()

        self.logger.info(f"Reference '{reference}': {target:,} parameters")

        records = {reference: self._reference_record(reference, target)}

        for model_name in self.models:
            if model_name == reference:
                continue

            result              = matcher.match(model_name, target)
            records[model_name] = asdict(result)

            self.logger.info(f"{model_name:<18} {result.parameters:>14,}  Δ {result.deviation_pct:+7.3f} %  (scale {result.scale:.4f}, {result.iterations} iterations)")

            for flag in result.flags:
                self.logger.warning(f"{model_name}: {flag}")

        self._write_results(records, self.records_path)
        self._write_report(records, target, matcher.out_channels)

        return records


class TrainingStage(SeedExpandedStage, QueuedTrainingStage):
    def __init__(self, config: BenchmarkConfig, entry_script: Path, run_tag: str, models: list[str], logger: Logger) -> None:
        items = self._expand(config, models)
        super().__init__(config=config, entry_script=entry_script, run_tag=run_tag, items=items, logger=logger)

    def _job(self, item: str) -> GpuJob:
        return GpuJob(
            name     = item,
            command  = self._seed_command(self.worker_action, item),
            log_path = self.stage_dir / item / self.worker_logname,
        )


class InferenceStage(SeedExpandedStage, QueuedInferenceStage):
    def __init__(self, config: BenchmarkConfig, entry_script: Path, run_tag: str, models: list[str], logger: Logger) -> None:
        items = self._expand(config, models)
        super().__init__(config=config, entry_script=entry_script, run_tag=run_tag, items=items, logger=logger)

    def _job(self, item: str) -> GpuJob:
        return GpuJob(
            name     = item,
            command  = self._seed_command(self.worker_action, item),
            log_path = self.stage_dir / item / self.worker_logname,
        )


class ComparisonStage(ExperimentStage):
    def __init__(self, config, run_tag: str, logger: Logger, reference_model: str, embed_images: bool) -> None:
        super().__init__(config=config, run_tag=run_tag, logger=logger)
        self.reference_model = reference_model
        self.embed_images    = embed_images

    def run(self) -> Path:
        self.logger.section("Comparison reports")

        collector = BenchmarkSeedCollector(run_dir=self.run_dir, logger=self.logger)
        records   = collector.collect()

        out_dir = self.run_dir / "comparison" / RunTag.now()

        report = ComparisonReport(
            records         = records,
            out_dir         = out_dir,
            reference_model = self.reference_model,
            embed_images    = self.embed_images,
            logger          = self.logger,
            seed_dispersion = collector.seed_dispersion,
        )

        written = report.write_all()

        self.logger.subsection("Reports written")
        for path in written:
            self.logger.info(f"{path}")

        return out_dir
