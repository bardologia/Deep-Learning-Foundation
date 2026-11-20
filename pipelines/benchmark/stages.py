
from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib     import Path

from configuration.benchmark                       import BenchmarkConfig
from configuration.dataset.gaussian             import GaussianConfig
from pipelines.benchmark.results                   import BenchmarkSeedCollector
from pipelines.shared.comparison.comparison_report import ComparisonReport
from pipelines.benchmark.sizing                    import SizeMatcher, SizeMatchResult
from pipelines.shared.model.model_builder          import ModelBuilder
from pipelines.shared.orchestration.unit_manifest  import UnidentifiedRuns, UnitLedger, UnitManifest
from pipelines.shared.training.run_naming          import RunNaming
from pipelines.shared.training.seed_sweep          import SeedSet
from tools.orchestration                           import ExperimentStage, GpuJob, QueuedInferenceStage, QueuedTrainingStage
from tools.data.io                                 import FileIO
from tools.monitoring.logger                       import Logger
from tools.runtime.git_state                       import RunProvenance
from tools.runtime.run_tag                         import RunTag


class BenchmarkUnitManifest:

    KIND = "benchmark"

    @staticmethod
    def identity(manifest: dict) -> tuple:
        return (manifest["model"], manifest["loss_component"], manifest["seed"])

    @classmethod
    def write(cls, run_directory: Path, run_tag: str, unit: str, model: str, loss_component: str | None, seed: int | None) -> Path:
        payload = {
            "run_tag"        : run_tag,
            "unit"           : unit,
            "model"          : model,
            "loss_component" : loss_component,
            "seed"           : seed,
        }

        return UnitManifest.write(run_directory, cls.KIND, payload)


class SeedExpandedStage:

    _N_GAUSSIANS : dict = {}

    @classmethod
    def components(cls, config: BenchmarkConfig) -> list[str | None]:
        components = list(config.sweep_loss_components)

        return components

    @classmethod
    def n_gaussians(cls, config: BenchmarkConfig) -> int:
        key = (str(config.paths.dataset_path), str(config.paths.parameters_path))

        if key not in cls._N_GAUSSIANS:
            cls._N_GAUSSIANS[key] = GaussianConfig.from_dataset(config.paths.dataset_path, config.paths.parameters_path).n_default_gaussians

        return cls._N_GAUSSIANS[key]

    @classmethod
    def unit_base(cls, config: BenchmarkConfig, model: str, component: str | None) -> str:
        return RunNaming.benchmark_unit(model, component, config.loss, cls.n_gaussians(config), config.augmentation)

    @classmethod
    def planned_units(cls, config: BenchmarkConfig, models: list[str]) -> list[tuple[str, tuple]]:
        components = cls.components(config)
        pairs      = [(model, component) for model in models for component in components]
        by_base    = {cls.unit_base(config, model, component): (model, component) for model, component in pairs}

        units = SeedSet.units(list(by_base.keys()), config.seeds)

        return [(name, (*by_base[base], seed)) for base, seed, name in units]

    def _expand(self, config: BenchmarkConfig, models: list[str]) -> list[str]:
        planned    = self.planned_units(config, models)
        self._unit = dict(planned)

        return [name for name, _ in planned]

    def _identity(self, name: str) -> tuple:
        return self._unit[name]

    def _reconcile(self, training_dir: Path) -> None:
        ledger = UnitLedger(training_dir, BenchmarkUnitManifest.KIND, BenchmarkUnitManifest.identity).build()

        self.unit_dirs    = {}
        self.unidentified = list(ledger.unidentified)

        for name in self._unit:
            directory = ledger.directory(self._identity(name))
            if directory is not None:
                self.unit_dirs[name] = directory.relative_to(training_dir).as_posix()

    def _log_unidentified(self) -> None:
        UnidentifiedRuns(self.stage_dir, self.unidentified, BenchmarkUnitManifest.KIND, {item: self.unit_dir(item) for item in self.items}).log(self.logger, type(self).__name__)

    def _worker_argv(self, action: str, name: str) -> list[str]:
        model, component, seed = self._identity(name)

        loss_args = ["--loss-component", component] if component is not None else []

        return ["--worker", action, "--model", model, *loss_args, *SeedSet.cli_args(seed), "--run-name", self.unit_name(name), "--run-tag", self.run_tag, "--run-dir", str(self.run_dir)]

    def _seed_command(self, action: str, name: str) -> list[str]:
        return [sys.executable, str(self.entry_script), *self._worker_argv(action, name)]

    def _job(self, item: str) -> GpuJob:
        return GpuJob(
            name     = item,
            command  = self._seed_command(self.worker_action, item),
            log_path = self.unit_dir(item) / self.worker_logname,
        )

    def _scored_splits(self) -> list[str]:
        return [self.config.inference.split] if self.config.runs_inference() else []


class SizeMatchStage(ExperimentStage):

    def __init__(self, config: BenchmarkConfig, run_tag: str, models: list[str], logger: Logger) -> None:
        super().__init__(config=config, run_tag=run_tag, logger=logger)
        self.models       = models
        self.records_path = self.run_dir / "pipeline" / "size_match.json"
        self.report_path  = self.run_dir / "pipeline" / "size_match_report.md"

    @staticmethod
    def resolve_reference(reference: str, models: list[str], heads: list[str]) -> str:
        if reference in models:
            return reference

        if "-" not in reference and len(heads) == 1:
            candidate = ModelBuilder.model_key(reference, heads[0])
            if candidate in models:
                return candidate

        return reference

    def _load_cached(self) -> dict | None:
        if not self.config.resume or not self.records_path.exists():
            return None

        records = FileIO.load_json(self.records_path)

        if not all(m in records for m in self.models):
            return None

        settings = self.config.size_match_settings()
        stale    = [m for m in self.models if records[m]["settings"] != settings]

        if stale:
            self.logger.warning(f"Cached size match of {stale} was produced under {records[stale[0]]['settings']}, current settings are {settings}; re-matching every model")
            return None

        return records

    def _reference_record(self, reference: str, target: int) -> dict:
        return asdict(SizeMatchResult(
            model         = reference,
            scale         = 1.0,
            overrides     = {},
            parameters    = target,
            target        = target,
            deviation_pct = 0.0,
            iterations    = 0,
            settings      = self.config.size_match_settings(),
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

        provenance = RunProvenance.collect()
        for record in records.values():
            record["provenance"] = provenance

        self._write_results(records, self.records_path)
        self._write_report(records, target, matcher.out_channels)

        return records


class TrainingStage(SeedExpandedStage, QueuedTrainingStage):

    def __init__(self, config: BenchmarkConfig, entry_script: Path, run_tag: str, models: list[str], logger: Logger) -> None:
        items = self._expand(config, models)
        super().__init__(config=config, entry_script=entry_script, run_tag=run_tag, items=items, logger=logger)
        self.unidentified = []

        if config.resume:
            self._reconcile(self.stage_dir)
            self._log_unidentified()


class InferenceStage(SeedExpandedStage, QueuedInferenceStage):

    def __init__(self, config: BenchmarkConfig, entry_script: Path, run_tag: str, models: list[str], logger: Logger) -> None:
        items = self._expand(config, models)
        super().__init__(config=config, entry_script=entry_script, run_tag=run_tag, items=items, logger=logger)
        self.unidentified = []

        if config.resume:
            self._reconcile(self.stage_dir)
            self._log_unidentified()


class ComparisonStage(ExperimentStage):

    def __init__(self, config, run_tag: str, logger: Logger, reference_model: str, embed_images: bool, inference_split: str, run_dir: Path | None = None, planned_units: dict[str, str] | None = None) -> None:
        super().__init__(config=config, run_tag=run_tag, logger=logger, run_dir=run_dir)
        self.reference_model = reference_model
        self.embed_images    = embed_images
        self.inference_split = inference_split
        self.planned_units   = planned_units

    def _check_coverage(self, runs: list) -> None:
        collected = [record.name for record in runs]
        scored    = [record.name for record in runs if record.has_inference]

        summary = {"Collected": len(collected), "With inference": len(scored), "Without inference": len(collected) - len(scored)}

        if self.planned_units is None:
            self.logger.kv_table(summary, title="Collection (planned units unknown: every run under the training directory is compared)")
            return

        planned_dirs = set(self.planned_units.values())
        missing      = [name for name, directory in self.planned_units.items() if directory not in collected]
        unplanned    = [name for name in collected if name not in planned_dirs]

        self.logger.kv_table({"Planned": len(self.planned_units), **summary, "Missing": len(missing), "Unplanned": len(unplanned)}, title="Collection against the plan")

        for name in missing:
            self.logger.warning(f"{name}: planned unit not collected, no run at training/{self.planned_units[name]}; absent from the reports")
        for name in unplanned:
            self.logger.warning(f"{name}: collected but planned by no unit; ranked in the reports beside the planned units")

    def run(self) -> Path:
        self.logger.section("Comparison reports")

        collector = BenchmarkSeedCollector(run_dir=self.run_dir, logger=self.logger, inference_split=self.inference_split)
        records   = collector.collect()

        self._check_coverage(collector.runs)

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
