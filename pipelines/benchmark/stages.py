
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib     import Path

from configuration.benchmark                       import BenchmarkConfig
from configuration.training.general.loss           import LossKind
from pipelines.benchmark.results                   import BenchmarkSeedCollector
from pipelines.shared.comparison.comparison_report import ComparisonReport
from pipelines.shared.orchestration.unit_manifest  import UnidentifiedRuns, UnitLedger, UnitManifest
from pipelines.shared.training.run_naming          import RunNaming
from pipelines.shared.training.seed_sweep          import SeedSet
from tools.orchestration.gpu_queue                 import GpuJob
from tools.orchestration.stages                    import ExperimentStage, QueuedInferenceStage, QueuedTrainingStage
from tools.monitoring.logger                       import Logger
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

    @classmethod
    def components(cls, config: BenchmarkConfig) -> list[str]:
        return list(config.sweep_loss_kinds)

    @classmethod
    def unit_base(cls, config: BenchmarkConfig, model: str, component: str) -> str:
        return RunNaming.tag(model, replace(config.loss, kind=LossKind(component)), config.augmentation)

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

    def __init__(self, config, run_tag: str, logger: Logger, embed_images: bool, inference_split: str, run_dir: Path | None = None, planned_units: dict[str, str] | None = None) -> None:
        super().__init__(config=config, run_tag=run_tag, logger=logger, run_dir=run_dir)
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
            embed_images    = self.embed_images,
            logger          = self.logger,
            seed_dispersion = collector.seed_dispersion,
        )

        written = report.write_all()

        self.logger.subsection("Reports written")
        for path in written:
            self.logger.info(f"{path}")

        return out_dir
