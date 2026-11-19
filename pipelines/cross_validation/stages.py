
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from configuration.cross_validation import CrossValidationConfig
from pipelines.cross_validation.cv_report import CrossValidationReport
from pipelines.cross_validation.folds             import FoldPlanner, FoldRunNaming, FoldTreeSurvey, FoldUnitManifest
from pipelines.cross_validation.workers           import FoldCollector
from pipelines.shared.inference.metadata          import CompletedInference
from pipelines.shared.inference.run_classifier    import InferenceLayout
from pipelines.shared.orchestration.unit_manifest import UnidentifiedRuns, UnitLedger
from pipelines.shared.training.seed_sweep         import SeedSet
from tools.orchestration.gpu_queue import GpuJob
from tools.orchestration.stages    import ExperimentStage, QueuedInferenceStage, QueuedTrainingStage
from tools.monitoring.logger       import Logger
from tools.runtime.completion      import CompletionMarker
from tools.runtime.run_tag         import RunTag


class FoldUnitStage:

    def _expand(self, config: CrossValidationConfig) -> list[str]:
        units      = FoldRunNaming(config).units()
        self._unit = {run_name: (fold_index, seed) for fold_index, seed, run_name in units}

        return [run_name for _, _, run_name in units]

    def _reconcile(self, folds_dir: Path) -> None:
        ledger = UnitLedger(folds_dir, FoldUnitManifest.KIND, FoldUnitManifest.identity).build()

        self.unit_dirs    = {}
        self.unidentified = list(ledger.unidentified)

        for name, identity in self._unit.items():
            directory = ledger.directory(identity)
            if directory is not None:
                self.unit_dirs[name] = directory.relative_to(folds_dir).as_posix()

    def _log_unidentified(self) -> None:
        UnidentifiedRuns(self.stage_dir, self.unidentified, FoldUnitManifest.KIND, {item: self.unit_dir(item) for item in self.items}).log(self.logger, type(self).__name__)

    def _worker_argv(self, action: str, name: str) -> list[str]:
        fold_index, seed = self._unit[name]

        return ["--worker", action, "--fold", str(fold_index), *SeedSet.cli_args(seed), "--run-name", self.unit_name(name), "--run-tag", self.run_tag, "--run-dir", str(self.run_dir)]

    def _job(self, item: str) -> GpuJob:
        return GpuJob(
            name     = item,
            command  = [sys.executable, str(self.entry_script), *self._worker_argv(self.worker_action, item)],
            log_path = self.unit_dir(item) / self.worker_logname,
        )

    def _scored_splits(self) -> list[str]:
        return list(self.config.inference_splits) if self.config.runs_inference() else []

    def _inference_base(self, directory: Path) -> Path:
        return InferenceLayout.base(directory)


class FoldTrainingStage(FoldUnitStage, QueuedTrainingStage):

    stage_subdir = "folds"

    def __init__(self, config: CrossValidationConfig, entry_script: Path, run_tag: str, logger: Logger) -> None:
        items = self._expand(config)
        super().__init__(config=config, entry_script=entry_script, run_tag=run_tag, items=items, logger=logger)
        self.unidentified = []

        if config.resume:
            self._reconcile(self.stage_dir)
            self._log_unidentified()

    def _config_kv(self) -> dict:
        return {
            "Model"      : self.config.backbone_name,
            "Folds"      : self.config.folds.n_folds,
            "Seeds"      : self.config.seeds or "—",
            "Runs"       : len(self.items),
            "Epochs"     : self.config.training.epochs,
            "Batch size" : self.config.training.batch_size,
            "GPUs"       : self.config.gpus,
            "Folds dir"  : str(self.stage_dir),
        }


class FoldInferenceStage(FoldUnitStage, QueuedInferenceStage):

    stage_subdir = "folds"

    def __init__(self, config: CrossValidationConfig, entry_script: Path, run_tag: str, logger: Logger) -> None:
        items = self._expand(config)
        super().__init__(config=config, entry_script=entry_script, run_tag=run_tag, items=items, logger=logger)
        self.splits       = config.inference_splits
        self.unidentified = []

        if config.resume:
            self._reconcile(self.stage_dir)
            self._log_unidentified()

    def _config_kv(self) -> dict:
        return {
            "Folds"  : self.config.folds.n_folds,
            "Seeds"  : self.config.seeds or "—",
            "Runs"   : len(self.items),
            "Splits" : ", ".join(self.splits),
            "GPUs"   : self.config.gpus,
        }

    def _has_inference(self, item: str) -> bool:
        if not self.config.resume:
            return False

        run_dir = self.unit_dir(item)

        return all(CompletedInference.latest(InferenceLayout.base(run_dir), split) is not None for split in self.splits)

    def _purge_unfinished(self, item: str) -> None:
        base = InferenceLayout.base(self.unit_dir(item))
        if not self.config.resume or not base.is_dir():
            return

        for candidate in base.iterdir():
            if candidate.is_dir() and not CompletionMarker.is_complete(candidate):
                self.logger.warning(f"{item}: unfinished inference '{candidate.name}' deleted before rerun")
                shutil.rmtree(candidate)


class FoldReportWriter:

    def __init__(self, run_dir: Path, splits: list[str], model_name: str, planner, embed_images: bool, logger: Logger) -> None:
        self.run_dir      = Path(run_dir)
        self.splits       = list(splits)
        self.model_name   = model_name
        self.planner      = planner
        self.embed_images = embed_images
        self.logger       = logger

    def write(self) -> Path:
        collector = FoldCollector(run_dir=self.run_dir, splits=self.splits, logger=self.logger)
        base_records, records_by_split = collector.collect_by_split()

        out_dir = self.run_dir / "reports" / RunTag.now()

        report = CrossValidationReport(
            base_records     = base_records,
            records_by_split = records_by_split,
            planner          = self.planner,
            out_dir          = out_dir,
            model_name       = self.model_name,
            embed_images     = self.embed_images,
            logger           = self.logger,
            seed_dispersion  = collector.seed_dispersion,
        )

        written = report.write_all()

        self.logger.subsection("Reports written")
        for path in written:
            self.logger.info(f"{path}")

        return out_dir


class CrossValidationReportStage(ExperimentStage):

    def __init__(self, config: CrossValidationConfig, run_tag: str, planner: FoldPlanner, logger: Logger, run_dir: Path | None = None) -> None:
        super().__init__(config=config, run_tag=run_tag, logger=logger, run_dir=run_dir)
        self.planner = planner

    def run(self) -> Path:
        self.logger.section("Cross-validation reports")

        splits = self.config.inference_splits if self.config.runs_inference() else []
        writer = FoldReportWriter(run_dir=self.run_dir, splits=splits, model_name=self.config.backbone_name, planner=self.planner, embed_images=self.config.comparison.embed_images, logger=self.logger)

        return writer.write()


class CrossValidationReportRebuilder:

    def __init__(self, run_dir: Path, logger: Logger) -> None:
        self.run_dir = Path(run_dir)
        self.logger  = logger
        self.survey  = FoldTreeSurvey(self.run_dir)

    def rebuild(self) -> Path:
        fold_runs  = self.survey.fold_runs()
        first      = fold_runs[min(fold_runs)]
        model_name = self.survey.model_name(first)
        splits     = self.survey.splits(fold_runs)
        planner    = self.survey.planner(fold_runs)

        self.logger.section("Cross-validation reports")
        self.logger.info(f"Rebuilding the cross-fold report of {self.run_dir} from its fold tree: {len(fold_runs)} fold(s) of '{model_name}', scored splits {', '.join(splits) or 'none'}")

        writer = FoldReportWriter(run_dir=self.run_dir, splits=splits, model_name=model_name, planner=planner, embed_images=False, logger=self.logger)

        return writer.write()
