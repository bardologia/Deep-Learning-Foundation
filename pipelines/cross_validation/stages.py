"""Stages of the cross-validation pipeline.

Holds the fold x seed unit expansion shared by the local and cluster paths, the
reconciliation of planned units against the fold run directories through their
manifests, the queued training and inference stages, the report writer and
stage that aggregate the finished folds, and the rebuilder that writes that
report for an existing run from its fold tree alone.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from configuration.cross_validation import CrossValidationConfig
from pipelines.cross_validation.cv_report import CrossValidationReport
from pipelines.cross_validation.folds             import FoldFamily, FoldPlanner, FoldRunNaming, FoldTreeSurvey, FoldUnitManifest
from pipelines.cross_validation.workers           import FoldCollector
from pipelines.shared.orchestration.unit_manifest import UnitLedger
from pipelines.shared.training.seed_sweep         import SeedSet
from tools.orchestration.gpu_queue import GpuJob
from tools.orchestration.stages    import ExperimentStage, QueuedInferenceStage, QueuedTrainingStage
from tools.monitoring.logger       import Logger
from tools.runtime.completion      import CompletionMarker
from tools.runtime.run_tag         import RunTag


class FoldUnitStage:
    """Unit expansion, on-disk reconciliation and worker command construction shared by fold stages.

    Attributes:
        unit_dirs    : Existing directory per unit name, relative to the folds directory, for the units a resume found through their manifests.
        unidentified : Run directories under the folds directory that carry no cross-validation manifest and are left untouched.
    """

    def _expand(self, config: CrossValidationConfig) -> list[str]:
        """Expands the sweep into fold x seed units and caches their coordinates.

        Args:
            config : Cross-validation configuration.

        Returns:
            Run name of every unit, in launch order.
        """
        units      = FoldRunNaming(config).units()
        self._unit = {run_name: (fold_index, seed) for fold_index, seed, run_name in units}

        return [run_name for _, _, run_name in units]

    def _reconcile(self, folds_dir: Path) -> None:
        """Maps every planned unit onto the run directory its manifest identifies, when one exists.

        Args:
            folds_dir : Directory holding the fold run directories.
        """
        ledger = UnitLedger(folds_dir, FoldUnitManifest.KIND, FoldUnitManifest.identity).build()

        self.unit_dirs    = {}
        self.unidentified = list(ledger.unidentified)

        for name, identity in self._unit.items():
            directory = ledger.directory(identity)
            if directory is not None:
                self.unit_dirs[name] = directory.relative_to(folds_dir).as_posix()

    def _worker_argv(self, action: str, name: str) -> list[str]:
        """Returns the worker argv running one action on one unit.

        Args:
            action : Worker action, 'train' or 'infer'.
            name   : Run name of the unit.

        Returns:
            Argument list identifying the fold, seed, run directory and the sweep run directory.
        """
        fold_index, seed = self._unit[name]

        return ["--worker", action, "--fold", str(fold_index), *SeedSet.cli_args(seed), "--run-name", self.unit_name(name), "--run-tag", self.run_tag, "--run-dir", str(self.run_dir)]

    def _job(self, item: str) -> GpuJob:
        """Returns the GPU-queued job that runs one unit through the entry script."""
        return GpuJob(
            name     = item,
            command  = [sys.executable, str(self.entry_script), *self._worker_argv(self.worker_action, item)],
            log_path = self.unit_dir(item) / self.worker_logname,
        )

    def _scored_splits(self) -> list[str]:
        """Returns the splits every fold scores, empty when the run compares folds on their checkpoints alone."""
        return list(self.config.inference_splits) if self.config.runs_inference() else []

    def _inference_base(self, directory: Path) -> Path:
        """Returns the directory a fold run's inference stamps live under for the configured family."""
        return FoldFamily.inference_base(directory, self.config.training_type)


class FoldTrainingStage(FoldUnitStage, QueuedTrainingStage):
    """Trains every fold x seed unit through the shared GPU queue."""

    stage_subdir = "folds"

    def __init__(self, config: CrossValidationConfig, entry_script: Path, run_tag: str, logger: Logger) -> None:
        """Expands the units, reconciles them against the folds directory and configures the queued training stage.

        Args:
            config       : Cross-validation configuration.
            entry_script : Entry point re-invoked for each unit.
            run_tag      : Tag of the sweep run directory.
            logger       : Logger used for progress reporting.
        """
        items = self._expand(config)
        super().__init__(config=config, entry_script=entry_script, run_tag=run_tag, items=items, logger=logger)

        if config.resume:
            self._reconcile(self.stage_dir)

    def _config_kv(self) -> dict:
        """Returns the stage configuration summary logged before launching."""
        return {
            "Model"      : FoldFamily.model_label(self.config),
            "Folds"      : self.config.folds.n_folds,
            "Seeds"      : self.config.seeds or "—",
            "Runs"       : len(self.items),
            "Epochs"     : self.config.training.epochs,
            "Batch size" : self.config.training.batch_size,
            "GPUs"       : self.config.gpus,
            "Folds dir"  : str(self.stage_dir),
        }


class FoldInferenceStage(FoldUnitStage, QueuedInferenceStage):
    """Runs inference for every unit on each configured split.

    Attributes:
        splits : Dataset splits inference is run on.
    """

    stage_subdir = "folds"

    def __init__(self, config: CrossValidationConfig, entry_script: Path, run_tag: str, logger: Logger) -> None:
        """Expands the units, reconciles them against the folds directory and configures the queued inference stage.

        Args:
            config       : Cross-validation configuration.
            entry_script : Entry point re-invoked for each unit.
            run_tag      : Tag of the sweep run directory.
            logger       : Logger used for progress reporting.
        """
        items = self._expand(config)
        super().__init__(config=config, entry_script=entry_script, run_tag=run_tag, items=items, logger=logger)
        self.splits = config.inference_splits

        if config.resume:
            self._reconcile(self.stage_dir)

    def _config_kv(self) -> dict:
        """Returns the stage configuration summary logged before launching."""
        return {
            "Folds"  : self.config.folds.n_folds,
            "Seeds"  : self.config.seeds or "—",
            "Runs"   : len(self.items),
            "Splits" : ", ".join(self.splits),
            "GPUs"   : self.config.gpus,
        }

    def _has_inference(self, item: str) -> bool:
        """Returns whether a unit already holds completed inference on every split."""
        if not self.config.resume:
            return False

        run_dir = self.unit_dir(item)

        return all(FoldFamily.split_inference_dir(run_dir, self.config.training_type, split) is not None for split in self.splits)

    def _purge_unfinished(self, item: str) -> None:
        """Deletes a unit's inference directories that carry no completion marker."""
        base = FoldFamily.inference_base(self.unit_dir(item), self.config.training_type)
        if not self.config.resume or not base.is_dir():
            return

        for candidate in base.iterdir():
            if candidate.is_dir() and not CompletionMarker.is_complete(candidate):
                self.logger.warning(f"{item}: unfinished inference '{candidate.name}' deleted before rerun")
                shutil.rmtree(candidate)


class FoldReportWriter:
    """Collects a run's folds and writes the cross-fold reports into a new stamp.

    Shared by the pipeline's report stage, which describes the run from its
    configuration, and the rebuilder, which describes it from the fold tree.

    Attributes:
        run_dir       : Cross-validation run directory holding the folds directory.
        splits        : Dataset splits whose fold inference metrics are reported, empty to report training alone.
        training_type : Training family the folds were trained in, deciding where their inference lives.
        model_name    : Label of the model every fold trained.
        planner       : Fold plan embedded in the report, from the configuration or from the stored regions.
        embed_images  : Whether the per-split comparison pages inline their figures.
        logger        : Logger used for progress reporting.
    """

    def __init__(self, run_dir: Path, splits: list[str], training_type: str, model_name: str, planner, embed_images: bool, logger: Logger) -> None:
        """Binds the writer to the run and the description of its sweep."""
        self.run_dir       = Path(run_dir)
        self.splits        = list(splits)
        self.training_type = training_type
        self.model_name    = model_name
        self.planner       = planner
        self.embed_images  = embed_images
        self.logger        = logger

    def write(self) -> Path:
        """Collects the folds and writes the reports into a timestamped directory under the run.

        Returns:
            Directory the reports were written to.

        Raises:
            ValueError : If a fold is missing, has no checkpoint data, or lacks metrics on a reported split.
        """
        collector = FoldCollector(run_dir=self.run_dir, splits=self.splits, training_type=self.training_type, logger=self.logger)
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
    """Collects the finished folds and writes the cross-fold reports.

    Attributes:
        planner : Planner describing the fold blocks, embedded in the report.
    """

    def __init__(self, config: CrossValidationConfig, run_tag: str, planner: FoldPlanner, logger: Logger, run_dir: Path | None = None) -> None:
        """Binds the stage to the sweep run and its fold plan.

        Args:
            config  : Cross-validation configuration.
            run_tag : Tag of the sweep run directory.
            planner : Planner describing the fold blocks.
            logger  : Logger used for progress reporting.
            run_dir : Explicit run directory, or None to derive it from the configured log base and the run tag.
        """
        super().__init__(config=config, run_tag=run_tag, logger=logger, run_dir=run_dir)
        self.planner = planner

    def run(self) -> Path:
        """Writes the reports into a timestamped directory under the sweep run.

        Returns:
            Directory the reports were written to.
        """
        self.logger.section("Cross-validation reports")

        splits = self.config.inference_splits if self.config.runs_inference() else []
        writer = FoldReportWriter(run_dir=self.run_dir, splits=splits, training_type=self.config.training_type, model_name=FoldFamily.model_label(self.config), planner=self.planner, embed_images=self.config.comparison.embed_images, logger=self.logger)

        return writer.write()


class CrossValidationReportRebuilder:
    """Writes the cross-fold report of an existing run from its fold tree alone.

    Nothing beyond the folds directory is needed: the family, the model name,
    the split regions and the scored splits are read off the fold runs, so a
    run whose pipeline directory was never written or never copied, or that
    sits in a sequence workspace, is reported in place. The writer aborts as the
    pipeline's own report stage does when a fold has no checkpoint or no
    metrics on a scored split. Figures are linked, not embedded.

    Attributes:
        run_dir : Cross-validation run directory holding the folds directory.
        logger  : Logger used for progress reporting.
        survey  : Reader of the fold tree.
    """

    def __init__(self, run_dir: Path, logger: Logger) -> None:
        """Binds the rebuilder to a run directory and a logger."""
        self.run_dir = Path(run_dir)
        self.logger  = logger
        self.survey  = FoldTreeSurvey(self.run_dir)

    def rebuild(self) -> Path:
        """Writes a fresh report stamp for the run and returns its directory."""
        fold_runs     = self.survey.fold_runs()
        first         = fold_runs[min(fold_runs)]
        training_type = self.survey.training_type(first)
        model_name    = self.survey.model_name(first)
        splits        = self.survey.splits(fold_runs, training_type)
        planner       = self.survey.planner(fold_runs)

        self.logger.section("Cross-validation reports")
        self.logger.info(f"Rebuilding the cross-fold report of {self.run_dir} from its fold tree: {len(fold_runs)} fold(s) of {training_type} '{model_name}', scored splits {', '.join(splits) or 'none'}")

        writer = FoldReportWriter(run_dir=self.run_dir, splits=splits, training_type=training_type, model_name=model_name, planner=planner, embed_images=False, logger=self.logger)

        return writer.write()
