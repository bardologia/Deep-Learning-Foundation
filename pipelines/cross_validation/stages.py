"""Stages of the cross-validation pipeline.

Holds the fold x seed unit expansion shared by the local and cluster paths, the
queued training and inference stages, and the report stage that aggregates the
finished folds.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from configuration.cross_validation import CrossValidationConfig
from pipelines.cross_validation.cv_report import CrossValidationReport
from pipelines.cross_validation.folds     import FoldFamily, FoldPlanner, FoldRunNaming
from pipelines.cross_validation.workers   import FoldCollector
from pipelines.shared.training.seed_sweep import SeedSet
from tools.orchestration.gpu_queue import GpuJob
from tools.orchestration.stages    import ExperimentStage, QueuedInferenceStage, QueuedTrainingStage
from tools.monitoring.logger       import Logger
from tools.runtime.completion      import CompletionMarker
from tools.runtime.run_tag         import RunTag


class FoldUnitStage:
    """Unit expansion and worker command construction shared by fold stages."""

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

    def _worker_argv(self, action: str, name: str) -> list[str]:
        """Returns the worker argv running one action on one unit.

        Args:
            action : Worker action, 'train' or 'infer'.
            name   : Run name of the unit.

        Returns:
            Argument list identifying the fold, seed, and sweep run directory.
        """
        fold_index, seed = self._unit[name]

        return ["--worker", action, "--fold", str(fold_index), *SeedSet.cli_args(seed), "--run-tag", self.run_tag, "--run-dir", str(self.run_dir)]

    def _job(self, item: str) -> GpuJob:
        """Returns the GPU-queued job that runs one unit through the entry script."""
        return GpuJob(
            name     = item,
            command  = [sys.executable, str(self.entry_script), *self._worker_argv(self.worker_action, item)],
            log_path = self.stage_dir / item / self.worker_logname,
        )


class FoldTrainingStage(FoldUnitStage, QueuedTrainingStage):
    """Trains every fold x seed unit through the shared GPU queue."""

    stage_subdir = "folds"

    def __init__(self, config: CrossValidationConfig, entry_script: Path, run_tag: str, logger: Logger) -> None:
        """Expands the units and configures the queued training stage.

        Args:
            config       : Cross-validation configuration.
            entry_script : Entry point re-invoked for each unit.
            run_tag      : Tag of the sweep run directory.
            logger       : Logger used for progress reporting.
        """
        items = self._expand(config)
        super().__init__(config=config, entry_script=entry_script, run_tag=run_tag, items=items, logger=logger)

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
        """Expands the units and configures the queued inference stage.

        Args:
            config       : Cross-validation configuration.
            entry_script : Entry point re-invoked for each unit.
            run_tag      : Tag of the sweep run directory.
            logger       : Logger used for progress reporting.
        """
        items = self._expand(config)
        super().__init__(config=config, entry_script=entry_script, run_tag=run_tag, items=items, logger=logger)
        self.splits = config.inference_splits

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

        run_dir = self.stage_dir / item

        return all(FoldFamily.split_inference_dir(run_dir, self.config.training_type, split) is not None for split in self.splits)

    def _purge_unfinished(self, item: str) -> None:
        """Deletes a unit's inference directories that carry no completion marker."""
        base = FoldFamily.inference_base(self.stage_dir / item, self.config.training_type)
        if not self.config.resume or not base.is_dir():
            return

        for candidate in base.iterdir():
            if candidate.is_dir() and not CompletionMarker.is_complete(candidate):
                self.logger.warning(f"{item}: unfinished inference '{candidate.name}' deleted before rerun")
                shutil.rmtree(candidate)


class CrossValidationReportStage(ExperimentStage):
    """Collects the finished folds and writes the cross-fold reports.

    Attributes:
        planner : Planner describing the fold blocks, embedded in the report.
    """

    def __init__(self, config: CrossValidationConfig, run_tag: str, planner: FoldPlanner, logger: Logger) -> None:
        """Binds the stage to the sweep run and its fold plan.

        Args:
            config  : Cross-validation configuration.
            run_tag : Tag of the sweep run directory.
            planner : Planner describing the fold blocks.
            logger  : Logger used for progress reporting.
        """
        super().__init__(config=config, run_tag=run_tag, logger=logger)
        self.planner = planner

    def run(self) -> Path:
        """Writes the reports into a timestamped directory under the sweep run.

        Returns:
            Directory the reports were written to.
        """
        self.logger.section("Cross-validation reports")

        splits     = self.config.inference_splits if self.config.runs_inference() else []
        model_name = FoldFamily.model_label(self.config)

        collector = FoldCollector(run_dir=self.run_dir, splits=splits, training_type=self.config.training_type, logger=self.logger)
        base_records, records_by_split = collector.collect_by_split()

        out_dir = self.run_dir / "reports" / RunTag.now()

        report = CrossValidationReport(
            base_records     = base_records,
            records_by_split = records_by_split,
            planner          = self.planner,
            out_dir          = out_dir,
            model_name       = model_name,
            embed_images     = self.config.comparison.embed_images,
            logger           = self.logger,
            seed_dispersion  = collector.seed_dispersion,
        )

        written = report.write_all()

        self.logger.subsection("Reports written")
        for path in written:
            self.logger.info(f"{path}")

        return out_dir
