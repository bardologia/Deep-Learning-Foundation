"""Staged orchestration of a k-fold cross-validation experiment.

Sequences fold training, per-split inference, and the cross-fold report, and
supports running only the prepare or compare stage when the units themselves
are dispatched to a cluster.
"""

from __future__ import annotations

from pathlib import Path

from configuration.cross_validation import CrossValidationConfig
from pipelines.cross_validation.folds               import FoldConfigFactory, FoldFamily, FoldPlanner
from pipelines.cross_validation.stages              import CrossValidationReportStage, FoldInferenceStage, FoldTrainingStage
from pipelines.shared.orchestration.staged_pipeline import StagedPipeline


class CrossValidationPipeline(StagedPipeline):
    """Runs the training, inference, and reporting stages of cross-validation.

    Attributes:
        factory : Builds the fold planner and per-fold configs.
    """

    LOGGER_NAME = "cross_validation_pipeline"
    STAGES      = ("full", "prepare", "compare")

    def __init__(self, config: CrossValidationConfig, entry_script: Path, resumed=None) -> None:
        """Binds the pipeline to its configuration and entry script.

        Args:
            config       : Cross-validation configuration.
            entry_script : Entry point the stages re-invoke for each unit.
            resumed      : Resumed-run description logged when the pipeline starts, or None for a fresh run.
        """
        super().__init__(config, entry_script, resumed=resumed)
        self.factory = FoldConfigFactory(config)

    @classmethod
    def validate_sweep(cls, config: CrossValidationConfig) -> None:
        """Checks the training family, stage, and inference splits are valid.

        Args:
            config : Cross-validation configuration to validate.

        Raises:
            ValueError : If the training family is unsupported.
            SystemExit : If the stage is unknown or a requested inference split has no fold-specific region.
        """
        FoldFamily.validate(config.training_type)

        if config.stage not in cls.STAGES:
            raise SystemExit(f"unknown cross-validation stage '{config.stage}'; valid: {', '.join(cls.STAGES)}")

        unknown = [split for split in config.inference_splits if split not in FoldPlanner.INFERENCE_SPLITS]
        if unknown:
            raise SystemExit(f"unknown inference split(s) {unknown}; each fold only holds an inference region for {', '.join(FoldPlanner.INFERENCE_SPLITS)}")

    def _validate_sweep(self) -> None:
        """Validates this pipeline's own configuration."""
        self.validate_sweep(self.config)

    def _log_unit_status(self) -> None:
        """Logs what every fold x seed unit already holds on disk, and which directories no unit claims.

        A full run gets the same table from its training stage; this is the
        prepare stage's version, so a resume can be inspected before anything
        is launched.
        """
        stage = FoldTrainingStage(config=self.config, entry_script=self.entry_script, run_tag=self.run_tag, logger=self.logger)

        stage.status_report().log(self.logger, "Unit status")

        for directory in stage.unidentified:
            self.logger.warning(f"{directory.relative_to(stage.stage_dir)}: run directory without a cross-validation manifest, left untouched")

    def _run_training(self) -> list[dict]:
        """Trains every fold x seed unit and records the stage status.

        Returns:
            One result dictionary per unit, each carrying a `status` key.
        """
        stage   = FoldTrainingStage(config=self.config, entry_script=self.entry_script, run_tag=self.run_tag, logger=self.logger)
        results = stage.run()

        failed = [r for r in results if r["status"] != "DONE"]
        self._mark_stage("training", "completed" if not failed else "partial")
        return results

    def _run_inference(self) -> list[dict]:
        """Runs inference on every unit and split, or skips when none are configured.

        Returns:
            One result dictionary per unit, empty when inference is skipped.
        """
        if not self.config.runs_inference():
            self.logger.subsection("Inference skipped: inference_splits is empty; folds are compared on their training checkpoints alone")
            self._mark_stage("inference", "skipped")
            return []

        stage   = FoldInferenceStage(config=self.config, entry_script=self.entry_script, run_tag=self.run_tag, logger=self.logger)
        results = stage.run()

        failed = [r for r in results if r["status"] == "FAILED"]
        self._mark_stage("inference", "completed" if not failed else "partial")
        return results

    def _run_reports(self, planner: FoldPlanner) -> Path:
        """Collects the fold runs and writes the cross-fold reports.

        Args:
            planner : Planner describing the fold blocks, embedded in the report.

        Returns:
            Directory the reports were written to.
        """
        stage   = CrossValidationReportStage(config=self.config, run_tag=self.run_tag, planner=planner, logger=self.logger)
        out_dir = stage.run()

        self._mark_stage("reports", "completed")
        return out_dir

    def run(self) -> None:
        """Runs the configured stage of the cross-validation experiment.

        The 'prepare' stage validates and logs the fold plan and the status
        of every unit on disk, 'compare' writes the reports from
        already-finished folds, and 'full' runs training, inference, and
        reporting in sequence.
        """
        try:
            self._validate_sweep()

            planner = self.factory.planner()

            self.logger.section("Cross-validation pipeline")
            self.logger.kv_table({
                "Run tag"          : self.run_tag,
                "Training type"    : self.config.training_type,
                "Model"            : FoldFamily.model_label(self.config),
                "Folds"            : self.config.folds.n_folds,
                "Row extent"       : f"[{self.config.folds.row_start}, {self.config.folds.row_end})",
                "Inference splits" : ", ".join(self.config.inference_splits),
                "GPUs"             : self.config.gpus,
                "Resume"           : self.config.resume,
                "Run dir"          : str(self.run_dir),
            }, title="Configuration")

            self._log_launch()

            if self.config.stage == "compare":
                reports_dir = self._run_reports(planner)
                self._mark_stage("pipeline", "completed")
                self.logger.info(f"Report stage done: {reports_dir}")
                return

            self.logger.subsection("Fold plan")
            rows = []
            for plan in planner.plans():
                train_regions = plan.split_regions.regions("train")
                rows.append({
                    "Fold"  : plan.fold_index,
                    "Test"  : f"[{plan.split_regions.regions('test')[0].row_start}, {plan.split_regions.regions('test')[0].row_end})",
                    "Val"   : f"[{plan.split_regions.regions('val')[0].row_start}, {plan.split_regions.regions('val')[0].row_end})",
                    "Train" : ", ".join(f"[{r.row_start}, {r.row_end})" for r in train_regions),
                })
            self.logger.metrics_table(rows, ["Fold", "Test", "Val", "Train"])

            if self.config.stage == "prepare":
                self._log_unit_status()
                self._mark_stage("prepare", "completed")
                self.logger.info(f"Prepare stage done: fold plan validated, unit status listed and resolved config written to {self.pipeline_dir}")
                return

            training_results  = self._run_training()
            inference_results = self._run_inference()
            reports_dir       = self._run_reports(planner)

            self._mark_stage("pipeline", "completed")

            self.logger.section("Pipeline summary")
            self.logger.kv_table({
                "Folds"         : self.config.folds.n_folds,
                "Trained"       : sum(1 for r in training_results if r["status"] == "DONE"),
                "Inferred"      : sum(1 for r in inference_results if r["status"] == "DONE"),
                "Reports"       : str(reports_dir),
                "Pipeline logs" : str(self.pipeline_dir),
            }, title="Done")
        finally:
            self.logger.close()
