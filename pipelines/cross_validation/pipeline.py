
from __future__ import annotations

from pathlib import Path

from configuration.cross_validation import CrossValidationConfig
from pipelines.cross_validation.folds               import FoldConfigFactory, FoldPlanner
from pipelines.cross_validation.stages              import CrossValidationReportStage, FoldInferenceStage, FoldTrainingStage
from pipelines.shared.orchestration.staged_pipeline import StagedPipeline


class CrossValidationPipeline(StagedPipeline):

    LOGGER_NAME = "cross_validation_pipeline"
    STAGES      = ("full", "prepare", "compare")

    def __init__(self, config: CrossValidationConfig, entry_script: Path, resumed=None, inherited: dict | None = None) -> None:
        super().__init__(config, entry_script, resumed=resumed)
        self.factory   = FoldConfigFactory(config)
        self.inherited = dict(inherited or {})

    @classmethod
    def validate_sweep(cls, config: CrossValidationConfig) -> None:
        if config.stage not in cls.STAGES:
            raise SystemExit(f"unknown cross-validation stage '{config.stage}'; valid: {', '.join(cls.STAGES)}")

        unknown = [split for split in config.inference_splits if split not in FoldPlanner.INFERENCE_SPLITS]
        if unknown:
            raise SystemExit(f"unknown inference split(s) {unknown}; each fold only holds an inference region for {', '.join(FoldPlanner.INFERENCE_SPLITS)}")

    def _validate_sweep(self) -> None:
        self._begin_stage("validation")
        self.validate_sweep(self.config)

    def _log_inherited(self) -> None:
        if not self.inherited:
            return

        self.logger.kv_table({f"curriculum.warmup.{name}": repr(value) for name, value in self.inherited.items()}, title=f"{len(self.inherited)} warmup field(s) inherited from the complete stage")

    def _log_unit_status(self) -> None:
        self._begin_stage("prepare")

        stage = FoldTrainingStage(config=self.config, entry_script=self.entry_script, run_tag=self.run_tag, logger=self.logger)

        stage.status_report().log(self.logger, "Unit status")

    def _run_training(self) -> list[dict]:
        self._begin_stage("training")

        stage   = FoldTrainingStage(config=self.config, entry_script=self.entry_script, run_tag=self.run_tag, logger=self.logger)
        results = stage.run()

        failed = [r for r in results if r["status"] != "DONE"]
        self._mark_stage("training", "completed" if not failed else "partial")
        return results

    def _run_inference(self) -> list[dict]:
        self._begin_stage("inference")

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
        self._begin_stage("reports")

        stage   = CrossValidationReportStage(config=self.config, run_tag=self.run_tag, planner=planner, logger=self.logger)
        out_dir = stage.run()

        self._mark_stage("reports", "completed")
        return out_dir

    def _execute(self) -> None:
        self._validate_sweep()

        planner = self.factory.planner()

        self.logger.section("Cross-validation pipeline")
        self.logger.kv_table({
            "Run tag"          : self.run_tag,
            "Model"            : self.config.backbone_name,
            "Folds"            : self.config.folds.n_folds,
            "Row extent"       : f"[{self.config.folds.row_start}, {self.config.folds.row_end})",
            "Inference splits" : ", ".join(self.config.inference_splits),
            "GPUs"             : self.config.gpus,
            "Resume"           : self.config.resume,
            "Run dir"          : str(self.run_dir),
        }, title="Configuration")

        self._log_launch()
        self._log_inherited()

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

        self._finish(training_results, inference_results, {"Folds": self.config.folds.n_folds, "Reports": str(reports_dir)})
