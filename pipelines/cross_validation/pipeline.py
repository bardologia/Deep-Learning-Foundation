from __future__ import annotations

from pathlib import Path

from configuration.cross_validation import CrossValidationConfig
from pipelines.cross_validation.folds               import FoldConfigFactory, FoldPlanner
from pipelines.cross_validation.stages              import CrossValidationReportStage, FoldInferenceStage, FoldTrainingStage
from pipelines.shared.orchestration.staged_pipeline import StagedPipeline


class CrossValidationPipeline(StagedPipeline):
    LOGGER_NAME = "cross_validation_pipeline"

    def __init__(self, config: CrossValidationConfig, entry_script: Path) -> None:
        super().__init__(config, entry_script)
        self.factory = FoldConfigFactory(config)

    def _run_training(self) -> list[dict]:
        stage   = FoldTrainingStage(config=self.config, entry_script=self.entry_script, run_tag=self.run_tag, logger=self.logger)
        results = stage.run()

        failed = [r for r in results if r["status"] != "DONE"]
        self._mark_stage("training", "completed" if not failed else "partial")
        return results

    def _run_inference(self, planner: FoldPlanner) -> list[dict]:
        if not self.config.runs_inference():
            self.logger.subsection(f"Inference skipped for training_type '{self.config.training_type}'; folds evaluated by reconstruction loss")
            self._mark_stage("inference", "skipped")
            return []

        stage   = FoldInferenceStage(config=self.config, entry_script=self.entry_script, run_tag=self.run_tag, planner=planner, logger=self.logger)
        results = stage.run()

        failed = [r for r in results if r["status"] == "FAILED"]
        self._mark_stage("inference", "completed" if not failed else "partial")
        return results

    def _run_reports(self, planner: FoldPlanner) -> Path:
        stage   = CrossValidationReportStage(config=self.config, run_tag=self.run_tag, planner=planner, logger=self.logger)
        out_dir = stage.run()

        self._mark_stage("reports", "completed")
        return out_dir

    def run(self) -> None:
        planner = self.factory.planner()

        self.logger.section("Cross-validation pipeline")
        self.logger.kv_table({
            "Run tag"          : self.run_tag,
            "Model"            : self.config.backbone_name,
            "Folds"            : self.config.folds.n_folds,
            "Azimuth extent"   : f"[{self.config.folds.azimuth_start}, {self.config.folds.azimuth_end})",
            "Inference splits" : ", ".join(self.config.inference_splits),
            "GPUs"             : self.config.gpus,
            "Resume"           : self.config.resume,
            "Run dir"          : str(self.run_dir),
        }, title="Configuration")

        self.logger.subsection("Fold plan")
        rows = []
        for plan in planner.plans():
            train_regions = plan.split_regions.regions("train")
            rows.append({
                "Fold"  : plan.fold_index,
                "Test"  : f"[{plan.split_regions.regions('test')[0].azimuth_start}, {plan.split_regions.regions('test')[0].azimuth_end})",
                "Val"   : f"[{plan.split_regions.regions('val')[0].azimuth_start}, {plan.split_regions.regions('val')[0].azimuth_end})",
                "Train" : ", ".join(f"[{r.azimuth_start}, {r.azimuth_end})" for r in train_regions),
            })
        self.logger.metrics_table(rows, ["Fold", "Test", "Val", "Train"])

        try:
            training_results  = self._run_training()
            inference_results = self._run_inference(planner)
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
