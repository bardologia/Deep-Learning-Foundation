from __future__ import annotations

from pathlib import Path

from configuration.patch_sweep                       import PatchSweepConfig
from pipelines.patch_sweep.planner                   import PatchSweepPlanner
from pipelines.patch_sweep.stages                    import SweepReportStage, SweepTrainingStage
from pipelines.shared.orchestration.staged_pipeline  import StagedPipeline


class PatchSweepPipeline(StagedPipeline):
    LOGGER_NAME = "patch_sweep_pipeline"

    def __init__(self, config: PatchSweepConfig, entry_script: Path) -> None:
        super().__init__(config, entry_script)

    def _run_training(self, planner: PatchSweepPlanner) -> list[dict]:
        stage   = SweepTrainingStage(config=self.config, entry_script=self.entry_script, run_tag=self.run_tag, planner=planner, logger=self.logger)
        results = stage.run()

        failed = [result for result in results if result["status"] != "DONE"]
        self._mark_stage("training", "completed" if not failed else "partial")

        return results

    def _run_report(self, planner: PatchSweepPlanner) -> Path:
        stage   = SweepReportStage(config=self.config, run_tag=self.run_tag, planner=planner, logger=self.logger)
        out_dir = stage.run()

        self._mark_stage("report", "completed")

        return out_dir

    def run(self) -> None:
        planner = PatchSweepPlanner(self.config)

        self.logger.section("Patch-size sweep pipeline")
        self.logger.kv_table({
            "Run tag"     : self.run_tag,
            "Model"       : f"{self.config.backbone_name}-{self.config.backbone_head}",
            **planner.summary(),
            "Secondaries" : ", ".join(self.config.paths.secondary_labels),
            "GPUs"        : self.config.gpus,
            "Resume"      : self.config.resume,
            "Run dir"     : str(self.run_dir),
        }, title="Configuration")

        self.logger.subsection("Sweep plan")
        scaling = self.config.training.scale_lr_with_batch
        rows    = [{
            "Unit"     : unit.name,
            "Dataset"  : unit.dataset,
            "Patch"    : f"{unit.patch_size[0]}x{unit.patch_size[1]}",
            "Stride"   : f"{unit.patch_stride[0]}x{unit.patch_stride[1]}",
            "Batch"    : unit.batch_size,
            "LR scale" : f"{unit.batch_size / unit.lr_reference_batch_size if scaling else 1.0:.2f}",
        } for unit in planner.units()]
        self.logger.metrics_table(rows, ["Unit", "Dataset", "Patch", "Stride", "Batch", "LR scale"])

        try:
            training_results = self._run_training(planner)
            report_dir       = self._run_report(planner)

            self._mark_stage("pipeline", "completed")

            self.logger.section("Pipeline summary")
            self.logger.kv_table({
                "Units"         : len(training_results),
                "Trained"       : sum(1 for result in training_results if result["status"] == "DONE"),
                "Report"        : str(report_dir),
                "Pipeline logs" : str(self.pipeline_dir),
            }, title="Done")
        finally:
            self.logger.close()
