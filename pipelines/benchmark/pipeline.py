
from __future__ import annotations

from pathlib import Path

from configuration.benchmark                        import BenchmarkConfig
from models                                         import BACKBONE_CONFIG_REGISTRY
from pipelines.shared.orchestration.staged_pipeline import StagedPipeline

from pipelines.benchmark.stages import ComparisonStage, InferenceStage, TrainingStage


class BenchmarkPipeline(StagedPipeline):

    LOGGER_NAME = "benchmark_pipeline"
    STAGES      = ("full", "prepare", "compare")

    def __init__(self, config: BenchmarkConfig, entry_script: Path, resumed=None) -> None:
        super().__init__(config, entry_script, resumed=resumed)
        self.models         = self.select_models(config)
        self.training_stage = None

    @staticmethod
    def select_models(config: BenchmarkConfig) -> list[str]:
        return [name for name in BACKBONE_CONFIG_REGISTRY if name not in config.skip_models]

    def _log_model_selection(self) -> None:
        skipped = [name for name in BACKBONE_CONFIG_REGISTRY if name not in self.models]

        self.logger.kv_table({
            "Selected"               : ", ".join(self.models),
            "Skipped by skip_models" : ", ".join(skipped) or "none",
        }, title=f"Model selection: {len(self.models)} benchmarked, {len(skipped)} skipped")

    def _log_unit_status(self) -> None:
        self._begin_stage("prepare")

        self.training_stage = TrainingStage(config=self.config, entry_script=self.entry_script, run_tag=self.run_tag, models=self.models, logger=self.logger)

        self.training_stage.status_report().log(self.logger, "Unit status")

    def _run_training(self) -> list[dict]:
        self._begin_stage("training")

        self.training_stage = TrainingStage(config=self.config, entry_script=self.entry_script, run_tag=self.run_tag, models=self.models, logger=self.logger)
        results             = self.training_stage.run()

        failed = [r for r in results if r["status"] != "DONE"]
        self._mark_stage("training", "completed" if not failed else "partial")
        return results

    def _run_inference(self) -> list[dict]:
        self._begin_stage("inference")

        if not self.config.runs_inference():
            self.logger.subsection("Inference skipped: infer_after is off; models are compared on their training checkpoints alone")
            self._mark_stage("inference", "skipped")
            return []

        stage   = InferenceStage(config=self.config, entry_script=self.entry_script, run_tag=self.run_tag, models=self.models, logger=self.logger)
        results = stage.run()

        failed = [r for r in results if r["status"] == "FAILED"]
        self._mark_stage("inference", "completed" if not failed else "partial")
        return results

    def _unit_plan(self) -> dict[str, str]:
        if self.training_stage is None:
            self.training_stage = TrainingStage(config=self.config, entry_script=self.entry_script, run_tag=self.run_tag, models=self.models, logger=self.logger)

        return {item: self.training_stage.unit_name(item) for item in self.training_stage.items}

    def _run_comparison(self) -> Path:
        self._begin_stage("comparison")

        stage   = ComparisonStage(config=self.config, run_tag=self.run_tag, logger=self.logger, embed_images=self.config.comparison.embed_images, inference_split=self.config.inference.split, planned_units=self._unit_plan())
        out_dir = stage.run()

        self._mark_stage("comparison", "completed")
        return out_dir

    def _execute(self) -> None:
        self.logger.section("Benchmark pipeline")
        self.logger.kv_table({
            "Run tag"   : self.run_tag,
            "Models"    : len(self.models),
            "GPUs"      : self.config.gpus,
            "Epochs"    : self.config.training.epochs,
            "Resume"    : self.config.resume,
            "Run dir"   : str(self.run_dir),
        }, title="Configuration")

        self._log_launch()
        self._log_model_selection()

        if self.config.stage == "compare":
            comparison_dir = self._run_comparison()
            self._mark_stage("pipeline", "completed")
            self.logger.info(f"Comparison stage done: {comparison_dir}")
            return

        if self.config.stage == "prepare":
            self._log_unit_status()
            self._mark_stage("prepare", "completed")
            self.logger.info(f"Prepare stage done: resolved configuration and unit status written to {self.pipeline_dir}")
            return

        training_results  = self._run_training()
        inference_results = self._run_inference()
        comparison_dir    = self._run_comparison()

        self._finish(training_results, inference_results, {"Models": len(self.models), "Comparison": str(comparison_dir)})
