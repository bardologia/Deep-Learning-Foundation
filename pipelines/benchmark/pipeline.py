
from __future__ import annotations

from pathlib import Path

from configuration.benchmark                        import BenchmarkConfig
from models                                         import BACKBONE_CONFIG_REGISTRY
from pipelines.shared.model.model_builder           import ModelBuilder
from pipelines.shared.orchestration.staged_pipeline import StagedPipeline

from pipelines.benchmark.stages import ComparisonStage, InferenceStage, MaxBatchStage, SizeMatchStage, TrainingStage


class BenchmarkPipeline(StagedPipeline):

    LOGGER_NAME = "benchmark_pipeline"
    STAGES      = ("full", "prepare", "compare")

    def __init__(self, config: BenchmarkConfig, entry_script: Path, resumed=None) -> None:
        super().__init__(config, entry_script, resumed=resumed)
        self.models         = self.select_models(config)
        self.training_stage = None

    @classmethod
    def select_models(cls, config: BenchmarkConfig) -> list[str]:
        skip = set(config.skip_models)
        keys = cls._model_keys(config)

        return [key for key in keys if key not in skip and ModelBuilder.split_key(key)[0] not in skip]

    @classmethod
    def _model_keys(cls, config: BenchmarkConfig) -> list[str]:
        return [ModelBuilder.model_key(name, head) for name in BACKBONE_CONFIG_REGISTRY for head in config.heads]

    @classmethod
    def validate_sweep(cls, config: BenchmarkConfig) -> None:
        config.size_match.reference_model = SizeMatchStage.resolve_reference(config.size_match.reference_model, cls.select_models(config), config.heads)

    def _validate_sweep(self) -> None:
        self._begin_stage("validation")
        self.validate_sweep(self.config)

    def _log_model_selection(self) -> None:
        skipped = [key for key in self._model_keys(self.config) if key not in self.models]

        self.logger.kv_table({
            "Selected"               : ", ".join(self.models),
            "Skipped by skip_models" : ", ".join(skipped) or "none",
        }, title=f"Model selection: {len(self.models)} benchmarked, {len(skipped)} skipped")

    def _run_max_batch(self) -> dict:
        self._begin_stage("max_batch")

        stage   = MaxBatchStage(config=self.config, entry_script=self.entry_script, run_tag=self.run_tag, models=self.models, logger=self.logger)
        records = stage.run()

        failed = [name for name, record in records.items() if record["status"] != "PASS"]
        self._mark_stage("max_batch", "completed" if not failed else "partial")

        if failed and len(failed) == len(records):
            raise SystemExit(f"max batch probing failed for every model ({', '.join(failed)}); see {stage.report_path}")
        if failed:
            self.logger.error(f"Max batch probing failed for {len(failed)}/{len(records)} models: {', '.join(failed)}; they cannot be trained")

        return records

    def _run_size_match(self) -> dict:
        self._begin_stage("size_match")

        stage   = SizeMatchStage(config=self.config, run_tag=self.run_tag, models=self.models, logger=self.logger)
        records = stage.run()

        self._mark_stage("size_match", "completed")
        return records

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

        stage   = ComparisonStage(config=self.config, run_tag=self.run_tag, logger=self.logger, reference_model=self.config.size_match.reference_model, embed_images=self.config.comparison.embed_images, inference_split=self.config.inference.split, planned_units=self._unit_plan())
        out_dir = stage.run()

        self._mark_stage("comparison", "completed")
        return out_dir

    def _execute(self) -> None:
        self.logger.section("Benchmark pipeline")
        self.logger.kv_table({
            "Run tag"   : self.run_tag,
            "Models"    : len(self.models),
            "GPUs"      : self.config.gpus,
            "Reference" : self.config.size_match.reference_model,
            "Epochs"    : self.config.training.epochs,
            "Resume"    : self.config.resume,
            "Run dir"   : str(self.run_dir),
        }, title="Configuration")

        self._log_launch()
        self._validate_sweep()
        self._log_model_selection()

        if self.config.stage == "compare":
            comparison_dir = self._run_comparison()
            self._mark_stage("pipeline", "completed")
            self.logger.info(f"Comparison stage done: {comparison_dir}")
            return

        self._run_size_match()

        self._run_max_batch()

        if self.config.stage == "prepare":
            self._log_unit_status()
            self._mark_stage("prepare", "completed")
            self.logger.info(f"Prepare stage done: size match, max batch and unit status written to {self.pipeline_dir}")
            return

        training_results  = self._run_training()
        inference_results = self._run_inference()
        comparison_dir    = self._run_comparison()

        self._finish(training_results, inference_results, {"Models": len(self.models), "Comparison": str(comparison_dir)})
