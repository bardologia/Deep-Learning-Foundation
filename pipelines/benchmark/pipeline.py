from __future__ import annotations

from pathlib import Path

from configuration.benchmark                        import BenchmarkConfig
from models                                         import BACKBONE_HEADS, config_registry
from pipelines.shared.model.model_builder           import ModelBuilder
from pipelines.shared.orchestration.staged_pipeline import StagedPipeline

from pipelines.benchmark.stages import ComparisonStage, InferenceStage, MaxBatchStage, SeedExpandedStage, SizeMatchStage, TrainingStage


class BenchmarkPipeline(StagedPipeline):
    LOGGER_NAME     = "benchmark_pipeline"
    SUPPORTED_TYPES = ("backbone", "jepa", "profile_autoencoder")

    def __init__(self, config: BenchmarkConfig, entry_script: Path) -> None:
        super().__init__(config, entry_script)
        skip        = set(config.skip_models)
        keys        = self._model_keys()
        self.models = [key for key in keys if key not in skip and ModelBuilder.split_key(key)[0] not in skip]

    def _registry(self) -> dict:
        return config_registry(self.config.training_type)

    def _model_keys(self) -> list[str]:
        if self.config.training_type != "backbone":
            return list(self._registry().keys())

        return [ModelBuilder.model_key(name, head) for name in self._registry() for head in self.config.heads]

    def _validate_sweep(self) -> None:
        if self.config.training_type not in self.SUPPORTED_TYPES:
            raise SystemExit(f"benchmark does not support training_type '{self.config.training_type}'; supported: {', '.join(self.SUPPORTED_TYPES)}")

        SeedExpandedStage.components(self.config)

        unknown = [head for head in self.config.heads if head not in BACKBONE_HEADS]
        if unknown:
            raise SystemExit(f"unknown head(s) {unknown}; valid: {', '.join(BACKBONE_HEADS)}")
        if self.config.training_type == "backbone" and not self.config.heads:
            raise SystemExit("heads is empty; select at least one output head to benchmark")

    def _run_max_batch(self) -> dict:
        stage   = MaxBatchStage(config=self.config, entry_script=self.entry_script, run_tag=self.run_tag, models=self.models, logger=self.logger)
        records = stage.run()

        self._mark_stage("max_batch", "completed")
        return records

    def _run_size_match(self) -> dict:
        stage   = SizeMatchStage(config=self.config, run_tag=self.run_tag, models=self.models, logger=self.logger)
        records = stage.run()

        self._mark_stage("size_match", "completed")
        return records

    def _run_training(self) -> list[dict]:
        stage   = TrainingStage(config=self.config, entry_script=self.entry_script, run_tag=self.run_tag, models=self.models, logger=self.logger)
        results = stage.run()

        failed = [r for r in results if r["status"] != "DONE"]
        self._mark_stage("training", "completed" if not failed else "partial")
        return results

    def _run_inference(self) -> list[dict]:
        stage   = InferenceStage(config=self.config, entry_script=self.entry_script, run_tag=self.run_tag, models=self.models, logger=self.logger)
        results = stage.run()

        failed = [r for r in results if r["status"] == "FAILED"]
        self._mark_stage("inference", "completed" if not failed else "partial")
        return results

    def _run_comparison(self) -> Path:
        stage   = ComparisonStage(config=self.config, run_tag=self.run_tag, logger=self.logger, reference_model=self.config.size_match.reference_model, embed_images=self.config.comparison.embed_images)
        out_dir = stage.run()

        self._mark_stage("comparison", "completed")
        return out_dir

    def run(self) -> None:
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

        try:
            self._validate_sweep()

            if self.config.runs_size_match():
                self._run_size_match()

            if self.config.runs_max_batch():
                self._run_max_batch()

            training_results  = self._run_training()
            inference_results = self._run_inference() if self.config.runs_inference() else []
            comparison_dir    = self._run_comparison()

            self._mark_stage("pipeline", "completed")

            self.logger.section("Pipeline summary")
            self.logger.kv_table({
                "Models"        : len(self.models),
                "Trained"       : sum(1 for r in training_results if r["status"] == "DONE"),
                "Inferred"      : sum(1 for r in inference_results if r["status"] == "DONE"),
                "Comparison"    : str(comparison_dir),
                "Pipeline logs" : str(self.pipeline_dir),
            }, title="Done")
        finally:
            self.logger.close()
