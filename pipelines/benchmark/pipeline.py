"""Staged architecture benchmark over the registered model families.

Runs capacity matching and batch probing, trains every selected model under the
same budget, optionally infers, and finally writes the comparison reports.
"""

from __future__ import annotations

from pathlib import Path

from configuration.benchmark                        import BenchmarkConfig
from models                                         import BACKBONE_HEADS, config_registry
from pipelines.shared.model.model_builder           import ModelBuilder
from pipelines.shared.orchestration.staged_pipeline import StagedPipeline

from pipelines.benchmark.stages import ComparisonStage, InferenceStage, MaxBatchStage, SeedExpandedStage, SizeMatchStage, TrainingStage


class BenchmarkPipeline(StagedPipeline):
    """Runs the architecture benchmark end to end or one of its stages.

    Attributes:
        models : Model keys selected for this benchmark, skip list applied.
    """

    LOGGER_NAME     = "benchmark_pipeline"
    SUPPORTED_TYPES = ("backbone", "jepa", "profile_autoencoder")
    STAGES          = ("full", "prepare", "compare")

    def __init__(self, config: BenchmarkConfig, entry_script: Path, resumed=None) -> None:
        """Resolves the run directory and the model keys this benchmark covers.

        Args:
            config       : Resolved benchmark configuration.
            entry_script : Entry script the stage workers re-invoke.
            resumed      : Resumed-run description logged when the pipeline starts, or None for a fresh run.
        """
        super().__init__(config, entry_script, resumed=resumed)
        self.models = self.select_models(config)

    @classmethod
    def select_models(cls, config: BenchmarkConfig) -> list[str]:
        """Returns the benchmarked model keys, dropping those the skip list names.

        Args:
            config : Benchmark configuration carrying the training type, heads and skip list.

        Returns:
            Model keys, by architecture name alone for non-backbone families and
            as backbone-and-head keys for backbones.
        """
        skip = set(config.skip_models)
        keys = cls._model_keys(config)

        return [key for key in keys if key not in skip and ModelBuilder.split_key(key)[0] not in skip]

    @classmethod
    def _model_keys(cls, config: BenchmarkConfig) -> list[str]:
        """Returns every registered model key of the training type, before skipping."""
        registry = config_registry(config.training_type)

        if config.training_type != "backbone":
            return list(registry.keys())

        return [ModelBuilder.model_key(name, head) for name in registry for head in config.heads]

    @classmethod
    def validate_sweep(cls, config: BenchmarkConfig) -> None:
        """Validates the sweep settings and resolves the size-match reference model.

        Args:
            config : Benchmark configuration, mutated in place when the reference model has to be expanded into a full model key.

        Raises:
            SystemExit : On an unsupported training type or stage, an invalid loss component, an unknown or empty head list, or a reference model that is not among the swept keys.
        """
        if config.training_type not in cls.SUPPORTED_TYPES:
            raise SystemExit(f"benchmark does not support training_type '{config.training_type}'; supported: {', '.join(cls.SUPPORTED_TYPES)}")

        if config.stage not in cls.STAGES:
            raise SystemExit(f"unknown benchmark stage '{config.stage}'; valid: {', '.join(cls.STAGES)}")

        SeedExpandedStage.components(config)

        unknown = [head for head in config.heads if head not in BACKBONE_HEADS]
        if unknown:
            raise SystemExit(f"unknown head(s) {unknown}; valid: {', '.join(BACKBONE_HEADS)}")
        if config.training_type == "backbone" and not config.heads:
            raise SystemExit("heads is empty; select at least one output head to benchmark")

        if config.runs_size_match():
            config.size_match.reference_model = SizeMatchStage.resolve_reference(config.size_match.reference_model, cls.select_models(config), config.heads)

    def _validate_sweep(self) -> None:
        """Validates this pipeline's own configuration before any stage runs."""
        self.validate_sweep(self.config)

    def _run_max_batch(self) -> dict:
        """Probes the maximum batch size of every model and records the outcome.

        Returns:
            The per-model probe records.

        Raises:
            SystemExit : If the probe failed for every model, leaving nothing to train.
        """
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
        """Matches every model's capacity to the reference and returns the per-model records."""
        stage   = SizeMatchStage(config=self.config, run_tag=self.run_tag, models=self.models, logger=self.logger)
        records = stage.run()

        self._mark_stage("size_match", "completed")
        return records

    def _log_unit_status(self) -> None:
        """Logs what every model x seed unit already holds on disk, and which directories no unit claims.

        A full run gets the same table from its training stage; this is the
        prepare stage's version, so a resume can be inspected before anything
        is launched.
        """
        stage = TrainingStage(config=self.config, entry_script=self.entry_script, run_tag=self.run_tag, models=self.models, logger=self.logger)

        stage.status_report().log(self.logger, "Unit status")

        for directory in stage.unidentified:
            self.logger.warning(f"{directory.relative_to(stage.stage_dir)}: run directory without a benchmark manifest, left untouched")

    def _run_training(self) -> list[dict]:
        """Trains every benchmark unit on the GPU pool and returns the job results."""
        stage   = TrainingStage(config=self.config, entry_script=self.entry_script, run_tag=self.run_tag, models=self.models, logger=self.logger)
        results = stage.run()

        failed = [r for r in results if r["status"] != "DONE"]
        self._mark_stage("training", "completed" if not failed else "partial")
        return results

    def _run_inference(self) -> list[dict]:
        """Runs inference over every trained unit and returns the job results."""
        stage   = InferenceStage(config=self.config, entry_script=self.entry_script, run_tag=self.run_tag, models=self.models, logger=self.logger)
        results = stage.run()

        failed = [r for r in results if r["status"] == "FAILED"]
        self._mark_stage("inference", "completed" if not failed else "partial")
        return results

    def _run_comparison(self) -> Path:
        """Writes the cross-model comparison reports and returns their output directory."""
        stage   = ComparisonStage(config=self.config, run_tag=self.run_tag, logger=self.logger, reference_model=self.config.size_match.reference_model, embed_images=self.config.comparison.embed_images)
        out_dir = stage.run()

        self._mark_stage("comparison", "completed")
        return out_dir

    def run(self) -> None:
        """Runs the configured stage, or the whole prepare-train-infer-compare sequence.

        Raises:
            SystemExit : If the sweep configuration is invalid or batch probing failed for every model.
        """
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

        self._log_resumed()

        try:
            self._validate_sweep()

            if self.config.stage == "compare":
                comparison_dir = self._run_comparison()
                self._mark_stage("pipeline", "completed")
                self.logger.info(f"Comparison stage done: {comparison_dir}")
                return

            if self.config.runs_size_match():
                self._run_size_match()

            self._run_max_batch()

            if self.config.stage == "prepare":
                self._log_unit_status()
                self._mark_stage("prepare", "completed")
                self.logger.info(f"Prepare stage done: size match, max batch and unit status written to {self.pipeline_dir}")
                return

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
