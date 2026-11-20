
from __future__ import annotations

from configuration.benchmark                 import BenchmarkConfig
from pipelines.shared.config.config_factory  import ConfigFactory
from pipelines.shared.training.seed_sweep    import SeedSet
from pipelines.shared.training.unit_resume   import UnitResume
from pipelines.shared.training.worker_base   import WorkerBase
from tools.data.io                           import FileIO


class BenchmarkWorker(WorkerBase):

    def __init__(self, config: BenchmarkConfig, run_tag: str) -> None:
        super().__init__(config, run_tag)
        self.factory = ConfigFactory(config)

    def _run_name(self, model_name: str, component: str | None, seed: int | None) -> str:
        from pipelines.benchmark.stages import SeedExpandedStage

        unit = SeedExpandedStage.unit_base(self.config, model_name, component)
        return unit if seed is None else SeedSet.run_name(unit, seed)

    def unit_run_name(self, model_name: str, component: str | None, seed: int | None, run_name: str | None) -> str:
        return run_name if run_name else self._run_name(model_name, component, seed)

    def _apply_loss_component(self, trainer_config, component: str | None) -> None:
        if component is None:
            return

        from pipelines.backbone.training.loss_terms import LossComponentCatalog

        trainer_config.curriculum = LossComponentCatalog.curriculum(component, base=self.config.loss)

    def _size_overrides(self, model_name: str) -> dict:
        size_match_path = self.run_dir / "pipeline" / "size_match.json"
        if not size_match_path.exists():
            return {}

        records = FileIO.load_json(size_match_path)

        return records[model_name]["overrides"]

class TrainingWorker(BenchmarkWorker):

    def _run_backbone(self, model_name: str, run_name: str, seed: int | None, loss_component: str | None) -> None:
        from pipelines.backbone.training.pipeline import TrainingPipeline
        from pipelines.shared.model.model_builder import ModelBuilder

        stage_dir    = self.run_dir / "training"
        name, head   = ModelBuilder.split_key(model_name)
        model_config = ModelBuilder.config_from_registry(name, self._size_overrides(model_name), head=head)

        trainer_config          = self.factory.training_trainer_config(logdir=stage_dir)
        self._apply_loss_component(trainer_config, loss_component)

        dataset_config              = self.factory.training_dataset_config()
        dataset_config.input_config = self.config.input

        pipeline = TrainingPipeline(
            trainer_config = trainer_config,
            dataset_config = dataset_config,
            backbone_name  = name,
            model_config   = model_config,
            seed           = self.config.seed if seed is None else seed,
            run_name       = run_name,
        )

        pipeline.run()

    def run(self, model_name: str, seed: int | None = None, loss_component: str | None = None, run_name: str | None = None) -> None:
        from pipelines.benchmark.stages import BenchmarkUnitManifest

        run_name = self.unit_run_name(model_name, loss_component, seed, run_name)

        BenchmarkUnitManifest.write(self.run_dir / "training" / run_name, run_tag=self.run_tag, unit=run_name, model=model_name, loss_component=loss_component, seed=seed)

        if self.skip_complete(self.run_dir / "training" / run_name, "benchmark_training", self.config.resume):
            return

        self._run_backbone(model_name, run_name, seed, loss_component)


class InferenceWorker(BenchmarkWorker):

    def run(self, model_name: str, seed: int | None = None, loss_component: str | None = None, run_name: str | None = None) -> None:
        from pipelines.backbone.inference.pipeline import InferencePipeline

        run_directory = self.run_dir / "training" / self.unit_run_name(model_name, loss_component, seed, run_name)

        if UnitResume(run_directory, enabled=self.config.resume).skip_inference(self.config.inference.split):
            return

        InferencePipeline(self.factory.inference_config(run_directory)).run()
