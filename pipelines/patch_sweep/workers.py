from __future__ import annotations

from configuration.patch_sweep              import PatchSweepConfig
from pipelines.patch_sweep.planner          import PatchSweepPlanner, SweepUnit
from pipelines.shared.config.config_factory import ConfigFactory
from pipelines.shared.model.model_builder   import ModelBuilder
from pipelines.shared.training.seed_sweep   import SeedSet
from pipelines.shared.training.worker_base  import WorkerBase


class SweepTrainingWorker(WorkerBase):
    def __init__(self, config: PatchSweepConfig, run_tag: str) -> None:
        super().__init__(config=config, run_tag=run_tag)

    def _apply_unit(self, unit: SweepUnit) -> None:
        self.config.paths.dataset_path    = unit.dataset_path
        self.config.paths.parameters_path = unit.parameters_path

        self.config.training.patch_size              = unit.patch_size
        self.config.training.patch_stride            = unit.patch_stride
        self.config.training.batch_size              = unit.batch_size
        self.config.training.lr_reference_batch_size = unit.lr_reference_batch_size

    def run(self, unit_name: str, seed: int | None = None) -> None:
        from pipelines.backbone.training.pipeline import TrainingPipeline

        unit = PatchSweepPlanner(self.config).unit(unit_name)
        self._apply_unit(unit)

        factory      = ConfigFactory(self.config)
        model_config = ModelBuilder.config_from_registry(self.config.backbone_name, self.config.model_overrides, head=self.config.backbone_head)

        trainer_config            = factory.training_trainer_config(logdir=self.run_dir / "training")
        trainer_config.curriculum = self.config.curriculum
        trainer_config.geometry   = self.config.geometry.resolved(self.config.paths.dataset_path, secondary_labels=factory._secondary_labels())

        dataset_config              = factory.training_dataset_config()
        dataset_config.input_config = self.config.input

        pipeline = TrainingPipeline(
            trainer_config = trainer_config,
            dataset_config = dataset_config,
            backbone_name  = self.config.backbone_name,
            model_config   = model_config,
            seed           = self.config.seed if seed is None else seed,
            run_name       = unit_name if seed is None else SeedSet.run_name(unit_name, seed),
        )

        pipeline.run(probe_config=self._probe_config())
