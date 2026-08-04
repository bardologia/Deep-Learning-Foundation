from __future__ import annotations

from pathlib import Path

from configuration.benchmark                 import BenchmarkConfig
from pipelines.shared.config.config_factory  import ConfigFactory
from pipelines.shared.training.seed_sweep    import SeedSet
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

        if model_name not in records:
            raise SystemExit(f"size_match.json present but missing an entry for '{model_name}'; the benchmark would silently train a capacity-unmatched model")

        return records[model_name]["overrides"]

    def _max_batch_size(self, model_name: str) -> int | None:
        path = self.run_dir / "pipeline" / "max_batch.json"
        if not path.exists():
            return None

        records = FileIO.load_json(path)

        if model_name not in records:
            raise SystemExit(f"max_batch.json present but missing an entry for '{model_name}'")

        entry = records[model_name]
        if entry.get("status") != "PASS" or not entry.get("batch_size"):
            raise SystemExit(f"model '{model_name}' has no usable measured batch size: {entry}")

        return int(entry["batch_size"])

    def _ae_entry_config(self, model_name: str, logdir: Path, run_name: str | None = None, seed: int | None = None):
        from configuration.training import ProfileAeEntryConfig

        return ProfileAeEntryConfig(
            run_name        = run_name or model_name,
            seed            = self.config.seed if seed is None else seed,
            logdir          = logdir,
            pixel_subsample = self.config.pixel_subsample,
            keep_empty_frac = self.config.keep_empty_frac,
            ae_model_name   = model_name,
            ae_loss         = self.config.ae_loss,
            paths           = self.config.paths,
            training        = self.config.training,
            overfit_check   = self.config.overfit_check,
        )

    def _jepa_entry_config(self, model_name: str, logdir: Path, run_name: str | None = None, seed: int | None = None):
        from configuration.training import JepaEntryConfig
        from pipelines.shared.model.model_builder import ModelBuilder

        jepa       = self.config.jepa
        name, head = ModelBuilder.split_key(model_name)

        return JepaEntryConfig(
            run_name                   = run_name or model_name,
            backbone_name              = name,
            backbone_head              = head,
            seed                       = self.config.seed if seed is None else seed,
            logdir                     = logdir,
            profile_autoencoder_logdir = jepa.profile_autoencoder_logdir,
            profile_autoencoder_run    = jepa.profile_autoencoder_run,
            profile_autoencoder_mode   = jepa.profile_autoencoder_mode,
            image_autoencoder_logdir   = jepa.image_autoencoder_logdir,
            image_autoencoder_run      = jepa.image_autoencoder_run,
            image_autoencoder_mode     = jepa.image_autoencoder_mode,
            target_provider            = jepa.target_provider,
            embedding_loss             = jepa.embedding_loss,
            param_loss                 = jepa.param_loss,
            geometry                   = self.config.geometry,
            paths                      = self.config.paths,
            training                   = self.config.training,
            normalization              = self.config.normalization,
            augmentation               = self.config.augmentation,
            overfit_check              = self.config.overfit_check,
        )


class MaxBatchWorker(BenchmarkWorker):
    def run(self, model_name: str) -> None:
        from pipelines.benchmark.batch_probe import MaxBatchProbe

        stage_dir   = self.run_dir / "max_batch"
        result_path = stage_dir / model_name / "max_batch_result.json"

        probe  = MaxBatchProbe(config=self.config, model_name=model_name, overrides=self._size_overrides(model_name), work_dir=stage_dir / model_name / "probe")
        result = probe.run()

        FileIO.save_json(result, result_path, indent=2)

        if result["status"] == "FAIL":
            raise SystemExit(1)


class TrainingWorker(BenchmarkWorker):
    def _run_profile_autoencoder(self, model_name: str, run_name: str, seed: int | None) -> None:
        from pipelines.profile_autoencoder.training.pipeline import TrainingPipeline

        entry = self._ae_entry_config(model_name, self.run_dir / "training", run_name=run_name, seed=seed)
        TrainingPipeline(entry).run()

    def _run_jepa(self, model_name: str, run_name: str, seed: int | None) -> None:
        from pipelines.jepa.training.pipeline import TrainingPipeline

        entry = self._jepa_entry_config(model_name, self.run_dir / "training", run_name=run_name, seed=seed)
        TrainingPipeline(entry).run()

    def _run_backbone(self, model_name: str, run_name: str, seed: int | None, loss_component: str | None) -> None:
        from models                               import BACKBONE_CONFIG_REGISTRY
        from pipelines.backbone.training.pipeline import TrainingPipeline
        from pipelines.shared.model.model_builder import ModelBuilder

        stage_dir    = self.run_dir / "training"
        name, head   = ModelBuilder.split_key(model_name)
        model_config = BACKBONE_CONFIG_REGISTRY[name]()
        model_config.head = head

        for attribute, value in self._size_overrides(model_name).items():
            setattr(model_config, attribute, value)

        measured_batch = self._max_batch_size(model_name)
        if measured_batch is not None:
            self.config.training.batch_size = measured_batch

        trainer_config          = self.factory.training_trainer_config(logdir=stage_dir)
        trainer_config.geometry = self.config.geometry.resolved(self.config.paths.dataset_path, secondary_labels=self.factory._secondary_labels())
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
            overfit_check  = self.config.overfit_check,
        )

        pipeline.run(probe_config=self._probe_config())

    def run(self, model_name: str, seed: int | None = None, loss_component: str | None = None) -> None:
        training_type = self.config.training_type
        if training_type not in ("backbone", "jepa", "profile_autoencoder"):
            raise ValueError(f"Unknown training_type '{training_type}', expected one of ['backbone', 'jepa', 'profile_autoencoder']")

        run_name = self._run_name(model_name, loss_component, seed)

        if training_type == "profile_autoencoder":
            self._run_profile_autoencoder(model_name, run_name, seed)
            return

        if training_type == "jepa":
            self._run_jepa(model_name, run_name, seed)
            return

        self._run_backbone(model_name, run_name, seed, loss_component)


class InferenceWorker(BenchmarkWorker):
    def run(self, model_name: str, seed: int | None = None, loss_component: str | None = None) -> None:
        from pipelines.backbone.inference.pipeline import InferencePipeline
        from pipelines.shared.inference.inference_components import InferenceComponentsResolver

        run_directory = self.run_dir / "training" / self._run_name(model_name, loss_component, seed)

        components = InferenceComponentsResolver.for_run(run_directory)
        pipeline   = InferencePipeline(self.factory.inference_config(run_directory), components=components)
        pipeline.run()
