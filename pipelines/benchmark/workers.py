"""Single-unit worker processes executed by the model benchmark sweep.

Each worker is spawned as its own process by the benchmark stages and performs
exactly one action for one (model, loss component, seed) unit: probing the
largest usable batch size, training it, or running inference on the resulting
run directory.
"""

from __future__ import annotations

from pathlib import Path

from configuration.benchmark                 import BenchmarkConfig
from pipelines.shared.config.config_factory  import ConfigFactory
from pipelines.shared.training.seed_sweep    import SeedSet
from pipelines.shared.training.worker_base   import WorkerBase
from tools.data.io                           import FileIO


class BenchmarkWorker(WorkerBase):
    """Shared state and config assembly for every benchmark worker action.

    Attributes:
        config  : Benchmark sweep configuration shared by all units.
        run_tag : Timestamped tag identifying the sweep run directory.
        factory : Builds trainer, dataset, and inference configs from `config`.
    """

    def __init__(self, config: BenchmarkConfig, run_tag: str) -> None:
        """Binds the worker to a benchmark sweep run.

        Args:
            config  : Benchmark sweep configuration.
            run_tag : Tag of the sweep run directory this unit belongs to.
        """
        super().__init__(config, run_tag)
        self.factory = ConfigFactory(config)

    def _run_name(self, model_name: str, component: str | None, seed: int | None) -> str:
        """Returns the run directory name for one benchmark unit."""
        from pipelines.benchmark.stages import SeedExpandedStage

        unit = SeedExpandedStage.unit_base(self.config, model_name, component)
        return unit if seed is None else SeedSet.run_name(unit, seed)

    def unit_run_name(self, model_name: str, component: str | None, seed: int | None, run_name: str | None) -> str:
        """Returns the run name the unit lives under: the one the scheduler handed over, else the composed one.

        Args:
            model_name : Registry key of the benchmarked model.
            component  : Loss component arm of the unit, or None.
            seed       : Seed of the unit, or None for an unseeded run.
            run_name   : Directory the scheduler found for this unit relative to the training directory, or None to compose the name.
        """
        return run_name if run_name else self._run_name(model_name, component, seed)

    def _apply_loss_component(self, trainer_config, component: str | None) -> None:
        """Replaces the trainer curriculum with the one isolating a loss component.

        Args:
            trainer_config : Trainer configuration mutated in place.
            component      : Name of the loss component arm, or None to keep the configured curriculum untouched.
        """
        if component is None:
            return

        from pipelines.backbone.training.loss_terms import LossComponentCatalog

        trainer_config.curriculum = LossComponentCatalog.curriculum(component, base=self.config.loss)

    def _size_overrides(self, model_name: str) -> dict:
        """Returns the capacity-matching model overrides recorded for a model.

        Args:
            model_name : Registry key of the benchmarked model.

        Returns:
            Override mapping from `size_match.json`, empty when the sweep did
            not run capacity matching.

        Raises:
            SystemExit : If `size_match.json` exists but holds no entry for the model, which would silently train a capacity-unmatched model.
        """
        size_match_path = self.run_dir / "pipeline" / "size_match.json"
        if not size_match_path.exists():
            return {}

        records = FileIO.load_json(size_match_path)

        if model_name not in records:
            raise SystemExit(f"size_match.json present but missing an entry for '{model_name}'; the benchmark would silently train a capacity-unmatched model")

        return records[model_name]["overrides"]

    def _max_batch_size(self, model_name: str) -> int | None:
        """Returns the measured largest usable batch size for a model.

        Args:
            model_name : Registry key of the benchmarked model.

        Returns:
            Batch size measured by the max-batch stage, or None when the stage
            did not run.

        Raises:
            SystemExit : If `max_batch.json` exists but holds no entry for the model, or the recorded probe did not pass.
        """
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

    def _lr_multiplier(self, model_name: str) -> float | None:
        """Returns the LR multiplier the range test selected for a model.

        Args:
            model_name : Registry key of the benchmarked model.

        Returns:
            Multiplier from the model's probe record, or None when the probe
            stage did not run or the test selected nothing.

        Raises:
            SystemExit : If `max_batch.json` exists but holds no entry for the model.
        """
        path = self.run_dir / "pipeline" / "max_batch.json"
        if not path.exists():
            return None

        records = FileIO.load_json(path)

        if model_name not in records:
            raise SystemExit(f"max_batch.json present but missing an entry for '{model_name}'")

        multiplier = records[model_name]["lr_multiplier"]

        return float(multiplier) if multiplier is not None else None

    def _ae_entry_config(self, model_name: str, logdir: Path, run_name: str | None = None, seed: int | None = None):
        """Builds the profile-autoencoder training entry config for one unit.

        Args:
            model_name : Profile-autoencoder registry key.
            logdir     : Directory the run writes its outputs under.
            run_name   : Run directory name, defaulting to the model name.
            seed       : Seed for this unit, defaulting to the sweep seed.

        Returns:
            A populated ProfileAeEntryConfig.
        """
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
        """Builds the JEPA training entry config for one unit.

        Args:
            model_name : Backbone registry key, optionally carrying a head suffix.
            logdir     : Directory the run writes its outputs under.
            run_name   : Run directory name, defaulting to the model name.
            seed       : Seed for this unit, defaulting to the sweep seed.

        Returns:
            A populated JepaEntryConfig with the backbone name and head split apart.
        """
        from configuration.training import JepaEntryConfig
        from pipelines.shared.model.model_builder import ModelBuilder

        jepa       = self.config.jepa
        name, head = ModelBuilder.split_key(model_name)

        return JepaEntryConfig(
            run_name      = run_name or model_name,
            backbone_name = name,
            backbone_head = head,
            seed          = self.config.seed if seed is None else seed,
            logdir        = logdir,
            paths         = self.config.paths,
            training      = self.config.training,
            normalization = self.config.normalization,
            augmentation  = self.config.augmentation,
            overfit_check = self.config.overfit_check,
            probe         = self.config.probe,
            **jepa.entry_kwargs(),
        )


class MaxBatchWorker(BenchmarkWorker):
    """Worker that measures the largest batch size a model fits into VRAM."""

    def _build_probe(self, model_name: str, work_dir: Path):
        """Returns the probe matching the sweep's training type.

        Backbone arms probe through the sweep-assembled context with the
        size-match overrides applied; JEPA and profile-autoencoder arms probe
        through their training pipeline's own preflight trainer builder.

        Args:
            model_name : Registry key of the model to probe.
            work_dir   : Directory the probe writes into.

        Returns:
            The configured probe instance.

        Raises:
            ValueError : If the configured training type has no probe.
        """
        from pipelines.benchmark.batch_probe import EntryMaxBatchProbe, MaxBatchProbe

        training_type = self.config.training_type

        if training_type == "backbone":
            return MaxBatchProbe(config=self.config, model_name=model_name, overrides=self._size_overrides(model_name), work_dir=work_dir)

        if training_type == "jepa":
            from pipelines.jepa.training.pipeline import TrainingPipeline

            entry = self._jepa_entry_config(model_name, work_dir)
            return EntryMaxBatchProbe(config=self.config, model_name=model_name, work_dir=work_dir, build_trainer=TrainingPipeline(entry).build_pretrain_trainer)

        if training_type == "profile_autoencoder":
            from pipelines.profile_autoencoder.training.pipeline import TrainingPipeline

            entry = self._ae_entry_config(model_name, work_dir)
            return EntryMaxBatchProbe(config=self.config, model_name=model_name, work_dir=work_dir, build_trainer=TrainingPipeline(entry).build_pretrain_trainer)

        raise ValueError(f"Unknown training_type '{training_type}', expected one of ['backbone', 'jepa', 'profile_autoencoder']")

    def run(self, model_name: str) -> None:
        """Probes and records the maximum usable batch size for one model.

        Args:
            model_name : Registry key of the model to probe.

        Raises:
            SystemExit : If the probe failed to find any usable batch size.
        """
        stage_dir   = self.run_dir / "max_batch"
        result_path = stage_dir / model_name / "max_batch_result.json"

        result = self._build_probe(model_name, stage_dir / model_name / "probe").run()

        FileIO.save_json(result, result_path, indent=2)

        if result["status"] == "FAIL":
            raise SystemExit(1)


class TrainingWorker(BenchmarkWorker):
    """Worker that trains one benchmark unit in the configured model family."""

    def _run_profile_autoencoder(self, model_name: str, run_name: str, seed: int | None) -> None:
        """Trains one profile-autoencoder unit under the probed batch size and LR multiplier."""
        from pipelines.profile_autoencoder.training.pipeline import TrainingPipeline

        self._apply_probed_overrides(model_name)

        entry = self._ae_entry_config(model_name, self.run_dir / "training", run_name=run_name, seed=seed)
        TrainingPipeline(entry).run()

    def _run_jepa(self, model_name: str, run_name: str, seed: int | None) -> None:
        """Trains one JEPA unit under the probed batch size and LR multiplier."""
        from pipelines.jepa.training.pipeline import TrainingPipeline

        self._apply_probed_overrides(model_name)

        entry = self._jepa_entry_config(model_name, self.run_dir / "training", run_name=run_name, seed=seed)
        TrainingPipeline(entry).run()

    def _run_backbone(self, model_name: str, run_name: str, seed: int | None, loss_component: str | None) -> None:
        """Trains one backbone unit under the measured batch size, probed LR multiplier and loss arm.

        Args:
            model_name     : Backbone registry key, optionally with a head suffix.
            run_name       : Run directory name for this unit.
            seed           : Seed for this unit, defaulting to the sweep seed.
            loss_component : Loss component arm to isolate, or None for the configured curriculum.
        """
        from pipelines.backbone.training.pipeline import TrainingPipeline
        from pipelines.shared.model.model_builder import ModelBuilder

        stage_dir    = self.run_dir / "training"
        name, head   = ModelBuilder.split_key(model_name)
        model_config = ModelBuilder.config_from_registry(name, self._size_overrides(model_name), head=head)

        self._apply_probed_overrides(model_name)

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
            overfit_check  = self.config.overfit_check,
        )

        pipeline.run(probe_config=self.config.probe)

    def _apply_probed_overrides(self, model_name: str) -> None:
        """Writes the model's probed batch size and LR multiplier into the shared training config.

        The entry configs of every family reference ``config.training``, so the
        overrides reach the unit regardless of the training type. Nothing is
        written when the probe stage did not run or found nothing.

        Args:
            model_name : Registry key of the benchmarked model.
        """
        measured_batch = self._max_batch_size(model_name)
        if measured_batch is not None:
            self.config.training.batch_size = measured_batch

        if self.config.find_lr.enabled:
            probed_multiplier = self._lr_multiplier(model_name)
            if probed_multiplier is not None:
                self.config.training.lr_multiplier = self.config.training.lr_multiplier * probed_multiplier

    def run(self, model_name: str, seed: int | None = None, loss_component: str | None = None, run_name: str | None = None) -> None:
        """Trains one benchmark unit, dispatching on the sweep training type.

        The unit's manifest is written first, so even a run that dies before
        its first epoch is identifiable when the benchmark is resumed; a unit
        that already finished is left as it is on a resume and refused when
        resume is off.

        Args:
            model_name     : Registry key of the model to train.
            seed           : Seed for this unit, defaulting to the sweep seed.
            loss_component : Loss component arm to isolate, backbone runs only.
            run_name       : Directory of the unit relative to the training directory, or None to compose it.

        Raises:
            ValueError : If the configured training type is not one of 'backbone', 'jepa', or 'profile_autoencoder'.
            SystemExit : If the unit already holds a completed run while resume is off.
        """
        from pipelines.benchmark.stages import BenchmarkUnitManifest

        training_type = self.config.training_type
        if training_type not in ("backbone", "jepa", "profile_autoencoder"):
            raise ValueError(f"Unknown training_type '{training_type}', expected one of ['backbone', 'jepa', 'profile_autoencoder']")

        run_name = self.unit_run_name(model_name, loss_component, seed, run_name)

        BenchmarkUnitManifest.write(self.run_dir / "training" / run_name, run_tag=self.run_tag, unit=run_name, model=model_name, loss_component=loss_component, seed=seed)

        if self.skip_complete(self.run_dir / "training" / run_name, "benchmark_training", self.config.resume):
            return

        if training_type == "profile_autoencoder":
            self._run_profile_autoencoder(model_name, run_name, seed)
            return

        if training_type == "jepa":
            self._run_jepa(model_name, run_name, seed)
            return

        self._run_backbone(model_name, run_name, seed, loss_component)


class InferenceWorker(BenchmarkWorker):
    """Worker that runs inference on one already-trained benchmark unit."""

    def run(self, model_name: str, seed: int | None = None, loss_component: str | None = None, run_name: str | None = None) -> None:
        """Runs the backbone inference pipeline over one trained unit.

        Args:
            model_name     : Registry key of the trained model.
            seed           : Seed of the unit, defaulting to the unseeded run name.
            loss_component : Loss component arm the unit was trained under.
            run_name       : Directory of the unit relative to the training directory, or None to compose it.
        """
        from pipelines.backbone.inference.pipeline import InferencePipeline
        from pipelines.shared.inference.inference_components import InferenceComponentsResolver

        run_directory = self.run_dir / "training" / self.unit_run_name(model_name, loss_component, seed, run_name)

        components = InferenceComponentsResolver.for_run(run_directory)
        pipeline   = InferencePipeline(self.factory.inference_config(run_directory), components=components)
        pipeline.run()
