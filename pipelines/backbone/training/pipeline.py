"""End-to-end backbone training pipeline: dataset, model, gate check and trainer.

Wires the patch dataset, the backbone under the configured head and the trainer
together, persists the run metadata and configurations, and runs the overfit
gate before the real training loop.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from configuration.dataset                         import DatasetConfig
from configuration.training                        import BackboneTrainerConfig, OverfitCheckConfig
from models                                        import get_backbone
from pipelines.backbone.dataset.pipeline           import DatasetPipeline
from pipelines.backbone.training.loss_terms        import LossComponentCatalog
from pipelines.backbone.training.trainer           import Trainer
from pipelines.shared.config.config_persistence    import BackboneModelConfigIO
from pipelines.shared.config.resolved_entry_config import ResolvedEntryConfigWriter
from pipelines.shared.config.run_metadata          import TrainingRunMetadata
from pipelines.shared.model.model_builder          import ModelBuilder
from pipelines.shared.training.overfit_check       import OverfitCheck
from tools.data.gaussians                          import GaussianAxis, GaussianHead
from tools.runtime.reproducibility                 import Reproducibility


class TrainingPipeline:
    """Runs one backbone training job from dataset preparation to test metrics.

    Attributes:
        trainer_config : Trainer, curriculum, gaussian and geometry configuration.
        dataset_config : Patch dataset configuration.
        backbone_name  : Registered backbone architecture name.
        model_config   : Resolved model configuration, or None to take registry defaults.
        seed           : Seed applied to every random source of the run.
        run_name       : Explicit run directory name, or None to derive one.
        overfit_check  : Configuration of the pre-training overfit gate.
        patch_size     : Patch size after the backbone's squaring requirement is applied.
    """

    MODEL_CONFIG_IO = BackboneModelConfigIO

    def __init__(
        self,
        trainer_config : BackboneTrainerConfig,
        dataset_config : DatasetConfig,
        backbone_name  : str,
        model_config   = None,
        seed           : int = 0,
        run_name       : str | None = None,
        overfit_check  : OverfitCheckConfig | None = None,
    ) -> None:
        """Seeds the run and reconciles the patch geometry with the backbone requirements.

        Args:
            trainer_config : Trainer, curriculum, gaussian and geometry configuration.
            dataset_config : Patch dataset configuration.
            backbone_name  : Registered backbone architecture name.
            model_config   : Resolved model configuration, or None for registry defaults.
            seed           : Seed applied to every random source of the run.
            run_name       : Explicit run directory name, or None to derive one.
            overfit_check  : Overfit gate configuration; a default one when omitted.
        """

        self.trainer_config = trainer_config
        self.dataset_config = dataset_config
        self.backbone_name  = backbone_name
        self.model_config   = model_config
        self.seed           = seed
        self.run_name       = run_name
        self.overfit_check  = overfit_check if overfit_check is not None else OverfitCheckConfig()

        patch                    = self.dataset_config.patch
        patch.size, patch.stride = ModelBuilder.square_patch(self._square_patch_backbones(), patch.size, patch.stride)
        self.patch_size          = patch.size

        Reproducibility.seed_everything(self.seed)

        self.dataset_config.n_gaussians = trainer_config.gaussian.n_default_gaussians

    def _square_patch_backbones(self) -> list[str]:
        """Returns the backbones whose square-patch requirement constrains the patch geometry."""
        return [self.backbone_name]

    def _build_dataset_pipeline(self, run_directory: Path, logger) -> DatasetPipeline:
        """Builds the patch dataset pipeline and derives the elevation axis from its profile length.

        Args:
            run_directory : Directory the dataset artefacts are written under.
            logger        : Logger passed to the dataset pipeline.

        Returns:
            The configured dataset pipeline, not yet run.
        """
        gaussian_cfg = self.trainer_config.gaussian

        dataset_pipeline = DatasetPipeline(
            config                 = self.dataset_config,
            training_run_directory = Path(run_directory),
            logger                 = logger,
            seed                   = self.seed,
            resume                 = self.trainer_config.training.resume,
        )

        self.dataset_config.z_axis = GaussianAxis.build(gaussian_cfg.x_min, gaussian_cfg.x_max, dataset_pipeline.layout.profile_length)

        return dataset_pipeline

    def _run_overfit_check(self, train_dataset, in_channels: int, out_channels: int, z_axis) -> None:
        """Runs the pre-training gate that must overfit a tiny subset before real training.

        The gate trains a sanitized copy of the model on a single-stage loss with
        active normalization disabled, so a failure points at the model or data
        rather than the curriculum.

        Args:
            train_dataset : Training dataset the gate draws its samples from.
            in_channels   : Model input channel count.
            out_channels  : Model output channel count.
            z_axis        : Z sample positions in metres, of shape (profile_length,).
        """
        check = OverfitCheck(self.overfit_check, self.run_metadata.run_directory, self.logger)
        if not check.enabled:
            return

        check.announce()

        gate_trainer_config = check.sanitized_trainer_config(self.trainer_config)

        gate_stage                          = gate_trainer_config.curriculum.initial_stage
        gate_stage.use_active_normalization = False

        gate_trainer_config.curriculum.enabled  = False
        gate_trainer_config.curriculum.warmup   = gate_stage
        gate_trainer_config.curriculum.complete = gate_stage

        check.record("curriculum.enabled", False)
        check.record("gate_loss_stage", "warmup" if self.trainer_config.curriculum.enabled else "complete")
        check.record("gate_stage.use_active_normalization", False)

        base_config       = self.model_config if self.model_config is not None else ModelBuilder.config_from_registry(self.backbone_name, {})
        gate_model_config = check.sanitized_model_config(base_config)

        gate_model, gate_model_cfg = self._model_factory()(self.backbone_name, config=gate_model_config, **self._model_overrides(in_channels, out_channels))

        gate_trainer = Trainer(
            model      = gate_model,
            model_cfg  = gate_model_cfg,
            z_axis     = z_axis,
            config     = gate_trainer_config,
            run_dir    = check.work_directory,
            logger     = self.logger,
            norm_stats = train_dataset.normalizer,
            emit_docs  = False,
        )

        check.run(gate_trainer, train_dataset)

    def _build_model(self, in_channels: int, out_channels: int):
        """Builds the backbone at the given widths and logs its architecture and size.

        Args:
            in_channels  : Model input channel count.
            out_channels : Model output channel count.

        Returns:
            The model and its resolved model configuration.
        """
        model, model_cfg = self._model_factory()(self.backbone_name, config=self.model_config, **self._model_overrides(in_channels, out_channels))

        n_total     = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        params      = f"{n_trainable:,} trainable" if n_total == n_trainable else f"{n_trainable:,} trainable ({n_total:,} total, {n_total - n_trainable:,} frozen)"

        self.logger.section("[Model Built]")
        self.logger.subsection(f"Architecture : {self._architecture_label()}")
        self.logger.subsection(f"Head         : {model_cfg.head}")
        self.logger.subsection(f"In Channels  : {in_channels}")
        self.logger.subsection(f"Out Channels : {out_channels}")
        self.logger.subsection(f"Parameters   : {params}")
        return model, model_cfg

    def _architecture_label(self) -> str:
        """Returns the architecture name shown in the model log banner."""
        return self.backbone_name

    def _model_overrides(self, in_channels: int, out_channels: int) -> dict:
        """Returns the channel widths plus any image-size override the backbone requires."""
        overrides = {"in_channels": in_channels, "out_channels": out_channels}
        overrides.update(ModelBuilder.image_size_override(self.backbone_name, self.patch_size))

        return overrides

    def _model_factory(self):
        """Returns the factory that instantiates the architecture family of this pipeline."""
        return get_backbone

    def _make_trainer(self, model, model_cfg, z_axis, norm_stats, run_dir: Path, logger, emit_docs: bool = True, writer=None):
        """Builds the trainer driving the training loop.

        Args:
            model      : The backbone to train.
            model_cfg  : Resolved model configuration.
            z_axis     : Z sample positions in metres, of shape (profile_length,).
            norm_stats : Normalizer of the dataset the model was fitted on.
            run_dir    : Directory the trainer writes checkpoints and logs into.
            logger     : Logger for the training loop.
            emit_docs  : Whether the trainer writes the architecture documentation.
            writer     : Optional TensorBoard writer.

        Returns:
            The configured trainer.
        """
        return Trainer(
            model      = model,
            model_cfg  = model_cfg,
            z_axis     = z_axis,
            config     = self.trainer_config,
            run_dir    = run_dir,
            logger     = logger,
            norm_stats = norm_stats,
            emit_docs  = emit_docs,
            writer     = writer,
        )

    def build_pretrain_trainer(self, work_dir: Path, logger):
        """Builds a throwaway trainer over the probe-union loss for the pretraining probes.

        Args:
            work_dir : Scratch directory the probe artefacts are written into.
            logger   : Logger for the probe run.

        Returns:
            Tuple of the trainer, the training dataset and the model.
        """
        work_dir         = Path(work_dir)
        dataset_pipeline = self._build_dataset_pipeline(work_dir, logger)

        _train_loader, _val_loader, _test_loader, datasets = dataset_pipeline.run()

        gaussian_cfg = self.trainer_config.gaussian
        dataset      = datasets["train"]
        in_channels  = dataset.input_channels
        out_channels = GaussianHead.total_channels(gaussian_cfg.params_per_gaussian, gaussian_cfg.n_default_gaussians)
        z_axis       = np.asarray(self.dataset_config.z_axis, dtype=np.float32)

        model, model_cfg = self._model_factory()(self.backbone_name, config=self.model_config, **self._model_overrides(in_channels, out_channels))

        trainer = self._make_trainer(model, model_cfg, z_axis, dataset.normalizer, work_dir, logger, emit_docs=False)
        trainer.criterion.set_curriculum(LossComponentCatalog.probe_union(self.trainer_config.curriculum), context="probe union of all curriculum stages")

        return trainer, dataset, model

    def run(self, probe_config=None, resolved_entry_config=None):
        """Prepares the data and model, runs the gate and trains, then saves test metrics.

        Args:
            probe_config          : Optional loss-scale probe configuration run before training.
            resolved_entry_config : Optional resolved entry config persisted into the run's docs directory.

        Returns:
            The training results returned by the trainer.
        """
        self.run_metadata = TrainingRunMetadata(
            trainer_config = self.trainer_config,
            model_name     = self.backbone_name,
            base_logdir    = Path(self.trainer_config.io.logdir),
            run_name       = self.run_name,
        )
        self.logger = self.run_metadata.logger

        if resolved_entry_config is not None:
            ResolvedEntryConfigWriter.save(resolved_entry_config, self.run_metadata.run_directory, self.logger)

        self.logger.section("[PyTorch Training Pipeline Execution]")
        self.logger.kv_table({
            "Seed"           : self.seed,
            "Gaussian slots" : self.trainer_config.gaussian.n_default_gaussians,
            "Z axis"         : f"[{self.trainer_config.gaussian.x_min:g}, {self.trainer_config.gaussian.x_max:g}] m (sample count set by the dataset profile length)",
        })

        self.dataset_pipeline = self._build_dataset_pipeline(self.run_metadata.run_directory, self.logger)

        gaussian_cfg = self.trainer_config.gaussian

        train_loader, val_loader, test_loader, datasets = self.dataset_pipeline.run()

        train_dataset = datasets["train"]
        in_channels   = train_dataset.input_channels
        n_gaussians   = gaussian_cfg.n_default_gaussians
        out_channels  = GaussianHead.total_channels(gaussian_cfg.params_per_gaussian, n_gaussians)
        z_axis        = np.asarray(self.dataset_config.z_axis, dtype=np.float32)
        x_axis_length = int(z_axis.size)

        model, model_cfg = self._build_model(in_channels=in_channels, out_channels=out_channels)

        self.run_metadata.save_trainer_config()
        self.run_metadata.save_model_config(model_cfg, self.backbone_name, config_io=self.MODEL_CONFIG_IO)

        self.run_metadata.save_run_summary(
            model_name       = self.backbone_name,
            in_channels      = in_channels,
            out_channels     = out_channels,
            x_axis_length    = x_axis_length,
            n_gaussians      = n_gaussians,
            seed             = self.seed,
        )

        try:
            self._run_overfit_check(train_dataset, in_channels, out_channels, z_axis)

            trainer = self._make_trainer(model, model_cfg, z_axis, train_dataset.normalizer, self.run_metadata.run_directory, self.logger, writer=self.run_metadata.writer)

            trainer.maybe_run_loss_probe(train_loader, probe_config)
            results = trainer.train(train_loader, val_loader, test_loader)
            self.run_metadata.save_test_metrics(trainer.test_metrics)
        finally:
            self.run_metadata.close()

        return results
