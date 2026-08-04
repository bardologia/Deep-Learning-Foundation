from __future__ import annotations

from pathlib import Path

import numpy as np

from configuration.dataset                      import DatasetConfig
from configuration.training                     import BackboneTrainerConfig, OverfitCheckConfig
from models                                     import get_backbone
from pipelines.backbone.dataset.pipeline        import DatasetPipeline
from pipelines.backbone.training.loss_terms     import LossComponentCatalog
from pipelines.backbone.training.trainer        import Trainer
from pipelines.shared.config.config_persistence import BackboneModelConfigIO
from pipelines.shared.config.run_metadata       import TrainingRunMetadata
from pipelines.shared.model.model_builder       import ModelBuilder
from pipelines.shared.training.overfit_check    import OverfitCheck
from tools.data.gaussians                       import GaussianAxis, GaussianHead
from tools.runtime.config_cli                   import ConfigCli
from tools.runtime.reproducibility              import Reproducibility


class TrainingPipeline:

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

        self.trainer_config = trainer_config
        self.dataset_config = dataset_config
        self.backbone_name  = backbone_name
        self.model_config   = model_config
        self.patch_size     = dataset_config.patch.size
        self.seed           = seed
        self.run_name       = run_name
        self.overfit_check  = overfit_check if overfit_check is not None else OverfitCheckConfig()

        Reproducibility.seed_everything(self.seed)

        self.dataset_config.n_gaussians = trainer_config.gaussian.n_default_gaussians

    @staticmethod
    def physics_geometry_active(trainer_config) -> bool:
        flags = ("use_coherence_resyn", "use_covariance_match", "use_capon_cycle")

        return any(getattr(cfg, flag) for cfg in trainer_config.curriculum.active_stages() for flag in flags)

    def _build_dataset_pipeline(self, run_directory: Path, logger) -> DatasetPipeline:
        gaussian_cfg = self.trainer_config.gaussian

        dataset_pipeline = DatasetPipeline(
            config                 = self.dataset_config,
            training_run_directory = Path(run_directory),
            logger                 = logger,
            seed                   = self.seed,
            height_axis_convention = self.trainer_config.geometry.height_axis_convention,
            build_geometry_field   = self.physics_geometry_active(self.trainer_config),
        )

        self.dataset_config.x_axis = GaussianAxis.build(gaussian_cfg.x_min, gaussian_cfg.x_max, dataset_pipeline.layout.profile_length)

        return dataset_pipeline

    def _run_overfit_check(self, train_dataset, in_channels: int, out_channels: int, x_axis) -> None:
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
            x_axis     = x_axis,
            config     = gate_trainer_config,
            run_dir    = check.work_directory,
            logger     = self.logger,
            norm_stats = train_dataset.normalizer,
            emit_docs  = False,
        )

        check.run(gate_trainer, train_dataset)

    def _build_model(self, in_channels: int, out_channels: int):
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
        return self.backbone_name

    def _model_overrides(self, in_channels: int, out_channels: int) -> dict:
        overrides = {"in_channels": in_channels, "out_channels": out_channels}
        overrides.update(ModelBuilder.image_size_override(self.backbone_name, self.patch_size))

        return overrides

    def _model_factory(self):
        return get_backbone

    def _make_trainer(self, model, model_cfg, x_axis, norm_stats, run_dir: Path, logger, emit_docs: bool = True):
        return Trainer(
            model      = model,
            model_cfg  = model_cfg,
            x_axis     = x_axis,
            config     = self.trainer_config,
            run_dir    = run_dir,
            logger     = logger,
            norm_stats = norm_stats,
            emit_docs  = emit_docs,
        )

    def build_pretrain_trainer(self, work_dir: Path, logger):
        work_dir         = Path(work_dir)
        dataset_pipeline = self._build_dataset_pipeline(work_dir, logger)

        _train_loader, _val_loader, _test_loader, datasets = dataset_pipeline.run()

        gaussian_cfg = self.trainer_config.gaussian
        dataset      = datasets["train"]
        in_channels  = dataset.input_channels
        out_channels = GaussianHead.total_channels(gaussian_cfg.params_per_gaussian, gaussian_cfg.n_default_gaussians)
        x_axis       = np.asarray(self.dataset_config.x_axis, dtype=np.float32)

        model, model_cfg = self._model_factory()(self.backbone_name, config=self.model_config, **self._model_overrides(in_channels, out_channels))

        trainer = self._make_trainer(model, model_cfg, x_axis, dataset.normalizer, work_dir, logger, emit_docs=False)
        trainer.criterion.set_curriculum(LossComponentCatalog.probe_union(self.trainer_config.curriculum), context="probe union of all curriculum stages")

        return trainer, dataset, model

    def run(self, probe_config=None, resolved_entry_config=None):
        self.run_metadata = TrainingRunMetadata(
            trainer_config = self.trainer_config,
            model_name     = self.backbone_name,
            base_logdir    = Path(self.trainer_config.io.logdir),
            run_name       = self.run_name,
        )
        self.logger = self.run_metadata.logger

        if resolved_entry_config is not None:
            resolved_path = self.run_metadata.run_directory / "docs" / "resolved_entry_config.json"
            ConfigCli.save_resolved(resolved_entry_config, resolved_path)
            self.logger.info(f"Resolved entry config saved: {resolved_path}")

        self.logger.section("[PyTorch Training Pipeline Execution]")
        self.logger.kv_table({
            "Seed"           : self.seed,
            "Gaussian slots" : self.trainer_config.gaussian.n_default_gaussians,
            "Elevation axis" : f"[{self.trainer_config.gaussian.x_min:g}, {self.trainer_config.gaussian.x_max:g}] m (sample count set by the dataset profile length)",
        })

        self.dataset_pipeline = self._build_dataset_pipeline(self.run_metadata.run_directory, self.logger)

        gaussian_cfg = self.trainer_config.gaussian

        train_loader, val_loader, test_loader, datasets = self.dataset_pipeline.run()

        train_dataset = datasets["train"]
        in_channels   = train_dataset.input_channels
        n_gaussians   = gaussian_cfg.n_default_gaussians
        out_channels  = GaussianHead.total_channels(gaussian_cfg.params_per_gaussian, n_gaussians)
        x_axis        = np.asarray(self.dataset_config.x_axis, dtype=np.float32)
        x_axis_length = int(x_axis.size)

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
            self._run_overfit_check(train_dataset, in_channels, out_channels, x_axis)

            trainer = self._make_trainer(model, model_cfg, x_axis, train_dataset.normalizer, self.run_metadata.run_directory, self.logger)

            trainer.maybe_run_loss_probe(train_loader, probe_config)
            results = trainer.train(train_loader, val_loader, test_loader)
            self.run_metadata.save_test_metrics(trainer.test_metrics)
        finally:
            self.run_metadata.close()

        return results
