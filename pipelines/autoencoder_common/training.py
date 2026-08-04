from __future__ import annotations

from pathlib import Path

from pipelines.shared.config.config_factory  import ConfigFactory
from pipelines.shared.config.run_metadata    import TrainingRunMetadata
from pipelines.shared.training.overfit_check import OverfitCheck
from tools.runtime.reproducibility           import Reproducibility


class AutoencoderTrainingPipeline:
    run_label       = None
    trainer_class   = None
    model_dim_label = "Model Dim"

    def __init__(self, entry_config, split_regions=None) -> None:
        self.entry   = entry_config
        self.factory = ConfigFactory(entry_config)
        Reproducibility.seed_everything(entry_config.seed)

        base = self.factory.training_trainer_config(logdir=entry_config.logdir)

        self.autoencoder_cfg = self._autoencoder_config(entry_config)
        self.ae_model_name   = entry_config.ae_model_name

        self.trainer_config          = self._build_trainer_config(base, entry_config)
        self.trainer_config.geometry = entry_config.geometry.resolved(entry_config.paths.dataset_path, secondary_labels=self.factory._secondary_labels())

        self.split_regions  = split_regions if split_regions is not None else self.factory.split_regions()
        self.dataset_config = self._build_dataset_config()

    def _build_dataset_config(self):
        config               = self.factory.training_dataset_config()
        config.split_regions = self.split_regions
        return config

    def _autoencoder_config(self, entry_config):
        raise NotImplementedError

    def _build_trainer_config(self, base, entry_config):
        raise NotImplementedError

    def _build_model(self, model_dim: int, config):
        raise NotImplementedError

    def _log_model(self, logger, model, model_dim: int) -> None:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.section("[Model Built]")
        logger.kv_table({
            "Architecture"         : self.ae_model_name,
            self.model_dim_label   : model_dim,
            "Embedding Dim"        : self.autoencoder_cfg.embedding_dim,
            "Embedding Norm"       : self.autoencoder_cfg.embedding_norm,
            "Trainable Parameters" : f"{n_params:,}",
        })

    def _prepare_data(self, run_directory, logger):
        raise NotImplementedError

    def _run_overfit_check(self, run_meta, logger, model_dim: int, x_axis, datasets) -> None:
        check = OverfitCheck(self.entry.overfit_check, run_meta.run_directory, logger)
        if not check.enabled:
            return

        check.announce()

        gate_arch_config    = check.sanitized_model_config(self.autoencoder_cfg)
        gate_trainer_config = check.sanitized_trainer_config(self.trainer_config)

        gate_model   = self._build_model(model_dim, gate_arch_config)
        gate_trainer = self.trainer_class(gate_model, gate_arch_config, x_axis, gate_trainer_config, check.work_directory, logger)

        check.run(gate_trainer, datasets["train"])

    def _save_metadata(self, run_meta, *args) -> None:
        raise NotImplementedError

    def _make_trainer(self, run_directory, logger, model, x_axis):
        return self.trainer_class(model, self.autoencoder_cfg, x_axis, self.trainer_config, run_directory, logger)

    def _train(self, run_meta, logger, model, x_axis, train_loader, val_loader, test_loader):
        trainer = self._make_trainer(run_meta.run_directory, logger, model, x_axis)
        try:
            results = trainer.train(train_loader, val_loader, test_loader)
        finally:
            run_meta.close()
        return results, run_meta.run_directory

    def build_pretrain_trainer(self, work_dir, logger):
        work_dir = Path(work_dir)

        _train_loader, _val_loader, _test_loader, x_axis, model_dim, datasets, _metadata_args = self._prepare_data(work_dir, logger)

        model   = self._build_model(model_dim, self.autoencoder_cfg)
        trainer = self._make_trainer(work_dir, logger, model, x_axis)

        return trainer, datasets["train"], model

    def run(self):
        run_meta = TrainingRunMetadata(self.trainer_config, self.run_label, Path(self.trainer_config.io.logdir), self.entry.run_name)
        logger   = run_meta.logger

        train_loader, val_loader, test_loader, x_axis, model_dim, datasets, metadata_args = self._prepare_data(run_meta.run_directory, logger)

        model = self._build_model(model_dim, self.autoencoder_cfg)
        self._log_model(logger, model, model_dim)

        self._save_metadata(run_meta, *metadata_args)

        try:
            self._run_overfit_check(run_meta, logger, model_dim, x_axis, datasets)
        except BaseException:
            run_meta.close()
            raise

        return self._train(run_meta, logger, model, x_axis, train_loader, val_loader, test_loader)
