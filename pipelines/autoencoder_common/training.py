"""Shared training orchestration for the profile and image autoencoders.

Subclasses declare the run label and trainer class, then supply the dataset,
architecture and trainer configuration hooks the orchestrator calls in order.
"""

from __future__ import annotations

from pathlib import Path

from pipelines.shared.config.config_factory  import ConfigFactory
from pipelines.shared.config.run_metadata    import TrainingRunMetadata
from pipelines.shared.training.overfit_check import OverfitCheck
from tools.runtime.reproducibility           import Reproducibility


class AutoencoderTrainingPipeline:
    """Prepares data, model and trainer for one autoencoder training run.

    Attributes:
        entry           : Entry configuration of the training script.
        factory         : Config factory deriving trainer and split configurations.
        autoencoder_cfg : Architecture configuration of the autoencoder.
        ae_model_name   : Registry name of the chosen architecture.
        trainer_config  : Fully resolved trainer configuration, including geometry.
        split_regions   : Region descriptors for the train, validation and test splits.
        dataset_config  : Dataset configuration built for this family.
    """

    run_label       = None
    trainer_class   = None
    model_dim_label = "Model Dim"

    def __init__(self, entry_config, split_regions=None) -> None:
        """Seeds the run and resolves the architecture, trainer and dataset configs.

        Args:
            entry_config  : Entry configuration of the training script.
            split_regions : Optional pre-computed split regions; derived from the config factory when omitted.
        """
        self.entry   = entry_config
        self.factory = ConfigFactory(entry_config)
        Reproducibility.seed_everything(entry_config.seed)

        base = self.factory.training_trainer_config(logdir=entry_config.logdir)

        self.autoencoder_cfg = self._autoencoder_config(entry_config)
        self.ae_model_name   = entry_config.ae_model_name

        self.trainer_config          = self._build_trainer_config(base, entry_config)

        self.split_regions  = split_regions if split_regions is not None else self.factory.split_regions()
        self.dataset_config = self._build_dataset_config()

    def _build_dataset_config(self):
        """Returns the family-specific dataset configuration; implemented by subclasses."""
        raise NotImplementedError

    def _autoencoder_config(self, entry_config):
        """Returns the architecture configuration for the run; implemented by subclasses."""
        raise NotImplementedError

    def _build_trainer_config(self, base, entry_config):
        """Returns the trainer configuration derived from the base config; implemented by subclasses."""
        raise NotImplementedError

    def _build_model(self, model_dim: int, config):
        """Returns an autoencoder instance for the given input dimension; implemented by subclasses."""
        raise NotImplementedError

    def _log_model(self, logger, model, model_dim: int) -> None:
        """Logs the architecture name, dimensions and trainable parameter count."""
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
        """Returns loaders, elevation axis, model dimension, datasets and metadata arguments; implemented by subclasses."""
        raise NotImplementedError

    def _run_overfit_check(self, run_meta, logger, model_dim: int, z_axis, datasets) -> None:
        """Runs the pre-training sanity gate on a sanitized copy of model and trainer configs.

        Args:
            run_meta  : Run metadata holding the run directory.
            logger    : Logger for the run.
            model_dim : Input dimension the gate model is built with.
            z_axis    : Z axis in metres of shape (profile_length,).
            datasets  : Mapping of split name to dataset; the train split feeds the gate.
        """
        check = OverfitCheck(self.entry.overfit_check, run_meta.run_directory, logger)
        if not check.enabled:
            return

        check.announce()

        gate_arch_config    = check.sanitized_model_config(self.autoencoder_cfg)
        gate_trainer_config = check.sanitized_trainer_config(self.trainer_config)

        gate_model   = self._build_model(model_dim, gate_arch_config)
        gate_trainer = self.trainer_class(gate_model, gate_arch_config, z_axis, gate_trainer_config, check.work_directory, logger)

        check.run(gate_trainer, datasets["train"])

    def _save_metadata(self, run_meta, *args) -> None:
        """Writes the run metadata files; implemented by subclasses."""
        raise NotImplementedError

    def _make_trainer(self, run_directory, logger, model, z_axis, writer=None):
        """Returns a trainer bound to the run directory, model and z axis."""
        return self.trainer_class(model, self.autoencoder_cfg, z_axis, self.trainer_config, run_directory, logger, writer)

    def _train(self, run_meta, logger, model, z_axis, train_loader, val_loader, test_loader):
        """Trains the model and returns the training results with the run directory."""
        trainer = self._make_trainer(run_meta.run_directory, logger, model, z_axis, writer=run_meta.writer)
        results = trainer.train(train_loader, val_loader, test_loader)

        return results, run_meta.run_directory

    def build_pretrain_trainer(self, work_dir, logger):
        """Builds a throwaway trainer for the pre-training batch and loader tuners.

        Args:
            work_dir : Scratch directory the tuning trainer writes into.
            logger   : Logger for the tuning phase.

        Returns:
            Tuple of the trainer, the training dataset and the model.
        """
        work_dir = Path(work_dir)

        _train_loader, _val_loader, _test_loader, z_axis, model_dim, datasets, _metadata_args = self._prepare_data(work_dir, logger)

        model   = self._build_model(model_dim, self.autoencoder_cfg)
        trainer = self._make_trainer(work_dir, logger, model, z_axis)

        return trainer, datasets["train"], model

    def run(self):
        """Executes data preparation, model build, overfit gate and training in order.

        Returns:
            Tuple of the training results and the run directory.
        """
        with TrainingRunMetadata(self.trainer_config, self.run_label, Path(self.trainer_config.io.logdir), self.entry.run_name) as run_meta:
            logger = run_meta.logger

            train_loader, val_loader, test_loader, z_axis, model_dim, datasets, metadata_args = self._prepare_data(run_meta.run_directory, logger)

            model = self._build_model(model_dim, self.autoencoder_cfg)
            self._log_model(logger, model, model_dim)

            self._save_metadata(run_meta, *metadata_args)
            self._run_overfit_check(run_meta, logger, model_dim, z_axis, datasets)

            return self._train(run_meta, logger, model, z_axis, train_loader, val_loader, test_loader)
