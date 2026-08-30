"""Training pipeline for the standalone profile autoencoder.

Specialises the shared autoencoder training pipeline: builds the profile model
from the registry, assembles the profile dataset loaders and persists the
architecture, dataset and run-summary metadata.
"""

from __future__ import annotations

from configuration.dataset                          import ProfileDatasetConfig
from configuration.training                         import ProfileAeTrainerConfig
from models.profile_autoencoder                     import PROFILE_AE_CONFIG_REGISTRY, get_profile_autoencoder
from pipelines.autoencoder_common.training          import AutoencoderTrainingPipeline
from pipelines.profile_autoencoder.dataset.pipeline import ProfileDatasetPipeline
from pipelines.shared.model.model_builder           import ModelBuilder
from pipelines.shared.training.run_naming           import RunNaming
from pipelines.shared.training.training_runner      import EntryConfigTrainRunner
from pipelines.profile_autoencoder.training.trainer import Trainer
from pipelines.shared.config.config_persistence     import ProfileAutoencoderConfigIO, ProfileDatasetConfigIO
from tools.runtime.run_tag                          import RunTag


class TrainingPipeline(AutoencoderTrainingPipeline):
    """Trains a profile autoencoder on curves rebuilt from extracted parameters."""

    run_label       = "profile_ae"
    trainer_class   = Trainer
    model_dim_label = "Profile Length"

    def _autoencoder_config(self, entry_config):
        """Builds the architecture config from the registry and entry overrides.

        Args:
            entry_config : Entry configuration carrying the model name, encoder and decoder learning rates and weight decays, and any architecture overrides.

        Returns:
            The populated profile-autoencoder configuration dataclass.
        """
        optimization = {
            "encoder_lr" : entry_config.encoder_lr,
            "decoder_lr" : entry_config.decoder_lr,
            "encoder_wd" : entry_config.encoder_wd,
            "decoder_wd" : entry_config.decoder_wd,
        }
        return ModelBuilder.config_from_registry(entry_config.ae_model_name, {**optimization, **entry_config.model_overrides}, registry=PROFILE_AE_CONFIG_REGISTRY)

    def _build_trainer_config(self, base, entry_config):
        """Assembles the trainer config from the shared base and entry settings.

        Args:
            base         : Shared configuration providing the Gaussian settings.
            entry_config : Entry configuration supplying the loss settings.

        Returns:
            The trainer configuration with the shared fields inherited.
        """
        trainer_config = ProfileAeTrainerConfig(
            gaussian    = base.gaussian,
            autoencoder = self.autoencoder_cfg,
            ae_loss     = entry_config.ae_loss,
        )
        trainer_config.inherit_shared_from(base)
        return trainer_config

    def _build_model(self, z_len: int, config):
        """Instantiates the autoencoder for a profile of `z_len` z bins."""
        config.profile_length = z_len
        model, _ = get_profile_autoencoder(self.ae_model_name, config)
        return model

    def _build_dataset_config(self) -> ProfileDatasetConfig:
        """Returns the dataset configuration derived from the entry configuration."""
        gaussian_cfg = self.trainer_config.gaussian
        training     = self.entry.training

        return ProfileDatasetConfig(
            preprocessing_run_directory = self.entry.paths.dataset_path,
            split_regions               = self.split_regions,
            parameters_path             = self.entry.paths.parameters_path,
            n_gaussians                 = gaussian_cfg.n_default_gaussians,
            x_min                       = gaussian_cfg.x_min,
            x_max                       = gaussian_cfg.x_max,
            pixel_subsample             = self.entry.pixel_subsample,
            keep_empty_frac             = self.entry.keep_empty_frac,
            amp_zero_thr                = self.entry.amp_zero_thr,
            stats_max_samples           = self.entry.stats_max_samples,
            batch_size                  = training.batch_size,
            num_workers                 = training.num_workers,
            prefetch_factor             = training.prefetch_factor,
            pin_memory                  = True,
            shuffle_train               = True,
            augmentation                = self.entry.profile_augmentation,
        )

    def _prepare_data(self, run_directory, logger):
        """Builds the profile loaders and elevation axis for a training run.

        Args:
            run_directory : Directory of the run being trained.
            logger        : Logger passed to the dataset pipeline.

        Returns:
            Tuple of the train, validation and test loaders, the elevation axis
            of shape (H,) in metres, its length, the datasets keyed by split,
            and the metadata payload for `_save_metadata`.
        """
        profile_config   = self.dataset_config
        dataset_pipeline = ProfileDatasetPipeline(profile_config, run_directory, logger=logger, seed=self.entry.seed, resume=self.trainer_config.training.resume)

        (train_loader, val_loader, test_loader), datasets, z_axis, z_len, _normalizer = dataset_pipeline.run()

        return train_loader, val_loader, test_loader, z_axis, z_len, datasets, (profile_config, z_len)

    def _save_metadata(self, run_meta, profile_config: ProfileDatasetConfig, z_len: int) -> None:
        """Persists trainer, architecture, dataset and summary metadata.

        Args:
            run_meta       : Run metadata writer.
            profile_config : Dataset configuration used for this run.
            z_len          : Profile length in z bins.
        """
        run_meta.save_trainer_config()
        ProfileAutoencoderConfigIO.save(self.autoencoder_cfg, self.ae_model_name, run_meta.metadata_directory)
        ProfileDatasetConfigIO.save(profile_config, run_meta.metadata_directory)
        run_meta.save_run_summary("profile_ae", in_channels=z_len, out_channels=self.autoencoder_cfg.embedding_dim, x_axis_length=z_len, seed=self.entry.seed)


class SingleTrainRunner(EntryConfigTrainRunner):
    """Runs one profile-autoencoder training from an entry configuration."""

    pipeline_class = TrainingPipeline

    @property
    def label(self) -> str:
        """Returns the autoencoder architecture name used as the run label."""
        return self.config.ae_model_name

    def _resolve_run_name(self) -> str:
        """Returns the run directory name, timestamped when none was given."""
        return RunNaming.compose(self.label, self.config.run_name or RunTag.now())
