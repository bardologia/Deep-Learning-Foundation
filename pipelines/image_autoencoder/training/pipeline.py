"""Training pipeline and single-run entry of the standalone image autoencoder.

The autoencoder is trained to reconstruct the same patch stack the backbone
consumes, so it reuses the backbone dataset preparation and persists its own
architecture configuration alongside the run summary.
"""

from __future__ import annotations

from configuration.training                       import ImageAeTrainerConfig
from models.image_autoencoder                     import IMAGE_AE_CONFIG_REGISTRY, get_image_autoencoder
from pipelines.autoencoder_common.training        import AutoencoderTrainingPipeline
from pipelines.backbone.dataset.pipeline          import BackboneDatasetPreparation
from pipelines.shared.model.model_builder         import ModelBuilder
from pipelines.shared.training.run_naming         import RunNaming
from pipelines.shared.training.training_runner    import EntryConfigTrainRunner
from pipelines.image_autoencoder.training.trainer import Trainer
from pipelines.shared.config.config_persistence   import ImageAutoencoderConfigIO
from tools.runtime.run_tag                        import RunTag


class TrainingPipeline(AutoencoderTrainingPipeline):
    """Trains an image autoencoder on the patch stacks of the preprocessed scene."""

    run_label       = "image_ae"
    trainer_class   = Trainer
    model_dim_label = "In Channels"

    def _autoencoder_config(self, entry_config):
        """Returns the architecture configuration of the selected image autoencoder."""
        return ModelBuilder.config_from_registry(entry_config.ae_model_name, entry_config.model_overrides, registry=IMAGE_AE_CONFIG_REGISTRY)

    def _build_trainer_config(self, base, entry_config):
        """Returns the trainer configuration carrying the architecture and loss settings.

        Args:
            base         : Shared training configuration whose common fields are inherited.
            entry_config : Entry configuration holding the autoencoder loss settings.
        """
        trainer_config = ImageAeTrainerConfig(
            gaussian          = base.gaussian,
            image_autoencoder = self.autoencoder_cfg,
            ae_loss           = entry_config.ae_loss,
        )
        trainer_config.inherit_shared_from(base)
        return trainer_config

    def _build_dataset_config(self):
        """Returns the dataset configuration of the run, rejecting input noise augmentation.

        Raises:
            ValueError : If `augmentation.p_noise` is positive, which would train the autoencoder to reproduce the injected noise realization because input and target are the same tensor.
        """
        config               = self.factory.training_dataset_config()
        config.split_regions = self.split_regions

        if config.augmentation.p_noise > 0.0:
            raise ValueError(f"augmentation.p_noise={config.augmentation.p_noise} is not usable for image-autoencoder training: PatchDataset injects the noise into the input tensor and the trainer reconstructs that same tensor, so the model would be trained to reproduce the noise realization instead of removing it. Set p_noise to 0.0; the profile autoencoder is the family that denoises, because its dataset returns a clean target alongside the noisy input.")

        return config

    def _build_model(self, in_channels: int, config):
        """Builds the autoencoder at the input width the dataset yields.

        Args:
            in_channels : Width of the assembled input patch stack.
            config      : Architecture configuration, mutated with the input width.

        Returns:
            The instantiated autoencoder.
        """
        config.in_channels = in_channels
        model, _ = get_image_autoencoder(self.ae_model_name, config)
        return model

    def _prepare_data(self, run_directory, logger):
        """Builds the train, validation and test loaders from the backbone dataset preparation.

        Args:
            run_directory : Directory the dataset metadata is written to.
            logger        : Logger the preparation reports through.

        Returns:
            The three loaders, the profile z axis, the input channel count,
            the datasets by split name, and the `(in_channels, z_len)` model
            dimensions.
        """
        loaders, datasets, z_axis, z_len      = BackboneDatasetPreparation(self.dataset_config, self.trainer_config, run_directory, logger, self.entry.seed).run()
        train_loader, val_loader, test_loader = loaders

        in_channels = datasets["train"].input_channels

        return train_loader, val_loader, test_loader, z_axis, in_channels, datasets, (in_channels, z_len)

    def _save_metadata(self, run_meta, in_channels: int, z_len: int) -> None:
        """Persists the trainer config, the architecture config and the run summary.

        Args:
            run_meta    : Run metadata writer of the training run.
            in_channels : Width of the assembled input patch stack.
            z_len       : Number of samples of the profile z axis.
        """
        run_meta.save_trainer_config()
        ImageAutoencoderConfigIO.save(self.autoencoder_cfg, self.ae_model_name, run_meta.metadata_directory)
        run_meta.save_run_summary("image_ae", in_channels=in_channels, out_channels=self.autoencoder_cfg.embedding_dim, x_axis_length=z_len, seed=self.entry.seed)


class SingleTrainRunner(EntryConfigTrainRunner):
    """Runs one image-autoencoder training job named after its architecture."""

    pipeline_class = TrainingPipeline

    @property
    def label(self) -> str:
        """Registered image-autoencoder architecture name used as the run label."""
        return self.config.ae_model_name

    def _resolve_run_name(self) -> str:
        """Returns the run directory name, the label joined with the configured name or a timestamp."""
        return RunNaming.compose(self.label, self.config.run_name or RunTag.now())
