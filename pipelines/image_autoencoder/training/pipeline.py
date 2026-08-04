from __future__ import annotations

from configuration.training                       import ImageAeTrainerConfig
from models.image_autoencoder                     import IMAGE_AE_CONFIG_REGISTRY, get_image_autoencoder
from pipelines.autoencoder_common.training        import AutoencoderTrainingPipeline
from pipelines.backbone.dataset.pipeline          import BackboneDatasetPreparation
from pipelines.shared.model.model_builder         import ModelBuilder
from pipelines.shared.training.training_runner    import EntryConfigTrainRunner
from pipelines.image_autoencoder.training.trainer import Trainer
from pipelines.shared.config.config_persistence   import ImageAutoencoderConfigIO


class TrainingPipeline(AutoencoderTrainingPipeline):
    run_label       = "image_ae"
    trainer_class   = Trainer
    model_dim_label = "In Channels"

    def _autoencoder_config(self, entry_config):
        return ModelBuilder.config_from_registry(entry_config.ae_model_name, entry_config.model_overrides, registry=IMAGE_AE_CONFIG_REGISTRY)

    def _build_trainer_config(self, base, entry_config):
        trainer_config = ImageAeTrainerConfig(
            gaussian          = base.gaussian,
            image_autoencoder = self.autoencoder_cfg,
            ae_loss           = entry_config.ae_loss,
        )
        trainer_config.inherit_shared_from(base)
        return trainer_config

    def _build_model(self, in_channels: int, config):
        config.in_channels = in_channels
        model, _ = get_image_autoencoder(self.ae_model_name, config)
        return model

    def _prepare_data(self, run_directory, logger):
        loaders, datasets, x_axis, x_len      = BackboneDatasetPreparation(self.dataset_config, self.trainer_config, run_directory, logger, self.entry.seed).run()
        train_loader, val_loader, test_loader = loaders

        in_channels = datasets["train"].input_channels

        return train_loader, val_loader, test_loader, x_axis, in_channels, datasets, (in_channels, x_len)

    def _save_metadata(self, run_meta, in_channels: int, x_len: int) -> None:
        run_meta.save_trainer_config()
        ImageAutoencoderConfigIO.save(self.autoencoder_cfg, self.ae_model_name, run_meta.metadata_directory)
        run_meta.save_run_summary("image_ae", in_channels=in_channels, out_channels=self.autoencoder_cfg.embedding_dim, x_axis_length=x_len, seed=self.entry.seed)


class SingleTrainRunner(EntryConfigTrainRunner):
    pipeline_class = TrainingPipeline

    @property
    def label(self) -> str:
        return self.config.ae_model_name
