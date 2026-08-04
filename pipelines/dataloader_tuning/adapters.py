from __future__ import annotations

from dataclasses import dataclass
from pathlib     import Path
from typing      import Callable

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import Dataset

from tools.data.gaussians import GaussianAxis, GaussianHead, GaussianMixture


FEED_MODES = ("synthetic", "profile_autoencoder", "image_autoencoder", "backbone")

DEFAULT_MODEL = {
    "synthetic"           : "mlp_ae",
    "profile_autoencoder" : "mlp_ae",
    "image_autoencoder"   : "conv2d_ae",
    "backbone"            : "resunet",
}


@dataclass
class FeedTarget:
    dataset        : Dataset
    model          : torch.nn.Module
    to_model_input : Callable[[object, torch.device], torch.Tensor]
    forward_loss   : Callable[[torch.nn.Module, torch.Tensor], torch.Tensor]
    model_name     : str
    sample_text    : str
    item_source    : str
    config_hint    : str


class FeedLosses:
    @staticmethod
    def reconstruction(model: torch.nn.Module, model_input: torch.Tensor) -> torch.Tensor:
        reconstruction, _ = model.reconstruct(model_input)
        return functional.mse_loss(reconstruction, model_input)

    @staticmethod
    def supervised(model: torch.nn.Module, model_input: torch.Tensor) -> torch.Tensor:
        prediction = model(model_input)
        return functional.mse_loss(prediction, torch.zeros_like(prediction))


class SyntheticCurveDataset(Dataset):
    def __init__(self, n_samples: int, profile_length: int, seed: int) -> None:
        self.n_samples      = int(n_samples)
        self.profile_length = int(profile_length)
        self.x_axis         = np.linspace(0.0, 1.0, self.profile_length, dtype=np.float32)
        self.rng_seed       = int(seed)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int):
        rng = np.random.default_rng(self.rng_seed + index)

        amplitudes = rng.uniform(0.2, 1.0, size=(1, 3)).astype(np.float32)
        means      = rng.uniform(0.2, 0.8, size=(1, 3)).astype(np.float32)
        sigmas     = rng.uniform(0.02, 0.08, size=(1, 3)).astype(np.float32)

        curve = GaussianMixture.evaluate_batch(self.x_axis, amplitudes, means, sigmas)[0]

        return curve.astype(np.float32)


class ProfileFeedAdapter:
    def __init__(self, config, work_dir: Path, logger) -> None:
        self.config     = config
        self.work_dir   = Path(work_dir)
        self.logger     = logger
        self.model_name = config.model_name or DEFAULT_MODEL["profile_autoencoder"]

    @staticmethod
    def to_model_input(batch, device: torch.device) -> torch.Tensor:
        return batch.to(device, non_blocking=True).unsqueeze(-1).unsqueeze(-1)

    def _training_pipeline(self):
        from configuration.training import ProfileAeEntryConfig
        from pipelines.profile_autoencoder.training.pipeline import TrainingPipeline

        entry = ProfileAeEntryConfig(ae_model_name=self.model_name, seed=self.config.seed)
        entry.paths.dataset_path    = self.config.paths.dataset_path
        entry.paths.parameters_path = self.config.paths.parameters_path
        entry.pixel_subsample       = self.config.pixel_subsample
        entry.keep_empty_frac       = self.config.keep_empty_frac

        return TrainingPipeline(entry)

    def _dataset(self, training_pipeline):
        from pipelines.profile_autoencoder.dataset.pipeline import ProfileDatasetPipeline

        profile_dataset_config = training_pipeline._profile_dataset_config()
        dataset_pipeline       = ProfileDatasetPipeline(profile_dataset_config, self.work_dir, logger=self.logger, seed=self.config.seed)

        (_loaders, datasets, _axis, profile_length, _normalizer) = dataset_pipeline.run()

        return datasets["train"], profile_length

    def _model(self, training_pipeline, profile_length: int):
        from models.profile_autoencoder import get_profile_autoencoder

        model, _ = get_profile_autoencoder(self.model_name, training_pipeline.autoencoder_cfg, profile_length=profile_length)
        return model

    def build(self) -> FeedTarget:
        training_pipeline       = self._training_pipeline()
        dataset, profile_length = self._dataset(training_pipeline)
        model                   = self._model(training_pipeline, profile_length)

        return FeedTarget(
            dataset        = dataset,
            model          = model,
            to_model_input = self.to_model_input,
            forward_loss   = FeedLosses.reconstruction,
            model_name     = self.model_name,
            sample_text    = f"profile curves, length {profile_length}, {len(dataset):,} samples",
            item_source    = "ProfileDataset.__getitem__ (per-profile GaussianMixture.evaluate_batch synthesis)",
            config_hint    = "configuration/dataset/profile_autoencoder.py ProfileDatasetConfig",
        )


class ImageFeedAdapter:
    def __init__(self, config, work_dir: Path, logger) -> None:
        self.config     = config
        self.work_dir   = Path(work_dir)
        self.logger     = logger
        self.model_name = config.model_name or DEFAULT_MODEL["image_autoencoder"]

    @staticmethod
    def to_model_input(batch, device: torch.device) -> torch.Tensor:
        return batch[0].to(device, non_blocking=True)

    def _training_pipeline(self):
        from configuration.training.image_autoencoder import ImageAeEntryConfig
        from pipelines.image_autoencoder.training.pipeline import TrainingPipeline

        entry = ImageAeEntryConfig(ae_model_name=self.model_name, seed=self.config.seed)
        entry.paths.dataset_path    = self.config.paths.dataset_path
        entry.paths.parameters_path = self.config.paths.parameters_path

        return TrainingPipeline(entry)

    def _dataset(self, training_pipeline):
        from pipelines.backbone.dataset.pipeline import DatasetPipeline

        dataset_config  = training_pipeline.dataset_config
        gaussian_config = training_pipeline.trainer_config.gaussian

        dataset_config.n_gaussians = gaussian_config.n_default_gaussians

        dataset_pipeline = DatasetPipeline(dataset_config, self.work_dir, logger=self.logger, seed=self.config.seed)
        profile_length   = dataset_pipeline.profile_length

        dataset_config.x_axis = GaussianAxis.build(gaussian_config.x_min, gaussian_config.x_max, profile_length)

        _train_loader, _val_loader, _test_loader, datasets = dataset_pipeline.run()

        return datasets["train"]

    def _model(self, training_pipeline, input_channels: int):
        from models.image_autoencoder import get_image_autoencoder

        model, _ = get_image_autoencoder(self.model_name, training_pipeline.autoencoder_cfg, in_channels=input_channels)
        return model

    def build(self) -> FeedTarget:
        training_pipeline = self._training_pipeline()
        dataset           = self._dataset(training_pipeline)
        input_channels    = dataset.input_channels
        model             = self._model(training_pipeline, input_channels)

        sample = np.asarray(dataset[0][0])

        return FeedTarget(
            dataset        = dataset,
            model          = model,
            to_model_input = self.to_model_input,
            forward_loss   = FeedLosses.reconstruction,
            model_name     = self.model_name,
            sample_text    = f"patches {tuple(sample.shape)}, {len(dataset):,} samples, {input_channels} channels",
            item_source    = "PatchDataset.__getitem__ (patch extraction + complex->representation conversion)",
            config_hint    = "configuration/dataset/general/dataset.py DatasetConfig",
        )


class SyntheticFeedAdapter:
    def __init__(self, config, work_dir: Path, logger) -> None:
        self.config     = config
        self.logger     = logger
        self.model_name = config.model_name or DEFAULT_MODEL["synthetic"]

    @staticmethod
    def to_model_input(batch, device: torch.device) -> torch.Tensor:
        return batch.to(device, non_blocking=True).unsqueeze(-1).unsqueeze(-1)

    def build(self) -> FeedTarget:
        from models.profile_autoencoder import get_profile_autoencoder

        dataset = SyntheticCurveDataset(self.config.synthetic_samples, self.config.synthetic_length, self.config.seed)
        model, _ = get_profile_autoencoder(self.model_name, None, profile_length=self.config.synthetic_length)

        return FeedTarget(
            dataset        = dataset,
            model          = model,
            to_model_input = self.to_model_input,
            forward_loss   = FeedLosses.reconstruction,
            model_name     = self.model_name,
            sample_text    = f"synthetic curves, length {self.config.synthetic_length}, {len(dataset):,} samples",
            item_source    = "SyntheticCurveDataset.__getitem__ (in-process Gaussian synthesis)",
            config_hint    = "configuration/benchmark/dataloader_tuning.py DataLoaderTuningEntryConfig",
        )


class BackboneFeedAdapter:
    def __init__(self, config, work_dir: Path, logger) -> None:
        self.config     = config
        self.work_dir   = Path(work_dir)
        self.logger     = logger
        self.model_name = config.model_name or DEFAULT_MODEL["backbone"]

    @staticmethod
    def to_model_input(batch, device: torch.device) -> torch.Tensor:
        return batch[0].to(device, non_blocking=True)

    def _config_factory(self):
        from configuration.training          import BackboneEntryConfig
        from pipelines.shared.config.config_factory import ConfigFactory
        from pipelines.shared.model.model_builder   import ModelBuilder

        name, head  = ModelBuilder.split_key(self.model_name)
        entry       = BackboneEntryConfig(backbone_name=name, backbone_head=head, seed=self.config.seed)
        entry.paths = self.config.paths

        return ConfigFactory(entry)

    def _dataset(self, factory):
        from pipelines.backbone.dataset.pipeline import DatasetPipeline

        dataset_config = factory.training_dataset_config()
        gaussian_cfg   = factory.training_trainer_config(logdir=self.work_dir).gaussian

        dataset_config.n_gaussians = gaussian_cfg.n_default_gaussians

        dataset_pipeline = DatasetPipeline(dataset_config, self.work_dir, logger=self.logger, seed=self.config.seed)
        profile_length   = dataset_pipeline.profile_length

        dataset_config.x_axis = GaussianAxis.build(gaussian_cfg.x_min, gaussian_cfg.x_max, profile_length)

        _train_loader, _val_loader, _test_loader, datasets = dataset_pipeline.run()

        return datasets["train"], dataset_config, gaussian_cfg

    def _model(self, dataset, dataset_config, gaussian_cfg):
        from models import BACKBONE_CONFIG_REGISTRY, get_backbone
        from pipelines.shared.model.model_builder import ModelBuilder

        name, head   = ModelBuilder.split_key(self.model_name)
        in_channels  = dataset.input_channels
        out_channels = GaussianHead.total_channels(gaussian_cfg.params_per_gaussian, gaussian_cfg.n_default_gaussians)

        overrides = {"in_channels": in_channels, "out_channels": out_channels, "head": head}
        overrides.update(ModelBuilder.image_size_override(name, dataset_config.patch.size))

        model, _ = get_backbone(name, config=BACKBONE_CONFIG_REGISTRY[name](), **overrides)
        return model, in_channels

    def build(self) -> FeedTarget:
        dataset, dataset_config, gaussian_cfg = self._dataset(self._config_factory())
        model, input_channels                 = self._model(dataset, dataset_config, gaussian_cfg)

        sample = np.asarray(dataset[0][0])

        return FeedTarget(
            dataset        = dataset,
            model          = model,
            to_model_input = self.to_model_input,
            forward_loss   = FeedLosses.supervised,
            model_name     = self.model_name,
            sample_text    = f"patches {tuple(sample.shape)}, {len(dataset):,} samples, {input_channels} channels",
            item_source    = "PatchDataset.__getitem__ (patch extraction + complex->representation conversion)",
            config_hint    = "configuration/dataset/general/dataset.py DatasetConfig",
        )


FEED_ADAPTERS = {
    "synthetic"           : SyntheticFeedAdapter,
    "profile_autoencoder" : ProfileFeedAdapter,
    "image_autoencoder"   : ImageFeedAdapter,
    "backbone"            : BackboneFeedAdapter,
}


def build_feed_target(config, work_dir: Path, logger) -> FeedTarget:
    if config.mode not in FEED_ADAPTERS:
        raise ValueError(f"Unknown tuning mode '{config.mode}'. Available: {list(FEED_ADAPTERS)}")

    return FEED_ADAPTERS[config.mode](config, work_dir, logger).build()
