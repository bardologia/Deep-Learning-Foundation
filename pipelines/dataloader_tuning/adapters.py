"""Feed targets the dataloader tuner benchmarks.

Each adapter assembles the dataset, model, and forward loss of one training
family so the tuner can time the GPU feed under different batch sizes, worker
counts, prefetch factors, and pinning settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib     import Path
from typing      import Callable

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import Dataset

from tools.data.gaussians import GaussianAxis, GaussianHead, GaussianMixture


@dataclass
class FeedTarget:
    """Everything the tuner needs to time one training family's input feed.

    Attributes:
        dataset        : Dataset the benchmark loader iterates over.
        model          : Model run on each batch to produce realistic GPU load.
        to_model_input : Moves a collated batch to the device and reshapes it into the model's input tensor.
        forward_loss   : Runs the model on a batch and returns a scalar loss.
        model_name     : Registry key of the benchmarked model.
        sample_text    : Human-readable description of the sample shape and count.
        item_source    : Description of where per-item cost is incurred.
        config_hint    : Configuration file and class governing the feed.
    """

    dataset        : Dataset
    model          : torch.nn.Module
    to_model_input : Callable[[object, torch.device], torch.Tensor]
    forward_loss   : Callable[[torch.nn.Module, torch.Tensor], torch.Tensor]
    model_name     : str
    sample_text    : str
    item_source    : str
    config_hint    : str


class FeedLosses:
    """Placeholder forward losses that exercise the model without real targets."""

    @staticmethod
    def reconstruction(model: torch.nn.Module, model_input: torch.Tensor) -> torch.Tensor:
        """Returns the mean squared reconstruction error of an autoencoder.

        Args:
            model       : Autoencoder exposing `reconstruct`.
            model_input : Batch already moved to the device.

        Returns:
            Scalar MSE between the reconstruction and the input.
        """
        reconstruction, _ = model.reconstruct(model_input)
        return functional.mse_loss(reconstruction, model_input)

    @staticmethod
    def supervised(model: torch.nn.Module, model_input: torch.Tensor) -> torch.Tensor:
        """Returns the mean squared prediction magnitude of a supervised model.

        Args:
            model       : Model called directly on the batch.
            model_input : Batch already moved to the device.

        Returns:
            Scalar MSE between the prediction and zeros, which stands in for a
            real target since the tuner only measures throughput.
        """
        prediction = model(model_input)
        return functional.mse_loss(prediction, torch.zeros_like(prediction))


class SyntheticCurveDataset(Dataset):
    """Three-Gaussian z profiles synthesised in process.

    Attributes:
        n_samples      : Number of curves the dataset exposes.
        profile_length : Number of z bins per curve.
        x_axis         : Normalised z axis of shape (profile_length,).
        rng_seed       : Base seed; each index draws from `rng_seed + index`.
    """

    def __init__(self, n_samples: int, profile_length: int, seed: int) -> None:
        """Configures the size and reproducibility of the synthetic dataset.

        Args:
            n_samples      : Number of curves the dataset exposes.
            profile_length : Number of z bins per curve.
            seed           : Base seed for the per-item generators.
        """
        self.n_samples      = int(n_samples)
        self.profile_length = int(profile_length)
        self.z_axis         = np.linspace(0.0, 1.0, self.profile_length, dtype=np.float32)
        self.rng_seed       = int(seed)

    def __len__(self) -> int:
        """Returns the number of curves in the dataset."""
        return self.n_samples

    def __getitem__(self, index: int):
        """Synthesises one curve from three randomly drawn Gaussians.

        Args:
            index : Sample index, which also seeds the draw.

        Returns:
            Float32 curve of shape (profile_length,).
        """
        rng = np.random.default_rng(self.rng_seed + index)

        amplitudes = rng.uniform(0.2, 1.0, size=(1, 3)).astype(np.float32)
        means      = rng.uniform(0.2, 0.8, size=(1, 3)).astype(np.float32)
        sigmas     = rng.uniform(0.02, 0.08, size=(1, 3)).astype(np.float32)

        curve = GaussianMixture.evaluate_batch(self.z_axis, amplitudes, means, sigmas)[0]

        return curve.astype(np.float32)


class ProfileFeedAdapter:
    """Builds the feed target for the profile-autoencoder family.

    Attributes:
        config     : Dataloader tuning entry configuration.
        work_dir   : Scratch directory the dataset pipeline writes into.
        logger     : Logger used for progress reporting.
        model_name : Registry key of the model to benchmark.
    """

    DEFAULT_MODEL = "mlp_ae"

    def __init__(self, config, work_dir: Path, logger) -> None:
        """Binds the adapter to the tuning configuration and scratch directory.

        Args:
            config   : Dataloader tuning entry configuration.
            work_dir : Scratch directory the dataset pipeline writes into.
            logger   : Logger used for progress reporting.
        """
        self.config     = config
        self.work_dir   = Path(work_dir)
        self.logger     = logger
        self.model_name = config.model_name or self.DEFAULT_MODEL

    @staticmethod
    def to_model_input(batch, device: torch.device) -> torch.Tensor:
        """Moves a profile batch to the device as shape (batch, profile_length, 1, 1).

        Args:
            batch  : Collated (noisy, clean) pair from ProfileDataset.
            device : Device the tensor is moved to.

        Returns:
            The noisy profiles with two trailing singleton spatial axes.
        """
        return batch[0].to(device, non_blocking=True).unsqueeze(-1).unsqueeze(-1)

    def _training_pipeline(self):
        """Builds the profile-autoencoder training pipeline supplying the configs."""
        from configuration.training import ProfileAeEntryConfig
        from pipelines.profile_autoencoder.training.pipeline import TrainingPipeline

        entry = ProfileAeEntryConfig(ae_model_name=self.model_name, seed=self.config.seed)
        entry.paths           = self.config.paths
        entry.pixel_subsample = self.config.pixel_subsample
        entry.keep_empty_frac = self.config.keep_empty_frac

        return TrainingPipeline(entry)

    def _dataset(self, training_pipeline):
        """Builds the training profile dataset.

        Args:
            training_pipeline : Pipeline supplying the dataset configuration.

        Returns:
            A tuple (dataset, profile_length) for the training split.
        """
        from pipelines.profile_autoencoder.dataset.pipeline import ProfileDatasetPipeline

        profile_dataset_config = training_pipeline.dataset_config
        dataset_pipeline       = ProfileDatasetPipeline(profile_dataset_config, self.work_dir, logger=self.logger, seed=self.config.seed)

        (_loaders, datasets, _axis, profile_length, _normalizer) = dataset_pipeline.run()

        return datasets["train"], profile_length

    def _model(self, training_pipeline, profile_length: int):
        """Instantiates the profile autoencoder for the given profile length."""
        from models.profile_autoencoder import get_profile_autoencoder

        model, _ = get_profile_autoencoder(self.model_name, training_pipeline.autoencoder_cfg, profile_length=profile_length)
        return model

    def build(self) -> FeedTarget:
        """Assembles the profile-autoencoder feed target.

        Returns:
            The feed target with a reconstruction loss.
        """
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
    """Builds the feed target for the image-autoencoder family.

    Attributes:
        config     : Dataloader tuning entry configuration.
        work_dir   : Scratch directory the dataset pipeline writes into.
        logger     : Logger used for progress reporting.
        model_name : Registry key of the model to benchmark.
    """

    DEFAULT_MODEL = "conv2d_ae"

    def __init__(self, config, work_dir: Path, logger) -> None:
        """Binds the adapter to the tuning configuration and scratch directory.

        Args:
            config   : Dataloader tuning entry configuration.
            work_dir : Scratch directory the dataset pipeline writes into.
            logger   : Logger used for progress reporting.
        """
        self.config     = config
        self.work_dir   = Path(work_dir)
        self.logger     = logger
        self.model_name = config.model_name or self.DEFAULT_MODEL

    @staticmethod
    def to_model_input(batch, device: torch.device) -> torch.Tensor:
        """Moves the input patches of shape (batch, channels, height, width) to the device."""
        return batch[0].to(device, non_blocking=True)

    def _training_pipeline(self):
        """Builds the image-autoencoder training pipeline supplying the configs."""
        from configuration.training.image_autoencoder import ImageAeEntryConfig
        from pipelines.image_autoencoder.training.pipeline import TrainingPipeline

        entry       = ImageAeEntryConfig(ae_model_name=self.model_name, seed=self.config.seed)
        entry.paths = self.config.paths

        return TrainingPipeline(entry)

    def _dataset(self, training_pipeline):
        """Builds the training patch dataset with the Gaussian axis resolved.

        Args:
            training_pipeline : Pipeline supplying the dataset and Gaussian configs.

        Returns:
            The training patch dataset.
        """
        from pipelines.backbone.dataset.pipeline import DatasetPipeline

        dataset_config  = training_pipeline.dataset_config
        gaussian_config = training_pipeline.trainer_config.gaussian

        dataset_config.n_gaussians = gaussian_config.n_default_gaussians

        dataset_pipeline = DatasetPipeline(dataset_config, self.work_dir, activity_threshold=gaussian_config.activity_threshold, logger=self.logger, seed=self.config.seed)
        profile_length   = dataset_pipeline.profile_length

        dataset_config.z_axis = GaussianAxis.build(gaussian_config.x_min, gaussian_config.x_max, profile_length)

        _train_loader, _val_loader, _test_loader, datasets = dataset_pipeline.run()

        return datasets["train"]

    def _model(self, training_pipeline, input_channels: int):
        """Instantiates the image autoencoder for the dataset's channel count."""
        from models.image_autoencoder import get_image_autoencoder

        model, _ = get_image_autoencoder(self.model_name, training_pipeline.autoencoder_cfg, in_channels=input_channels)
        return model

    def build(self) -> FeedTarget:
        """Assembles the image-autoencoder feed target.

        Returns:
            The feed target with a reconstruction loss.
        """
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
    """Builds a feed target backed by in-process synthetic curves.

    Serves as the control condition: no disk access, so the measured throughput
    isolates the collate, transfer, and forward cost.

    Attributes:
        config     : Dataloader tuning entry configuration.
        logger     : Logger used for progress reporting.
        model_name : Registry key of the model to benchmark.
    """

    DEFAULT_MODEL = "mlp_ae"

    def __init__(self, config, work_dir: Path, logger) -> None:
        """Binds the adapter to the tuning configuration.

        Args:
            config   : Dataloader tuning entry configuration.
            work_dir : Scratch directory, unused by this adapter.
            logger   : Logger used for progress reporting.
        """
        self.config     = config
        self.logger     = logger
        self.model_name = config.model_name or self.DEFAULT_MODEL

    @staticmethod
    def to_model_input(batch, device: torch.device) -> torch.Tensor:
        """Moves a curve batch to the device as shape (batch, profile_length, 1, 1)."""
        return batch.to(device, non_blocking=True).unsqueeze(-1).unsqueeze(-1)

    def build(self) -> FeedTarget:
        """Assembles the synthetic feed target.

        Returns:
            The feed target with a reconstruction loss.
        """
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
    """Builds the feed target for the supervised backbone family.

    Attributes:
        config     : Dataloader tuning entry configuration.
        work_dir   : Scratch directory the dataset pipeline writes into.
        logger     : Logger used for progress reporting.
        model_name : Backbone registry key, optionally with a head suffix.
    """

    DEFAULT_MODEL = "resunet"

    def __init__(self, config, work_dir: Path, logger) -> None:
        """Binds the adapter to the tuning configuration and scratch directory.

        Args:
            config   : Dataloader tuning entry configuration.
            work_dir : Scratch directory the dataset pipeline writes into.
            logger   : Logger used for progress reporting.
        """
        self.config     = config
        self.work_dir   = Path(work_dir)
        self.logger     = logger
        self.model_name = config.model_name or self.DEFAULT_MODEL

    @staticmethod
    def to_model_input(batch, device: torch.device) -> torch.Tensor:
        """Moves the input patches of shape (batch, channels, height, width) to the device."""
        return batch[0].to(device, non_blocking=True)

    def _config_factory(self):
        """Returns a config factory bound to a minimal backbone entry config."""
        from configuration.training          import BackboneEntryConfig
        from pipelines.shared.config.config_factory import ConfigFactory
        from pipelines.shared.model.model_builder   import ModelBuilder

        name, head  = ModelBuilder.split_key(self.model_name)
        entry       = BackboneEntryConfig(backbone_name=name, backbone_head=head, seed=self.config.seed)
        entry.paths = self.config.paths

        return ConfigFactory(entry)

    def _dataset(self, factory):
        """Builds the training patch dataset with patch size squared where required.

        Args:
            factory : Config factory supplying the dataset and Gaussian configs.

        Returns:
            A tuple (dataset, dataset_config, gaussian_config) for the training split.
        """
        from pipelines.backbone.dataset.pipeline  import DatasetPipeline
        from pipelines.shared.model.model_builder import ModelBuilder

        dataset_config = factory.training_dataset_config()
        gaussian_cfg   = factory.training_trainer_config(logdir=self.work_dir).gaussian

        dataset_config.n_gaussians = gaussian_cfg.n_default_gaussians

        name, _head              = ModelBuilder.split_key(self.model_name)
        patch                    = dataset_config.patch
        patch.size, patch.stride = ModelBuilder.square_patch([name], patch.size, patch.stride)

        dataset_pipeline = DatasetPipeline(dataset_config, self.work_dir, activity_threshold=gaussian_cfg.activity_threshold, logger=self.logger, seed=self.config.seed)
        profile_length   = dataset_pipeline.profile_length

        dataset_config.z_axis = GaussianAxis.build(gaussian_cfg.x_min, gaussian_cfg.x_max, profile_length)

        _train_loader, _val_loader, _test_loader, datasets = dataset_pipeline.run()

        return datasets["train"], dataset_config, gaussian_cfg

    def _model(self, dataset, dataset_config, gaussian_cfg):
        """Instantiates the backbone sized to the dataset and Gaussian head.

        Args:
            dataset        : Training dataset supplying the input channel count.
            dataset_config : Dataset configuration supplying the patch size.
            gaussian_cfg   : Gaussian configuration supplying the head width.

        Returns:
            A tuple (model, in_channels).
        """
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
        """Assembles the backbone feed target.

        Returns:
            The feed target with a supervised placeholder loss.
        """
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


class FeedTargetFactory:
    """Maps a tuning mode to the adapter that builds its feed target."""

    ADAPTERS = {
        "synthetic"           : SyntheticFeedAdapter,
        "profile_autoencoder" : ProfileFeedAdapter,
        "image_autoencoder"   : ImageFeedAdapter,
        "backbone"            : BackboneFeedAdapter,
    }

    @classmethod
    def modes(cls) -> tuple[str, ...]:
        """Returns the tuning modes the factory can build."""
        return tuple(cls.ADAPTERS)

    @classmethod
    def adapter(cls, mode: str):
        """Returns the adapter class for a tuning mode.

        Args:
            mode : Tuning mode name.

        Returns:
            The adapter class.

        Raises:
            ValueError : If the mode is unknown.
        """
        if mode not in cls.ADAPTERS:
            raise ValueError(f"Unknown tuning mode '{mode}'. Available: {list(cls.ADAPTERS)}")

        return cls.ADAPTERS[mode]

    @classmethod
    def model_name(cls, config) -> str:
        """Returns the model a sweep of the configured mode benchmarks: the configured one, else the mode's default.

        Raises:
            ValueError : If the mode is unknown.
        """
        return config.model_name or cls.adapter(config.mode).DEFAULT_MODEL

    @classmethod
    def build(cls, config, work_dir: Path, logger) -> FeedTarget:
        """Builds the feed target for the configured tuning mode.

        Args:
            config   : Dataloader tuning entry configuration naming the mode.
            work_dir : Scratch directory the dataset pipelines write into.
            logger   : Logger used for progress reporting.

        Returns:
            The assembled feed target.
        """
        return cls.adapter(config.mode)(config, work_dir, logger).build()
