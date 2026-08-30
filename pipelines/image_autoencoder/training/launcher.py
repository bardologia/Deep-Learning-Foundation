"""Cluster launcher for image-autoencoder seed sweeps."""

from __future__ import annotations

from configuration.training.image_autoencoder    import ImageAeEntryConfig
from pipelines.shared.training.training_launcher import SeedSweepClusterLauncher


class ImageAeTrainingLauncher(SeedSweepClusterLauncher):
    """Submits one cluster job per seed of an image-autoencoder training."""

    entry_config_class = ImageAeEntryConfig
    description        = "Fan an image autoencoder training out over a cluster: one job per seed"
    sweep_label        = "image_autoencoder"
