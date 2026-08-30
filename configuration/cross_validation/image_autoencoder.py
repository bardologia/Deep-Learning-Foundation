"""Image autoencoder specific settings for cross-validation runs."""

from __future__ import annotations

from dataclasses import dataclass, field

from configuration.training.image_autoencoder import ImageAeLossConfig


@dataclass
class ImageAeCvConfig:
    """Image autoencoder settings used when cross-validating that family.

    Attributes:
        ae_model_name : Image autoencoder family instantiated per fold.
        ae_loss       : Reconstruction loss configuration.
    """

    ae_model_name : str               = "conv2d_ae"
    ae_loss       : ImageAeLossConfig = field(default_factory=ImageAeLossConfig)
