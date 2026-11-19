
from __future__ import annotations

from dataclasses import dataclass, field

from configuration.training.image_autoencoder import ImageAeLossConfig


@dataclass
class ImageAeCvConfig:

    ae_model_name : str               = "conv2d_ae"
    ae_loss       : ImageAeLossConfig = field(default_factory=ImageAeLossConfig)
