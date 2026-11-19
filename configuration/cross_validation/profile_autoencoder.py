
from __future__ import annotations

from dataclasses import dataclass, field

from configuration.architectures.profile_autoencoder import ProfileAutoencoderBaseConfig
from configuration.dataset.profile_autoencoder       import ProfileAugmentationConfig
from configuration.training.profile_autoencoder      import ProfileAeLossConfig


@dataclass
class AeCvConfig:

    ae_model_name   : str                 = "mlp_ae"
    ae_loss         : ProfileAeLossConfig = field(default_factory=ProfileAeLossConfig)
    pixel_subsample : float               = 1.0
    keep_empty_frac : float               = 0.05

    profile_augmentation : ProfileAugmentationConfig = field(default_factory=ProfileAugmentationConfig)

    encoder_lr : float = ProfileAutoencoderBaseConfig.encoder_lr
    decoder_lr : float = ProfileAutoencoderBaseConfig.decoder_lr
    encoder_wd : float = ProfileAutoencoderBaseConfig.encoder_wd
    decoder_wd : float = ProfileAutoencoderBaseConfig.decoder_wd
