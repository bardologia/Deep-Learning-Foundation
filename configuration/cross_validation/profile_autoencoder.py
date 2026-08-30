"""Profile autoencoder specific settings for cross-validation runs."""

from __future__ import annotations

from dataclasses import dataclass, field

from configuration.dataset.profile_autoencoder      import ProfileAugmentationConfig
from configuration.training.profile_autoencoder     import ProfileAeLossConfig


@dataclass
class AeCvConfig:
    """Profile autoencoder settings used when cross-validating that family.

    Attributes:
        ae_model_name        : Profile autoencoder family instantiated per fold.
        ae_loss              : Reconstruction loss configuration.
        pixel_subsample      : Fraction of pixels retained from the dataset.
        keep_empty_frac      : Fraction of empty-profile pixels retained.
        profile_augmentation : Augmentation applied to the sampled profiles in every fold.
    """

    ae_model_name   : str                 = "mlp_ae"
    ae_loss         : ProfileAeLossConfig = field(default_factory=ProfileAeLossConfig)
    pixel_subsample : float               = 1.0
    keep_empty_frac : float               = 0.05

    profile_augmentation : ProfileAugmentationConfig = field(default_factory=ProfileAugmentationConfig)
