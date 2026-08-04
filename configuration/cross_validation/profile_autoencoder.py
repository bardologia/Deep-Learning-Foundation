from __future__ import annotations

from dataclasses import dataclass, field

from configuration.training.profile_autoencoder     import ProfileAeLossConfig


@dataclass
class AeCvConfig:
    ae_model_name   : str                 = "mlp_ae"
    ae_loss         : ProfileAeLossConfig = field(default_factory=ProfileAeLossConfig)
    pixel_subsample : float               = 1.0
    keep_empty_frac : float               = 0.05
