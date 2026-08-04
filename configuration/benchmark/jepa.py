from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

from configuration.training.jepa import EmbeddingLossConfig, default_param_loss


def _default_embedding_loss():
    return EmbeddingLossConfig()


def _default_param_loss():
    return default_param_loss()


@dataclass
class JepaBenchConfig:
    profile_autoencoder_logdir : Path       = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/profile_autoencoder")
    profile_autoencoder_run    : str | None = None
    profile_autoencoder_mode   : str        = "frozen"

    image_autoencoder_logdir : Path       = Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/image_autoencoder")
    image_autoencoder_run    : str | None = None
    image_autoencoder_mode   : str        = "frozen"

    target_provider : str    = "stopgrad"
    embedding_loss  : object = field(default_factory=_default_embedding_loss)
    param_loss      : object = field(default_factory=_default_param_loss)
