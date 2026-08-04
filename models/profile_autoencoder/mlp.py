from __future__ import annotations

import torch
import torch.nn as nn

from configuration.architectures     import MlpAutoencoderConfig
from models.profile_autoencoder.base import ProfileAutoencoderBase, ProfileAutoencoderBlocks


class MlpEncoder(nn.Module):
    def __init__(self, config: MlpAutoencoderConfig) -> None:
        super().__init__()
        self.net = ProfileAutoencoderBlocks.mlp_stack(
            in_ch      = config.profile_length,
            hidden     = config.hidden_dim,
            out_ch     = config.embedding_dim,
            depth      = config.depth,
            activation = config.activation,
            dropout    = config.dropout,
        )

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        return self.net(curve)


class MlpDecoder(nn.Module):
    def __init__(self, config: MlpAutoencoderConfig) -> None:
        super().__init__()
        self.net = ProfileAutoencoderBlocks.mlp_stack(
            in_ch      = config.embedding_dim,
            hidden     = config.hidden_dim,
            out_ch     = config.profile_length,
            depth      = config.depth,
            activation = config.activation,
            dropout    = config.dropout,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class MlpAutoencoder(ProfileAutoencoderBase):
    def __init__(self, config: MlpAutoencoderConfig | None = None) -> None:
        config = config if config is not None else MlpAutoencoderConfig()
        super().__init__(config, MlpEncoder(config), MlpDecoder(config))
