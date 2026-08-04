from __future__ import annotations

import torch
import torch.nn as nn

from configuration.architectures     import Transformer1dAutoencoderConfig
from models.profile_autoencoder.base import ProfileAutoencoderBase, ProfileAutoencoderBlocks


class Transformer1dEncoder(nn.Module):
    def __init__(self, config: Transformer1dAutoencoderConfig) -> None:
        super().__init__()
        self.embed = nn.Linear(config.profile_length, config.hidden_dim)
        layer      = nn.TransformerEncoderLayer(
            d_model         = config.hidden_dim,
            nhead           = config.num_heads,
            dim_feedforward = config.hidden_dim * 2,
            dropout         = config.dropout,
            activation      = config.activation,
            batch_first     = True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, config.depth))
        self.head    = nn.Linear(config.hidden_dim, config.embedding_dim)

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        seq, dims = ProfileAutoencoderBlocks.to_sequence(curve)
        tokens   = self.embed(seq).unsqueeze(1)
        attended = self.encoder(tokens).squeeze(1)
        z        = self.head(attended)
        return ProfileAutoencoderBlocks.from_sequence(z, dims)


class Transformer1dDecoder(nn.Module):
    def __init__(self, config: Transformer1dAutoencoderConfig) -> None:
        super().__init__()
        self.embed = nn.Linear(config.embedding_dim, config.hidden_dim)
        layer      = nn.TransformerEncoderLayer(
            d_model         = config.hidden_dim,
            nhead           = config.num_heads,
            dim_feedforward = config.hidden_dim * 2,
            dropout         = config.dropout,
            activation      = config.activation,
            batch_first     = True,
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=max(1, config.depth))
        self.head    = nn.Linear(config.hidden_dim, config.profile_length)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        seq, dims = ProfileAutoencoderBlocks.to_sequence(z)
        tokens   = self.embed(seq).unsqueeze(1)
        attended = self.decoder(tokens).squeeze(1)
        curve    = self.head(attended)
        return ProfileAutoencoderBlocks.from_sequence(curve, dims)


class Transformer1dAutoencoder(ProfileAutoencoderBase):
    def __init__(self, config: Transformer1dAutoencoderConfig | None = None) -> None:
        config = config if config is not None else Transformer1dAutoencoderConfig()
        if config.activation not in ("relu", "gelu"):
            raise ValueError(f"transformer1d_ae supports activations relu and gelu only (nn.TransformerEncoderLayer constraint); got '{config.activation}'.")

        super().__init__(config, Transformer1dEncoder(config), Transformer1dDecoder(config))
