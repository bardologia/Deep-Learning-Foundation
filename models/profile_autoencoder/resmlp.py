from __future__ import annotations

import torch
import torch.nn as nn

from configuration.architectures     import ResMlpAutoencoderConfig
from models.profile_autoencoder.base import ProfileAutoencoderBase, ProfileAutoencoderBlocks
from models.blocks                   import build_activation


class ResidualMlpBlock(nn.Module):
    def __init__(self, dim: int, activation: str, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.body = nn.Sequential(
            nn.Linear(dim, dim),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(self.norm(x))


class ResMlpEncoder(nn.Module):
    def __init__(self, config: ResMlpAutoencoderConfig) -> None:
        super().__init__()
        self.embed  = nn.Linear(config.profile_length, config.hidden_dim)
        self.blocks = nn.Sequential(*[
            ResidualMlpBlock(config.hidden_dim, config.activation, config.dropout)
            for _ in range(max(1, config.depth))
        ])
        self.head   = nn.Linear(config.hidden_dim, config.embedding_dim)

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        seq, dims = ProfileAutoencoderBlocks.to_sequence(curve)
        x = self.embed(seq)
        x = self.blocks(x)
        z = self.head(x)
        return ProfileAutoencoderBlocks.from_sequence(z, dims)


class ResMlpDecoder(nn.Module):
    def __init__(self, config: ResMlpAutoencoderConfig) -> None:
        super().__init__()
        self.embed  = nn.Linear(config.embedding_dim, config.hidden_dim)
        self.blocks = nn.Sequential(*[
            ResidualMlpBlock(config.hidden_dim, config.activation, config.dropout)
            for _ in range(max(1, config.depth))
        ])
        self.head   = nn.Linear(config.hidden_dim, config.profile_length)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        seq, dims = ProfileAutoencoderBlocks.to_sequence(z)
        x     = self.embed(seq)
        x     = self.blocks(x)
        curve = self.head(x)
        return ProfileAutoencoderBlocks.from_sequence(curve, dims)


class ResMlpAutoencoder(ProfileAutoencoderBase):
    def __init__(self, config: ResMlpAutoencoderConfig | None = None) -> None:
        config = config if config is not None else ResMlpAutoencoderConfig()
        super().__init__(config, ResMlpEncoder(config), ResMlpDecoder(config))
