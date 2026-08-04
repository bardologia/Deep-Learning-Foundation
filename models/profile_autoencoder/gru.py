from __future__ import annotations

import torch
import torch.nn as nn

from configuration.architectures     import GruAutoencoderConfig
from models.profile_autoencoder.base import ProfileAutoencoderBase, ProfileAutoencoderBlocks


class GruEncoder(nn.Module):
    def __init__(self, config: GruAutoencoderConfig) -> None:
        super().__init__()
        self.num_layers  = max(1, config.depth)
        self.directions  = 2 if config.bidirectional else 1
        self.hidden_dim  = config.hidden_dim
        self.rnn         = nn.GRU(
            input_size    = 1,
            hidden_size   = config.hidden_dim,
            num_layers    = self.num_layers,
            batch_first   = True,
            bidirectional = config.bidirectional,
            dropout       = config.dropout if self.num_layers > 1 else 0.0,
        )
        self.head        = nn.Linear(config.hidden_dim * self.directions, config.embedding_dim)

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        seq, dims = ProfileAutoencoderBlocks.to_sequence(curve)
        _, h_n = self.rnn(seq.unsqueeze(-1))
        last   = h_n.view(self.num_layers, self.directions, seq.shape[0], self.hidden_dim)[-1]
        merged = last.permute(1, 0, 2).reshape(seq.shape[0], self.directions * self.hidden_dim)
        z      = self.head(merged)
        return ProfileAutoencoderBlocks.from_sequence(z, dims)


class GruDecoder(nn.Module):
    def __init__(self, config: GruAutoencoderConfig) -> None:
        super().__init__()
        self.length     = config.profile_length
        self.num_layers = max(1, config.depth)
        self.project    = nn.Linear(config.embedding_dim, config.hidden_dim)
        self.rnn        = nn.GRU(
            input_size  = config.hidden_dim,
            hidden_size = config.hidden_dim,
            num_layers  = self.num_layers,
            batch_first = True,
            dropout     = config.dropout if self.num_layers > 1 else 0.0,
        )
        self.head       = nn.Linear(config.hidden_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        seq, dims = ProfileAutoencoderBlocks.to_sequence(z)
        seed   = self.project(seq).unsqueeze(1).expand(-1, self.length, -1)
        out, _ = self.rnn(seed)
        curve  = self.head(out).squeeze(-1)
        return ProfileAutoencoderBlocks.from_sequence(curve, dims)


class GruAutoencoder(ProfileAutoencoderBase):
    def __init__(self, config: GruAutoencoderConfig | None = None) -> None:
        config = config if config is not None else GruAutoencoderConfig()
        super().__init__(config, GruEncoder(config), GruDecoder(config))
