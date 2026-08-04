from __future__ import annotations

import torch
import torch.nn as nn

from configuration.architectures     import TcnAutoencoderConfig
from models.profile_autoencoder.base import ProfileAutoencoderBase, ProfileAutoencoderBlocks
from models.blocks                   import build_activation


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, activation: str, dropout: float) -> None:
        super().__init__()
        pad       = dilation * (kernel_size - 1) // 2
        self.body = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
        )
        self.act  = build_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.body(x))


class TcnEncoder(nn.Module):
    def __init__(self, config: TcnAutoencoderConfig) -> None:
        super().__init__()
        pad       = config.seq_kernel_size // 2
        self.stem = nn.Conv1d(1, config.seq_channels, config.seq_kernel_size, padding=pad)
        self.body = nn.Sequential(*[
            DilatedResidualBlock(config.seq_channels, config.seq_kernel_size, 2 ** i, config.activation, config.dropout)
            for i in range(max(1, config.depth))
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(config.seq_channels, config.embedding_dim)

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        seq, dims = ProfileAutoencoderBlocks.to_sequence(curve)
        feats = self.stem(seq.unsqueeze(1))
        feats = self.body(feats)
        z     = self.head(self.pool(feats).squeeze(-1))
        return ProfileAutoencoderBlocks.from_sequence(z, dims)


class TcnDecoder(nn.Module):
    def __init__(self, config: TcnAutoencoderConfig) -> None:
        super().__init__()
        self.length       = config.profile_length
        self.seq_channels = config.seq_channels
        pad               = config.seq_kernel_size // 2
        self.project      = nn.Linear(config.embedding_dim, config.seq_channels * config.profile_length)
        self.body         = nn.Sequential(*[
            DilatedResidualBlock(config.seq_channels, config.seq_kernel_size, 2 ** i, config.activation, config.dropout)
            for i in range(max(1, config.depth))
        ])
        self.head         = nn.Conv1d(config.seq_channels, 1, config.seq_kernel_size, padding=pad)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        seq, dims = ProfileAutoencoderBlocks.to_sequence(z)
        feats = self.project(seq).reshape(-1, self.seq_channels, self.length)
        feats = self.body(feats)
        curve = self.head(feats).squeeze(1)
        return ProfileAutoencoderBlocks.from_sequence(curve, dims)


class TcnAutoencoder(ProfileAutoencoderBase):
    def __init__(self, config: TcnAutoencoderConfig | None = None) -> None:
        config = config if config is not None else TcnAutoencoderConfig()
        super().__init__(config, TcnEncoder(config), TcnDecoder(config))
