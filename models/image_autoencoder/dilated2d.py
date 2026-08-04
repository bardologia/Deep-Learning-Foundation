from __future__ import annotations

import torch
import torch.nn as nn

from configuration.architectures   import DilatedConv2dImageAutoencoderConfig
from models.image_autoencoder.base import ImageAutoencoderBase
from models.blocks                 import build_activation, build_norm2d


class Dilated2dResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, activation: str, normalization: str, dropout: float) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            build_norm2d(normalization, channels),
            build_activation(activation),
            nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            build_norm2d(normalization, channels),
            nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity(),
        )
        self.act  = build_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.body(x))


class DilatedConv2dImageEncoder(nn.Module):
    def __init__(self, config: DilatedConv2dImageAutoencoderConfig) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(config.in_channels, config.base_channels, kernel_size=3, padding=1, bias=False),
            build_norm2d(config.normalization, config.base_channels),
            build_activation(config.activation),
        )
        self.body = nn.Sequential(*[
            Dilated2dResidualBlock(config.base_channels, 2 ** i, config.activation, config.normalization, config.dropout)
            for i in range(max(1, config.dilation_depth))
        ])
        self.to_embedding = nn.Conv2d(config.base_channels, config.embedding_dim, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.stem(image)
        x = self.body(x)
        return self.to_embedding(x)


class DilatedConv2dImageDecoder(nn.Module):
    def __init__(self, config: DilatedConv2dImageAutoencoderConfig) -> None:
        super().__init__()
        self.from_embedding = nn.Conv2d(config.embedding_dim, config.base_channels, kernel_size=1)
        self.body           = nn.Sequential(*[
            Dilated2dResidualBlock(config.base_channels, 2 ** i, config.activation, config.normalization, config.dropout)
            for i in range(max(1, config.dilation_depth))
        ])
        self.head           = nn.Conv2d(config.base_channels, config.in_channels, kernel_size=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.from_embedding(z)
        x = self.body(x)
        return self.head(x)


class DilatedConv2dImageAutoencoder(ImageAutoencoderBase):
    def __init__(self, config: DilatedConv2dImageAutoencoderConfig | None = None) -> None:
        config = config if config is not None else DilatedConv2dImageAutoencoderConfig()
        super().__init__(config, DilatedConv2dImageEncoder(config), DilatedConv2dImageDecoder(config))
