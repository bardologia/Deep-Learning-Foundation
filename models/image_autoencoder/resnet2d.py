from __future__ import annotations

import torch
import torch.nn as nn

from configuration.architectures   import ResNet2dImageAutoencoderConfig
from models.image_autoencoder.base import ImageAutoencoderBase
from models.blocks                 import ResidualConvBlock, build_upsample, downsample_stages


class ResNet2dImageEncoder(nn.Module):
    def __init__(self, config: ResNet2dImageAutoencoderConfig) -> None:
        super().__init__()
        n_stages = downsample_stages(config.downsample_factor)

        stem = [ResidualConvBlock(config.in_channels, config.base_channels, config.dropout, config.activation, config.normalization, first_unit=True)]
        for _ in range(max(0, config.depth - 1)):
            stem.append(ResidualConvBlock(config.base_channels, config.base_channels, config.dropout, config.activation, config.normalization))
        self.stem = nn.Sequential(*stem)

        downs    = []
        channels = config.base_channels
        for _ in range(n_stages):
            nxt = channels * 2
            downs.append(ResidualConvBlock(channels, nxt, config.dropout, config.activation, config.normalization, stride=2))
            for _ in range(max(0, config.depth - 1)):
                downs.append(ResidualConvBlock(nxt, nxt, config.dropout, config.activation, config.normalization))
            channels = nxt
        self.downsample = nn.Sequential(*downs)

        self.to_embedding        = nn.Conv2d(channels, config.embedding_dim, kernel_size=1)
        self.bottleneck_channels = channels

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.stem(image)
        x = self.downsample(x)
        return self.to_embedding(x)


class ResNet2dImageDecoder(nn.Module):
    def __init__(self, config: ResNet2dImageAutoencoderConfig, bottleneck_channels: int) -> None:
        super().__init__()
        n_stages = downsample_stages(config.downsample_factor)

        self.from_embedding = nn.Conv2d(config.embedding_dim, bottleneck_channels, kernel_size=1)

        ups      = []
        channels = bottleneck_channels
        for _ in range(n_stages):
            nxt = channels // 2
            ups.append(build_upsample(config.upsample_mode, channels, nxt, scale_factor=2))
            ups.append(ResidualConvBlock(nxt, nxt, config.dropout, config.activation, config.normalization))
            channels = nxt
        self.upsample = nn.Sequential(*ups)

        refine = []
        for _ in range(max(0, config.depth - 1)):
            refine.append(ResidualConvBlock(channels, channels, config.dropout, config.activation, config.normalization))
        self.refine = nn.Sequential(*refine)

        self.head = nn.Conv2d(channels, config.in_channels, kernel_size=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.from_embedding(z)
        x = self.upsample(x)
        x = self.refine(x)
        return self.head(x)


class ResNet2dImageAutoencoder(ImageAutoencoderBase):
    def __init__(self, config: ResNet2dImageAutoencoderConfig | None = None) -> None:
        config  = config if config is not None else ResNet2dImageAutoencoderConfig()
        encoder = ResNet2dImageEncoder(config)
        decoder = ResNet2dImageDecoder(config, encoder.bottleneck_channels)
        super().__init__(config, encoder, decoder)
