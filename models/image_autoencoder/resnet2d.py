"""Residual image autoencoder built from the shared pre-activation residual blocks."""

from __future__ import annotations

import torch
import torch.nn as nn

from configuration.architectures   import ResNet2dImageAutoencoderConfig
from models.image_autoencoder.base import ImageAutoencoderBase
from models.blocks                 import LayerFactory, ResidualConvBlock


class ResNet2dImageEncoder(nn.Module):
    """Residual encoder that halves the grid with strided residual blocks.

    Attributes:
        stem                : Full-resolution residual blocks applied before downsampling.
        downsample          : The strided residual stages reducing the grid.
        to_embedding        : 1x1 projection onto ``embedding_dim``.
        bottleneck_channels : Channel width reached just before the projection.
    """

    def __init__(self, config: ResNet2dImageAutoencoderConfig) -> None:
        """Builds the residual stem, the strided ladder and the embedding projection.

        Args:
            config : Autoencoder configuration giving widths, depth and downsample factor.
        """
        super().__init__()
        n_stages = LayerFactory.downsample_stages(config.downsample_factor)

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
        """Encodes an image stack into the latent feature map.

        Args:
            image : Input stack of shape (batch, in_channels, height, width).

        Returns:
            Latent of shape
            (batch, embedding_dim, height / downsample_factor, width / downsample_factor).
        """
        x = self.stem(image)
        x = self.downsample(x)
        return self.to_embedding(x)


class ResNet2dImageDecoder(nn.Module):
    """Residual decoder mirroring the encoder ladder back to input resolution.

    Attributes:
        from_embedding : 1x1 projection from ``embedding_dim`` to the bottleneck width.
        upsample       : The upsampling and residual stages restoring the input grid.
        refine         : Full-resolution residual blocks applied after upsampling.
        head           : 1x1 projection onto ``in_channels``.
    """

    def __init__(self, config: ResNet2dImageAutoencoderConfig, bottleneck_channels: int) -> None:
        """Builds the projection, the upsampling ladder, the refinement blocks and the head.

        Args:
            config              : Autoencoder configuration giving widths, depth and upsample mode.
            bottleneck_channels : Channel width the encoder reached before its projection.
        """
        super().__init__()
        n_stages = LayerFactory.downsample_stages(config.downsample_factor)

        self.from_embedding = nn.Conv2d(config.embedding_dim, bottleneck_channels, kernel_size=1)

        ups      = []
        channels = bottleneck_channels
        for _ in range(n_stages):
            nxt = channels // 2
            ups.append(LayerFactory.build_upsample(config.upsample_mode, channels, nxt, scale_factor=2))
            ups.append(ResidualConvBlock(nxt, nxt, config.dropout, config.activation, config.normalization))
            channels = nxt
        self.upsample = nn.Sequential(*ups)

        refine = []
        for _ in range(max(0, config.depth - 1)):
            refine.append(ResidualConvBlock(channels, channels, config.dropout, config.activation, config.normalization))
        self.refine = nn.Sequential(*refine)

        self.head = nn.Conv2d(channels, config.in_channels, kernel_size=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstructs an image stack from the latent feature map.

        Args:
            z : Latent of shape (batch, embedding_dim, latent_height, latent_width).

        Returns:
            Reconstruction of shape
            (batch, in_channels, latent_height * downsample_factor, latent_width * downsample_factor).
        """
        x = self.from_embedding(z)
        x = self.upsample(x)
        x = self.refine(x)
        return self.head(x)


class ResNet2dImageAutoencoder(ImageAutoencoderBase):
    """Image autoencoder pairing the residual encoder and decoder."""

    def __init__(self, config: ResNet2dImageAutoencoderConfig | None = None) -> None:
        """Builds the encoder and decoder and hands them to the base class.

        Args:
            config : Autoencoder configuration; defaults to ``ResNet2dImageAutoencoderConfig()``.
        """
        config  = config if config is not None else ResNet2dImageAutoencoderConfig()
        encoder = ResNet2dImageEncoder(config)
        decoder = ResNet2dImageDecoder(config, encoder.bottleneck_channels)
        super().__init__(config, encoder, decoder)
