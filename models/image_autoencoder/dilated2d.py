"""Image autoencoder that widens its receptive field with dilation instead of pooling.

The grid is never reduced, so the latent keeps the input resolution and the model
serves as the resolution-preserving arm of the autoencoder zoo.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from configuration.architectures   import DilatedConv2dImageAutoencoderConfig, ImageAutoencoderBaseConfig
from models.image_autoencoder.base import ImageAutoencoderBase
from models.blocks                 import LayerFactory


class Dilated2dResidualBlock(nn.Module):
    """Residual block of two dilated 3x3 convolutions at fixed channel width.

    Attributes:
        body : The two dilated convolutions with normalization and optional dropout.
        act  : Activation applied to the residual sum.
    """

    def __init__(self, channels: int, dilation: int, activation: str, normalization: str, dropout: float) -> None:
        """Builds the dilated residual branch.

        Args:
            channels      : Channel width entering and leaving the block.
            dilation      : Dilation rate of both convolutions; padding matches it so the grid is preserved.
            activation    : Activation name resolved by ``LayerFactory.build_activation``.
            normalization : 2D normalization name resolved by ``LayerFactory.build_norm2d``.
            dropout       : Spatial dropout probability; disabled at 0.
        """
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            LayerFactory.build_norm2d(normalization, channels),
            LayerFactory.build_activation(activation),
            nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            LayerFactory.build_norm2d(normalization, channels),
            nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity(),
        )
        self.act  = LayerFactory.build_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the activated residual sum.

        Args:
            x : Feature map of shape (batch, channels, height, width).

        Returns:
            Feature map of the same shape as ``x``.
        """
        return self.act(x + self.body(x))


class DilatedConv2dImageEncoder(nn.Module):
    """Encoder stacking residual blocks with exponentially growing dilation.

    Attributes:
        stem         : Convolution, normalization and activation lifting the input to ``base_channels``.
        body         : The dilated residual blocks, with dilation 1, 2, 4, ... per block.
        to_embedding : 1x1 projection onto ``embedding_dim``.
    """

    def __init__(self, config: DilatedConv2dImageAutoencoderConfig) -> None:
        """Builds the stem, the dilated block stack and the embedding projection.

        Args:
            config : Autoencoder configuration giving the base width and dilation depth.
        """
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(config.in_channels, config.base_channels, kernel_size=3, padding=1, bias=False),
            LayerFactory.build_norm2d(config.normalization, config.base_channels),
            LayerFactory.build_activation(config.activation),
        )
        self.body = nn.Sequential(*[
            Dilated2dResidualBlock(config.base_channels, 2 ** i, config.activation, config.normalization, config.dropout)
            for i in range(max(1, config.dilation_depth))
        ])
        self.to_embedding = nn.Conv2d(config.base_channels, config.embedding_dim, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Encodes an image stack into a latent at the input resolution.

        Args:
            image : Input stack of shape (batch, in_channels, height, width).

        Returns:
            Latent of shape (batch, embedding_dim, height, width).
        """
        x = self.stem(image)
        x = self.body(x)
        return self.to_embedding(x)


class DilatedConv2dImageDecoder(nn.Module):
    """Decoder mirroring the dilated block stack back to the input channels.

    Attributes:
        from_embedding : 1x1 projection from ``embedding_dim`` to ``base_channels``.
        body           : The dilated residual blocks.
        head           : 1x1 projection onto ``in_channels``.
    """

    def __init__(self, config: DilatedConv2dImageAutoencoderConfig) -> None:
        """Builds the projection, the dilated block stack and the output head.

        Args:
            config : Autoencoder configuration giving the base width and dilation depth.
        """
        super().__init__()
        self.from_embedding = nn.Conv2d(config.embedding_dim, config.base_channels, kernel_size=1)
        self.body           = nn.Sequential(*[
            Dilated2dResidualBlock(config.base_channels, 2 ** i, config.activation, config.normalization, config.dropout)
            for i in range(max(1, config.dilation_depth))
        ])
        self.head           = nn.Conv2d(config.base_channels, config.in_channels, kernel_size=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstructs an image stack from the latent feature map.

        Args:
            z : Latent of shape (batch, embedding_dim, height, width).

        Returns:
            Reconstruction of shape (batch, in_channels, height, width).
        """
        x = self.from_embedding(z)
        x = self.body(x)
        return self.head(x)


class DilatedConv2dImageAutoencoder(ImageAutoencoderBase):
    """Image autoencoder pairing the dilated encoder and decoder.

    Rejects configurations that set fields this architecture ignores, so a sweep can
    never silently run a variant that has no effect.
    """

    def __init__(self, config: DilatedConv2dImageAutoencoderConfig | None = None) -> None:
        """Builds the encoder and decoder after checking the inapplicable fields.

        Args:
            config : Autoencoder configuration; defaults to ``DilatedConv2dImageAutoencoderConfig()``.

        Raises:
            ValueError : If ``downsample_factor`` or ``depth`` has been moved off its base default, since this architecture reads neither.
        """
        config = config if config is not None else DilatedConv2dImageAutoencoderConfig()
        if config.downsample_factor != ImageAutoencoderBaseConfig.downsample_factor:
            raise ValueError(f"dilated2d_ae holds the input grid through dilated convolutions and never reads downsample_factor; leave the field at {ImageAutoencoderBaseConfig.downsample_factor}, got {config.downsample_factor}.")
        if config.depth != ImageAutoencoderBaseConfig.depth:
            raise ValueError(f"dilated2d_ae stacks dilation_depth residual blocks and never reads depth; leave the field at {ImageAutoencoderBaseConfig.depth}, got {config.depth}.")

        super().__init__(config, DilatedConv2dImageEncoder(config), DilatedConv2dImageDecoder(config))
