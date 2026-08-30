"""Minimal-context baselines that bound how much spatial reach the task needs.

PixelMLPNet sees a single pixel through 1x1 convolutions only; LocalCNN widens that
to a short stack of small odd-kernel blocks at full resolution. Neither downsamples,
so both isolate the contribution of pure per-pixel inversion against the U-shaped backbones.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from configuration.architectures import LocalCNNConfig, PixelMLPNetConfig
from ..blocks                    import ConvBlock, LayerFactory, OutputHeadsMixin


class PixelMLPNet(nn.Module, OutputHeadsMixin):
    """Per-pixel MLP baseline built entirely from 1x1 convolutions.

    Attributes:
        config             : Architecture configuration driving widths and head choice.
        trunk              : The stacked 1x1 layers producing the embedding.
        embedding_channels : Channel width of the embedding fed to the output head.
    """

    def __init__(self, config: PixelMLPNetConfig | None = None):
        """Builds the 1x1 trunk and the output head.

        Args:
            config : Architecture configuration; defaults to ``PixelMLPNetConfig()``.

        Raises:
            ValueError : If ``features`` is empty.
        """
        super().__init__()
        self.config = config if config is not None else PixelMLPNetConfig()

        if len(self.config.features) == 0:
            raise ValueError("features must contain at least one channel size")

        layers   = []
        channels = self.config.in_channels
        for feature_size in self.config.features:
            layers.append(nn.Conv2d(channels, feature_size, kernel_size=1, bias=self.config.conv_bias))
            layers.append(LayerFactory.build_norm2d(self.config.normalization, feature_size))
            layers.append(LayerFactory.build_activation(self.config.activation))
            if self.config.dropout > 0:
                layers.append(nn.Dropout(self.config.dropout))
            channels = feature_size

        self.trunk              = nn.Sequential(*layers)
        self.embedding_channels = channels
        self._build_output_head()

        LayerFactory.initialize_weights(module=self, mode=self.config.init_mode)

    def encode_decode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the embedding at input resolution.

        Args:
            x : Input feature stack of shape (batch, in_channels, height, width).

        Returns:
            Embedding of shape (batch, embedding_channels, height, width).
        """
        return self.trunk(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Maps the input stack to per-pixel outputs through the configured head.

        Args:
            x : Input feature stack of shape (batch, in_channels, height, width).

        Returns:
            Tensor of shape (batch, out_channels, height, width) holding the
            interleaved Gaussian parameters, or the raw embedding when the head
            is ``none``.
        """
        return self._head_forward(self.encode_decode(x))


class LocalCNN(nn.Module, OutputHeadsMixin):
    """Full-resolution convolutional baseline with a bounded receptive field.

    Attributes:
        config             : Architecture configuration driving widths, kernels and head choice.
        trunk              : The stacked convolution blocks producing the embedding.
        embedding_channels : Channel width of the embedding fed to the output head.
    """

    def __init__(self, config: LocalCNNConfig | None = None):
        """Builds the convolution block stack and the output head.

        Args:
            config : Architecture configuration; defaults to ``LocalCNNConfig()``.

        Raises:
            ValueError : If ``features`` is empty, or if ``block_kernels`` is given but does not have one entry per feature size.
        """
        super().__init__()
        self.config = config if config is not None else LocalCNNConfig()

        if len(self.config.features) == 0:
            raise ValueError("features must contain at least one channel size")

        kernels = self.config.block_kernels if self.config.block_kernels is not None else [3] * len(self.config.features)
        if len(kernels) != len(self.config.features):
            raise ValueError(f"block_kernels has {len(kernels)} entries but features has {len(self.config.features)}; every block needs exactly one kernel size")

        blocks   = []
        channels = self.config.in_channels
        for feature_size, kernel_size in zip(self.config.features, kernels):
            blocks.append(
                ConvBlock(
                    input_channels  = channels,
                    output_channels = feature_size,
                    dropout         = self.config.dropout,
                    activation      = self.config.activation,
                    normalization   = self.config.normalization,
                    bias            = self.config.conv_bias,
                    kernel_size     = kernel_size,
                )
            )
            channels = feature_size

        self.trunk              = nn.Sequential(*blocks)
        self.embedding_channels = channels
        self._build_output_head()

        LayerFactory.initialize_weights(module=self, mode=self.config.init_mode)

    def encode_decode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the embedding at input resolution.

        Args:
            x : Input feature stack of shape (batch, in_channels, height, width).

        Returns:
            Embedding of shape (batch, embedding_channels, height, width).
        """
        return self.trunk(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Maps the input stack to per-pixel outputs through the configured head.

        Args:
            x : Input feature stack of shape (batch, in_channels, height, width).

        Returns:
            Tensor of shape (batch, out_channels, height, width) holding the
            interleaved Gaussian parameters, or the raw embedding when the head
            is ``none``.
        """
        return self._head_forward(self.encode_decode(x))
