"""Residual U-shaped backbones sharing one encoder-decoder skeleton.

``ResUNet`` downsamples with strided residual blocks; ``UNetSkip`` keeps the blocks
at stride one and pools instead. Both are the reference architectures of the project.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from configuration.architectures import ResUNetConfig, UNetSkipConfig
from ..blocks                    import FeatureGeometry, LayerFactory, OutputHeadsMixin, ResidualConvBlock


class ResUNetBackbone(nn.Module, OutputHeadsMixin):
    """Residual encoder-decoder trunk shared by ResUNet and UNetSkip.

    Subclasses attach the output head and initialize the weights; this class only
    provides the layers and the ``encode_decode`` path.

    Attributes:
        config             : Architecture configuration driving widths, depth and head choice.
        downsample_mode    : Either ``stride`` or ``maxpool``, selecting how resolution drops.
        embedding_channels : Channel width of the decoded embedding fed to the output head.
    """

    def __init__(self, config, downsample: str = "stride"):
        """Builds the residual encoder, the bottleneck and the concatenating decoder.

        Args:
            config     : Architecture configuration exposing ``features``, ``in_channels``, ``bottleneck_factor``, ``upsample_mode`` and the block hyperparameters.
            downsample : Downsampling strategy, ``stride`` or ``maxpool``.

        Raises:
            ValueError : If ``features`` is empty or ``downsample`` is unknown.
        """
        super().__init__()
        self.config          = config
        self.downsample_mode = downsample

        if len(config.features) == 0:
            raise ValueError("features must contain at least one channel size")
        if downsample not in ("stride", "maxpool"):
            raise ValueError(f"Unknown downsample mode '{downsample}'. Available: stride, maxpool")

        feature_sizes       = config.features
        bottleneck_channels = feature_sizes[-1] * config.bottleneck_factor

        self.encoder_blocks    = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        channels = config.in_channels
        for index, feature_size in enumerate(feature_sizes):
            stride     = 1 if downsample == "maxpool" or index == 0 else 2
            first_unit = index == 0

            self.encoder_blocks.append(
                ResidualConvBlock(
                    input_channels  = channels,
                    output_channels = feature_size,
                    dropout         = config.dropout,
                    activation      = config.activation,
                    normalization   = config.normalization,
                    bias            = config.conv_bias,
                    stride          = stride,
                    first_unit      = first_unit,
                )
            )
            if downsample == "maxpool":
                self.downsample_layers.append(nn.MaxPool2d(kernel_size=2))
            channels = feature_size

        self.bottleneck = ResidualConvBlock(
            input_channels  = feature_sizes[-1],
            output_channels = bottleneck_channels,
            dropout         = config.dropout,
            activation      = config.activation,
            normalization   = config.normalization,
            bias            = config.conv_bias,
            stride          = 2 if downsample == "stride" else 1,
        )

        reversed_features    = [bottleneck_channels] + feature_sizes[::-1]
        self.upsample_layers = nn.ModuleList()
        self.decoder_blocks  = nn.ModuleList()
        for index in range(len(reversed_features) - 1):
            self.upsample_layers.append(
                LayerFactory.build_upsample(
                    mode         = config.upsample_mode,
                    in_channels  = reversed_features[index],
                    out_channels = reversed_features[index + 1],
                )
            )
            self.decoder_blocks.append(
                ResidualConvBlock(
                    input_channels  = reversed_features[index + 1] * 2,
                    output_channels = reversed_features[index + 1],
                    dropout         = config.dropout,
                    activation      = config.activation,
                    normalization   = config.normalization,
                    bias            = config.conv_bias,
                )
            )

        self.embedding_channels = feature_sizes[0]

    def encode_decode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the embedding at input resolution.

        Args:
            x : Input feature stack of shape (batch, in_channels, height, width).

        Returns:
            Embedding of shape (batch, embedding_channels, height, width).
        """
        skip_connections: list[torch.Tensor] = []
        if self.downsample_mode == "stride":
            for encoder_block in self.encoder_blocks:
                x = encoder_block(x)
                skip_connections.append(x)
        else:
            for encoder_block, downsample in zip(self.encoder_blocks, self.downsample_layers):
                x = encoder_block(x)
                skip_connections.append(x)
                x = downsample(x)

        x = self.bottleneck(x)

        for upsample, decoder_block, skip in zip(self.upsample_layers, self.decoder_blocks, reversed(skip_connections)):
            x = upsample(x)
            x = FeatureGeometry.match_spatial_size(source=x, reference=skip)
            x = torch.cat([skip, x], dim=1)
            x = decoder_block(x)

        return x


class ResUNet(ResUNetBackbone):
    """Residual U-Net that drops resolution with strided residual blocks."""

    def __init__(self, config: ResUNetConfig | None = None):
        """Builds the strided residual trunk and its output head.

        Args:
            config : Architecture configuration; defaults to ``ResUNetConfig()``.
        """
        super().__init__(config if config is not None else ResUNetConfig())
        self._build_output_head()

        LayerFactory.initialize_weights(module=self, mode=self.config.init_mode)

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


class UNetSkip(ResUNetBackbone):
    """Residual U-Net that drops resolution with max pooling between stride-one blocks."""

    def __init__(self, config: UNetSkipConfig | None = None):
        """Builds the pooled residual trunk and its output head.

        Args:
            config : Architecture configuration; defaults to ``UNetSkipConfig()``.
        """
        super().__init__(config if config is not None else UNetSkipConfig(), downsample="maxpool")
        self._build_output_head()

        LayerFactory.initialize_weights(module=self, mode=self.config.init_mode)

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
