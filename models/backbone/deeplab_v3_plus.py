"""DeepLabV3+ backbone and its atrous spatial pyramid pooling module.

Pairs a residual convolutional encoder with an ASPP context module and the
DeepLabV3+ low-level-feature decoder, upsampling back to the input resolution
before the shared output head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

from configuration.architectures import DeepLabV3PlusConfig
from ..blocks                    import ConvNormAct, LayerFactory, OutputHeadsMixin, ResidualConvBlock


class ASPP(nn.Module):
    """Atrous spatial pyramid pooling with a global image-level branch.

    Runs one 1x1 branch, one dilated 3x3 branch per atrous rate and a global
    average-pooled branch, concatenates them and projects the result back to
    ``output_channels``.
    """

    def __init__(
        self,
        input_channels:  int,
        output_channels: int,
        atrous_rates:    tuple,
        dropout:         float,
        activation:      str,
        normalization:   str,
        bias:            bool,
    ):
        """Builds the parallel atrous branches and the fusion projection.

        Args:
            input_channels  : Channel count of the encoder feature map.
            output_channels : Channel count of each branch and of the fused output.
            atrous_rates    : Dilation rates of the 3x3 branches.
            dropout         : Dropout2d probability applied after the fusion projection.
            activation      : Activation name passed to ``LayerFactory``.
            normalization   : Normalisation kind passed to ``LayerFactory``.
            bias            : Whether the branch convolutions carry a bias term.
        """
        super().__init__()
        self.branches = nn.ModuleList([ConvNormAct(input_channels, output_channels, kernel_size=1, activation=activation, normalization=normalization, bias=bias)])

        for rate in atrous_rates:
            self.branches.append(ConvNormAct(input_channels, output_channels, kernel_size=3, dilation=rate, activation=activation, normalization=normalization, bias=bias))

        self.pool_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=bias),
            LayerFactory.build_norm2d("group", output_channels),
            LayerFactory.build_activation(activation),
        )

        n_branches   = len(self.branches) + 1
        self.project = nn.Sequential(
            ConvNormAct(n_branches * output_channels, output_channels, kernel_size=1, activation=activation, normalization=normalization, bias=bias),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the fused multi-scale context features.

        Args:
            x : Encoder feature map of shape (batch, input_channels, height, width).

        Returns:
            Feature map of shape (batch, output_channels, height, width).
        """
        branch_outputs = [branch(x) for branch in self.branches]

        pooled = self.pool_branch(x)
        pooled = functional.interpolate(pooled, size=x.shape[2:], mode="bilinear", align_corners=False)
        branch_outputs.append(pooled)

        return self.project(torch.cat(branch_outputs, dim=1))


class DeepLabV3Plus(nn.Module, OutputHeadsMixin):
    """DeepLabV3+ backbone with a residual encoder and ASPP context module.

    A stride-2 stem halves the resolution, residual stages widen the channels
    with pooling between the intermediate stages, ASPP aggregates multi-scale
    context, and the decoder fuses it with a projected low-level feature map
    before bilinear upsampling back to the input size.

    Attributes:
        config             : Architecture configuration in use.
        embedding_channels : Channel count of the decoder embedding handed to the output head, half the last encoder width.
    """

    def __init__(self, config: DeepLabV3PlusConfig | None = None):
        """Builds the stem, residual encoder, ASPP, decoder and output head.

        Args:
            config : Architecture configuration; a default ``DeepLabV3PlusConfig`` is used when omitted.

        Raises:
            ValueError : If ``features`` holds fewer than two channel sizes.
        """
        super().__init__()
        if config is None:
            config = DeepLabV3PlusConfig()
        self.config = config

        if len(config.features) < 2:
            raise ValueError("features must contain at least two channel sizes")

        feature_sizes      = config.features
        aspp_channels      = feature_sizes[-1] // 2
        low_level_channels = max(feature_sizes[0] // 2, 16)

        self.stem = nn.Sequential(
            nn.Conv2d(config.in_channels, feature_sizes[0], kernel_size=3, stride=2, padding=1, bias=config.conv_bias),
            LayerFactory.build_norm2d(config.normalization, feature_sizes[0]),
            LayerFactory.build_activation(config.activation),
        )

        self.encoder_stages    = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        channels = feature_sizes[0]
        for index, feature_size in enumerate(feature_sizes):
            self.encoder_stages.append(
                ResidualConvBlock(
                    input_channels  = channels,
                    output_channels = feature_size,
                    dropout         = config.dropout,
                    activation      = config.activation,
                    normalization   = config.normalization,
                    bias            = config.conv_bias,
                )
            )
            self.downsample_layers.append(nn.MaxPool2d(kernel_size=2) if 0 < index < len(feature_sizes) - 1 else nn.Identity())
            channels = feature_size

        self.aspp = ASPP(
            input_channels  = feature_sizes[-1],
            output_channels = aspp_channels,
            atrous_rates    = config.atrous_rates,
            dropout         = config.dropout,
            activation      = config.activation,
            normalization   = config.normalization,
            bias            = config.conv_bias,
        )

        self.low_level_projection = ConvNormAct(feature_sizes[0], low_level_channels, kernel_size=1, activation=config.activation, normalization=config.normalization, bias=config.conv_bias)

        self.decoder_blocks = nn.Sequential(
            ConvNormAct(aspp_channels + low_level_channels, aspp_channels, kernel_size=3, activation=config.activation, normalization=config.normalization, bias=config.conv_bias),
            ConvNormAct(aspp_channels, aspp_channels, kernel_size=3, activation=config.activation, normalization=config.normalization, bias=config.conv_bias),
        )

        self.embedding_channels = aspp_channels
        self._build_output_head()

        LayerFactory.initialize_weights(module=self, mode=config.init_mode)

    def encode_decode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the decoder embedding for an input patch.

        Args:
            x : Input stack of shape (batch, in_channels, height, width).

        Returns:
            Feature map of shape (batch, features[-1] // 2, height, width),
            bilinearly resized back to the spatial size of ``x``.
        """
        input_size = x.shape[2:]

        x         = self.stem(x)
        low_level = None
        for index, (stage, downsample) in enumerate(zip(self.encoder_stages, self.downsample_layers)):
            x = downsample(x)
            x = stage(x)
            if index == 0:
                low_level = x

        x = self.aspp(x)
        x = functional.interpolate(x, size=low_level.shape[2:], mode="bilinear", align_corners=False)

        low_level = self.low_level_projection(low_level)

        x = torch.cat([x, low_level], dim=1)
        x = self.decoder_blocks(x)
        return functional.interpolate(x, size=input_size, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the head output for an input patch.

        Args:
            x : Input stack of shape (batch, in_channels, height, width).

        Returns:
            Head output of shape (batch, out_channels, height, width); for the
            Gaussian heads the channels are the per-slot parameters interleaved
            as amplitude, mu and sigma.
        """
        return self._head_forward(self.encode_decode(x))
