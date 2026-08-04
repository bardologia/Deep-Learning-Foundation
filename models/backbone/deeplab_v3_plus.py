from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

from configuration.architectures import DeepLabV3PlusConfig
from ..blocks                    import build_activation, build_norm2d, initialize_weights
from ..blocks                    import ConvNormAct, OutputHeadsMixin, ResidualConvBlock


class ASPP(nn.Module):
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
        super().__init__()
        self.branches = nn.ModuleList([ConvNormAct(input_channels, output_channels, kernel_size=1, activation=activation, normalization=normalization, bias=bias)])

        for rate in atrous_rates:
            self.branches.append(ConvNormAct(input_channels, output_channels, kernel_size=3, dilation=rate, activation=activation, normalization=normalization, bias=bias))

        self.pool_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=bias),
            build_norm2d("group", output_channels),
            build_activation(activation),
        )

        n_branches   = len(self.branches) + 1
        self.project = nn.Sequential(
            ConvNormAct(n_branches * output_channels, output_channels, kernel_size=1, activation=activation, normalization=normalization, bias=bias),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outputs = [branch(x) for branch in self.branches]

        pooled = self.pool_branch(x)
        pooled = functional.interpolate(pooled, size=x.shape[2:], mode="bilinear", align_corners=False)
        branch_outputs.append(pooled)

        return self.project(torch.cat(branch_outputs, dim=1))


class DeepLabV3Plus(nn.Module, OutputHeadsMixin):
    def __init__(self, config: DeepLabV3PlusConfig | None = None):
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
            build_norm2d(config.normalization, feature_sizes[0]),
            build_activation(config.activation),
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

        initialize_weights(module=self, mode=config.init_mode)

    def encode_decode(self, x: torch.Tensor) -> torch.Tensor:
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
        return self._head_forward(self.encode_decode(x))
