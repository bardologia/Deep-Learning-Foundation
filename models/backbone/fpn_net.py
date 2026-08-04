from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

from configuration.architectures import FPNNetConfig
from ..blocks                    import build_activation, build_norm2d, initialize_weights
from ..blocks                    import OutputHeadsMixin, ResidualConvBlock


class SegmentationBlock(nn.Module):
    def __init__(self, channels: int, n_stages: int, convs_per_stage: int, activation: str, normalization: str, bias: bool):
        super().__init__()

        layers = []

        if n_stages == 0:
            for _ in range(convs_per_stage):
                layers += [
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias),
                    build_norm2d(normalization, channels),
                    build_activation(activation),
                ]

        for _ in range(n_stages):
            for _ in range(convs_per_stage):
                layers += [
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias),
                    build_norm2d(normalization, channels),
                    build_activation(activation),
                ]
            layers += [nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class FPNNet(nn.Module, OutputHeadsMixin):
    def __init__(self, config: FPNNetConfig | None = None):
        super().__init__()
        if config is None:
            config = FPNNetConfig()
        self.config = config

        if len(config.features) < 2:
            raise ValueError("features must contain at least two channel sizes")

        feature_sizes    = config.features
        pyramid_channels = config.pyramid_channels

        self.encoder_blocks    = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        channels = config.in_channels
        for index, feature_size in enumerate(feature_sizes):
            self.downsample_layers.append(nn.Identity() if index == 0 else nn.MaxPool2d(kernel_size=2))
            self.encoder_blocks.append(
                ResidualConvBlock(
                    input_channels  = channels,
                    output_channels = feature_size,
                    dropout         = config.dropout,
                    activation      = config.activation,
                    normalization   = config.normalization,
                    bias            = config.conv_bias,
                    first_unit      = (index == 0),
                )
            )
            channels = feature_size

        self.lateral_convs = nn.ModuleList([nn.Conv2d(feature_size, pyramid_channels, kernel_size=1) for feature_size in feature_sizes])
        self.smooth_convs  = nn.ModuleList([nn.Conv2d(pyramid_channels, pyramid_channels, kernel_size=3, padding=1, bias=config.conv_bias) for _ in feature_sizes])

        self.segmentation_blocks = nn.ModuleList([
            SegmentationBlock(pyramid_channels, index, config.segmentation_convs, config.activation, config.normalization, config.conv_bias)
            for index in range(len(feature_sizes))
        ])

        self.fuse_block = nn.Sequential(
            nn.Conv2d(pyramid_channels, pyramid_channels, kernel_size=3, padding=1, bias=config.conv_bias),
            build_norm2d(config.normalization, pyramid_channels),
            build_activation(config.activation),
            nn.Dropout2d(config.dropout),
        )

        self.embedding_channels = pyramid_channels
        self._build_output_head()

        initialize_weights(module=self, mode=config.init_mode)

    def encode_decode(self, x: torch.Tensor) -> torch.Tensor:
        encoder_outputs = []
        for downsample, encoder_block in zip(self.downsample_layers, self.encoder_blocks):
            x = downsample(x)
            x = encoder_block(x)
            encoder_outputs.append(x)

        pyramid = [self.lateral_convs[-1](encoder_outputs[-1])]
        for index in range(len(encoder_outputs) - 2, -1, -1):
            lateral  = self.lateral_convs[index](encoder_outputs[index])
            top_down = functional.interpolate(pyramid[0], size=lateral.shape[2:], mode="bilinear", align_corners=False)
            pyramid.insert(0, lateral + top_down)

        pyramid = [smooth(level) for smooth, level in zip(self.smooth_convs, pyramid)]

        target_size = pyramid[0].shape[2:]
        fused       = None
        for segmentation_block, level in zip(self.segmentation_blocks, pyramid):
            out = segmentation_block(level)

            if out.shape[2:] != target_size:
                out = functional.interpolate(out, size=target_size, mode="bilinear", align_corners=False)

            fused = out if fused is None else fused + out

        return self.fuse_block(fused)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._head_forward(self.encode_decode(x))
