"""Feature pyramid network backbone and its per-level segmentation block.

Builds a residual encoder, fuses the levels top-down through lateral 1x1
projections, brings every level to the finest resolution with repeated
segmentation blocks and sums them into a single embedding.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

from configuration.architectures import FPNNetConfig
from ..blocks                    import LayerFactory, OutputHeadsMixin, ResidualConvBlock


class SegmentationBlock(nn.Module):
    """Repeated 3x3 convolutions with one 2x upsampling per pyramid stage.

    The number of stages equals the depth of the pyramid level, so every level
    is lifted to the resolution of the finest level.
    """

    def __init__(self, channels: int, n_stages: int, convs_per_stage: int, activation: str, normalization: str, bias: bool):
        """Builds the convolution and upsampling sequence for one pyramid level.

        Args:
            channels        : Channel count, preserved through the block.
            n_stages        : Number of 2x upsampling stages; zero yields a single convolution group with no upsampling.
            convs_per_stage : Convolutions applied before each upsampling step.
            activation      : Activation name passed to ``LayerFactory``.
            normalization   : Normalisation kind passed to ``LayerFactory``.
            bias            : Whether the convolutions carry a bias term.
        """
        super().__init__()

        layers = []

        for _ in range(max(n_stages, 1)):
            for _ in range(convs_per_stage):
                layers += [
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias),
                    LayerFactory.build_norm2d(normalization, channels),
                    LayerFactory.build_activation(activation),
                ]
            if n_stages > 0:
                layers += [nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the upsampled features for one pyramid level.

        Args:
            x : Pyramid level of shape (batch, channels, height, width).

        Returns:
            Feature map of shape
            (batch, channels, height * 2**n_stages, width * 2**n_stages).
        """
        return self.layers(x)


class FPNNet(nn.Module, OutputHeadsMixin):
    """Feature pyramid backbone summing per-level segmentation branches.

    Residual encoder stages halve the resolution from the second stage onwards,
    lateral 1x1 convolutions bring every stage to ``pyramid_channels``, a
    top-down path adds the coarser levels into the finer ones, and the smoothed
    levels are upsampled and summed before a final fusion block.

    Attributes:
        config             : Architecture configuration in use.
        embedding_channels : Channel count of the fused embedding handed to the output head, equal to ``pyramid_channels``.
    """

    def __init__(self, config: FPNNetConfig | None = None):
        """Builds the encoder, lateral and smoothing convolutions, and the head.

        Args:
            config : Architecture configuration; a default ``FPNNetConfig`` is used when omitted.

        Raises:
            ValueError : If ``features`` holds fewer than two channel sizes.
        """
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
            LayerFactory.build_norm2d(config.normalization, pyramid_channels),
            LayerFactory.build_activation(config.activation),
            nn.Dropout2d(config.dropout),
        )

        self.embedding_channels = pyramid_channels
        self._build_output_head()

        LayerFactory.initialize_weights(module=self, mode=config.init_mode)

    def encode_decode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the fused pyramid embedding for an input patch.

        Args:
            x : Input stack of shape (batch, in_channels, height, width).

        Returns:
            Feature map of shape (batch, pyramid_channels, height, width) at the
            resolution of the finest pyramid level.
        """
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
        """Returns the head output for an input patch.

        Args:
            x : Input stack of shape (batch, in_channels, height, width).

        Returns:
            Head output of shape (batch, out_channels, height, width); for the
            Gaussian heads the channels are the per-slot parameters interleaved
            as amplitude, mu and sigma.
        """
        return self._head_forward(self.encode_decode(x))
