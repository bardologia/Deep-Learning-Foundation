"""LinkNet backbone: residual strided encoder with additive decoder skips.

The decoder adds each stage output back onto the matching encoder feature map
instead of concatenating it, which keeps the decoder narrow relative to a U-Net.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from configuration.architectures import LinkNetConfig
from ..blocks                    import FeatureGeometry, LayerFactory, OutputHeadsMixin

class ResidualEncoderBlock(nn.Module):
    """Residual block that halves the spatial resolution with a strided convolution.

    Attributes:
        main_path     : Strided 3x3 convolution followed by a second 3x3 convolution.
        shortcut_path : Strided 1x1 projection matching the main path shape.
        activation    : Activation applied to the summed paths.
    """

    def __init__(
        self,
        input_channels:  int,
        output_channels: int,
        dropout:         float = 0.0,
        activation:      str   = "relu",
        normalization:   str   = "batch",
        bias:            bool  = False,
    ):
        """Builds the strided residual unit.

        Args:
            input_channels  : Channel width of the incoming feature map.
            output_channels : Channel width produced by the block.
            dropout         : Spatial dropout probability appended to the main path; disabled at 0.
            activation      : Activation name resolved by ``LayerFactory.build_activation``.
            normalization   : 2D normalization name resolved by ``LayerFactory.build_norm2d``.
            bias            : Whether the convolutions carry a bias term.
        """
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels  = input_channels,
                out_channels = output_channels,
                kernel_size  = 3,
                stride       = 2,
                padding      = 1,
                bias         = bias,
            ),
            LayerFactory.build_norm2d(normalization, output_channels),
            LayerFactory.build_activation(activation),
            nn.Conv2d(
                in_channels  = output_channels,
                out_channels = output_channels,
                kernel_size  = 3,
                padding      = 1,
                bias         = bias,
            ),
            LayerFactory.build_norm2d(normalization, output_channels),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.main_path = nn.Sequential(*layers)

        self.shortcut_path = nn.Sequential(
            nn.Conv2d(
                in_channels  = input_channels,
                out_channels = output_channels,
                kernel_size  = 1,
                stride       = 2,
                bias         = bias,
            ),
            LayerFactory.build_norm2d(normalization, output_channels),
        )
        self.activation = LayerFactory.build_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the activated residual sum at half the input resolution.

        Args:
            x : Feature map of shape (batch, input_channels, height, width).

        Returns:
            Feature map of shape (batch, output_channels, ceil(height / 2), ceil(width / 2)).
        """
        return self.activation(self.main_path(x) + self.shortcut_path(x))


class BottleneckDecoderBlock(nn.Module):
    """Decoder block that compresses channels, upsamples by two and re-expands.

    Attributes:
        layers : Sequence of 1x1 compression, transposed convolution and 1x1 expansion.
    """

    def __init__(
        self,
        input_channels:    int,
        output_channels:   int,
        compression_ratio: int   = 4,
        activation:        str   = "relu",
        normalization:     str   = "batch",
        bias:              bool  = False,
    ):
        """Builds the compress-upsample-expand stack.

        Args:
            input_channels    : Channel width of the incoming feature map.
            output_channels   : Channel width produced by the block.
            compression_ratio : Divisor applied to ``input_channels`` for the internal width.
            activation        : Activation name resolved by ``LayerFactory.build_activation``.
            normalization     : 2D normalization name resolved by ``LayerFactory.build_norm2d``.
            bias              : Whether the convolutions carry a bias term.
        """
        super().__init__()
        compressed_channels = max(1, input_channels // compression_ratio)
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels  = input_channels,
                out_channels = compressed_channels,
                kernel_size  = 1,
                bias         = bias,
            ),
            LayerFactory.build_norm2d(normalization, compressed_channels),
            LayerFactory.build_activation(activation),
            nn.ConvTranspose2d(
                in_channels    = compressed_channels,
                out_channels   = compressed_channels,
                kernel_size    = 3,
                stride         = 2,
                padding        = 1,
                output_padding = 1,
                bias           = bias,
            ),
            LayerFactory.build_norm2d(normalization, compressed_channels),
            LayerFactory.build_activation(activation),
            nn.Conv2d(
                in_channels  = compressed_channels,
                out_channels = output_channels,
                kernel_size  = 1,
                bias         = bias,
            ),
            LayerFactory.build_norm2d(normalization, output_channels),
            LayerFactory.build_activation(activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the upsampled feature map.

        Args:
            x : Feature map of shape (batch, input_channels, height, width).

        Returns:
            Feature map of shape (batch, output_channels, height * 2, width * 2).
        """
        return self.layers(x)


class LinkNet(nn.Module, OutputHeadsMixin):
    """LinkNet backbone predicting per-pixel Gaussian mixture parameters.

    Attributes:
        config             : Architecture configuration driving widths, depth and head choice.
        embedding_channels : Channel width of the decoded embedding fed to the output head.
    """

    def __init__(self, config: LinkNetConfig | None = None):
        """Builds the stem, the strided encoder ladder, the decoder and the output head.

        Args:
            config : Architecture configuration; defaults to ``LinkNetConfig()``.

        Raises:
            ValueError : If ``features`` is empty, or if ``initial_kernel_size`` or ``decoder_bottleneck_ratio`` is not positive.
        """
        super().__init__()
        if config is None:
            config = LinkNetConfig()
        self.config = config

        if len(config.features) == 0:
            raise ValueError("features must contain at least one channel size")
        if config.initial_kernel_size <= 0:
            raise ValueError("initial_kernel_size must be a positive integer")
        if config.decoder_bottleneck_ratio <= 0:
            raise ValueError("decoder_bottleneck_ratio must be a positive integer")

        feature_sizes = config.features
        kernel_size   = config.initial_kernel_size
        padding       = kernel_size // 2

        self.initial_conv = nn.Sequential(
            nn.Conv2d(
                in_channels  = config.in_channels,
                out_channels = feature_sizes[0],
                kernel_size  = kernel_size,
                stride       = 1,
                padding      = padding,
                bias         = config.conv_bias,
            ),
            LayerFactory.build_norm2d(config.normalization, feature_sizes[0]),
            LayerFactory.build_activation(config.activation),
        )

        self.encoder_stages = nn.ModuleList()
        channels            = feature_sizes[0]
        for feature_size in feature_sizes:
            self.encoder_stages.append(
                ResidualEncoderBlock(
                    input_channels  = channels,
                    output_channels = feature_size,
                    dropout         = config.dropout,
                    activation      = config.activation,
                    normalization   = config.normalization,
                    bias            = config.conv_bias,
                )
            )
            channels = feature_size

        self.decoder_stages = nn.ModuleList()
        for index in range(len(feature_sizes) - 1, 0, -1):
            self.decoder_stages.append(
                BottleneckDecoderBlock(
                    input_channels    = feature_sizes[index],
                    output_channels   = feature_sizes[index - 1],
                    compression_ratio = config.decoder_bottleneck_ratio,
                    activation        = config.activation,
                    normalization     = config.normalization,
                    bias              = config.conv_bias,
                )
            )
        self.decoder_stages.append(
            BottleneckDecoderBlock(
                input_channels    = feature_sizes[0],
                output_channels   = feature_sizes[0],
                compression_ratio = config.decoder_bottleneck_ratio,
                activation        = config.activation,
                normalization     = config.normalization,
                bias              = config.conv_bias,
            )
        )

        self.embedding_channels = feature_sizes[0]
        self._build_output_head()

        LayerFactory.initialize_weights(module=self, mode=config.init_mode)

    def encode_decode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the decoded embedding resampled back to the input resolution.

        Args:
            x : Input feature stack of shape (batch, in_channels, height, width).

        Returns:
            Embedding of shape (batch, embedding_channels, height, width).
        """
        original_input = x
        x = self.initial_conv(x)

        encoder_outputs: list[torch.Tensor] = []
        for encoder_stage in self.encoder_stages:
            x = encoder_stage(x)
            encoder_outputs.append(x)

        for stage_index, decoder_stage in enumerate(self.decoder_stages):
            decoded = decoder_stage(x)

            if stage_index + 1 < len(encoder_outputs):
                target_skip = encoder_outputs[-(stage_index + 2)]
                decoded     = FeatureGeometry.match_spatial_size(source=decoded, reference=target_skip)
                x           = decoded + target_skip
            else:
                x = decoded

        return FeatureGeometry.match_spatial_size(source=x, reference=original_input)

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
