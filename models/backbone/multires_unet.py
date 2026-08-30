"""MultiResUNet backbone with inception-style blocks and residual skip paths.

Each block chains three 3x3 convolutions of growing width and concatenates them, so
one block spans several effective receptive fields; skip connections pass through
residual ResPath chains before meeting the decoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from configuration.architectures import MultiResUNetConfig
from ..blocks                    import FeatureGeometry, LayerFactory, OutputHeadsMixin


class MultiResBlock(nn.Module):
    """Inception-style block concatenating three chained 3x3 convolutions.

    The three branch widths split ``output_channels`` as roughly 1/6, 1/3 and the
    remainder, approximating 3x3, 5x5 and 7x7 receptive fields at a lower cost.

    Attributes:
        concat_norm : Normalization applied to the concatenated branches.
        shortcut    : 1x1 projection of the input to ``output_channels``.
        activation  : Activation applied to the summed concatenation and shortcut.
        dropout     : Spatial dropout, or identity when disabled.
    """

    def __init__(self, input_channels: int, output_channels: int, dropout: float, activation: str, normalization: str, bias: bool):
        """Builds the three chained branches, the shortcut and the output normalization.

        Args:
            input_channels  : Channel width of the incoming feature map.
            output_channels : Total channel width produced by the block.
            dropout         : Spatial dropout probability; disabled at 0.
            activation      : Activation name resolved by ``LayerFactory.build_activation``.
            normalization   : 2D normalization name resolved by ``LayerFactory.build_norm2d``.
            bias            : Whether the convolutions carry a bias term.
        """
        super().__init__()
        c1 = output_channels // 6
        c2 = output_channels // 3
        c3 = output_channels - c1 - c2

        self.conv1 = self._conv(input_channels, c1, activation, normalization, bias)
        self.conv2 = self._conv(c1, c2, activation, normalization, bias)
        self.conv3 = self._conv(c2, c3, activation, normalization, bias)

        self.concat_norm = LayerFactory.build_norm2d(normalization, output_channels)
        self.shortcut    = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=bias),
            LayerFactory.build_norm2d(normalization, output_channels),
        )
        self.activation = LayerFactory.build_activation(activation)
        self.dropout    = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    @staticmethod
    def _conv(input_channels: int, output_channels: int, activation: str, normalization: str, bias: bool) -> nn.Sequential:
        """Returns a padded 3x3 convolution followed by normalization and activation."""
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=bias),
            LayerFactory.build_norm2d(normalization, output_channels),
            LayerFactory.build_activation(activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the fused multi-receptive-field response.

        Args:
            x : Feature map of shape (batch, input_channels, height, width).

        Returns:
            Feature map of shape (batch, output_channels, height, width).
        """
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)

        out = self.concat_norm(torch.cat([out1, out2, out3], dim=1))
        out = self.activation(out + self.shortcut(x))

        return self.dropout(out)


class ResPath(nn.Module):
    """Chain of residual 3x3 units applied to a skip connection before the decoder.

    Attributes:
        convs      : The 3x3 residual branches, one per chain step.
        shortcuts  : The matching 1x1 projections.
        activation : Activation applied after each step.
    """

    def __init__(self, channels: int, length: int, activation: str, normalization: str, bias: bool):
        """Builds ``length`` residual units that keep the channel width fixed.

        Args:
            channels      : Channel width carried through the whole chain.
            length        : Number of residual units; deeper encoder levels use shorter chains.
            activation    : Activation name resolved by ``LayerFactory.build_activation``.
            normalization : 2D normalization name resolved by ``LayerFactory.build_norm2d``.
            bias          : Whether the convolutions carry a bias term.
        """
        super().__init__()
        self.convs     = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        for _ in range(length):
            self.convs.append(nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias),
                LayerFactory.build_norm2d(normalization, channels),
            ))
            self.shortcuts.append(nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1, bias=bias),
                LayerFactory.build_norm2d(normalization, channels),
            ))
        self.activation = LayerFactory.build_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the skip feature map after the residual chain.

        Args:
            x : Skip feature map of shape (batch, channels, height, width).

        Returns:
            Feature map of the same shape as ``x``.
        """
        for conv, shortcut in zip(self.convs, self.shortcuts):
            x = self.activation(conv(x) + shortcut(x))
        return x


class MultiResUNet(nn.Module, OutputHeadsMixin):
    """MultiResUNet backbone predicting per-pixel Gaussian mixture parameters.

    Attributes:
        config             : Architecture configuration driving widths, depth and head choice.
        embedding_channels : Channel width of the decoded embedding fed to the output head.
    """

    def __init__(self, config: MultiResUNetConfig | None = None):
        """Builds the encoder, the ResPath skips, the bottleneck, the decoder and the head.

        Args:
            config : Architecture configuration; defaults to ``MultiResUNetConfig()``.

        Raises:
            ValueError : If ``features`` is empty.
        """
        super().__init__()
        if config is None:
            config = MultiResUNetConfig()
        self.config = config

        if len(config.features) == 0:
            raise ValueError("features must contain at least one channel size")

        feature_sizes       = config.features
        n_levels            = len(feature_sizes)
        bottleneck_channels = feature_sizes[-1] * config.bottleneck_factor

        self.encoder_blocks    = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        self.res_paths         = nn.ModuleList()
        channels = config.in_channels
        for index, feature_size in enumerate(feature_sizes):
            self.encoder_blocks.append(MultiResBlock(channels, feature_size, config.dropout, config.activation, config.normalization, config.conv_bias))
            self.downsample_layers.append(nn.MaxPool2d(kernel_size=2))
            self.res_paths.append(ResPath(feature_size, n_levels - index, config.activation, config.normalization, config.conv_bias))
            channels = feature_size

        self.bottleneck = MultiResBlock(feature_sizes[-1], bottleneck_channels, config.dropout, config.activation, config.normalization, config.conv_bias)

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
            self.decoder_blocks.append(MultiResBlock(reversed_features[index + 1] * 2, reversed_features[index + 1], config.dropout, config.activation, config.normalization, config.conv_bias))

        self.embedding_channels = feature_sizes[0]
        self._build_output_head()

        LayerFactory.initialize_weights(module=self, mode=config.init_mode)

    def encode_decode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the embedding at input resolution.

        Args:
            x : Input feature stack of shape (batch, in_channels, height, width).

        Returns:
            Embedding of shape (batch, embedding_channels, height, width).
        """
        skip_connections: list[torch.Tensor] = []
        for encoder_block, downsample, res_path in zip(self.encoder_blocks, self.downsample_layers, self.res_paths):
            x = encoder_block(x)
            skip_connections.append(res_path(x))
            x = downsample(x)

        x = self.bottleneck(x)

        for upsample, decoder_block, skip in zip(self.upsample_layers, self.decoder_blocks, reversed(skip_connections)):
            x = upsample(x)
            x = FeatureGeometry.match_spatial_size(source=x, reference=skip)
            x = torch.cat([skip, x], dim=1)
            x = decoder_block(x)

        return x

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
