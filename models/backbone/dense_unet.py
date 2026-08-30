"""DenseNet-style U-Net backbone and the dense blocks it is built from.

Follows the Tiramisu layout: dense blocks concatenate every layer output onto
the running feature stack, transition layers pool between resolutions, and only
the newly grown features are carried into the decoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from configuration.architectures import DenseUNetConfig
from ..blocks                    import FeatureGeometry, LayerFactory, OutputHeadsMixin


class DenseLayer(nn.Module):
    """Single pre-activation dense layer producing ``growth_rate`` new channels."""

    def __init__(self, input_channels: int, growth_rate: int, dropout: float, activation: str, normalization: str, bias: bool):
        """Builds the norm, activation, 3x3 convolution and optional dropout.

        Args:
            input_channels : Channel count of the incoming concatenated stack.
            growth_rate    : Number of feature channels this layer contributes.
            dropout        : Dropout2d probability; the layer is omitted when not positive.
            activation     : Activation name passed to ``LayerFactory``.
            normalization  : Normalisation kind passed to ``LayerFactory``.
            bias           : Whether the convolution carries a bias term.
        """
        super().__init__()
        layers = [
            LayerFactory.build_norm2d(normalization, input_channels),
            LayerFactory.build_activation(activation),
            nn.Conv2d(input_channels, growth_rate, kernel_size=3, padding=1, bias=bias),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the newly grown features for a feature map.

        Args:
            x : Feature map of shape (batch, input_channels, height, width).

        Returns:
            Feature map of shape (batch, growth_rate, height, width).
        """
        return self.layers(x)


class DenseBlock(nn.Module):
    """Stack of dense layers whose outputs are concatenated layer by layer.

    Each layer sees the input concatenated with every previous layer output, and
    the block returns only the concatenation of the new features.

    Attributes:
        new_channels : Total channels grown by the block, ``n_layers * growth_rate``.
    """

    def __init__(self, input_channels: int, growth_rate: int, n_layers: int, dropout: float, activation: str, normalization: str, bias: bool):
        """Builds the dense layers with their growing input widths.

        Args:
            input_channels : Channel count entering the first layer.
            growth_rate    : Channels contributed by each layer.
            n_layers       : Number of dense layers in the block.
            dropout        : Dropout2d probability inside each layer.
            activation     : Activation name passed to ``LayerFactory``.
            normalization  : Normalisation kind passed to ``LayerFactory``.
            bias           : Whether the layer convolutions carry a bias term.
        """
        super().__init__()
        self.layers = nn.ModuleList([
            DenseLayer(input_channels + index * growth_rate, growth_rate, dropout, activation, normalization, bias)
            for index in range(n_layers)
        ])
        self.new_channels = n_layers * growth_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the concatenated new features grown by the block.

        Args:
            x : Feature map of shape (batch, input_channels, height, width).

        Returns:
            Feature map of shape (batch, new_channels, height, width).
        """
        new_features = []
        for layer in self.layers:
            out = layer(x)
            new_features.append(out)
            x = torch.cat([x, out], dim=1)
        return torch.cat(new_features, dim=1)


class TransitionDown(nn.Module):
    """Pre-activation 1x1 bottleneck followed by 2x2 max pooling."""

    def __init__(self, channels: int, dropout: float, activation: str, normalization: str, bias: bool):
        """Builds the norm, activation, 1x1 convolution, dropout and pooling.

        Args:
            channels      : Channel count, preserved through the transition.
            dropout       : Dropout2d probability; the layer is omitted when not positive.
            activation    : Activation name passed to ``LayerFactory``.
            normalization : Normalisation kind passed to ``LayerFactory``.
            bias          : Whether the convolution carries a bias term.
        """
        super().__init__()
        layers = [
            LayerFactory.build_norm2d(normalization, channels),
            LayerFactory.build_activation(activation),
            nn.Conv2d(channels, channels, kernel_size=1, bias=bias),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers.append(nn.MaxPool2d(kernel_size=2))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the half-resolution feature map.

        Args:
            x : Feature map of shape (batch, channels, height, width).

        Returns:
            Feature map of shape (batch, channels, height/2, width/2).
        """
        return self.layers(x)


class DenseUNet(nn.Module, OutputHeadsMixin):
    """U-Net whose encoder and decoder stages are densely connected blocks.

    A 3x3 stem lifts the input to three growth units, dense blocks and
    transition-down layers form the encoder, transposed convolutions upsample in
    the decoder, and each decoder dense block consumes the upsampled features
    concatenated with the matching encoder skip.

    Attributes:
        config             : Architecture configuration in use.
        embedding_channels : Channel count of the decoder embedding handed to the output head, the new-feature width of the last decoder block.
    """

    def __init__(self, config: DenseUNetConfig | None = None):
        """Builds the stem, dense encoder, bottleneck, dense decoder and head.

        Args:
            config : Architecture configuration; a default ``DenseUNetConfig`` is used when omitted.

        Raises:
            ValueError : If ``block_layers`` is empty.
        """
        super().__init__()
        if config is None:
            config = DenseUNetConfig()
        self.config = config

        if len(config.block_layers) == 0:
            raise ValueError("block_layers must contain at least one entry")

        growth        = config.growth_rate
        stem_channels = 3 * growth

        self.stem = nn.Conv2d(config.in_channels, stem_channels, kernel_size=3, padding=1, bias=config.conv_bias)

        self.dense_down    = nn.ModuleList()
        self.trans_down    = nn.ModuleList()
        skip_channel_sizes = []
        channels           = stem_channels
        for n_layers in config.block_layers:
            block = DenseBlock(channels, growth, n_layers, config.dropout, config.activation, config.normalization, config.conv_bias)
            self.dense_down.append(block)
            channels += block.new_channels
            skip_channel_sizes.append(channels)
            self.trans_down.append(TransitionDown(channels, config.dropout, config.activation, config.normalization, config.conv_bias))

        self.bottleneck         = DenseBlock(channels, growth, config.bottleneck_layers, config.dropout, config.activation, config.normalization, config.conv_bias)
        bottleneck_new_channels = self.bottleneck.new_channels

        self.trans_up = nn.ModuleList()
        self.dense_up = nn.ModuleList()
        up_channels   = bottleneck_new_channels
        for skip_channels, n_layers in zip(reversed(skip_channel_sizes), reversed(config.block_layers)):
            self.trans_up.append(nn.ConvTranspose2d(up_channels, up_channels, kernel_size=2, stride=2))
            block = DenseBlock(up_channels + skip_channels, growth, n_layers, config.dropout, config.activation, config.normalization, config.conv_bias)
            self.dense_up.append(block)
            up_channels = block.new_channels

        self.embedding_channels = up_channels
        self._build_output_head()

        LayerFactory.initialize_weights(module=self, mode=config.init_mode)

    def encode_decode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the decoder embedding for an input patch.

        Args:
            x : Input stack of shape (batch, in_channels, height, width).

        Returns:
            Feature map of shape (batch, embedding_channels, height, width).
        """
        x = self.stem(x)

        skip_connections: list[torch.Tensor] = []
        for block, transition in zip(self.dense_down, self.trans_down):
            new_features = block(x)
            x            = torch.cat([x, new_features], dim=1)
            skip_connections.append(x)
            x = transition(x)

        x = self.bottleneck(x)

        for transition, block, skip in zip(self.trans_up, self.dense_up, reversed(skip_connections)):
            x = transition(x)
            x = FeatureGeometry.match_spatial_size(source=x, reference=skip)
            x = torch.cat([x, skip], dim=1)
            x = block(x)

        return x

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
