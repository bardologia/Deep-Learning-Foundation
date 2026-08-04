from __future__ import annotations

import torch
import torch.nn as nn

from configuration.architectures import DenseUNetConfig
from ..blocks                    import build_activation, build_norm2d, initialize_weights
from ..blocks                    import OutputHeadsMixin, match_spatial_size


class DenseLayer(nn.Module):
    def __init__(self, input_channels: int, growth_rate: int, dropout: float, activation: str, normalization: str, bias: bool):
        super().__init__()
        layers = [
            build_norm2d(normalization, input_channels),
            build_activation(activation),
            nn.Conv2d(input_channels, growth_rate, kernel_size=3, padding=1, bias=bias),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DenseBlock(nn.Module):
    def __init__(self, input_channels: int, growth_rate: int, n_layers: int, dropout: float, activation: str, normalization: str, bias: bool):
        super().__init__()
        self.layers = nn.ModuleList([
            DenseLayer(input_channels + index * growth_rate, growth_rate, dropout, activation, normalization, bias)
            for index in range(n_layers)
        ])
        self.new_channels = n_layers * growth_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = []
        for layer in self.layers:
            out = layer(x)
            new_features.append(out)
            x = torch.cat([x, out], dim=1)
        return torch.cat(new_features, dim=1)


class TransitionDown(nn.Module):
    def __init__(self, channels: int, dropout: float, activation: str, normalization: str, bias: bool):
        super().__init__()
        layers = [
            build_norm2d(normalization, channels),
            build_activation(activation),
            nn.Conv2d(channels, channels, kernel_size=1, bias=bias),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers.append(nn.MaxPool2d(kernel_size=2))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DenseUNet(nn.Module, OutputHeadsMixin):
    def __init__(self, config: DenseUNetConfig | None = None):
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

        initialize_weights(module=self, mode=config.init_mode)

    def encode_decode(self, x: torch.Tensor) -> torch.Tensor:
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
            x = match_spatial_size(source=x, reference=skip)
            x = torch.cat([x, skip], dim=1)
            x = block(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._head_forward(self.encode_decode(x))
