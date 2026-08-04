from __future__ import annotations

import torch
import torch.nn as nn

from configuration.architectures import MultiResUNetConfig
from ..blocks                    import build_activation, build_norm2d, build_upsample, initialize_weights
from ..blocks                    import OutputHeadsMixin, match_spatial_size


class MultiResBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, dropout: float, activation: str, normalization: str, bias: bool):
        super().__init__()
        c1 = output_channels // 6
        c2 = output_channels // 3
        c3 = output_channels - c1 - c2

        def conv(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=bias),
                build_norm2d(normalization, out_ch),
                build_activation(activation),
            )

        self.conv1 = conv(input_channels, c1)
        self.conv2 = conv(c1, c2)
        self.conv3 = conv(c2, c3)

        self.concat_norm = build_norm2d(normalization, output_channels)
        self.shortcut    = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=bias),
            build_norm2d(normalization, output_channels),
        )
        self.activation = build_activation(activation)
        self.dropout    = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)

        out = self.concat_norm(torch.cat([out1, out2, out3], dim=1))
        out = self.activation(out + self.shortcut(x))

        return self.dropout(out)


class ResPath(nn.Module):
    def __init__(self, channels: int, length: int, activation: str, normalization: str, bias: bool):
        super().__init__()
        self.convs     = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        for _ in range(length):
            self.convs.append(nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias),
                build_norm2d(normalization, channels),
            ))
            self.shortcuts.append(nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1, bias=bias),
                build_norm2d(normalization, channels),
            ))
        self.activation = build_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv, shortcut in zip(self.convs, self.shortcuts):
            x = self.activation(conv(x) + shortcut(x))
        return x


class MultiResUNet(nn.Module, OutputHeadsMixin):
    def __init__(self, config: MultiResUNetConfig | None = None):
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
                build_upsample(
                    mode         = config.upsample_mode,
                    in_channels  = reversed_features[index],
                    out_channels = reversed_features[index + 1],
                )
            )
            self.decoder_blocks.append(MultiResBlock(reversed_features[index + 1] * 2, reversed_features[index + 1], config.dropout, config.activation, config.normalization, config.conv_bias))

        self.embedding_channels = feature_sizes[0]
        self._build_output_head()

        initialize_weights(module=self, mode=config.init_mode)

    def encode_decode(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections: list[torch.Tensor] = []
        for encoder_block, downsample, res_path in zip(self.encoder_blocks, self.downsample_layers, self.res_paths):
            x = encoder_block(x)
            skip_connections.append(res_path(x))
            x = downsample(x)

        x = self.bottleneck(x)

        for upsample, decoder_block, skip in zip(self.upsample_layers, self.decoder_blocks, reversed(skip_connections)):
            x = upsample(x)
            x = match_spatial_size(source=x, reference=skip)
            x = torch.cat([skip, x], dim=1)
            x = decoder_block(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._head_forward(self.encode_decode(x))
