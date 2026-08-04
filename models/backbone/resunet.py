from __future__ import annotations

import torch
import torch.nn as nn

from configuration.architectures import ResUNetConfig, UNetSkipConfig
from ..blocks                    import build_upsample, initialize_weights
from ..blocks                    import OutputHeadsMixin, ResidualConvBlock, match_spatial_size


class ResUNetBackbone(nn.Module, OutputHeadsMixin):
    def __init__(self, config, downsample: str = "stride"):
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
            if downsample == "stride":
                stride     = 1 if index == 0 else 2
                first_unit = index == 0
            else:
                stride     = 1
                first_unit = False

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
                build_upsample(
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
            x = match_spatial_size(source=x, reference=skip)
            x = torch.cat([skip, x], dim=1)
            x = decoder_block(x)

        return x


class ResUNet(ResUNetBackbone):
    def __init__(self, config: ResUNetConfig | None = None):
        super().__init__(config if config is not None else ResUNetConfig())
        self._build_output_head()

        initialize_weights(module=self, mode=self.config.init_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._head_forward(self.encode_decode(x))


class UNetSkip(ResUNetBackbone):
    def __init__(self, config: UNetSkipConfig | None = None):
        super().__init__(config if config is not None else UNetSkipConfig(), downsample="maxpool")
        self._build_output_head()

        initialize_weights(module=self, mode=self.config.init_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._head_forward(self.encode_decode(x))
