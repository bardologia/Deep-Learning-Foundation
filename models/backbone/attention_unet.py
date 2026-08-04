from __future__ import annotations

import torch
import torch.nn as nn

from configuration.architectures import AttentionUNetConfig
from ..blocks                    import ConvBlock, OutputHeadsMixin, build_norm2d, build_upsample, initialize_weights, match_spatial_size

class AttentionGate(nn.Module):
    def __init__(
        self,
        gate_channels:         int,
        skip_channels:         int,
        intermediate_channels: int,
        normalization:         str  = "batch",
        bias:                  bool = False,
    ):
        super().__init__()
        self.gate_projection = nn.Sequential(
            nn.Conv2d(
                in_channels  = gate_channels,
                out_channels = intermediate_channels,
                kernel_size  = 1,
                bias         = bias,
            ),
            build_norm2d(normalization, intermediate_channels),
        )
        self.skip_projection = nn.Sequential(
            nn.Conv2d(
                in_channels  = skip_channels,
                out_channels = intermediate_channels,
                kernel_size  = 2,
                stride       = 2,
                bias         = bias,
            ),
            build_norm2d(normalization, intermediate_channels),
        )
        self.attention_score = nn.Sequential(
            nn.Conv2d(
                in_channels  = intermediate_channels,
                out_channels = 1,
                kernel_size  = 1,
                bias         = True,
            ),
            nn.Sigmoid(),
        )
        self.output_transform = nn.Sequential(
            nn.Conv2d(
                in_channels  = skip_channels,
                out_channels = skip_channels,
                kernel_size  = 1,
                bias         = bias,
            ),
            build_norm2d(normalization, skip_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate_signal: torch.Tensor, skip_connection: torch.Tensor) -> torch.Tensor:
        projected_gate = self.gate_projection(gate_signal)
        projected_skip = self.skip_projection(skip_connection)

        projected_gate = match_spatial_size(source=projected_gate, reference=projected_skip)

        attention_weights = self.attention_score(self.relu(projected_gate + projected_skip))
        attention_weights = match_spatial_size(source=attention_weights, reference=skip_connection)

        return self.output_transform(skip_connection * attention_weights)

class AttentionUNet(nn.Module, OutputHeadsMixin):
    def __init__(self, config: AttentionUNetConfig | None = None):
        super().__init__()
        if config is None:
            config = AttentionUNetConfig()
        self.config = config

        if len(config.features) == 0:
            raise ValueError("features must contain at least one channel size")
        if config.attention_intermediate_ratio <= 0:
            raise ValueError("attention_intermediate_ratio must be greater than 0")

        feature_sizes       = config.features
        bottleneck_channels = feature_sizes[-1] * config.bottleneck_factor

        self.encoder_blocks    = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        channels = config.in_channels
        for feature_size in feature_sizes:
            self.encoder_blocks.append(
                ConvBlock(
                    input_channels  = channels,
                    output_channels = feature_size,
                    dropout         = config.dropout,
                    activation      = config.activation,
                    normalization   = config.normalization,
                    bias            = config.conv_bias,
                )
            )
            self.downsample_layers.append(nn.MaxPool2d(kernel_size=2))
            channels = feature_size

        self.bottleneck = ConvBlock(
            input_channels  = feature_sizes[-1],
            output_channels = bottleneck_channels,
            dropout         = config.dropout,
            activation      = config.activation,
            normalization   = config.normalization,
            bias            = config.conv_bias,
        )

        reversed_features    = [bottleneck_channels] + feature_sizes[::-1]
        self.upsample_layers = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        self.decoder_blocks  = nn.ModuleList()
        for index in range(len(reversed_features) - 1):
            decoder_channels      = reversed_features[index + 1]
            intermediate_channels = max(1, int(decoder_channels * config.attention_intermediate_ratio))

            self.upsample_layers.append(
                build_upsample(
                    mode         = config.upsample_mode,
                    in_channels  = reversed_features[index],
                    out_channels = decoder_channels,
                )
            )
            
            self.attention_gates.append(
                AttentionGate(
                    gate_channels         = reversed_features[index],
                    skip_channels         = decoder_channels,
                    intermediate_channels = intermediate_channels,
                    normalization         = config.normalization,
                    bias                  = config.conv_bias,
                )
            )
            
            self.decoder_blocks.append(
                ConvBlock(
                    input_channels  = decoder_channels * 2,
                    output_channels = decoder_channels,
                    dropout         = config.dropout,
                    activation      = config.activation,
                    normalization   = config.normalization,
                    bias            = config.conv_bias,
                )
            )

        self.embedding_channels = feature_sizes[0]
        self._build_output_head()

        initialize_weights(module=self, mode=config.init_mode)

    def encode_decode(self, x: torch.Tensor) -> torch.Tensor:
        original_input = x
        skip_connections: list[torch.Tensor] = []
        for encoder_block, downsample in zip(self.encoder_blocks, self.downsample_layers):
            x = encoder_block(x)
            skip_connections.append(x)
            x = downsample(x)

        x = self.bottleneck(x)

        for upsample, attention_gate, decoder_block, skip in zip(
            self.upsample_layers,
            self.attention_gates,
            self.decoder_blocks,
            reversed(skip_connections),
        ):
            skip = attention_gate(gate_signal=x, skip_connection=skip)
            x    = upsample(x)
            x    = match_spatial_size(source=x, reference=skip)
            x    = torch.cat([skip, x], dim=1)
            x    = decoder_block(x)

        return match_spatial_size(source=x, reference=original_input)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._head_forward(self.encode_decode(x))
