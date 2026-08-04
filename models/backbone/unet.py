from __future__ import annotations

import torch
import torch.nn as nn

from configuration.architectures import UNetConfig
from ..blocks                    import initialize_weights
from ..blocks                    import ConvBlock, Decoder, Encoder, OutputHeadsMixin


class UNetBackbone(nn.Module, OutputHeadsMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if len(config.features) == 0:
            raise ValueError("features must contain at least one channel size")

        feature_sizes       = config.features
        bottleneck_channels = feature_sizes[-1] * config.bottleneck_factor

        self.encoder = Encoder(
            input_channels = config.in_channels,
            feature_sizes  = feature_sizes,
            dropout        = config.dropout,
            activation     = config.activation,
            normalization  = config.normalization,
            bias           = config.conv_bias,
        )

        self.bottleneck = ConvBlock(
            input_channels  = feature_sizes[-1],
            output_channels = bottleneck_channels,
            dropout         = config.dropout,
            activation      = config.activation,
            normalization   = config.normalization,
            bias            = config.conv_bias,
        )

        decoder_feature_sizes = [bottleneck_channels] + feature_sizes[::-1]
        self.decoder = Decoder(
            feature_sizes = decoder_feature_sizes,
            dropout       = config.dropout,
            activation    = config.activation,
            normalization = config.normalization,
            bias          = config.conv_bias,
            upsample_mode = config.upsample_mode,
        )

        self.embedding_channels = feature_sizes[0]

    def encode_decode(self, x: torch.Tensor) -> torch.Tensor:
        x, skip_connections = self.encoder(x)
        x = self.bottleneck(x)
        return self.decoder(x, skip_connections[::-1])


class UNet(UNetBackbone):
    def __init__(self, config: UNetConfig | None = None):
        super().__init__(config if config is not None else UNetConfig())
        self._build_output_head()

        initialize_weights(module=self, mode=self.config.init_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._head_forward(self.encode_decode(x))
