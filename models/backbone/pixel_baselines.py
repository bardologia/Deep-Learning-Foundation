from __future__ import annotations

import torch
import torch.nn as nn

from configuration.architectures import LocalCNNConfig, PixelMLPNetConfig
from ..blocks                    import ConvBlock, OutputHeadsMixin, build_activation, build_norm2d, initialize_weights


class PixelMLPNet(nn.Module, OutputHeadsMixin):
    def __init__(self, config: PixelMLPNetConfig | None = None):
        super().__init__()
        self.config = config if config is not None else PixelMLPNetConfig()

        if len(self.config.features) == 0:
            raise ValueError("features must contain at least one channel size")

        layers   = []
        channels = self.config.in_channels
        for feature_size in self.config.features:
            layers.append(nn.Conv2d(channels, feature_size, kernel_size=1, bias=self.config.conv_bias))
            layers.append(build_norm2d(self.config.normalization, feature_size))
            layers.append(build_activation(self.config.activation))
            if self.config.dropout > 0:
                layers.append(nn.Dropout(self.config.dropout))
            channels = feature_size

        self.trunk              = nn.Sequential(*layers)
        self.embedding_channels = channels
        self._build_output_head()

        initialize_weights(module=self, mode=self.config.init_mode)

    def encode_decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._head_forward(self.encode_decode(x))


class LocalCNN(nn.Module, OutputHeadsMixin):
    def __init__(self, config: LocalCNNConfig | None = None):
        super().__init__()
        self.config = config if config is not None else LocalCNNConfig()

        if len(self.config.features) == 0:
            raise ValueError("features must contain at least one channel size")

        kernels = self.config.block_kernels if self.config.block_kernels is not None else [3] * len(self.config.features)
        if len(kernels) != len(self.config.features):
            raise ValueError(f"block_kernels has {len(kernels)} entries but features has {len(self.config.features)}; every block needs exactly one kernel size")

        blocks   = []
        channels = self.config.in_channels
        for feature_size, kernel_size in zip(self.config.features, kernels):
            blocks.append(
                ConvBlock(
                    input_channels  = channels,
                    output_channels = feature_size,
                    dropout         = self.config.dropout,
                    activation      = self.config.activation,
                    normalization   = self.config.normalization,
                    bias            = self.config.conv_bias,
                    kernel_size     = kernel_size,
                )
            )
            channels = feature_size

        self.trunk              = nn.Sequential(*blocks)
        self.embedding_channels = channels
        self._build_output_head()

        initialize_weights(module=self, mode=self.config.init_mode)

    def encode_decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._head_forward(self.encode_decode(x))
