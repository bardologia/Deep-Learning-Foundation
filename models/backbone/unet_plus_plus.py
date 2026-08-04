from __future__ import annotations

import torch
import torch.nn as nn

from configuration.architectures import UNetPlusPlusConfig
from ..blocks                    import ConvBlock, OutputHeadsMixin, initialize_weights, match_spatial_size

class UNetPlusPlus(nn.Module, OutputHeadsMixin):
    def __init__(self, config: UNetPlusPlusConfig | None = None):
        super().__init__()
        if config is None:
            config = UNetPlusPlusConfig()
        self.config = config

        feature_sizes = config.features
        if len(feature_sizes) != 4:
            raise ValueError("features must contain exactly 4 channel sizes")

        level_0, level_1, level_2, level_3 = feature_sizes
        bottleneck_width = level_3 * config.bottleneck_factor
        dropout          = config.dropout
        act              = config.activation
        norm             = config.normalization
        bias             = config.conv_bias

        self.pool           = nn.MaxPool2d(2)
        self._upsample_mode = config.upsample_mode
        if config.upsample_mode == "convtranspose":
            unique_channels = sorted(set([level_1, level_2, level_3, bottleneck_width]))
            self.upsample_modules = nn.ModuleDict({
                str(c): nn.ConvTranspose2d(c, c, kernel_size=2, stride=2) for c in unique_channels
            })
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.encoder_0_0 = ConvBlock(config.in_channels, level_0, dropout, act, norm, bias)
        self.encoder_1_0 = ConvBlock(level_0, level_1, dropout, act, norm, bias)
        self.encoder_2_0 = ConvBlock(level_1, level_2, dropout, act, norm, bias)
        self.encoder_3_0 = ConvBlock(level_2, level_3, dropout, act, norm, bias)
        self.encoder_4_0 = ConvBlock(level_3, bottleneck_width, dropout, act, norm, bias)

        self.dense_0_1 = ConvBlock(level_0 + level_1, level_0, dropout, act, norm, bias)
        self.dense_1_1 = ConvBlock(level_1 + level_2, level_1, dropout, act, norm, bias)
        self.dense_2_1 = ConvBlock(level_2 + level_3, level_2, dropout, act, norm, bias)
        self.dense_3_1 = ConvBlock(level_3 + bottleneck_width, level_3, dropout, act, norm, bias)

        self.dense_0_2 = ConvBlock(level_0 * 2 + level_1, level_0, dropout, act, norm, bias)
        self.dense_1_2 = ConvBlock(level_1 * 2 + level_2, level_1, dropout, act, norm, bias)
        self.dense_2_2 = ConvBlock(level_2 * 2 + level_3, level_2, dropout, act, norm, bias)

        self.dense_0_3 = ConvBlock(level_0 * 3 + level_1, level_0, dropout, act, norm, bias)
        self.dense_1_3 = ConvBlock(level_1 * 3 + level_2, level_1, dropout, act, norm, bias)

        self.dense_0_4 = ConvBlock(level_0 * 4 + level_1, level_0, dropout, act, norm, bias)

        self.embedding_channels = level_0
        self._build_output_head()

        initialize_weights(module=self, mode=config.init_mode)

    def _upsample_and_match(self, source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if self._upsample_mode == "convtranspose":
            upsampled = self.upsample_modules[str(source.shape[1])](source)
        else:
            upsampled = self.upsample(source)
        return match_spatial_size(source=upsampled, reference=reference)

    def encode_decode(self, x: torch.Tensor) -> torch.Tensor:
        node_0_0 = self.encoder_0_0(x)
        node_1_0 = self.encoder_1_0(self.pool(node_0_0))
        node_2_0 = self.encoder_2_0(self.pool(node_1_0))
        node_3_0 = self.encoder_3_0(self.pool(node_2_0))
        node_4_0 = self.encoder_4_0(self.pool(node_3_0))

        up_1_0   = self._upsample_and_match(source=node_1_0, reference=node_0_0)
        node_0_1 = self.dense_0_1(torch.cat([node_0_0, up_1_0], dim=1))
        up_2_0   = self._upsample_and_match(source=node_2_0, reference=node_1_0)
        node_1_1 = self.dense_1_1(torch.cat([node_1_0, up_2_0], dim=1))
        up_3_0   = self._upsample_and_match(source=node_3_0, reference=node_2_0)
        node_2_1 = self.dense_2_1(torch.cat([node_2_0, up_3_0], dim=1))
        up_4_0   = self._upsample_and_match(source=node_4_0, reference=node_3_0)
        node_3_1 = self.dense_3_1(torch.cat([node_3_0, up_4_0], dim=1))

        up_1_1   = self._upsample_and_match(source=node_1_1, reference=node_0_0)
        node_0_2 = self.dense_0_2(torch.cat([node_0_0, node_0_1, up_1_1], dim=1))
        up_2_1   = self._upsample_and_match(source=node_2_1, reference=node_1_0)
        node_1_2 = self.dense_1_2(torch.cat([node_1_0, node_1_1, up_2_1], dim=1))
        up_3_1   = self._upsample_and_match(source=node_3_1, reference=node_2_0)
        node_2_2 = self.dense_2_2(torch.cat([node_2_0, node_2_1, up_3_1], dim=1))

        up_1_2   = self._upsample_and_match(source=node_1_2, reference=node_0_0)
        node_0_3 = self.dense_0_3(torch.cat([node_0_0, node_0_1, node_0_2, up_1_2], dim=1))
        up_2_2   = self._upsample_and_match(source=node_2_2, reference=node_1_0)
        node_1_3 = self.dense_1_3(torch.cat([node_1_0, node_1_1, node_1_2, up_2_2], dim=1))

        up_1_3   = self._upsample_and_match(source=node_1_3, reference=node_0_0)
        node_0_4 = self.dense_0_4(torch.cat([node_0_0, node_0_1, node_0_2, node_0_3, up_1_3], dim=1))

        return node_0_4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._head_forward(self.encode_decode(x))
