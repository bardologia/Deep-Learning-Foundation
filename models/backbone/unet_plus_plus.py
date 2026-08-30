"""UNet++ backbone with a dense grid of nested decoder skip pathways.

Node (i, j) consumes every shallower node on its own level plus the upsampled node
(i + 1, j - 1), so encoder features reach the output through progressively refined paths.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from configuration.architectures import UNetPlusPlusConfig
from ..blocks                    import ConvBlock, FeatureGeometry, LayerFactory, OutputHeadsMixin

class UNetPlusPlus(nn.Module, OutputHeadsMixin):
    """UNet++ backbone predicting per-pixel Gaussian mixture parameters.

    Attributes:
        config             : Architecture configuration driving widths, depth and head choice.
        embedding_channels : Channel width of the final dense node fed to the output head.
    """

    def __init__(self, config: UNetPlusPlusConfig | None = None):
        """Builds the encoder column, the dense decoder nodes and the output head.

        Args:
            config : Architecture configuration; defaults to ``UNetPlusPlusConfig()``.

        Raises:
            ValueError : If ``features`` does not hold exactly four channel sizes.
        """
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
            level_widths          = {"level_1": level_1, "level_2": level_2, "level_3": level_3, "bottleneck": bottleneck_width}
            self.upsample_modules = nn.ModuleDict({
                level: nn.ConvTranspose2d(width, width, kernel_size=2, stride=2) for level, width in level_widths.items()
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

        LayerFactory.initialize_weights(module=self, mode=config.init_mode)

    def _upsample_and_match(self, source: torch.Tensor, reference: torch.Tensor, level: str) -> torch.Tensor:
        """Upsamples a node by two and resizes it onto the reference grid.

        Args:
            source    : Feature map to upsample, of shape (batch, channels, height, width).
            reference : Feature map whose spatial size the result must match.
            level     : Key selecting the transposed convolution for this level; ignored in bilinear mode.

        Returns:
            Feature map with the channels of ``source`` and the spatial size of ``reference``.
        """
        if self._upsample_mode == "convtranspose":
            upsampled = self.upsample_modules[level](source)
        else:
            upsampled = self.upsample(source)
        return FeatureGeometry.match_spatial_size(source=upsampled, reference=reference)

    def encode_decode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the embedding at input resolution.

        Args:
            x : Input feature stack of shape (batch, in_channels, height, width).

        Returns:
            Embedding of shape (batch, embedding_channels, height, width).
        """
        node_0_0 = self.encoder_0_0(x)
        node_1_0 = self.encoder_1_0(self.pool(node_0_0))
        node_2_0 = self.encoder_2_0(self.pool(node_1_0))
        node_3_0 = self.encoder_3_0(self.pool(node_2_0))
        node_4_0 = self.encoder_4_0(self.pool(node_3_0))

        up_1_0   = self._upsample_and_match(source=node_1_0, reference=node_0_0, level="level_1")
        node_0_1 = self.dense_0_1(torch.cat([node_0_0, up_1_0], dim=1))
        up_2_0   = self._upsample_and_match(source=node_2_0, reference=node_1_0, level="level_2")
        node_1_1 = self.dense_1_1(torch.cat([node_1_0, up_2_0], dim=1))
        up_3_0   = self._upsample_and_match(source=node_3_0, reference=node_2_0, level="level_3")
        node_2_1 = self.dense_2_1(torch.cat([node_2_0, up_3_0], dim=1))
        up_4_0   = self._upsample_and_match(source=node_4_0, reference=node_3_0, level="bottleneck")
        node_3_1 = self.dense_3_1(torch.cat([node_3_0, up_4_0], dim=1))

        up_1_1   = self._upsample_and_match(source=node_1_1, reference=node_0_0, level="level_1")
        node_0_2 = self.dense_0_2(torch.cat([node_0_0, node_0_1, up_1_1], dim=1))
        up_2_1   = self._upsample_and_match(source=node_2_1, reference=node_1_0, level="level_2")
        node_1_2 = self.dense_1_2(torch.cat([node_1_0, node_1_1, up_2_1], dim=1))
        up_3_1   = self._upsample_and_match(source=node_3_1, reference=node_2_0, level="level_3")
        node_2_2 = self.dense_2_2(torch.cat([node_2_0, node_2_1, up_3_1], dim=1))

        up_1_2   = self._upsample_and_match(source=node_1_2, reference=node_0_0, level="level_1")
        node_0_3 = self.dense_0_3(torch.cat([node_0_0, node_0_1, node_0_2, up_1_2], dim=1))
        up_2_2   = self._upsample_and_match(source=node_2_2, reference=node_1_0, level="level_2")
        node_1_3 = self.dense_1_3(torch.cat([node_1_0, node_1_1, node_1_2, up_2_2], dim=1))

        up_1_3   = self._upsample_and_match(source=node_1_3, reference=node_0_0, level="level_1")
        node_0_4 = self.dense_0_4(torch.cat([node_0_0, node_0_1, node_0_2, node_0_3, up_1_3], dim=1))

        return node_0_4

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
