"""U2-Net Lite backbone built from nested residual U-blocks.

Every encoder and decoder stage is itself a small U-Net (RSU); the bridge uses a
dilated variant that keeps the resolution fixed while widening the receptive field.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

from configuration.architectures import U2NetLiteConfig
from ..blocks                    import ConvNormAct, LayerFactory, OutputHeadsMixin


class RSU(nn.Module):
    """Residual U-block: a small pooled U-Net whose output is added to its input projection.

    Attributes:
        height : Depth of the nested U, counting the encoder convolutions.
    """

    def __init__(self, height: int, input_channels: int, mid_channels: int, output_channels: int, dropout: float, activation: str, normalization: str, bias: bool):
        """Builds the nested encoder, the dilated bottom convolution and the nested decoder.

        Args:
            height          : Depth of the nested U; must be at least 2.
            input_channels  : Channel width of the incoming feature map.
            mid_channels    : Channel width inside the nested U.
            output_channels : Channel width produced by the block.
            dropout         : Spatial dropout probability applied to the output; disabled at 0.
            activation      : Activation name resolved by ``LayerFactory.build_activation``.
            normalization   : 2D normalization name resolved by ``LayerFactory.build_norm2d``.
            bias            : Whether the convolutions carry a bias term.

        Raises:
            ValueError : If ``height`` is smaller than 2.
        """
        super().__init__()
        if height < 2:
            raise ValueError(f"RSU height must be at least 2; got {height}")
        self.height = height

        self.conv_in = ConvNormAct(input_channels, output_channels, dilation=1, activation=activation, normalization=normalization, bias=bias)

        self.encoder_convs = nn.ModuleList([ConvNormAct(output_channels, mid_channels, dilation=1, activation=activation, normalization=normalization, bias=bias)])
        for _ in range(height - 2):
            self.encoder_convs.append(ConvNormAct(mid_channels, mid_channels, dilation=1, activation=activation, normalization=normalization, bias=bias))

        self.bottom_conv = ConvNormAct(mid_channels, mid_channels, dilation=2, activation=activation, normalization=normalization, bias=bias)

        self.decoder_convs = nn.ModuleList()
        for index in range(height - 1):
            out_channels = output_channels if index == height - 2 else mid_channels
            self.decoder_convs.append(ConvNormAct(mid_channels * 2, out_channels, dilation=1, activation=activation, normalization=normalization, bias=bias))

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs the nested U and adds its output to the input projection.

        Args:
            x : Feature map of shape (batch, input_channels, height, width).

        Returns:
            Feature map of shape (batch, output_channels, height, width).
        """
        x_in = self.conv_in(x)

        encoder_outputs = []
        x = x_in
        for index, conv in enumerate(self.encoder_convs):
            x = conv(x)
            encoder_outputs.append(x)
            if index < self.height - 2:
                x = functional.max_pool2d(x, kernel_size=2, ceil_mode=True)

        x = self.bottom_conv(x)

        for index, conv in enumerate(self.decoder_convs):
            skip = encoder_outputs[-(index + 1)]
            if x.shape[2:] != skip.shape[2:]:
                x = functional.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = conv(torch.cat([x, skip], dim=1))

        return self.dropout(x + x_in)


class RSUDilated(nn.Module):
    """Residual U-block that widens the receptive field with dilation instead of pooling.

    Dilation rates run 1, 2, 4 down the encoder, 8 at the bottom and 4, 2, 1 back up,
    so the spatial resolution never changes.
    """

    def __init__(self, input_channels: int, mid_channels: int, output_channels: int, dropout: float, activation: str, normalization: str, bias: bool):
        """Builds the dilated encoder, bottom and decoder convolutions.

        Args:
            input_channels  : Channel width of the incoming feature map.
            mid_channels    : Channel width inside the block.
            output_channels : Channel width produced by the block.
            dropout         : Spatial dropout probability applied to the output; disabled at 0.
            activation      : Activation name resolved by ``LayerFactory.build_activation``.
            normalization   : 2D normalization name resolved by ``LayerFactory.build_norm2d``.
            bias            : Whether the convolutions carry a bias term.
        """
        super().__init__()
        self.conv_in = ConvNormAct(input_channels, output_channels, dilation=1, activation=activation, normalization=normalization, bias=bias)

        self.encoder_convs = nn.ModuleList([
            ConvNormAct(output_channels, mid_channels, dilation=1, activation=activation, normalization=normalization, bias=bias),
            ConvNormAct(mid_channels,    mid_channels, dilation=2, activation=activation, normalization=normalization, bias=bias),
            ConvNormAct(mid_channels,    mid_channels, dilation=4, activation=activation, normalization=normalization, bias=bias),
        ])

        self.bottom_conv = ConvNormAct(mid_channels, mid_channels, dilation=8, activation=activation, normalization=normalization, bias=bias)

        self.decoder_convs = nn.ModuleList([
            ConvNormAct(mid_channels * 2, mid_channels,    dilation=4, activation=activation, normalization=normalization, bias=bias),
            ConvNormAct(mid_channels * 2, mid_channels,    dilation=2, activation=activation, normalization=normalization, bias=bias),
            ConvNormAct(mid_channels * 2, output_channels, dilation=1, activation=activation, normalization=normalization, bias=bias),
        ])

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs the dilated U and adds its output to the input projection.

        Args:
            x : Feature map of shape (batch, input_channels, height, width).

        Returns:
            Feature map of shape (batch, output_channels, height, width).
        """
        x_in = self.conv_in(x)

        encoder_outputs = []
        x = x_in
        for conv in self.encoder_convs:
            x = conv(x)
            encoder_outputs.append(x)

        x = self.bottom_conv(x)

        for index, conv in enumerate(self.decoder_convs):
            skip = encoder_outputs[-(index + 1)]
            x    = conv(torch.cat([x, skip], dim=1))

        return self.dropout(x + x_in)


class U2NetLite(nn.Module, OutputHeadsMixin):
    """U2-Net backbone predicting per-pixel Gaussian mixture parameters.

    Attributes:
        config             : Architecture configuration driving widths, nesting depths and head choice.
        embedding_channels : Channel width of the decoded embedding fed to the output head.
    """

    def __init__(self, config: U2NetLiteConfig | None = None):
        """Builds the three RSU encoder stages, the dilated bridge, the decoder and the head.

        Args:
            config : Architecture configuration; defaults to ``U2NetLiteConfig()``.

        Raises:
            ValueError : If ``features`` does not hold exactly four channel sizes or ``rsu_heights`` does not hold exactly three nesting depths.
        """
        super().__init__()
        if config is None:
            config = U2NetLiteConfig()
        self.config = config

        if len(config.features) != 4:
            raise ValueError(f"features must contain exactly four channel sizes; got {len(config.features)}")
        if len(config.rsu_heights) != 3:
            raise ValueError(f"rsu_heights must contain exactly three nested-U depths, one per encoder and mirrored decoder stage; got {len(config.rsu_heights)}")

        f0, f1, f2, f3 = config.features
        heights        = config.rsu_heights

        self.encoder_stages = nn.ModuleList([
            RSU(heights[0], config.in_channels, self._mid_channels(f0), f0, config.dropout, config.activation, config.normalization, config.conv_bias),
            RSU(heights[1], f0,                 self._mid_channels(f1), f1, config.dropout, config.activation, config.normalization, config.conv_bias),
            RSU(heights[2], f1,                 self._mid_channels(f2), f2, config.dropout, config.activation, config.normalization, config.conv_bias),
        ])

        self.bridge = RSUDilated(f2, self._mid_channels(f3), f3, config.dropout, config.activation, config.normalization, config.conv_bias)

        self.decoder_stages = nn.ModuleList([
            RSU(heights[2], f3 + f2, self._mid_channels(f2), f2, config.dropout, config.activation, config.normalization, config.conv_bias),
            RSU(heights[1], f2 + f1, self._mid_channels(f1), f1, config.dropout, config.activation, config.normalization, config.conv_bias),
            RSU(heights[0], f1 + f0, self._mid_channels(f0), f0, config.dropout, config.activation, config.normalization, config.conv_bias),
        ])

        self.embedding_channels = f0
        self._build_output_head()

        LayerFactory.initialize_weights(module=self, mode=config.init_mode)

    @staticmethod
    def _mid_channels(channels: int) -> int:
        """Returns the internal RSU width for a stage of the given output width."""
        return max(channels // 4, 8)

    def encode_decode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the embedding at input resolution.

        Args:
            x : Input feature stack of shape (batch, in_channels, height, width).

        Returns:
            Embedding of shape (batch, embedding_channels, height, width).
        """
        encoder_outputs = []
        for stage in self.encoder_stages:
            x = stage(x)
            encoder_outputs.append(x)
            x = functional.max_pool2d(x, kernel_size=2, ceil_mode=True)

        x = self.bridge(x)

        for stage, skip in zip(self.decoder_stages, reversed(encoder_outputs)):
            x = functional.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = stage(torch.cat([x, skip], dim=1))

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
