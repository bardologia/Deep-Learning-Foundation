"""Plain U-Net backbone assembled from the shared Encoder and Decoder blocks."""

from __future__ import annotations

import torch
import torch.nn as nn

from configuration.architectures import UNetConfig
from ..blocks                    import ConvBlock, Decoder, Encoder, LayerFactory, OutputHeadsMixin


class UNetBackbone(nn.Module, OutputHeadsMixin):
    """Encoder-bottleneck-decoder trunk without an output head.

    Attributes:
        config             : Architecture configuration driving widths, depth and head choice.
        embedding_channels : Channel width of the decoded embedding fed to the output head.
    """

    def __init__(self, config):
        """Builds the pooled encoder, the bottleneck block and the concatenating decoder.

        Args:
            config : Architecture configuration exposing ``features``, ``in_channels``, ``bottleneck_factor``, ``upsample_mode`` and the block hyperparameters.

        Raises:
            ValueError : If ``features`` is empty.
        """
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
        """Returns the embedding at input resolution.

        Args:
            x : Input feature stack of shape (batch, in_channels, height, width).

        Returns:
            Embedding of shape (batch, embedding_channels, height, width).
        """
        x, skip_connections = self.encoder(x)
        x = self.bottleneck(x)
        return self.decoder(x, skip_connections[::-1])


class UNet(UNetBackbone):
    """U-Net backbone predicting per-pixel Gaussian mixture parameters."""

    def __init__(self, config: UNetConfig | None = None):
        """Builds the U-Net trunk and its output head.

        Args:
            config : Architecture configuration; defaults to ``UNetConfig()``.
        """
        super().__init__(config if config is not None else UNetConfig())
        self._build_output_head()

        LayerFactory.initialize_weights(module=self, mode=self.config.init_mode)

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
