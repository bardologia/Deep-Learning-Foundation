"""Attention U-Net backbone and its additive skip-connection attention gate.

Implements the Oktay et al. attention gate on top of the standard encoder,
bottleneck, decoder structure shared by the backbone family.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from configuration.architectures import AttentionUNetConfig
from ..blocks                    import ConvBlock, FeatureGeometry, LayerFactory, OutputHeadsMixin

class AttentionGate(nn.Module):
    """Additive attention gate that reweights a U-Net skip connection.

    Projects the coarse decoder gate signal and the finer skip tensor into a
    shared intermediate space, forms a single-channel sigmoid attention map from
    their sum, and rescales the skip connection with it before a final
    projection.
    """

    def __init__(
        self,
        gate_channels:         int,
        skip_channels:         int,
        intermediate_channels: int,
        normalization:         str  = "batch",
        bias:                  bool = False,
    ):
        """Builds the gate, skip, scoring and output projections.

        Args:
            gate_channels         : Channel count of the coarse decoder gate signal.
            skip_channels         : Channel count of the encoder skip connection, also the channel count of the returned tensor.
            intermediate_channels : Width of the shared space in which the gate and skip projections are summed.
            normalization         : Normalisation kind passed to ``LayerFactory``.
            bias                  : Whether the projection convolutions carry a bias term.
        """
        super().__init__()
        self.gate_projection = nn.Sequential(
            nn.Conv2d(
                in_channels  = gate_channels,
                out_channels = intermediate_channels,
                kernel_size  = 1,
                bias         = bias,
            ),
            LayerFactory.build_norm2d(normalization, intermediate_channels),
        )
        self.skip_projection = nn.Sequential(
            nn.Conv2d(
                in_channels  = skip_channels,
                out_channels = intermediate_channels,
                kernel_size  = 2,
                stride       = 2,
                bias         = bias,
            ),
            LayerFactory.build_norm2d(normalization, intermediate_channels),
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
            LayerFactory.build_norm2d(normalization, skip_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate_signal: torch.Tensor, skip_connection: torch.Tensor) -> torch.Tensor:
        """Returns the attention-weighted skip connection.

        Args:
            gate_signal     : Decoder feature map of shape (batch, gate_channels, height/2, width/2).
            skip_connection : Encoder feature map of shape (batch, skip_channels, height, width).

        Returns:
            Gated skip features of shape (batch, skip_channels, height, width).
        """
        projected_gate = self.gate_projection(gate_signal)
        projected_skip = self.skip_projection(skip_connection)

        projected_gate = FeatureGeometry.match_spatial_size(source=projected_gate, reference=projected_skip)

        attention_weights = self.attention_score(self.relu(projected_gate + projected_skip))
        attention_weights = FeatureGeometry.match_spatial_size(source=attention_weights, reference=skip_connection)

        return self.output_transform(skip_connection * attention_weights)

class AttentionUNet(nn.Module, OutputHeadsMixin):
    """U-Net whose skip connections pass through additive attention gates.

    Encoder blocks halve the resolution at every stage, the decoder upsamples
    back and concatenates each skip after it has been reweighted by an
    ``AttentionGate``. The final decoder feature map feeds the output head
    selected in the configuration.

    Attributes:
        config             : Architecture configuration in use.
        embedding_channels : Channel count of the decoder embedding handed to the output head.
    """

    def __init__(self, config: AttentionUNetConfig | None = None):
        """Builds the encoder, bottleneck, attention-gated decoder and head.

        Args:
            config : Architecture configuration; a default ``AttentionUNetConfig`` is used when omitted.

        Raises:
            ValueError : If ``features`` is empty or ``attention_intermediate_ratio`` is not positive.
        """
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
                LayerFactory.build_upsample(
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

        LayerFactory.initialize_weights(module=self, mode=config.init_mode)

    def encode_decode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the decoder embedding for an input patch.

        Args:
            x : Input stack of shape (batch, in_channels, height, width).

        Returns:
            Feature map of shape (batch, features[0], height, width), resized
            back to the spatial size of ``x``.
        """
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
            x    = FeatureGeometry.match_spatial_size(source=x, reference=skip)
            x    = torch.cat([skip, x], dim=1)
            x    = decoder_block(x)

        return FeatureGeometry.match_spatial_size(source=x, reference=original_input)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the head output for an input patch.

        Args:
            x : Input stack of shape (batch, in_channels, height, width).

        Returns:
            Head output of shape (batch, out_channels, height, width); for the
            Gaussian heads the channels are the per-slot parameters interleaved
            as amplitude, mu and sigma.
        """
        return self._head_forward(self.encode_decode(x))
