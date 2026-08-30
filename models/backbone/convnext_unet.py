"""ConvNeXt U-Net backbone and the ConvNeXt stage it is assembled from.

Replaces the plain convolution blocks of the U-Net skeleton with ConvNeXt
blocks, sharing one stochastic-depth schedule across encoder, bottleneck and
decoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from configuration.architectures import ConvNeXtUNetConfig
from ..blocks                    import ChannelLayerNorm, ConvNeXtBlock, DropPathSchedule, FeatureGeometry, LayerFactory, OutputHeadsMixin


class ConvNeXtStage(nn.Module):
    """Sequence of ConvNeXt blocks preceded by an optional channel projection.

    The 1x1 projection is an identity when the input and output channel counts
    already agree, so the stage can be reused for both encoder widening and
    decoder narrowing after concatenation.
    """

    def __init__(self, input_channels: int, output_channels: int, n_blocks: int, ffn_ratio: float, drop_path_rates: list[float], ffn_activation: str, layer_scale_init: float):
        """Builds the channel projection and the stack of ConvNeXt blocks.

        Args:
            input_channels   : Channel count of the incoming feature map.
            output_channels  : Channel count of every block in the stage.
            n_blocks         : Number of ConvNeXt blocks stacked in the stage.
            ffn_ratio        : Expansion ratio of the pointwise feed-forward branch.
            drop_path_rates  : Stochastic-depth rate for each block, one entry per block in the stage.
            ffn_activation   : Activation name used inside the feed-forward branch.
            layer_scale_init : Initial value of the per-channel LayerScale gain.
        """
        super().__init__()
        self.projection = nn.Conv2d(input_channels, output_channels, kernel_size=1) if input_channels != output_channels else nn.Identity()
        self.blocks     = nn.Sequential(*[
            ConvNeXtBlock(output_channels, ffn_ratio, drop_path_rates[index], ffn_activation, layer_scale_init)
            for index in range(n_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the stage output for a feature map.

        Args:
            x : Feature map of shape (batch, input_channels, height, width).

        Returns:
            Feature map of shape (batch, output_channels, height, width).
        """
        return self.blocks(self.projection(x))


class ConvNeXtUNet(nn.Module, OutputHeadsMixin):
    """U-Net whose encoder, bottleneck and decoder stages are ConvNeXt blocks.

    A 3x3 stem lifts the input to the first feature width, strided 2x2
    convolutions downsample between stages, transposed convolutions upsample in
    the decoder, and stochastic depth is distributed across all stages by a
    single ``DropPathSchedule``.

    Attributes:
        config             : Architecture configuration in use.
        embedding_channels : Channel count of the decoder embedding handed to the output head.
    """

    def __init__(self, config: ConvNeXtUNetConfig | None = None):
        """Builds the stem, ConvNeXt encoder, bottleneck, decoder and head.

        Args:
            config : Architecture configuration; a default ``ConvNeXtUNetConfig`` is used when omitted.

        Raises:
            ValueError : If ``features`` is empty.
        """
        super().__init__()
        if config is None:
            config = ConvNeXtUNetConfig()
        self.config = config

        if len(config.features) == 0:
            raise ValueError("features must contain at least one channel size")

        feature_sizes       = config.features
        n_stages            = len(feature_sizes)
        bottleneck_channels = feature_sizes[-1] * config.bottleneck_factor

        total_blocks = config.blocks_per_stage * (2 * n_stages + 1)
        schedule     = DropPathSchedule(total_blocks, config.stochastic_depth_rate, config.blocks_per_stage)

        self.stem = nn.Sequential(
            nn.Conv2d(config.in_channels, feature_sizes[0], kernel_size=3, padding=1, bias=config.conv_bias),
            ChannelLayerNorm(feature_sizes[0]),
        )

        self.encoder_stages    = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        channels = feature_sizes[0]
        for stage_index, feature_size in enumerate(feature_sizes):
            self.encoder_stages.append(ConvNeXtStage(channels, feature_size, config.blocks_per_stage, config.ffn_ratio, schedule.stage(stage_index), config.ffn_activation, config.layer_scale_init))
            self.downsample_layers.append(nn.Sequential(
                ChannelLayerNorm(feature_size),
                nn.Conv2d(feature_size, feature_size, kernel_size=2, stride=2),
            ))
            channels = feature_size

        self.bottleneck = ConvNeXtStage(feature_sizes[-1], bottleneck_channels, config.blocks_per_stage, config.ffn_ratio, schedule.stage(n_stages), config.ffn_activation, config.layer_scale_init)

        reversed_features    = [bottleneck_channels] + feature_sizes[::-1]
        self.upsample_layers = nn.ModuleList()
        self.decoder_stages  = nn.ModuleList()
        for index in range(len(reversed_features) - 1):
            self.upsample_layers.append(nn.ConvTranspose2d(reversed_features[index], reversed_features[index + 1], kernel_size=2, stride=2))
            self.decoder_stages.append(ConvNeXtStage(reversed_features[index + 1] * 2, reversed_features[index + 1], config.blocks_per_stage, config.ffn_ratio, schedule.stage(n_stages + 1 + index), config.ffn_activation, config.layer_scale_init))

        self.embedding_channels = feature_sizes[0]
        self._build_output_head()

        LayerFactory.initialize_weights(module=self, mode=config.init_mode)

    def _head_activation(self) -> str:
        """Returns the activation name the output head shares with the blocks."""
        return self.config.ffn_activation

    def encode_decode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the decoder embedding for an input patch.

        Args:
            x : Input stack of shape (batch, in_channels, height, width).

        Returns:
            Feature map of shape (batch, features[0], height, width).
        """
        x = self.stem(x)

        skip_connections: list[torch.Tensor] = []
        for stage, downsample in zip(self.encoder_stages, self.downsample_layers):
            x = stage(x)
            skip_connections.append(x)
            x = downsample(x)

        x = self.bottleneck(x)

        for upsample, stage, skip in zip(self.upsample_layers, self.decoder_stages, reversed(skip_connections)):
            x = upsample(x)
            x = FeatureGeometry.match_spatial_size(source=x, reference=skip)
            x = torch.cat([skip, x], dim=1)
            x = stage(x)

        return x

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
