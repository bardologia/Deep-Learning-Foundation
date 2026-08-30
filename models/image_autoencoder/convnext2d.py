"""ConvNeXt image autoencoder with a shared stochastic-depth schedule.

The drop-path ramp is computed once over the encoder and decoder blocks together, so
the two halves see a single monotone schedule rather than two independent ones.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from configuration.architectures   import ConvNeXt2dImageAutoencoderConfig
from models.image_autoencoder.base import ImageAutoencoderBase
from models.blocks                 import ChannelLayerNorm, ConvNeXtBlock, DropPathSchedule, LayerFactory


FFN_RATIO        = 4.0
LAYER_SCALE_INIT = 1e-6


class ConvNeXt2dImageEncoder(nn.Module):
    """ConvNeXt encoder alternating channel-norm downsampling and ConvNeXt blocks.

    Attributes:
        stem                : 3x3 convolution lifting the input to ``base_channels``.
        start               : Full-resolution ConvNeXt blocks before any downsampling.
        downsample          : The halving stages, each followed by its own ConvNeXt blocks.
        to_embedding        : 1x1 projection onto ``embedding_dim``.
        bottleneck_channels : Channel width reached just before the projection.
    """

    def __init__(self, config: ConvNeXt2dImageAutoencoderConfig, drop_path_rates: list[float]) -> None:
        """Builds the stem, the downsampling ladder and the embedding projection.

        Args:
            config          : Autoencoder configuration giving widths, depth and downsample factor.
            drop_path_rates : One stochastic depth probability per encoder block, in the order the blocks are created.
        """
        super().__init__()
        n_stages = LayerFactory.downsample_stages(config.downsample_factor)
        rates    = iter(drop_path_rates)

        self.stem  = nn.Conv2d(config.in_channels, config.base_channels, kernel_size=3, padding=1)
        self.start = nn.Sequential(*[
            ConvNeXtBlock(config.base_channels, FFN_RATIO, next(rates), config.activation, LAYER_SCALE_INIT)
            for _ in range(max(1, config.depth))
        ])

        downs    = []
        channels = config.base_channels
        for _ in range(n_stages):
            nxt = channels * 2
            downs.append(ChannelLayerNorm(channels))
            downs.append(nn.Conv2d(channels, nxt, kernel_size=2, stride=2))
            for _ in range(max(1, config.depth)):
                downs.append(ConvNeXtBlock(nxt, FFN_RATIO, next(rates), config.activation, LAYER_SCALE_INIT))
            channels = nxt
        self.downsample = nn.Sequential(*downs)

        self.to_embedding        = nn.Conv2d(channels, config.embedding_dim, kernel_size=1)
        self.bottleneck_channels = channels

    @staticmethod
    def block_count(config: ConvNeXt2dImageAutoencoderConfig) -> int:
        """Returns how many ConvNeXt blocks the encoder will hold for a configuration."""
        return max(1, config.depth) * (LayerFactory.downsample_stages(config.downsample_factor) + 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Encodes an image stack into the latent feature map.

        Args:
            image : Input stack of shape (batch, in_channels, height, width).

        Returns:
            Latent of shape
            (batch, embedding_dim, height / downsample_factor, width / downsample_factor).
        """
        x = self.stem(image)
        x = self.start(x)
        x = self.downsample(x)
        return self.to_embedding(x)


class ConvNeXt2dImageDecoder(nn.Module):
    """ConvNeXt decoder mirroring the encoder ladder back to input resolution.

    Attributes:
        from_embedding : 1x1 projection from ``embedding_dim`` to the bottleneck width.
        upsample       : The upsampling stages, each followed by its own ConvNeXt blocks.
        refine         : Full-resolution ConvNeXt blocks applied after upsampling.
        head           : 1x1 projection onto ``in_channels``.
    """

    def __init__(self, config: ConvNeXt2dImageAutoencoderConfig, bottleneck_channels: int, drop_path_rates: list[float]) -> None:
        """Builds the projection, the upsampling ladder, the refinement blocks and the head.

        Args:
            config              : Autoencoder configuration giving widths, depth and upsample mode.
            bottleneck_channels : Channel width the encoder reached before its projection.
            drop_path_rates     : One stochastic depth probability per decoder block, in the order the blocks are created.
        """
        super().__init__()
        n_stages = LayerFactory.downsample_stages(config.downsample_factor)
        rates    = iter(drop_path_rates)

        self.from_embedding = nn.Conv2d(config.embedding_dim, bottleneck_channels, kernel_size=1)

        ups      = []
        channels = bottleneck_channels
        for _ in range(n_stages):
            nxt = channels // 2
            ups.append(LayerFactory.build_upsample(config.upsample_mode, channels, nxt, scale_factor=2))
            for _ in range(max(1, config.depth)):
                ups.append(ConvNeXtBlock(nxt, FFN_RATIO, next(rates), config.activation, LAYER_SCALE_INIT))
            channels = nxt
        self.upsample = nn.Sequential(*ups)

        refine = []
        for _ in range(max(0, config.depth - 1)):
            refine.append(ConvNeXtBlock(channels, FFN_RATIO, next(rates), config.activation, LAYER_SCALE_INIT))
        self.refine = nn.Sequential(*refine)

        self.head = nn.Conv2d(channels, config.in_channels, kernel_size=1)

    @staticmethod
    def block_count(config: ConvNeXt2dImageAutoencoderConfig) -> int:
        """Returns how many ConvNeXt blocks the decoder will hold for a configuration."""
        return max(1, config.depth) * LayerFactory.downsample_stages(config.downsample_factor) + max(0, config.depth - 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstructs an image stack from the latent feature map.

        Args:
            z : Latent of shape (batch, embedding_dim, latent_height, latent_width).

        Returns:
            Reconstruction of shape
            (batch, in_channels, latent_height * downsample_factor, latent_width * downsample_factor).
        """
        x = self.from_embedding(z)
        x = self.upsample(x)
        x = self.refine(x)
        return self.head(x)


class ConvNeXt2dImageAutoencoder(ImageAutoencoderBase):
    """Image autoencoder pairing the ConvNeXt encoder and decoder.

    Rejects configurations that set fields this architecture ignores, so a sweep can
    never silently run a variant that has no effect.
    """

    def __init__(self, config: ConvNeXt2dImageAutoencoderConfig | None = None) -> None:
        """Splits the joint drop-path schedule and builds both halves.

        Args:
            config : Autoencoder configuration; defaults to ``ConvNeXt2dImageAutoencoderConfig()``.

        Raises:
            ValueError : If ``normalization`` is not ``layernorm`` or ``dropout`` is not 0, since the blocks hardcode LayerNorm and regularize through ``stochastic_depth_rate`` alone.
        """
        config = config if config is not None else ConvNeXt2dImageAutoencoderConfig()
        if config.normalization != "layernorm":
            raise ValueError(f"convnext2d_ae hardcodes LayerNorm and does not honor normalization='{config.normalization}'; leave the field at 'layernorm'.")
        if config.dropout != 0.0:
            raise ValueError(f"convnext2d_ae has no dropout layers and regularizes through stochastic_depth_rate; leave dropout at 0.0 (got {config.dropout}).")

        encoder_blocks = ConvNeXt2dImageEncoder.block_count(config)
        decoder_blocks = ConvNeXt2dImageDecoder.block_count(config)
        rates          = DropPathSchedule(encoder_blocks + decoder_blocks, config.stochastic_depth_rate, 1).rates

        encoder = ConvNeXt2dImageEncoder(config, rates[:encoder_blocks])
        decoder = ConvNeXt2dImageDecoder(config, encoder.bottleneck_channels, rates[encoder_blocks:])
        super().__init__(config, encoder, decoder)
