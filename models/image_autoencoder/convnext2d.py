from __future__ import annotations

import torch
import torch.nn as nn

from configuration.architectures   import ConvNeXt2dImageAutoencoderConfig
from models.image_autoencoder.base import ImageAutoencoderBase
from models.blocks                 import ChannelLayerNorm, ConvNeXtBlock, build_upsample, downsample_stages


FFN_RATIO        = 4.0
LAYER_SCALE_INIT = 1e-6


class ConvNeXt2dImageEncoder(nn.Module):
    def __init__(self, config: ConvNeXt2dImageAutoencoderConfig) -> None:
        super().__init__()
        n_stages = downsample_stages(config.downsample_factor)

        self.stem  = nn.Conv2d(config.in_channels, config.base_channels, kernel_size=3, padding=1)
        self.start = nn.Sequential(*[
            ConvNeXtBlock(config.base_channels, FFN_RATIO, config.stochastic_depth_rate, config.activation, LAYER_SCALE_INIT)
            for _ in range(max(1, config.depth))
        ])

        downs    = []
        channels = config.base_channels
        for _ in range(n_stages):
            nxt = channels * 2
            downs.append(ChannelLayerNorm(channels))
            downs.append(nn.Conv2d(channels, nxt, kernel_size=2, stride=2))
            for _ in range(max(1, config.depth)):
                downs.append(ConvNeXtBlock(nxt, FFN_RATIO, config.stochastic_depth_rate, config.activation, LAYER_SCALE_INIT))
            channels = nxt
        self.downsample = nn.Sequential(*downs)

        self.to_embedding        = nn.Conv2d(channels, config.embedding_dim, kernel_size=1)
        self.bottleneck_channels = channels

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.stem(image)
        x = self.start(x)
        x = self.downsample(x)
        return self.to_embedding(x)


class ConvNeXt2dImageDecoder(nn.Module):
    def __init__(self, config: ConvNeXt2dImageAutoencoderConfig, bottleneck_channels: int) -> None:
        super().__init__()
        n_stages = downsample_stages(config.downsample_factor)

        self.from_embedding = nn.Conv2d(config.embedding_dim, bottleneck_channels, kernel_size=1)

        ups      = []
        channels = bottleneck_channels
        for _ in range(n_stages):
            nxt = channels // 2
            ups.append(build_upsample(config.upsample_mode, channels, nxt, scale_factor=2))
            for _ in range(max(1, config.depth)):
                ups.append(ConvNeXtBlock(nxt, FFN_RATIO, config.stochastic_depth_rate, config.activation, LAYER_SCALE_INIT))
            channels = nxt
        self.upsample = nn.Sequential(*ups)

        refine = []
        for _ in range(max(0, config.depth - 1)):
            refine.append(ConvNeXtBlock(channels, FFN_RATIO, config.stochastic_depth_rate, config.activation, LAYER_SCALE_INIT))
        self.refine = nn.Sequential(*refine)

        self.head = nn.Conv2d(channels, config.in_channels, kernel_size=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.from_embedding(z)
        x = self.upsample(x)
        x = self.refine(x)
        return self.head(x)


class ConvNeXt2dImageAutoencoder(ImageAutoencoderBase):
    def __init__(self, config: ConvNeXt2dImageAutoencoderConfig | None = None) -> None:
        config = config if config is not None else ConvNeXt2dImageAutoencoderConfig()
        if config.normalization != "layernorm":
            raise ValueError(f"convnext2d_ae hardcodes LayerNorm and does not honor normalization='{config.normalization}'; leave the field at 'layernorm'.")
        if config.dropout != 0.0:
            raise ValueError(f"convnext2d_ae has no dropout layers and regularizes through stochastic_depth_rate; leave dropout at 0.0 (got {config.dropout}).")

        encoder = ConvNeXt2dImageEncoder(config)
        decoder = ConvNeXt2dImageDecoder(config, encoder.bottleneck_channels)
        super().__init__(config, encoder, decoder)
