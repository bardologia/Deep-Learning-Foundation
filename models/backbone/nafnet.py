from __future__ import annotations

import torch
import torch.nn as nn

from configuration.architectures import NAFNetConfig
from ..blocks                    import ChannelLayerNorm, OutputHeadsMixin, initialize_weights, match_spatial_size


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first, second = x.chunk(2, dim=1)
        return first * second


class SimplifiedChannelAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.pool       = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.projection(self.pool(x))


class NAFBlock(nn.Module):
    def __init__(self, channels: int, dw_expand: int, ffn_expand: int, dropout: float):
        super().__init__()
        dw_channels  = channels * dw_expand
        ffn_channels = channels * ffn_expand

        self.norm_1    = ChannelLayerNorm(channels)
        self.expand    = nn.Conv2d(channels, dw_channels, kernel_size=1)
        self.depthwise = nn.Conv2d(dw_channels, dw_channels, kernel_size=3, padding=1, groups=dw_channels)
        self.gate_1    = SimpleGate()
        self.attention = SimplifiedChannelAttention(dw_channels // 2)
        self.project_1 = nn.Conv2d(dw_channels // 2, channels, kernel_size=1)
        self.dropout_1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm_2    = ChannelLayerNorm(channels)
        self.ffn       = nn.Conv2d(channels, ffn_channels, kernel_size=1)
        self.gate_2    = SimpleGate()
        self.project_2 = nn.Conv2d(ffn_channels // 2, channels, kernel_size=1)
        self.dropout_2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.beta  = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch = self.norm_1(x)
        branch = self.expand(branch)
        branch = self.depthwise(branch)
        branch = self.gate_1(branch)
        branch = self.attention(branch)
        branch = self.project_1(branch)
        branch = self.dropout_1(branch)
        x      = x + branch * self.beta

        branch = self.norm_2(x)
        branch = self.ffn(branch)
        branch = self.gate_2(branch)
        branch = self.project_2(branch)
        branch = self.dropout_2(branch)
        return x + branch * self.gamma


class NAFNet(nn.Module, OutputHeadsMixin):
    conv_head_kernel_size = 3

    def __init__(self, config: NAFNetConfig | None = None):
        super().__init__()
        self.config = config if config is not None else NAFNetConfig()

        if len(self.config.enc_blocks) != len(self.config.dec_blocks):
            raise ValueError(f"enc_blocks ({self.config.enc_blocks}) and dec_blocks ({self.config.dec_blocks}) must have the same number of stages")
        if len(self.config.enc_blocks) == 0:
            raise ValueError("enc_blocks must contain at least one stage")
        if (self.config.width * self.config.dw_expand) % 2 != 0 or (self.config.width * self.config.ffn_expand) % 2 != 0:
            raise ValueError(f"width ({self.config.width}) times dw_expand ({self.config.dw_expand}) and ffn_expand ({self.config.ffn_expand}) must be even; SimpleGate splits channels in half")

        width = self.config.width
        self.intro = nn.Conv2d(self.config.in_channels, width, kernel_size=3, padding=1)

        self.encoder_stages    = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        channels = width
        for n_blocks in self.config.enc_blocks:
            self.encoder_stages.append(self._stage(channels, n_blocks))
            self.downsample_layers.append(nn.Conv2d(channels, channels * 2, kernel_size=2, stride=2))
            channels *= 2

        self.middle_stage = self._stage(channels, self.config.middle_blocks)

        self.upsample_layers = nn.ModuleList()
        self.decoder_stages  = nn.ModuleList()
        for n_blocks in self.config.dec_blocks:
            self.upsample_layers.append(nn.Sequential(nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False), nn.PixelShuffle(2)))
            channels //= 2
            self.decoder_stages.append(self._stage(channels, n_blocks))

        self.embedding_channels = channels
        self._build_output_head()

        initialize_weights(module=self, mode=self.config.init_mode)

    def _head_activation(self) -> str:
        return "gelu"

    def _stage(self, channels: int, n_blocks: int) -> nn.Sequential:
        return nn.Sequential(*[NAFBlock(channels, self.config.dw_expand, self.config.ffn_expand, self.config.dropout) for _ in range(n_blocks)])

    def encode_decode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.intro(x)

        skip_connections = []
        for stage, downsample in zip(self.encoder_stages, self.downsample_layers):
            x = stage(x)
            skip_connections.append(x)
            x = downsample(x)

        x = self.middle_stage(x)

        for upsample, stage, skip in zip(self.upsample_layers, self.decoder_stages, skip_connections[::-1]):
            x = upsample(x)
            x = match_spatial_size(source=x, reference=skip)
            x = x + skip
            x = stage(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._head_forward(self.encode_decode(x))
