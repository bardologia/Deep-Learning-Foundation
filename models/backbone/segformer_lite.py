from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

from configuration.architectures import SegFormerLiteConfig
from ..blocks                    import DropPath, OutputHeadsMixin, build_activation, initialize_weights


class OverlapPatchEmbedding(nn.Module):
    def __init__(self, input_channels: int, embedding_dim: int, kernel_size: int, stride: int):
        super().__init__()
        self.projection = nn.Conv2d(input_channels, embedding_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
        self.norm       = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x          = self.projection(x)
        _, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, h, w


class EfficientSelfAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, sr_ratio: int, attention_dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.sr_ratio  = sr_ratio

        if sr_ratio > 1:
            self.spatial_reduction = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_norm           = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        if self.sr_ratio > 1:
            B, N, C = x.shape
            kv = x.transpose(1, 2).reshape(B, C, h, w)
            kv = self.spatial_reduction(kv)
            kv = kv.flatten(2).transpose(1, 2)
            kv = self.sr_norm(kv)
        else:
            kv = x

        out, _ = self.attention(x, kv, kv, need_weights=False)
        return out


class MixFFN(nn.Module):
    def __init__(self, embedding_dim: int, mlp_ratio: float, dropout: float, ffn_activation: str):
        super().__init__()
        hidden_dim = int(embedding_dim * mlp_ratio)

        self.fc1        = nn.Conv2d(embedding_dim, hidden_dim, kernel_size=1)
        self.dwconv     = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.activation = build_activation(ffn_activation)
        self.fc2        = nn.Conv2d(hidden_dim, embedding_dim, kernel_size=1)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        B, N, C = x.shape

        x = x.transpose(1, 2).reshape(B, C, h, w)
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.flatten(2).transpose(1, 2)

        return self.dropout(x)


class SegFormerBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, sr_ratio: int, mlp_ratio: float, dropout: float, attention_dropout: float, drop_path: float, ffn_activation: str):
        super().__init__()
        self.norm1     = nn.LayerNorm(embedding_dim)
        self.attention = EfficientSelfAttention(embedding_dim, num_heads, sr_ratio, attention_dropout)
        self.norm2     = nn.LayerNorm(embedding_dim)
        self.ffn       = MixFFN(embedding_dim, mlp_ratio, dropout, ffn_activation)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x = x + self.drop_path(self.attention(self.norm1(x), h, w))
        x = x + self.drop_path(self.ffn(self.norm2(x), h, w))
        return x


class SegFormerStage(nn.Module):
    def __init__(self, blocks: list[SegFormerBlock], embedding_dim: int):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.norm   = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, h, w)

        x = self.norm(x)
        B, N, C = x.shape
        return x.transpose(1, 2).reshape(B, C, h, w)


class SegFormerLite(nn.Module, OutputHeadsMixin):
    def __init__(self, config: SegFormerLiteConfig | None = None):
        super().__init__()
        if config is None:
            config = SegFormerLiteConfig()
        self.config = config

        dims     = config.embedding_dims
        n_stages = len(dims)
        if not (len(config.depths) == len(config.num_heads) == len(config.sr_ratios) == n_stages):
            raise ValueError("embedding_dims, depths, num_heads, and sr_ratios must have the same length")

        kernel_sizes = [7] + [3] * (n_stages - 1)
        strides      = [4] + [2] * (n_stages - 1)

        total_blocks    = sum(config.depths)
        drop_path_rates = [config.stochastic_depth_rate * index / max(total_blocks - 1, 1) for index in range(total_blocks)]

        self.patch_embeddings = nn.ModuleList()
        self.encoder_stages   = nn.ModuleList()
        channels    = config.in_channels
        block_index = 0
        for stage_index in range(n_stages):
            self.patch_embeddings.append(OverlapPatchEmbedding(channels, dims[stage_index], kernel_sizes[stage_index], strides[stage_index]))

            blocks = []
            for _ in range(config.depths[stage_index]):
                blocks.append(SegFormerBlock(
                    embedding_dim     = dims[stage_index],
                    num_heads         = config.num_heads[stage_index],
                    sr_ratio          = config.sr_ratios[stage_index],
                    mlp_ratio         = config.mlp_ratio,
                    dropout           = config.dropout,
                    attention_dropout = config.attention_dropout,
                    drop_path         = drop_path_rates[block_index],
                    ffn_activation    = config.ffn_activation,
                ))
                block_index += 1

            self.encoder_stages.append(SegFormerStage(blocks, dims[stage_index]))
            channels = dims[stage_index]

        self.decode_projections = nn.ModuleList([nn.Conv2d(dim, config.decoder_channels, kernel_size=1) for dim in dims])

        self.fuse = nn.Sequential(
            nn.Conv2d(n_stages * config.decoder_channels, config.decoder_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(config.decoder_channels),
            build_activation(config.ffn_activation),
            nn.Dropout2d(config.dropout),
        )

        self.embedding_channels = config.decoder_channels
        self._build_output_head()

        initialize_weights(module=self, mode=config.init_mode)

    def _head_activation(self) -> str:
        return self.config.ffn_activation

    def _fused_embedding(self, x: torch.Tensor) -> torch.Tensor:
        stage_outputs = []
        for patch_embedding, stage in zip(self.patch_embeddings, self.encoder_stages):
            x, h, w = patch_embedding(x)
            x       = stage(x, h, w)
            stage_outputs.append(x)

        target_size = stage_outputs[0].shape[2:]

        fused = []
        for projection, features in zip(self.decode_projections, stage_outputs):
            features = projection(features)
            if features.shape[2:] != target_size:
                features = functional.interpolate(features, size=target_size, mode="bilinear", align_corners=False)
            fused.append(features)

        return self.fuse(torch.cat(fused, dim=1))

    def encode_decode(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]
        embedding  = self._fused_embedding(x)
        return functional.interpolate(embedding, size=input_size, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]
        out        = self._head_forward(self._fused_embedding(x))
        return functional.interpolate(out, size=input_size, mode="bilinear", align_corners=False)
