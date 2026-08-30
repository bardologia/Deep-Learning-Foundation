"""SegFormer-style hierarchical transformer backbone with an all-MLP decoder.

Four token stages of shrinking resolution are produced by overlapping patch
embeddings; each stage is projected to a common width, resampled to the finest
stage grid, concatenated and fused before being upsampled back to input resolution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

from configuration.architectures import SegFormerLiteConfig
from ..blocks                    import DropPath, LayerFactory, OutputHeadsMixin


class OverlapPatchEmbedding(nn.Module):
    """Strided convolution that tokenizes a feature map with overlapping patches.

    Attributes:
        projection : Strided convolution mapping pixels to tokens.
        norm       : Layer normalization applied to the flattened tokens.
    """

    def __init__(self, input_channels: int, embedding_dim: int, kernel_size: int, stride: int):
        """Builds the patch projection and token normalization.

        Args:
            input_channels : Channel width of the incoming feature map.
            embedding_dim  : Token width produced by the projection.
            kernel_size    : Patch kernel size; the padding is half of it, so patches overlap.
            stride         : Stride of the projection, equal to the resolution reduction factor.
        """
        super().__init__()
        self.projection = nn.Conv2d(input_channels, embedding_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
        self.norm       = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """Tokenizes a feature map and reports the token grid shape.

        Args:
            x : Feature map of shape (batch, input_channels, height, width).

        Returns:
            Tuple of the token sequence of shape (batch, h * w, embedding_dim) and
            the token grid height and width.
        """
        x          = self.projection(x)
        _, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, h, w


class EfficientSelfAttention(nn.Module):
    """Self-attention whose keys and values are spatially reduced before attending.

    Attributes:
        attention : The underlying multi-head attention.
        sr_ratio  : Spatial reduction factor; 1 disables the reduction.
    """

    def __init__(self, embedding_dim: int, num_heads: int, sr_ratio: int, attention_dropout: float):
        """Builds the attention and, when reduction is active, the key/value downsampler.

        Args:
            embedding_dim     : Token width.
            num_heads         : Number of attention heads.
            sr_ratio          : Spatial reduction factor applied to keys and values.
            attention_dropout : Dropout probability on the attention weights.
        """
        super().__init__()
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.sr_ratio  = sr_ratio

        if sr_ratio > 1:
            self.spatial_reduction = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_norm           = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Attends over spatially reduced keys and values.

        Args:
            x : Token sequence of shape (batch, h * w, embedding_dim).
            h : Token grid height.
            w : Token grid width.

        Returns:
            Token sequence of the same shape as ``x``.
        """
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
    """Feed-forward block that injects positional information via a depthwise convolution.

    Attributes:
        fc1    : 1x1 expansion convolution.
        dwconv : Depthwise 3x3 convolution acting on the token grid.
        fc2    : 1x1 projection back to the token width.
    """

    def __init__(self, embedding_dim: int, mlp_ratio: float, dropout: float, ffn_activation: str):
        """Builds the expansion, depthwise mixing and projection layers.

        Args:
            embedding_dim  : Token width.
            mlp_ratio      : Hidden width as a multiple of ``embedding_dim``.
            dropout        : Dropout probability applied inside and at the output.
            ffn_activation : Activation name resolved by ``LayerFactory.build_activation``.
        """
        super().__init__()
        hidden_dim = int(embedding_dim * mlp_ratio)

        self.fc1        = nn.Conv2d(embedding_dim, hidden_dim, kernel_size=1)
        self.dwconv     = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.activation = LayerFactory.build_activation(ffn_activation)
        self.fc2        = nn.Conv2d(hidden_dim, embedding_dim, kernel_size=1)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Applies the convolutional feed-forward transform on the token grid.

        Args:
            x : Token sequence of shape (batch, h * w, embedding_dim).
            h : Token grid height.
            w : Token grid width.

        Returns:
            Token sequence of the same shape as ``x``.
        """
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
    """Pre-norm transformer block pairing efficient attention with a Mix-FFN."""

    def __init__(self, embedding_dim: int, num_heads: int, sr_ratio: int, mlp_ratio: float, dropout: float, attention_dropout: float, drop_path: float, ffn_activation: str):
        """Builds the attention and feed-forward sublayers with their normalizations.

        Args:
            embedding_dim     : Token width.
            num_heads         : Number of attention heads.
            sr_ratio          : Spatial reduction factor for keys and values.
            mlp_ratio         : Hidden width of the Mix-FFN as a multiple of ``embedding_dim``.
            dropout           : Dropout probability in the feed-forward path.
            attention_dropout : Dropout probability on the attention weights.
            drop_path         : Stochastic depth probability for both residual branches.
            ffn_activation    : Activation name used in the Mix-FFN.
        """
        super().__init__()
        self.norm1     = nn.LayerNorm(embedding_dim)
        self.attention = EfficientSelfAttention(embedding_dim, num_heads, sr_ratio, attention_dropout)
        self.norm2     = nn.LayerNorm(embedding_dim)
        self.ffn       = MixFFN(embedding_dim, mlp_ratio, dropout, ffn_activation)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Applies the attention and feed-forward residual branches.

        Args:
            x : Token sequence of shape (batch, h * w, embedding_dim).
            h : Token grid height.
            w : Token grid width.

        Returns:
            Token sequence of the same shape as ``x``.
        """
        x = x + self.drop_path(self.attention(self.norm1(x), h, w))
        x = x + self.drop_path(self.ffn(self.norm2(x), h, w))
        return x


class SegFormerStage(nn.Module):
    """Sequence of SegFormer blocks returning a feature map instead of tokens."""

    def __init__(self, blocks: list[SegFormerBlock], embedding_dim: int):
        """Registers the stage blocks and the closing normalization.

        Args:
            blocks        : The transformer blocks making up this stage.
            embedding_dim : Token width, used by the closing layer normalization.
        """
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.norm   = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Runs the stage and reshapes the tokens back into a feature map.

        Args:
            x : Token sequence of shape (batch, h * w, embedding_dim).
            h : Token grid height.
            w : Token grid width.

        Returns:
            Feature map of shape (batch, embedding_dim, h, w).
        """
        for block in self.blocks:
            x = block(x, h, w)

        x = self.norm(x)
        B, N, C = x.shape
        return x.transpose(1, 2).reshape(B, C, h, w)


class SegFormerLite(nn.Module, OutputHeadsMixin):
    """SegFormer backbone predicting per-pixel Gaussian mixture parameters.

    Attributes:
        config             : Architecture configuration driving stage widths, depths and head choice.
        embedding_channels : Channel width of the fused embedding fed to the output head.
    """

    def __init__(self, config: SegFormerLiteConfig | None = None):
        """Builds the patch embeddings, the encoder stages, the all-MLP decoder and the head.

        Args:
            config : Architecture configuration; defaults to ``SegFormerLiteConfig()``.

        Raises:
            ValueError : If ``embedding_dims``, ``depths``, ``num_heads`` and ``sr_ratios`` do not all have the same length.
        """
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
            LayerFactory.build_activation(config.ffn_activation),
            nn.Dropout2d(config.dropout),
        )

        self.embedding_channels = config.decoder_channels
        self._build_output_head()

        LayerFactory.initialize_weights(module=self, mode=config.init_mode)

    def _head_activation(self) -> str:
        """Returns the activation name used inside the output head."""
        return self.config.ffn_activation

    def _fused_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the fused decoder embedding on the first stage grid.

        Args:
            x : Input feature stack of shape (batch, in_channels, height, width).

        Returns:
            Feature map of shape (batch, decoder_channels, height // 4, width // 4).
        """
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
        """Returns the embedding at input resolution.

        Args:
            x : Input feature stack of shape (batch, in_channels, height, width).

        Returns:
            Embedding of shape (batch, embedding_channels, height, width).
        """
        input_size = x.shape[2:]
        embedding  = self._fused_embedding(x)
        return functional.interpolate(embedding, size=input_size, mode="bilinear", align_corners=False)

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
