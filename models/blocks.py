
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional


class DropPath(nn.Module):

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob     = 1.0 - self.drop_prob
        shape         = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor_(random_tensor + keep_prob)
        return x / keep_prob * random_tensor


class DropPathSchedule:

    def __init__(self, total_blocks: int, rate: float, blocks_per_stage: int):
        self.rates            = [rate * index / max(total_blocks - 1, 1) for index in range(total_blocks)]
        self.blocks_per_stage = blocks_per_stage

    def stage(self, stage_index: int) -> list[float]:
        start = stage_index * self.blocks_per_stage
        stop  = start + self.blocks_per_stage
        return self.rates[start:stop]


class LayerFactory:

    @staticmethod
    def build_activation(name: str) -> nn.Module:
        factories = {
            "relu"        : lambda: nn.ReLU(inplace=True),
            "leaky_relu"  : lambda: nn.LeakyReLU(inplace=True),
            "gelu"        : lambda: nn.GELU(),
            "elu"         : lambda: nn.ELU(inplace=True),
            "silu"        : lambda: nn.SiLU(inplace=True),
        }
        return factories[name]()

    @staticmethod
    def group_count(num_features: int) -> int:
        num_groups = min(32, num_features)
        while num_features % num_groups != 0:
            num_groups -= 1
        return num_groups

    @staticmethod
    def build_norm2d(name: str, num_features: int) -> nn.Module:
        factories = {
            "batch"    : lambda: nn.BatchNorm2d(num_features),
            "instance" : lambda: nn.InstanceNorm2d(num_features, affine=True),
            "group"    : lambda: nn.GroupNorm(LayerFactory.group_count(num_features), num_features),
            "none"     : lambda: nn.Identity(),
        }
        return factories[name]()

    @staticmethod
    def build_upsample(mode: str, in_channels: int, out_channels: int, scale_factor: int = 2) -> nn.Module:
        if mode == "convtranspose":
            return nn.ConvTranspose2d(
                in_channels  = in_channels,
                out_channels = out_channels,
                kernel_size  = scale_factor,
                stride       = scale_factor,
            )
        return {"bilinear": lambda: nn.Sequential(
            nn.Upsample(
                scale_factor  = scale_factor,
                mode          = "bilinear",
                align_corners = False,
            ),
            nn.Conv2d(
                in_channels  = in_channels,
                out_channels = out_channels,
                kernel_size  = 1,
                bias         = False,
            ),
        )}[mode]()

    @staticmethod
    def initialize_weights(module: nn.Module, mode: str) -> None:
        if mode == "default":
            return
        for m in module.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d)):
                if mode == "kaiming":
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif mode == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                if mode == "kaiming":
                    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                elif mode == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.MultiheadAttention):
                if mode == "kaiming":
                    nn.init.kaiming_normal_(m.in_proj_weight, mode="fan_in", nonlinearity="relu")
                elif mode == "xavier":
                    nn.init.xavier_uniform_(m.in_proj_weight)
                if m.in_proj_bias is not None:
                    nn.init.zeros_(m.in_proj_bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.LayerNorm)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)


class FeatureGeometry:

    @staticmethod
    def match_spatial_size(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if source.shape[2:] != reference.shape[2:]:
            return functional.interpolate(
                input         = source,
                size          = reference.shape[2:],
                mode          = "bilinear",
                align_corners = False,
            )
        return source

    @staticmethod
    def tokens_to_feature_map(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch_size, _, channels = tokens.shape
        return tokens.transpose(1, 2).view(batch_size, channels, height, width)

    @staticmethod
    def resize_positional_embedding(embedding: torch.Tensor, grid_size: int, height: int, width: int) -> torch.Tensor:
        if height == grid_size and width == grid_size:
            return embedding

        resized = embedding.transpose(1, 2).view(1, -1, grid_size, grid_size)
        resized = functional.interpolate(
            input         = resized,
            size          = (height, width),
            mode          = "bilinear",
            align_corners = False,
        )
        return resized.flatten(2).transpose(1, 2)


class ConvNormAct(nn.Module):

    def __init__(
        self,
        input_channels:  int,
        output_channels: int,
        kernel_size:     int = 3,
        dilation:        int = 1,
        activation:      str = "relu",
        normalization:   str = "batch",
        bias:            bool = False,
    ):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=bias),
            LayerFactory.build_norm2d(normalization, output_channels),
            LayerFactory.build_activation(activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ConvBlock(nn.Module):

    def __init__(
        self,
        input_channels:  int,
        output_channels: int,
        dropout:         float = 0.0,
        activation:      str   = "relu",
        normalization:   str   = "batch",
        bias:            bool  = False,
        kernel_size:     int   = 3,
    ):
        super().__init__()
        padding = kernel_size // 2
        layers  = [
            nn.Conv2d(
                in_channels  = input_channels,
                out_channels = output_channels,
                kernel_size  = kernel_size,
                padding      = padding,
                bias         = bias,
            ),
            LayerFactory.build_norm2d(normalization, output_channels),
            LayerFactory.build_activation(activation),
            nn.Conv2d(
                in_channels  = output_channels,
                out_channels = output_channels,
                kernel_size  = kernel_size,
                padding      = padding,
                bias         = bias,
            ),
            LayerFactory.build_norm2d(normalization, output_channels),
            LayerFactory.build_activation(activation),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ResidualConvBlock(nn.Module):

    def __init__(
        self,
        input_channels:  int,
        output_channels: int,
        dropout:         float = 0.0,
        activation:      str   = "relu",
        normalization:   str   = "batch",
        bias:            bool  = False,
        stride:          int   = 1,
        first_unit:      bool  = False,
    ):
        super().__init__()
        layers = []
        if not first_unit:
            layers.append(LayerFactory.build_norm2d(normalization, input_channels))
            layers.append(LayerFactory.build_activation(activation))

        layers.append(
            nn.Conv2d(
                in_channels  = input_channels,
                out_channels = output_channels,
                kernel_size  = 3,
                stride       = stride,
                padding      = 1,
                bias         = bias,
            )
        )
        layers.append(LayerFactory.build_norm2d(normalization, output_channels))
        layers.append(LayerFactory.build_activation(activation))
        layers.append(
            nn.Conv2d(
                in_channels  = output_channels,
                out_channels = output_channels,
                kernel_size  = 3,
                padding      = 1,
                bias         = bias,
            )
        )
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.layers = nn.Sequential(*layers)

        if input_channels != output_channels or stride != 1:
            self.shortcut = nn.Conv2d(
                in_channels  = input_channels,
                out_channels = output_channels,
                kernel_size  = 1,
                stride       = stride,
                bias         = bias,
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x) + self.shortcut(x)


class ChannelLayerNorm(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):

    def __init__(self, channels: int, ffn_ratio: float, drop_path: float, ffn_activation: str, layer_scale_init: float):
        super().__init__()
        hidden = int(channels * ffn_ratio)

        self.dwconv     = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.norm       = nn.LayerNorm(channels)
        self.fc1        = nn.Linear(channels, hidden)
        self.activation = LayerFactory.build_activation(ffn_activation)
        self.fc2        = nn.Linear(hidden, channels)
        self.drop_path  = DropPath(drop_path)

        self.gamma = nn.Parameter(layer_scale_init * torch.ones(channels)) if layer_scale_init > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)

        return residual + self.drop_path(x)


class OutputHeadMixin:

    conv_head_kernel_size : int = 1

    def _build_output_head(self) -> None:
        self.output_head = nn.Conv2d(
            in_channels  = self.embedding_channels,
            out_channels = self.config.out_channels,
            kernel_size  = self.conv_head_kernel_size,
            padding      = self.conv_head_kernel_size // 2,
        )

    def _head_forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.output_head(embedding)

    def head_parameters(self) -> list:
        return list(self.output_head.parameters())


class Encoder(nn.Module):

    def __init__(
        self,
        input_channels: int,
        feature_sizes:  list[int],
        dropout:        float = 0.0,
        activation:     str   = "relu",
        normalization:  str   = "batch",
        bias:           bool  = False,
    ):
        super().__init__()
        self.conv_blocks = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        channels = input_channels
        for feature_size in feature_sizes:
            self.conv_blocks.append(
                ConvBlock(
                    input_channels  = channels,
                    output_channels = feature_size,
                    dropout         = dropout,
                    activation      = activation,
                    normalization   = normalization,
                    bias            = bias,
                )
            )
            self.downsample_layers.append(nn.MaxPool2d(kernel_size=2))
            channels = feature_size

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        skip_connections: list[torch.Tensor] = []
        for conv_block, downsample in zip(self.conv_blocks, self.downsample_layers):
            x = conv_block(x)
            skip_connections.append(x)
            x = downsample(x)
        return x, skip_connections


class Decoder(nn.Module):

    def __init__(
        self,
        feature_sizes: list[int],
        dropout:       float = 0.0,
        activation:    str   = "relu",
        normalization: str   = "batch",
        bias:          bool  = False,
        upsample_mode: str   = "convtranspose",
    ):
        super().__init__()
        self.upsample_layers = nn.ModuleList()
        self.conv_blocks     = nn.ModuleList()
        for index in range(len(feature_sizes) - 1):
            self.upsample_layers.append(
                LayerFactory.build_upsample(
                    mode         = upsample_mode,
                    in_channels  = feature_sizes[index],
                    out_channels = feature_sizes[index + 1],
                )
            )
            self.conv_blocks.append(
                ConvBlock(
                    input_channels  = feature_sizes[index + 1] * 2,
                    output_channels = feature_sizes[index + 1],
                    dropout         = dropout,
                    activation      = activation,
                    normalization   = normalization,
                    bias            = bias,
                )
            )

    def forward(self, x: torch.Tensor, skip_connections: list[torch.Tensor]) -> torch.Tensor:
        for upsample, conv_block, skip in zip(self.upsample_layers, self.conv_blocks, skip_connections):
            x = upsample(x)
            x = FeatureGeometry.match_spatial_size(source=x, reference=skip)
            x = torch.cat([skip, x], dim=1)
            x = conv_block(x)
        return x


class AttentionMath:

    @staticmethod
    def scaled_dot_product(query, key, value, scale, attention_dropout, attention_bias=None):
        dropout_p = attention_dropout.p if attention_dropout.training else 0.0

        return functional.scaled_dot_product_attention(query, key, value, attn_mask=attention_bias, dropout_p=dropout_p, scale=scale)


class MultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        embedding_dim:     int,
        num_heads:         int,
        dropout:           float = 0.0,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embedding_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.query_key_value   = nn.Linear(embedding_dim, embedding_dim * 3)
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.output_dropout    = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, embedding_dim = x.shape
        qkv = self.query_key_value(x)
        qkv = qkv.reshape(batch_size, sequence_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)

        attended  = AttentionMath.scaled_dot_product(query, key, value, self.scale, self.attention_dropout).transpose(1, 2)
        attended  = attended.reshape(batch_size, sequence_length, embedding_dim)
        projected = self.output_projection(attended)
        return self.output_dropout(projected)


class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_dim:     int,
        num_heads:         int,
        mlp_ratio:         float = 4.0,
        dropout:           float = 0.0,
        attention_dropout: float = 0.0,
        ffn_activation:    str   = "gelu",
        drop_path_rate:    float = 0.0,
    ):
        super().__init__()
        self.norm_1    = nn.LayerNorm(embedding_dim)
        self.attention = MultiHeadSelfAttention(
            embedding_dim     = embedding_dim,
            num_heads         = num_heads,
            dropout           = dropout,
            attention_dropout = attention_dropout,
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm_2    = nn.LayerNorm(embedding_dim)

        hidden_dim = int(embedding_dim * mlp_ratio)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            LayerFactory.build_activation(ffn_activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm_1(x)
        x          = x + self.drop_path(self.attention(normalized))
        normalized = self.norm_2(x)
        x          = x + self.drop_path(self.feed_forward(normalized))
        return x


class PatchEmbedding(nn.Module):

    def __init__(
        self,
        in_channels:   int,
        embedding_dim: int,
        patch_size:    int,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(
            in_channels  = in_channels,
            out_channels = embedding_dim,
            kernel_size  = patch_size,
            stride       = patch_size,
        )
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.projection(x)
        grid_height, grid_width = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, grid_height, grid_width

