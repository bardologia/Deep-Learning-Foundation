"""Shared neural building blocks reused across every model family in the project.

Holds the layer factories, geometry helpers, convolutional and transformer blocks, and
the output-head mixin that turns a per-pixel embedding into Gaussian mixture parameters.
Components live here rather than in a single architecture so no block is ever duplicated.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as functional


class EmbeddingNorm:
    """Mixin normalizing an autoencoder embedding along its channel axis.

    The host class must expose a ``config`` with an ``embedding_norm`` field.

    Attributes:
        EMBEDDING_NORMS : Accepted values of ``config.embedding_norm``.
    """

    EMBEDDING_NORMS = ("none", "l2", "layernorm")

    def normalize_embedding(self, z: torch.Tensor) -> torch.Tensor:
        """Normalizes the embedding according to the configured mode.

        Args:
            z : Embedding of shape (batch, embedding_dim, ...), normalized over dim 1.

        Returns:
            Embedding of the same shape: unchanged for ``none``, unit-norm per pixel
            for ``l2``, or zero-mean unit-variance across channels for ``layernorm``.
        """
        kind = self.config.embedding_norm
        if kind == "none":
            return z
        if kind == "l2":
            return functional.normalize(z, dim=1, eps=1e-6)

        mean = z.mean(dim=1, keepdim=True)
        var  = z.var(dim=1, keepdim=True, unbiased=False)
        return (z - mean) / torch.sqrt(var + 1e-6)


class DropPath(nn.Module):
    """Stochastic depth: drops whole residual branches per sample during training.

    Attributes:
        drop_prob : Probability of zeroing a sample's branch output.
    """

    def __init__(self, drop_prob: float = 0.0):
        """Stores the per-sample drop probability.

        Args:
            drop_prob : Probability of dropping the branch; 0 makes the module a no-op.
        """
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Drops the branch for a random subset of samples and rescales the survivors.

        Args:
            x : Residual branch output whose first axis is the batch.

        Returns:
            Tensor of the same shape as ``x``, returned unchanged in eval mode or
            when ``drop_prob`` is 0.
        """
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob     = 1.0 - self.drop_prob
        shape         = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor_(random_tensor + keep_prob)
        return x / keep_prob * random_tensor


class DropPathSchedule:
    """Linear stochastic depth schedule sliced into equal-sized stages.

    Attributes:
        rates            : Drop probability per block, rising linearly from 0 to ``rate``.
        blocks_per_stage : Number of consecutive blocks each stage consumes.
    """

    def __init__(self, total_blocks: int, rate: float, blocks_per_stage: int):
        """Builds the linear ramp over all blocks.

        Args:
            total_blocks     : Total number of blocks sharing the schedule.
            rate             : Drop probability reached by the last block.
            blocks_per_stage : Number of blocks per stage slice.

        Raises:
            ValueError : If ``blocks_per_stage`` is smaller than 1.
        """
        if blocks_per_stage < 1:
            raise ValueError(f"blocks_per_stage must be >= 1, got {blocks_per_stage}")

        self.rates            = [rate * index / max(total_blocks - 1, 1) for index in range(total_blocks)]
        self.blocks_per_stage = blocks_per_stage

    def stage(self, stage_index: int) -> list[float]:
        """Returns the drop probabilities belonging to one stage.

        Args:
            stage_index : Zero-based index of the stage.

        Returns:
            The ``blocks_per_stage`` rates assigned to that stage.

        Raises:
            ValueError : If the stage reaches past the end of the schedule.
        """
        start = stage_index * self.blocks_per_stage
        stop  = start + self.blocks_per_stage
        if stop > len(self.rates):
            raise ValueError(f"stage {stage_index} needs rates {start}:{stop} but the schedule holds only {len(self.rates)}")

        return self.rates[start:stop]


class LayerFactory:
    """Builds activations, normalizations and upsamplers from configuration names."""

    @staticmethod
    def downsample_stages(downsample_factor: int) -> int:
        """Returns the number of halving stages that realize a downsampling factor.

        Args:
            downsample_factor : Total spatial reduction, which must be a power of two.

        Returns:
            log2 of ``downsample_factor``.

        Raises:
            ValueError : If the factor is below 1 or is not a power of two.
        """
        if downsample_factor < 1:
            raise ValueError(f"downsample_factor must be >= 1, got {downsample_factor}")

        stages = math.log2(downsample_factor)
        if stages != int(stages):
            raise ValueError(f"downsample_factor must be a power of two (1, 2, 4, 8, ...), got {downsample_factor}")

        return int(stages)

    @staticmethod
    def build_activation(name: str) -> nn.Module:
        """Returns a fresh activation module for a configuration name.

        Args:
            name : One of ``relu``, ``leaky_relu``, ``gelu``, ``elu`` or ``silu``.

        Returns:
            The corresponding activation module.

        Raises:
            ValueError : If the name is not one of the supported activations.
        """
        factories = {
            "relu"        : lambda: nn.ReLU(inplace=True),
            "leaky_relu"  : lambda: nn.LeakyReLU(inplace=True),
            "gelu"        : lambda: nn.GELU(),
            "elu"         : lambda: nn.ELU(inplace=True),
            "silu"        : lambda: nn.SiLU(inplace=True),
        }
        if name not in factories:
            raise ValueError(f"Unknown activation '{name}'. Available: {list(factories.keys())}")
        return factories[name]()

    @staticmethod
    def build_norm2d(name: str, num_features: int) -> nn.Module:
        """Returns a 2D normalization layer for a configuration name.

        For ``group``, the group count starts at 32 and is reduced until it divides
        ``num_features``.

        Args:
            name         : One of ``batch``, ``instance``, ``group`` or ``none``.
            num_features : Channel width the layer normalizes.

        Returns:
            The corresponding normalization module, or ``nn.Identity`` for ``none``.

        Raises:
            ValueError : If the name is not one of the supported normalizations.
        """
        if name == "batch":
            return nn.BatchNorm2d(num_features)
        if name == "instance":
            return nn.InstanceNorm2d(num_features, affine=True)
        if name == "group":
            num_groups = min(32, num_features)
            while num_features % num_groups != 0:
                num_groups -= 1
            return nn.GroupNorm(num_groups, num_features)
        if name == "none":
            return nn.Identity()
        raise ValueError(f"Unknown normalization '{name}'. Available: batch, instance, group, none")

    @staticmethod
    def sinusoidal_position_table(num_positions: int, dim: int) -> torch.Tensor:
        """Returns the fixed sine-cosine positional encoding table.

        Args:
            num_positions : Number of sequence positions encoded.
            dim           : Feature width of each position; odd widths leave the final channel of the cosine half unused.

        Returns:
            Tensor of shape (1, num_positions, dim) holding the standard
            transformer sinusoidal encoding.
        """
        position = torch.arange(num_positions, dtype=torch.float32).unsqueeze(1)
        rate     = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))

        table = torch.zeros(num_positions, dim)
        table[:, 0::2] = torch.sin(position * rate)
        table[:, 1::2] = torch.cos(position * rate)[:, : dim // 2]

        return table.unsqueeze(0)

    @staticmethod
    def build_upsample(mode: str, in_channels: int, out_channels: int, scale_factor: int = 2) -> nn.Module:
        """Returns a module that upsamples and rechannels a feature map.

        Args:
            mode         : Either ``convtranspose`` for a learned transposed convolution or ``bilinear`` for interpolation followed by a 1x1 projection.
            in_channels  : Channel width of the incoming feature map.
            out_channels : Channel width produced by the module.
            scale_factor : Spatial upsampling factor per axis.

        Returns:
            The upsampling module.

        Raises:
            ValueError : If the mode is unknown.
        """
        if mode == "convtranspose":
            return nn.ConvTranspose2d(
                in_channels  = in_channels,
                out_channels = out_channels,
                kernel_size  = scale_factor,
                stride       = scale_factor,
            )
        if mode == "bilinear":
            return nn.Sequential(
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
            )
        raise ValueError(f"Unknown upsample mode '{mode}'. Available: convtranspose, bilinear")

    @staticmethod
    def initialize_weights(module: nn.Module, mode: str) -> None:
        """Reinitializes convolution, linear, attention and normalization weights in place.

        Normalization layers always get unit weights and zero biases; ``default``
        leaves every parameter at PyTorch's own initialization.

        Args:
            module : Root module whose submodules are reinitialized.
            mode   : One of ``default``, ``kaiming`` or ``xavier``.

        Raises:
            ValueError : If the mode is unknown.
        """
        if mode not in ("default", "kaiming", "xavier"):
            raise ValueError(f"Unknown init_mode '{mode}'. Available: default, kaiming, xavier")
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
    """Reshaping helpers that keep feature maps, tokens and embeddings aligned."""

    @staticmethod
    def match_spatial_size(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Resamples a feature map onto the spatial grid of a reference tensor.

        Args:
            source    : Feature map of shape (batch, channels, height, width).
            reference : Tensor whose trailing two axes give the target grid.

        Returns:
            ``source`` unchanged when the grids already agree, otherwise a bilinearly
            resampled copy of shape (batch, channels, reference_height, reference_width).
        """
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
        """Reshapes a token sequence into a feature map.

        Args:
            tokens : Token sequence of shape (batch, height * width, channels).
            height : Token grid height.
            width  : Token grid width.

        Returns:
            Feature map of shape (batch, channels, height, width).
        """
        batch_size, _, channels = tokens.shape
        return tokens.transpose(1, 2).view(batch_size, channels, height, width)

    @staticmethod
    def resize_positional_embedding(embedding: torch.Tensor, grid_size: int, height: int, width: int) -> torch.Tensor:
        """Resamples a square positional embedding onto a different token grid.

        Args:
            embedding : Positional embedding of shape (1, grid_size ** 2, channels).
            grid_size : Side length of the grid the embedding was learned on.
            height    : Target token grid height.
            width     : Target token grid width.

        Returns:
            Positional embedding of shape (1, height * width, channels), returned
            unchanged when the target grid is already the learned one.
        """
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
    """Single convolution followed by normalization and activation.

    Attributes:
        layers : The convolution, normalization and activation in sequence.
    """

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
        """Builds the convolution with padding chosen to preserve the spatial shape.

        Args:
            input_channels  : Channel width of the incoming feature map.
            output_channels : Channel width produced by the block.
            kernel_size     : Convolution kernel size.
            dilation        : Dilation rate of the convolution.
            activation      : Activation name resolved by ``LayerFactory.build_activation``.
            normalization   : 2D normalization name resolved by ``LayerFactory.build_norm2d``.
            bias            : Whether the convolution carries a bias term.
        """
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=bias),
            LayerFactory.build_norm2d(normalization, output_channels),
            LayerFactory.build_activation(activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the convolved, normalized and activated feature map.

        Args:
            x : Feature map of shape (batch, input_channels, height, width).

        Returns:
            Feature map of shape (batch, output_channels, height, width).
        """
        return self.layers(x)


class ConvBlock(nn.Module):
    """Two padded convolutions, each followed by normalization and activation.

    The workhorse block of the U-Net family; the spatial grid is preserved.

    Attributes:
        layers : The two convolution-norm-activation pairs plus optional dropout.
    """

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
        """Builds the two-convolution block.

        Args:
            input_channels  : Channel width of the incoming feature map.
            output_channels : Channel width produced by the block.
            dropout         : Spatial dropout probability appended to the block; disabled at 0.
            activation      : Activation name resolved by ``LayerFactory.build_activation``.
            normalization   : 2D normalization name resolved by ``LayerFactory.build_norm2d``.
            bias            : Whether the convolutions carry a bias term.
            kernel_size     : Odd convolution kernel size, so padding preserves the grid.

        Raises:
            ValueError : If ``kernel_size`` is not a positive odd integer.
        """
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"ConvBlock kernel_size={kernel_size} must be a positive odd integer so padding preserves the spatial shape")

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
        """Returns the twice-convolved feature map.

        Args:
            x : Feature map of shape (batch, input_channels, height, width).

        Returns:
            Feature map of shape (batch, output_channels, height, width).
        """
        return self.layers(x)


class ResidualConvBlock(nn.Module):
    """Pre-activation residual unit with an optional strided shortcut.

    Attributes:
        layers   : The residual main path of two 3x3 convolutions.
        shortcut : 1x1 projection when the width or stride changes, identity otherwise.
    """

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
        """Builds a pre-activation residual unit with a matching shortcut.

        Args:
            input_channels  : Channel width of the incoming feature map.
            output_channels : Channel width produced by the block.
            dropout         : Spatial dropout probability on the main path; disabled at 0.
            activation      : Activation name resolved by ``LayerFactory.build_activation``.
            normalization   : 2D normalization name resolved by ``LayerFactory.build_norm2d``.
            bias            : Whether the convolutions carry a bias term.
            stride          : Stride of the first convolution, halving the grid when 2.
            first_unit      : When True, the leading normalization and activation are skipped because the input comes straight from the network input.
        """
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
        """Returns the sum of the residual path and the shortcut.

        Args:
            x : Feature map of shape (batch, input_channels, height, width).

        Returns:
            Feature map of shape
            (batch, output_channels, height // stride, width // stride).
        """
        return self.layers(x) + self.shortcut(x)


class ChannelLayerNorm(nn.Module):
    """Layer normalization over the channel axis of a 4D feature map.

    Attributes:
        norm : The underlying layer normalization over ``channels``.
    """

    def __init__(self, channels: int):
        """Builds the layer normalization.

        Args:
            channels : Channel width being normalized.
        """
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalizes each pixel across its channels.

        Args:
            x : Feature map of shape (batch, channels, height, width).

        Returns:
            Feature map of the same shape as ``x``.
        """
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class AttentionPool1d(nn.Module):
    """Learned attention pooling over the length axis of a 1-D feature map.

    Attributes:
        score : 1x1 convolution producing one attention logit per position.
    """

    def __init__(self, channels: int):
        """Builds the scoring convolution.

        Args:
            channels : Channel width of the pooled features.
        """
        super().__init__()
        self.score = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pools features by their softmax-normalized attention scores.

        Args:
            x : Features of shape (batch, channels, length).

        Returns:
            Pooled features of shape (batch, channels, 1).
        """
        weights = torch.softmax(self.score(x), dim=-1)
        return (x * weights).sum(dim=-1, keepdim=True)


class TokenMeanPool(nn.Module):
    """Mean pooling over a token sequence of shape (batch, tokens, dim)."""

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Returns the token mean of shape (batch, dim)."""
        return tokens.mean(dim=1)


class TokenMaxPool(nn.Module):
    """Max pooling over a token sequence of shape (batch, tokens, dim)."""

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Returns the per-feature token maximum of shape (batch, dim)."""
        return tokens.max(dim=1).values


class TokenAttentionPool(nn.Module):
    """Learned attention pooling over a token sequence.

    Attributes:
        score : Linear layer producing one attention logit per token.
    """

    def __init__(self, dim: int):
        """Builds the scoring layer.

        Args:
            dim : Feature width of the pooled tokens.
        """
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Pools tokens by their softmax-normalized attention scores.

        Args:
            tokens : Token sequence of shape (batch, tokens, dim).

        Returns:
            Pooled features of shape (batch, dim).
        """
        weights = torch.softmax(self.score(tokens), dim=1)
        return (tokens * weights).sum(dim=1)


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block: 7x7 depthwise convolution, layer norm and an inverted MLP.

    Attributes:
        gamma     : Learned per-channel layer scale of shape (channels,), or None when ``layer_scale_init`` is not positive.
        drop_path : Stochastic depth applied to the residual branch.
    """

    def __init__(self, channels: int, ffn_ratio: float, drop_path: float, ffn_activation: str, layer_scale_init: float):
        """Builds the depthwise convolution and the inverted bottleneck MLP.

        Args:
            channels         : Channel width entering and leaving the block.
            ffn_ratio        : Hidden width of the MLP as a multiple of ``channels``.
            drop_path        : Stochastic depth probability for the residual branch.
            ffn_activation   : Activation name resolved by ``LayerFactory.build_activation``.
            layer_scale_init : Initial value of the learned layer scale; 0 disables it.
        """
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
        """Applies the ConvNeXt residual branch.

        Args:
            x : Feature map of shape (batch, channels, height, width).

        Returns:
            Feature map of the same shape as ``x``.
        """
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


class PixelMLP(nn.Module):
    """Two-layer MLP applied independently at every pixel via 1x1 convolutions.

    Attributes:
        mlp : The 1x1 convolution, activation and 1x1 convolution in sequence.
    """

    def __init__(
        self,
        in_channels:     int,
        hidden_channels: int,
        out_channels:    int,
        activation:      str = "relu",
    ):
        """Builds the per-pixel MLP.

        Args:
            in_channels     : Channel width of the incoming embedding.
            hidden_channels : Hidden width of the MLP.
            out_channels    : Channel width produced by the MLP.
            activation      : Activation name resolved by ``LayerFactory.build_activation``.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=True),
            LayerFactory.build_activation(activation),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the MLP at each pixel.

        Args:
            x : Embedding of shape (batch, in_channels, height, width).

        Returns:
            Tensor of shape (batch, out_channels, height, width).
        """
        return self.mlp(x)


class OutputHeadsMixin:
    """Mixin turning a per-pixel embedding into the configured output head.

    The host model must set ``self.embedding_channels`` before calling
    ``_build_output_head`` and expose a ``config`` carrying ``head``,
    ``out_channels``, ``params_per_gaussian`` and ``activation``. Gaussian heads emit
    ``n_gaussians * params_per_gaussian`` channels, with the parameters of one
    Gaussian stored contiguously.

    Attributes:
        HEADS                 : Accepted values of ``config.head``.
        conv_head_kernel_size : Kernel size of the plain convolutional head.
    """

    HEADS                 : tuple = ("none", "conv", "multihead", "per_gaussian", "set_pred")
    conv_head_kernel_size : int   = 1

    def _resolve_gaussian_layout(self) -> None:
        """Splits ``out_channels`` into a Gaussian count and a parameter count.

        Raises:
            ValueError : If ``out_channels`` is not divisible by ``params_per_gaussian``.
        """
        n_params = self.config.params_per_gaussian
        if self.config.out_channels % n_params != 0:
            raise ValueError(f"out_channels ({self.config.out_channels}) must be divisible by params_per_gaussian ({n_params})")
        self.n_gaussians = self.config.out_channels // n_params
        self.n_params    = n_params

    def _head_activation(self) -> str:
        """Returns the activation name used inside the output head."""
        return self.config.activation

    def _build_conv_head(self) -> None:
        """Builds the single convolution mapping the embedding to ``out_channels``."""
        self.output_head = nn.Conv2d(
            in_channels  = self.embedding_channels,
            out_channels = self.config.out_channels,
            kernel_size  = self.conv_head_kernel_size,
            padding      = self.conv_head_kernel_size // 2,
        )

    def _build_triple_heads(self) -> None:
        """Builds one per-pixel MLP each for amplitude, mean and width.

        Raises:
            ValueError : If ``params_per_gaussian`` is not exactly 3.
        """
        if self.n_params != 3:
            raise ValueError(f"the multihead output head stacks exactly amplitude, mu and sigma, so params_per_gaussian must be 3, got {self.n_params}")

        self.head_amp   = PixelMLP(self.embedding_channels, self.hidden_channels, self.n_gaussians, self._head_activation())
        self.head_mu    = PixelMLP(self.embedding_channels, self.hidden_channels, self.n_gaussians, self._head_activation())
        self.head_sigma = PixelMLP(self.embedding_channels, self.hidden_channels, self.n_gaussians, self._head_activation())

    def _triple_head_forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Runs the three parameter heads and interleaves them per Gaussian.

        Args:
            embedding : Embedding of shape (batch, embedding_channels, height, width).

        Returns:
            Tensor of shape (batch, n_gaussians * 3, height, width) ordered as
            amplitude, mean, width for each Gaussian in turn.
        """
        amp   = self.head_amp(embedding)
        mu    = self.head_mu(embedding)
        sigma = self.head_sigma(embedding)

        B, K, H, W = amp.shape
        out = torch.stack([amp, mu, sigma], dim=2)
        return out.reshape(B, K * 3, H, W)

    def _build_per_gaussian_heads(self) -> None:
        """Builds one per-pixel MLP per Gaussian slot, each emitting all its parameters."""
        self.gaussian_heads = nn.ModuleList([
            PixelMLP(self.embedding_channels, self.hidden_channels, self.n_params, self._head_activation())
            for _ in range(self.n_gaussians)
        ])

    def _per_gaussian_forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Runs every slot head and stacks the slots along the channel axis.

        Args:
            embedding : Embedding of shape (batch, embedding_channels, height, width).

        Returns:
            Tensor of shape (batch, n_gaussians * n_params, height, width) with the
            parameters of each Gaussian contiguous.
        """
        head_outputs = [head(embedding) for head in self.gaussian_heads]

        B, _, H, W = head_outputs[0].shape
        out = torch.stack(head_outputs, dim=1)
        return out.reshape(B, self.n_gaussians * self.n_params, H, W)

    def _build_set_prediction_heads(self) -> None:
        """Builds the per-slot parameter heads plus a per-slot existence gate.

        The gate blends each slot's amplitude towards the learned ``amp_off`` offset,
        so an unused slot can be switched off rather than forced onto a component.
        """
        self._build_per_gaussian_heads()
        self.existence_head = PixelMLP(self.embedding_channels, self.hidden_channels, self.n_gaussians, self._head_activation())
        self.amp_off        = nn.Parameter(torch.zeros(self.n_gaussians))

    def _set_prediction_forward(self, embedding: torch.Tensor, existence_embedding: torch.Tensor | None = None) -> torch.Tensor:
        """Runs the slot heads and gates each slot's amplitude by its existence score.

        Args:
            embedding           : Embedding of shape (batch, embedding_channels, height, width) feeding the parameter heads.
            existence_embedding : Separate embedding of shape (batch, channels, height, width) feeding the existence gate; when None the gate reads ``embedding``. Dual-trunk models pass their second trunk here.

        Returns:
            Tensor of shape (batch, n_gaussians * n_params, height, width) with each
            slot's amplitude interpolated between the predicted value and ``amp_off``.
        """
        gate_source  = embedding if existence_embedding is None else existence_embedding
        head_outputs = [head(embedding) for head in self.gaussian_heads]
        gate         = torch.sigmoid(self.existence_head(gate_source))

        B, _, H, W = head_outputs[0].shape
        out = torch.stack(head_outputs, dim=1)

        off = self.amp_off.view(1, self.n_gaussians, 1, 1)
        amp = gate * out[:, :, 0] + (1.0 - gate) * off
        out = torch.cat([amp[:, :, None], out[:, :, 1:]], dim=2)

        return out.reshape(B, self.n_gaussians * self.n_params, H, W)

    def _build_output_head(self) -> None:
        """Builds the head selected by ``config.head``.

        The ``none`` head adds no layers, so the model returns its raw embedding.

        Raises:
            ValueError : If ``config.head`` is not one of ``HEADS``, or if the chosen Gaussian head cannot be laid out over ``out_channels``.
        """
        head = self.config.head
        if head not in self.HEADS:
            raise ValueError(f"Unknown head '{head}'. Available: {list(self.HEADS)}")

        if head == "none":
            return

        self.hidden_channels = max(self.embedding_channels // 2, 16)

        if head == "conv":
            self._build_conv_head()
            return

        self._resolve_gaussian_layout()
        if head == "multihead":
            self._build_triple_heads()
        elif head == "per_gaussian":
            self._build_per_gaussian_heads()
        else:
            self._build_set_prediction_heads()

    def _head_forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Applies the configured head to an embedding.

        Args:
            embedding : Embedding of shape (batch, embedding_channels, height, width).

        Returns:
            Tensor of shape (batch, out_channels, height, width), or ``embedding``
            itself when the head is ``none``.
        """
        head = self.config.head
        if head == "none":
            return embedding
        if head == "conv":
            return self.output_head(embedding)
        if head == "multihead":
            return self._triple_head_forward(embedding)
        if head == "per_gaussian":
            return self._per_gaussian_forward(embedding)
        return self._set_prediction_forward(embedding)

    def head_parameters(self) -> list:
        """Returns the parameters belonging to the output head only.

        Used to give the head its own optimizer group, separate from the trunk.

        Returns:
            List of parameters of the active head, empty for the ``none`` head.
        """
        head = self.config.head
        if head == "none":
            return []
        if head == "conv":
            return list(self.output_head.parameters())
        if head == "multihead":
            return list(self.head_amp.parameters()) + list(self.head_mu.parameters()) + list(self.head_sigma.parameters())
        if head == "per_gaussian":
            return list(self.gaussian_heads.parameters())
        return list(self.gaussian_heads.parameters()) + list(self.existence_head.parameters()) + [self.amp_off]


class Encoder(nn.Module):
    """U-Net encoder alternating convolution blocks and max pooling.

    Attributes:
        conv_blocks       : One convolution block per level.
        downsample_layers : The 2x max pooling layer following each block.
    """

    def __init__(
        self,
        input_channels: int,
        feature_sizes:  list[int],
        dropout:        float = 0.0,
        activation:     str   = "relu",
        normalization:  str   = "batch",
        bias:           bool  = False,
    ):
        """Builds one convolution block and pooling layer per feature size.

        Args:
            input_channels : Channel width of the network input.
            feature_sizes  : Channel width of each encoder level, coarsening downwards.
            dropout        : Spatial dropout probability inside each block; disabled at 0.
            activation     : Activation name resolved by ``LayerFactory.build_activation``.
            normalization  : 2D normalization name resolved by ``LayerFactory.build_norm2d``.
            bias           : Whether the convolutions carry a bias term.
        """
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
        """Runs the encoder and collects the pre-pooling skip connections.

        Args:
            x : Input feature stack of shape (batch, input_channels, height, width).

        Returns:
            Tuple of the pooled bottom feature map of shape
            (batch, feature_sizes[-1], height / 2 ** levels, width / 2 ** levels) and
            the per-level skip connections ordered from finest to coarsest.
        """
        skip_connections: list[torch.Tensor] = []
        for conv_block, downsample in zip(self.conv_blocks, self.downsample_layers):
            x = conv_block(x)
            skip_connections.append(x)
            x = downsample(x)
        return x, skip_connections


class Decoder(nn.Module):
    """U-Net decoder alternating upsampling and convolution over concatenated skips.

    Attributes:
        upsample_layers : One upsampling module per level.
        conv_blocks     : The convolution block consuming each concatenated skip.
    """

    def __init__(
        self,
        feature_sizes: list[int],
        dropout:       float = 0.0,
        activation:    str   = "relu",
        normalization: str   = "batch",
        bias:          bool  = False,
        upsample_mode: str   = "convtranspose",
    ):
        """Builds one upsampling and convolution pair per decoder level.

        Args:
            feature_sizes : Channel widths from the bottleneck outwards, so entry 0 is the bottleneck and the rest mirror the encoder in reverse.
            dropout       : Spatial dropout probability inside each block; disabled at 0.
            activation    : Activation name resolved by ``LayerFactory.build_activation``.
            normalization : 2D normalization name resolved by ``LayerFactory.build_norm2d``.
            bias          : Whether the convolutions carry a bias term.
            upsample_mode : Upsampling strategy passed to ``LayerFactory.build_upsample``.
        """
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
        """Upsamples through the levels, concatenating one skip connection at each.

        Args:
            x                : Bottleneck feature map of shape (batch, feature_sizes[0], height, width).
            skip_connections : Encoder skips ordered from coarsest to finest.

        Returns:
            Feature map of shape (batch, feature_sizes[-1], full_height, full_width).
        """
        for upsample, conv_block, skip in zip(self.upsample_layers, self.conv_blocks, skip_connections):
            x = upsample(x)
            x = FeatureGeometry.match_spatial_size(source=x, reference=skip)
            x = torch.cat([skip, x], dim=1)
            x = conv_block(x)
        return x


class AttentionTap:
    """Process-wide switch that captures attention weights for analysis.

    Disabled by default, so training and inference pay nothing; the attention-capture
    analysis enables it around a forward pass and reads the collected maps afterwards.
    """

    _sink = None

    @classmethod
    def enable(cls, sink: list) -> None:
        """Starts appending attention weights to the given list."""
        cls._sink = sink

    @classmethod
    def disable(cls) -> None:
        """Stops capturing attention weights."""
        cls._sink = None

    @classmethod
    def record(cls, weights) -> None:
        """Appends a detached CPU copy of the attention weights when capture is enabled.

        Args:
            weights : Softmaxed attention weights of shape (batch, heads, queries, keys).
        """
        if cls._sink is not None:
            cls._sink.append(weights.detach().cpu())


class AttentionMath:
    """Scaled dot-product attention routed through the capture tap."""

    @staticmethod
    def scaled_dot_product(query, key, value, scale, attention_dropout, attention_bias=None):
        """Computes scaled dot-product attention and records the weights.

        Args:
            query             : Queries of shape (batch, heads, queries, head_dim).
            key               : Keys of shape (batch, heads, keys, head_dim).
            value             : Values of shape (batch, heads, keys, head_dim).
            scale             : Multiplier applied to the query-key products, usually head_dim ** -0.5.
            attention_dropout : Dropout module applied to the softmaxed weights.
            attention_bias    : Additive bias broadcast onto the attention logits, used for relative position and window masking, or None.

        Returns:
            Attended values of shape (batch, heads, queries, head_dim).
        """
        attention_weights = (query @ key.transpose(-2, -1)) * scale

        if attention_bias is not None:
            attention_weights = attention_weights + attention_bias

        attention_weights = attention_weights.softmax(dim=-1)
        AttentionTap.record(attention_weights)
        attention_weights = attention_dropout(attention_weights)

        return attention_weights @ value


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention over a token sequence.

    Attributes:
        num_heads : Number of attention heads.
        head_dim  : Channel width per head.
        scale     : Multiplier applied to the query-key products.
    """

    def __init__(
        self,
        embedding_dim:     int,
        num_heads:         int,
        dropout:           float = 0.0,
        attention_dropout: float = 0.0,
    ):
        """Builds the fused qkv projection and the output projection.

        Args:
            embedding_dim     : Token width.
            num_heads         : Number of attention heads.
            dropout           : Dropout probability on the output projection.
            attention_dropout : Dropout probability on the attention weights.

        Raises:
            ValueError : If ``embedding_dim`` is not divisible by ``num_heads``.
        """
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim  = embedding_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.query_key_value   = nn.Linear(embedding_dim, embedding_dim * 3)
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.output_dropout    = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attends over the whole token sequence.

        Args:
            x : Token sequence of shape (batch, sequence_length, embedding_dim).

        Returns:
            Token sequence of the same shape as ``x``.
        """
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
    """Pre-norm transformer block with full self-attention and an MLP."""

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
        """Builds the attention and feed-forward sublayers with their normalizations.

        Args:
            embedding_dim     : Token width.
            num_heads         : Number of attention heads.
            mlp_ratio         : Hidden width of the MLP as a multiple of ``embedding_dim``.
            dropout           : Dropout probability in the attention output and the MLP.
            attention_dropout : Dropout probability on the attention weights.
            ffn_activation    : Activation name resolved by ``LayerFactory.build_activation``.
            drop_path_rate    : Stochastic depth probability for both residual branches.
        """
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
        """Applies the attention and feed-forward residual branches.

        Args:
            x : Token sequence of shape (batch, sequence_length, embedding_dim).

        Returns:
            Token sequence of the same shape as ``x``.
        """
        normalized = self.norm_1(x)
        x          = x + self.drop_path(self.attention(normalized))
        normalized = self.norm_2(x)
        x          = x + self.drop_path(self.feed_forward(normalized))
        return x


class PatchEmbedding(nn.Module):
    """Non-overlapping patch tokenizer for transformer trunks.

    Attributes:
        patch_size : Side length of a patch, in pixels.
        projection : Strided convolution mapping each patch to a token.
        norm       : Layer normalization applied to the tokens.
    """

    def __init__(
        self,
        in_channels:   int,
        embedding_dim: int,
        patch_size:    int,
    ):
        """Builds the patch projection and token normalization.

        Args:
            in_channels   : Channel width of the incoming feature map.
            embedding_dim : Token width produced by the projection.
            patch_size    : Side length of a patch; also the projection stride.
        """
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
        """Tokenizes a feature map and reports the token grid shape.

        Args:
            x : Feature map of shape (batch, in_channels, height, width).

        Returns:
            Tuple of the normalized tokens of shape
            (batch, grid_height * grid_width, embedding_dim) and the token grid
            height and width.
        """
        x = self.projection(x)
        grid_height, grid_width = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, grid_height, grid_width

