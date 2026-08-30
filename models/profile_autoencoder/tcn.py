"""Temporal convolutional profile autoencoder with exponentially dilated blocks.

Dilations double with depth, so the receptive field over the elevation axis
grows geometrically while the profile length stays fixed throughout the stack.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from configuration.architectures     import TcnAutoencoderConfig
from models.profile_autoencoder.base import ProfileAutoencoderBase, ProfileAutoencoderBlocks
from models.blocks                   import LayerFactory


class DilatedResidualBlock(nn.Module):
    """Residual block of two dilated 1-D convolutions with post-activation.

    Attributes:
        body : Dilated convolution, optional normalization, activation, dropout, dilated convolution.
        act  : Activation applied after the residual sum.
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int, activation: str, dropout: float, normalization: str) -> None:
        """Builds the dilated convolution pair with length-preserving padding.

        Args:
            channels      : Channel width, identical on input and output.
            kernel_size   : Convolution kernel length along the elevation axis.
            dilation      : Spacing between kernel taps.
            activation    : Name of the activation resolved by ``LayerFactory``.
            dropout       : Dropout probability between the two convolutions.
            normalization : Sequence normalization after each dilated convolution, one of none or batchnorm.

        Raises:
            ValueError : If the normalization mode is unsupported.
        """
        super().__init__()
        pad    = dilation * (kernel_size - 1) // 2
        layers = [nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)]

        norm = ProfileAutoencoderBlocks.norm1d(normalization, channels, "tcn_ae", ("none", "batchnorm"))
        if norm is not None:
            layers.append(norm)

        layers.append(LayerFactory.build_activation(activation))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation))

        second_norm = ProfileAutoencoderBlocks.norm1d(normalization, channels, "tcn_ae", ("none", "batchnorm"))
        if second_norm is not None:
            layers.append(second_norm)

        self.body = nn.Sequential(*layers)
        self.act  = LayerFactory.build_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the activated residual sum for features of shape (pixels, channels, length)."""
        return self.act(x + self.body(x))


class TcnEncoder(nn.Module):
    """Dilated convolutional encoder pooling a profile to a latent code.

    Attributes:
        stem : Convolution lifting the single-channel profile to seq_channels.
        body : Stack of dilated residual blocks with dilations 1, 2, 4, ...
        pool : Length-axis pooling to a single feature vector.
        head : Linear projection of the pooled features to the latent dimension.
    """

    def __init__(self, config: TcnAutoencoderConfig) -> None:
        """Builds the stem, dilated stack and latent head.

        Args:
            config : Architecture configuration supplying channel width, kernel size, depth, activation, dropout, normalization, pooling and embedding dimension.
        """
        super().__init__()
        pad       = config.seq_kernel_size // 2
        self.stem = nn.Conv1d(1, config.seq_channels, config.seq_kernel_size, padding=pad)
        self.body = nn.Sequential(*[
            DilatedResidualBlock(config.seq_channels, config.seq_kernel_size, 2 ** i, config.activation, config.dropout, config.normalization)
            for i in range(max(1, config.depth))
        ])
        self.pool = ProfileAutoencoderBlocks.pool1d(config.pooling, config.seq_channels, "tcn_ae")
        self.head = nn.Linear(config.seq_channels, config.embedding_dim)

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        """Encodes profiles into latent codes.

        Args:
            curve : Profile map of shape (batch, profile_length, height, width).

        Returns:
            Latent map of shape (batch, embedding_dim, height, width).
        """
        seq, dims = ProfileAutoencoderBlocks.to_sequence(curve)
        feats = self.stem(seq.unsqueeze(1))
        feats = self.body(feats)
        z     = self.head(self.pool(feats).squeeze(-1))
        return ProfileAutoencoderBlocks.from_sequence(z, dims)


class TcnDecoder(nn.Module):
    """Dilated convolutional decoder expanding a latent code to a profile.

    Attributes:
        length       : Profile length in elevation bins.
        seq_channels : Channel width of the dilated stack.
        project      : Linear lift of the latent code to a full-length feature map.
        body         : Stack of dilated residual blocks with dilations 1, 2, 4, ...
        head         : Convolution collapsing the features to one output channel.
    """

    def __init__(self, config: TcnAutoencoderConfig) -> None:
        """Builds the latent projection, dilated stack and output convolution.

        Args:
            config : Architecture configuration supplying profile length, channel width, kernel size, depth, activation, dropout and embedding dimension.
        """
        super().__init__()
        self.length       = config.profile_length
        self.seq_channels = config.seq_channels
        pad               = config.seq_kernel_size // 2
        self.project      = nn.Linear(config.embedding_dim, config.seq_channels * config.profile_length)
        self.body         = nn.Sequential(*[
            DilatedResidualBlock(config.seq_channels, config.seq_kernel_size, 2 ** i, config.activation, config.dropout, config.normalization)
            for i in range(max(1, config.depth))
        ])
        self.head         = nn.Conv1d(config.seq_channels, 1, config.seq_kernel_size, padding=pad)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes latent codes back into profiles.

        Args:
            z : Latent map of shape (batch, embedding_dim, height, width).

        Returns:
            Profile map of shape (batch, profile_length, height, width).
        """
        seq, dims = ProfileAutoencoderBlocks.to_sequence(z)
        feats = self.project(seq).reshape(-1, self.seq_channels, self.length)
        feats = self.body(feats)
        curve = self.head(feats).squeeze(1)
        return ProfileAutoencoderBlocks.from_sequence(curve, dims)


class TcnAutoencoder(ProfileAutoencoderBase):
    """Profile autoencoder pairing the dilated convolutional encoder and decoder."""

    def __init__(self, config: TcnAutoencoderConfig | None = None) -> None:
        """Assembles the autoencoder, using default settings when config is None."""
        config = config if config is not None else TcnAutoencoderConfig()
        super().__init__(config, TcnEncoder(config), TcnDecoder(config))
