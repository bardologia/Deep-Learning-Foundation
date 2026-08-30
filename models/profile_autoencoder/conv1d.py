"""Profile autoencoder built from plain 1-D convolutions.

The encoder convolves each elevation profile and pools it to a latent vector;
the decoder projects the code back to a full-length feature sequence and
convolves it down to a single reconstructed profile.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from configuration.architectures     import Conv1dAutoencoderConfig
from models.profile_autoencoder.base import ProfileAutoencoderBase, ProfileAutoencoderBlocks
from models.blocks                   import LayerFactory


class Conv1dStack:
    """Constructor of the length-preserving 1-D convolution body."""

    @staticmethod
    def channels(config: Conv1dAutoencoderConfig) -> list[int]:
        """Returns the per-layer channel widths of the encoder body."""
        return list(config.seq_channels_list) if config.seq_channels_list else [config.seq_channels] * max(1, config.depth)

    @staticmethod
    def body(config: Conv1dAutoencoderConfig, widths: list[int], in_ch: int) -> nn.Sequential:
        """Builds the convolution, normalization, activation and dropout stack.

        Args:
            config : Architecture configuration supplying kernel size, activation, normalization and dropout.
            widths : Output channel width of each convolution layer.
            in_ch  : Channel width of the incoming features.

        Returns:
            Sequential module preserving the length axis of features of shape
            (pixels, channels, length).

        Raises:
            ValueError : If the normalization mode is unsupported.
        """
        pad    = config.seq_kernel_size // 2
        layers = []
        prev   = in_ch

        for width in widths:
            layers.append(nn.Conv1d(prev, width, config.seq_kernel_size, padding=pad))

            norm = ProfileAutoencoderBlocks.norm1d(config.normalization, width, "conv1d_ae", ("none", "batchnorm"))
            if norm is not None:
                layers.append(norm)

            layers.append(LayerFactory.build_activation(config.activation))
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            prev = width

        return nn.Sequential(*layers)


class Conv1dEncoder(nn.Module):
    """1-D convolutional encoder pooling each profile to a latent code.

    Attributes:
        body : Convolution, normalization, activation and dropout stack.
        pool : Length-axis pooling to a single feature vector.
        head : Linear projection of the pooled features to the latent dimension.
    """

    def __init__(self, config: Conv1dAutoencoderConfig) -> None:
        """Builds the convolutional body, pooling and latent head.

        Args:
            config : Architecture configuration supplying channel widths, kernel size, depth, activation, normalization, dropout, pooling and embedding dimension.
        """
        super().__init__()
        widths    = Conv1dStack.channels(config)
        self.body = Conv1dStack.body(config, widths, in_ch=1)
        self.pool = ProfileAutoencoderBlocks.pool1d(config.pooling, widths[-1], "conv1d_ae")
        self.head = nn.Linear(widths[-1], config.embedding_dim)

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        """Encodes profiles into latent codes.

        Args:
            curve : Profile map of shape (batch, profile_length, height, width).

        Returns:
            Latent map of shape (batch, embedding_dim, height, width).
        """
        seq, dims = ProfileAutoencoderBlocks.to_sequence(curve)
        feats = self.pool(self.body(seq.unsqueeze(1)))
        z     = self.head(feats.squeeze(-1))
        return ProfileAutoencoderBlocks.from_sequence(z, dims)


class Conv1dDecoder(nn.Module):
    """1-D convolutional decoder expanding a latent code to a profile.

    Attributes:
        length         : Profile length in elevation bins.
        first_channels : Channel width the latent code is lifted to.
        project        : Linear lift of the latent code to a full-length feature map.
        body           : Convolution stack over the reversed encoder widths, ending in a single output channel.
    """

    def __init__(self, config: Conv1dAutoencoderConfig) -> None:
        """Builds the latent projection and convolutional body.

        Args:
            config : Architecture configuration supplying profile length, channel widths, kernel size, depth, activation, normalization, dropout and embedding dimension.
        """
        super().__init__()
        widths              = list(reversed(Conv1dStack.channels(config)))
        pad                 = config.seq_kernel_size // 2
        self.length         = config.profile_length
        self.first_channels = widths[0]
        self.project        = nn.Linear(config.embedding_dim, widths[0] * config.profile_length)

        hidden = Conv1dStack.body(config, widths[1:], in_ch=widths[0])
        final  = nn.Conv1d(widths[-1], 1, config.seq_kernel_size, padding=pad)
        self.body = nn.Sequential(*hidden, final)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes latent codes back into profiles.

        Args:
            z : Latent map of shape (batch, embedding_dim, height, width).

        Returns:
            Profile map of shape (batch, profile_length, height, width).
        """
        seq, dims = ProfileAutoencoderBlocks.to_sequence(z)
        feats = self.project(seq).reshape(-1, self.first_channels, self.length)
        curve = self.body(feats).squeeze(1)
        return ProfileAutoencoderBlocks.from_sequence(curve, dims)


class Conv1dAutoencoder(ProfileAutoencoderBase):
    """Profile autoencoder pairing the 1-D convolutional encoder and decoder."""

    def __init__(self, config: Conv1dAutoencoderConfig | None = None) -> None:
        """Assembles the autoencoder, using default settings when config is None."""
        config = config if config is not None else Conv1dAutoencoderConfig()
        super().__init__(config, Conv1dEncoder(config), Conv1dDecoder(config))
