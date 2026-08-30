"""Profile autoencoder built from per-pixel multilayer perceptrons.

Both halves are stacks of 1x1 convolutions, so the whole z profile is
treated as the channel dimension and every pixel is compressed independently.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from configuration.architectures     import MlpAutoencoderConfig
from models.profile_autoencoder.base import ProfileAutoencoderBase, ProfileAutoencoderBlocks


class MlpEncoder(nn.Module):
    """Per-pixel MLP mapping a profile to a latent code.

    Attributes:
        net : 1x1 convolution stack from profile_length to embedding_dim.
    """

    def __init__(self, config: MlpAutoencoderConfig) -> None:
        """Builds the encoder MLP.

        Args:
            config : Architecture configuration supplying profile length, hidden widths, embedding dimension, depth, activation, dropout and normalization.
        """
        super().__init__()
        self.net = ProfileAutoencoderBlocks.mlp_stack(
            in_ch         = config.profile_length,
            widths        = ProfileAutoencoderBlocks.hidden_widths(config.hidden_dims, config.hidden_dim, config.depth),
            out_ch        = config.embedding_dim,
            activation    = config.activation,
            dropout       = config.dropout,
            normalization = config.normalization,
        )

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        """Maps a profile map (batch, profile_length, height, width) to a latent map (batch, embedding_dim, height, width)."""
        return self.net(curve)


class MlpDecoder(nn.Module):
    """Per-pixel MLP mapping a latent code back to a profile.

    Attributes:
        net : 1x1 convolution stack from embedding_dim to profile_length.
    """

    def __init__(self, config: MlpAutoencoderConfig) -> None:
        """Builds the decoder MLP.

        Args:
            config : Architecture configuration supplying profile length, hidden widths, embedding dimension, depth, activation, dropout and normalization.
        """
        super().__init__()
        self.net = ProfileAutoencoderBlocks.mlp_stack(
            in_ch         = config.embedding_dim,
            widths        = list(reversed(ProfileAutoencoderBlocks.hidden_widths(config.hidden_dims, config.hidden_dim, config.depth))),
            out_ch        = config.profile_length,
            activation    = config.activation,
            dropout       = config.dropout,
            normalization = config.normalization,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Maps a latent map (batch, embedding_dim, height, width) to a profile map (batch, profile_length, height, width)."""
        return self.net(z)


class MlpAutoencoder(ProfileAutoencoderBase):
    """Profile autoencoder pairing the per-pixel MLP encoder and decoder."""

    def __init__(self, config: MlpAutoencoderConfig | None = None) -> None:
        """Assembles the autoencoder, using default settings when config is None."""
        config = config if config is not None else MlpAutoencoderConfig()
        super().__init__(config, MlpEncoder(config), MlpDecoder(config))
