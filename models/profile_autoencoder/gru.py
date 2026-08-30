"""Recurrent profile autoencoder built on gated recurrent units.

The encoder reads each elevation profile bin by bin and projects the final
hidden state to a latent code; the decoder broadcasts that code across the
profile length and unrolls a GRU to emit one amplitude per elevation bin.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from configuration.architectures     import GruAutoencoderConfig
from models.profile_autoencoder.base import ProfileAutoencoderBase, ProfileAutoencoderBlocks


class GruEncoder(nn.Module):
    """GRU encoder mapping an elevation profile to a latent code.

    Attributes:
        num_layers : Number of stacked GRU layers.
        directions : Two when the GRU is bidirectional, one otherwise.
        hidden_dim : Hidden width of each GRU direction.
        rnn        : The recurrent stack reading one elevation bin per step.
        head       : Linear projection of the final hidden state to the latent dim.
    """

    def __init__(self, config: GruAutoencoderConfig) -> None:
        """Builds the recurrent stack and latent head.

        Args:
            config : Architecture configuration supplying hidden dimension, depth, bidirectionality, dropout and embedding dimension.
        """
        super().__init__()
        self.num_layers  = max(1, config.depth)
        self.directions  = 2 if config.bidirectional else 1
        self.hidden_dim  = config.hidden_dim
        self.rnn         = nn.GRU(
            input_size    = 1,
            hidden_size   = config.hidden_dim,
            num_layers    = self.num_layers,
            batch_first   = True,
            bidirectional = config.bidirectional,
            dropout       = config.dropout if self.num_layers > 1 else 0.0,
        )
        self.head        = nn.Linear(config.hidden_dim * self.directions, config.embedding_dim)

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        """Encodes profiles into latent codes from the last GRU layer state.

        Args:
            curve : Profile map of shape (batch, profile_length, height, width).

        Returns:
            Latent map of shape (batch, embedding_dim, height, width).
        """
        seq, dims = ProfileAutoencoderBlocks.to_sequence(curve)
        _, h_n = self.rnn(seq.unsqueeze(-1))
        last   = h_n.view(self.num_layers, self.directions, seq.shape[0], self.hidden_dim)[-1]
        merged = last.permute(1, 0, 2).reshape(seq.shape[0], self.directions * self.hidden_dim)
        z      = self.head(merged)
        return ProfileAutoencoderBlocks.from_sequence(z, dims)


class GruDecoder(nn.Module):
    """GRU decoder unrolling a latent code into an elevation profile.

    Attributes:
        length     : Profile length in elevation bins.
        num_layers : Number of stacked GRU layers.
        project    : Linear lift of the latent code to the GRU input width.
        rnn        : Unidirectional recurrent stack unrolled over the profile length.
        head       : Linear layer emitting one amplitude per elevation bin.
    """

    def __init__(self, config: GruAutoencoderConfig) -> None:
        """Builds the latent projection, recurrent stack and output head.

        Args:
            config : Architecture configuration supplying profile length, hidden dimension, depth, dropout and embedding dimension.
        """
        super().__init__()
        self.length     = config.profile_length
        self.num_layers = max(1, config.depth)
        self.project    = nn.Linear(config.embedding_dim, config.hidden_dim)
        self.rnn        = nn.GRU(
            input_size  = config.hidden_dim,
            hidden_size = config.hidden_dim,
            num_layers  = self.num_layers,
            batch_first = True,
            dropout     = config.dropout if self.num_layers > 1 else 0.0,
        )
        self.head       = nn.Linear(config.hidden_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes latent codes back into profiles.

        Args:
            z : Latent map of shape (batch, embedding_dim, height, width).

        Returns:
            Profile map of shape (batch, profile_length, height, width).
        """
        seq, dims = ProfileAutoencoderBlocks.to_sequence(z)
        seed   = self.project(seq).unsqueeze(1).expand(-1, self.length, -1)
        out, _ = self.rnn(seed)
        curve  = self.head(out).squeeze(-1)
        return ProfileAutoencoderBlocks.from_sequence(curve, dims)


class GruAutoencoder(ProfileAutoencoderBase):
    """Profile autoencoder pairing the GRU encoder and decoder."""

    def __init__(self, config: GruAutoencoderConfig | None = None) -> None:
        """Assembles the autoencoder, using default settings when config is None."""
        config = config if config is not None else GruAutoencoderConfig()
        super().__init__(config, GruEncoder(config), GruDecoder(config))
