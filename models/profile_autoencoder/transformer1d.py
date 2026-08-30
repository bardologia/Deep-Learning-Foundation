"""Profile autoencoder wrapping the whole profile in a single attention token.

Each z profile is embedded into one token, passed through a stack of
``nn.TransformerEncoderLayer`` blocks, and projected to the latent code or back
to profile length, so attention acts as a gated feed-forward refinement.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from configuration.architectures     import Transformer1dAutoencoderConfig
from models.profile_autoencoder.base import ProfileAutoencoderBase, ProfileAutoencoderBlocks


class Transformer1dEncoder(nn.Module):
    """Transformer encoder mapping a profile token to a latent code.

    Attributes:
        embed   : Linear embedding from profile length to hidden width.
        encoder : Stack of transformer encoder layers over the single token.
        head    : Linear projection to the latent dimension.
    """

    def __init__(self, config: Transformer1dAutoencoderConfig) -> None:
        """Builds the token embedding, transformer stack and latent head.

        Args:
            config : Architecture configuration supplying profile length, hidden dimension, head count, depth, dropout, activation, feed-forward ratio and embedding dimension.
        """
        super().__init__()
        self.embed = nn.Linear(config.profile_length, config.hidden_dim)
        layer      = nn.TransformerEncoderLayer(
            d_model         = config.hidden_dim,
            nhead           = config.num_heads,
            dim_feedforward = int(config.hidden_dim * config.ffn_ratio),
            dropout         = config.dropout,
            activation      = config.activation,
            batch_first     = True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, config.depth))
        self.head    = nn.Linear(config.hidden_dim, config.embedding_dim)

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        """Encodes profiles into latent codes.

        Args:
            curve : Profile map of shape (batch, profile_length, height, width).

        Returns:
            Latent map of shape (batch, embedding_dim, height, width).
        """
        seq, dims = ProfileAutoencoderBlocks.to_sequence(curve)
        tokens   = self.embed(seq).unsqueeze(1)
        attended = self.encoder(tokens).squeeze(1)
        z        = self.head(attended)
        return ProfileAutoencoderBlocks.from_sequence(z, dims)


class Transformer1dDecoder(nn.Module):
    """Transformer decoder mapping a latent token back to a profile.

    Attributes:
        embed   : Linear embedding from the latent dimension to hidden width.
        decoder : Stack of transformer encoder layers over the single token.
        head    : Linear projection to the profile length.
    """

    def __init__(self, config: Transformer1dAutoencoderConfig) -> None:
        """Builds the latent embedding, transformer stack and profile head.

        Args:
            config : Architecture configuration supplying profile length, hidden dimension, head count, depth, dropout, activation, feed-forward ratio and embedding dimension.
        """
        super().__init__()
        self.embed = nn.Linear(config.embedding_dim, config.hidden_dim)
        layer      = nn.TransformerEncoderLayer(
            d_model         = config.hidden_dim,
            nhead           = config.num_heads,
            dim_feedforward = int(config.hidden_dim * config.ffn_ratio),
            dropout         = config.dropout,
            activation      = config.activation,
            batch_first     = True,
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=max(1, config.depth))
        self.head    = nn.Linear(config.hidden_dim, config.profile_length)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes latent codes back into profiles.

        Args:
            z : Latent map of shape (batch, embedding_dim, height, width).

        Returns:
            Profile map of shape (batch, profile_length, height, width).
        """
        seq, dims = ProfileAutoencoderBlocks.to_sequence(z)
        tokens   = self.embed(seq).unsqueeze(1)
        attended = self.decoder(tokens).squeeze(1)
        curve    = self.head(attended)
        return ProfileAutoencoderBlocks.from_sequence(curve, dims)


class Transformer1dAutoencoder(ProfileAutoencoderBase):
    """Profile autoencoder pairing the single-token transformer halves."""

    def __init__(self, config: Transformer1dAutoencoderConfig | None = None) -> None:
        """Assembles the autoencoder and validates the activation choice.

        Args:
            config : Architecture configuration; defaults are used when None.

        Raises:
            ValueError : If the activation is neither 'relu' nor 'gelu', the only options ``nn.TransformerEncoderLayer`` accepts.
        """
        config = config if config is not None else Transformer1dAutoencoderConfig()
        if config.activation not in ("relu", "gelu"):
            raise ValueError(f"transformer1d_ae supports activations relu and gelu only (nn.TransformerEncoderLayer constraint); got '{config.activation}'.")

        super().__init__(config, Transformer1dEncoder(config), Transformer1dDecoder(config))
