"""Profile autoencoder built from pre-norm residual MLP blocks.

Each z profile is embedded into a hidden vector, refined by a stack of
residual blocks, and projected to the latent code or back to profile length.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from configuration.architectures     import ResMlpAutoencoderConfig
from models.profile_autoencoder.base import ProfileAutoencoderBase, ProfileAutoencoderBlocks
from models.blocks                   import LayerFactory


class ResidualMlpBlock(nn.Module):
    """Pre-norm residual block of two linear layers.

    Attributes:
        norm : Normalization applied before the block body, an identity when the configuration disables it.
        body : Linear, activation, dropout, linear stack.
    """

    def __init__(self, dim: int, activation: str, dropout: float, normalization: str) -> None:
        """Builds the normalization and two-layer body.

        Args:
            dim           : Feature width, identical on input and output.
            activation    : Name of the activation resolved by ``LayerFactory``.
            dropout       : Dropout probability applied between the linear layers.
            normalization : Pre-norm mode, one of layernorm or none.

        Raises:
            ValueError : If the normalization mode is unsupported.
        """
        super().__init__()
        if normalization not in ("layernorm", "none"):
            raise ValueError(f"resmlp_ae supports normalization modes ['layernorm', 'none'] only; got '{normalization}'.")

        self.norm = nn.LayerNorm(dim) if normalization == "layernorm" else nn.Identity()
        self.body = nn.Sequential(
            nn.Linear(dim, dim),
            LayerFactory.build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the residual sum of the input of shape (pixels, dim) and its transformed copy."""
        return x + self.body(self.norm(x))


class ResMlpEncoder(nn.Module):
    """Residual-MLP encoder mapping a profile to a latent code.

    Attributes:
        embed  : Linear embedding from profile length to hidden width.
        blocks : Stack of residual MLP blocks.
        head   : Linear projection to the latent dimension.
    """

    def __init__(self, config: ResMlpAutoencoderConfig) -> None:
        """Builds the embedding, residual stack and latent head.

        Args:
            config : Architecture configuration supplying profile length, hidden dimension, embedding dimension, depth, activation and dropout.
        """
        super().__init__()
        self.embed  = nn.Linear(config.profile_length, config.hidden_dim)
        self.blocks = nn.Sequential(*[
            ResidualMlpBlock(config.hidden_dim, config.activation, config.dropout, config.normalization)
            for _ in range(max(1, config.depth))
        ])
        self.head   = nn.Linear(config.hidden_dim, config.embedding_dim)

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        """Encodes profiles into latent codes.

        Args:
            curve : Profile map of shape (batch, profile_length, height, width).

        Returns:
            Latent map of shape (batch, embedding_dim, height, width).
        """
        seq, dims = ProfileAutoencoderBlocks.to_sequence(curve)
        x = self.embed(seq)
        x = self.blocks(x)
        z = self.head(x)
        return ProfileAutoencoderBlocks.from_sequence(z, dims)


class ResMlpDecoder(nn.Module):
    """Residual-MLP decoder mapping a latent code back to a profile.

    Attributes:
        embed  : Linear embedding from the latent dimension to hidden width.
        blocks : Stack of residual MLP blocks.
        head   : Linear projection to the profile length.
    """

    def __init__(self, config: ResMlpAutoencoderConfig) -> None:
        """Builds the latent embedding, residual stack and profile head.

        Args:
            config : Architecture configuration supplying profile length, hidden dimension, embedding dimension, depth, activation and dropout.
        """
        super().__init__()
        self.embed  = nn.Linear(config.embedding_dim, config.hidden_dim)
        self.blocks = nn.Sequential(*[
            ResidualMlpBlock(config.hidden_dim, config.activation, config.dropout, config.normalization)
            for _ in range(max(1, config.depth))
        ])
        self.head   = nn.Linear(config.hidden_dim, config.profile_length)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes latent codes back into profiles.

        Args:
            z : Latent map of shape (batch, embedding_dim, height, width).

        Returns:
            Profile map of shape (batch, profile_length, height, width).
        """
        seq, dims = ProfileAutoencoderBlocks.to_sequence(z)
        x     = self.embed(seq)
        x     = self.blocks(x)
        curve = self.head(x)
        return ProfileAutoencoderBlocks.from_sequence(curve, dims)


class ResMlpAutoencoder(ProfileAutoencoderBase):
    """Profile autoencoder pairing the residual-MLP encoder and decoder."""

    def __init__(self, config: ResMlpAutoencoderConfig | None = None) -> None:
        """Assembles the autoencoder, using default settings when config is None."""
        config = config if config is not None else ResMlpAutoencoderConfig()
        super().__init__(config, ResMlpEncoder(config), ResMlpDecoder(config))
