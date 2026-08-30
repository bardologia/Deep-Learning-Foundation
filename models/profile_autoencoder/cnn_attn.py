"""Profile autoencoder combining a 1-D convolutional stem with self-attention.

The encoder convolves each elevation profile, splits it into patch tokens and
attends over them before mean-pooling to a latent code; the decoder broadcasts
the code back to tokens and reassembles the profile from decoded patches.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from configuration.architectures     import CnnAttnAutoencoderConfig
from models.profile_autoencoder.base import ProfileAutoencoderBase, ProfileAutoencoderBlocks
from models.blocks                   import LayerFactory, TransformerBlock


class CnnAttnPositional:
    """Constructor of the token positional embedding shared by both halves."""

    @staticmethod
    def build(module: nn.Module, config: CnnAttnAutoencoderConfig, num_tokens: int) -> None:
        """Attaches the configured positional embedding as ``pos_embed``.

        A learned embedding is a zero-initialised parameter, the sinusoidal
        variant a fixed non-persistent buffer, and ``none`` leaves the
        attribute at None so the addition is skipped.

        Args:
            module     : Encoder or decoder receiving the attribute.
            config     : Architecture configuration supplying pos_embedding and hidden_dim.
            num_tokens : Number of patch tokens per profile.

        Raises:
            ValueError : If the positional embedding mode is unknown.
        """
        if config.pos_embedding == "learned":
            module.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, config.hidden_dim))
        elif config.pos_embedding == "sinusoidal":
            module.register_buffer("pos_embed", LayerFactory.sinusoidal_position_table(num_tokens, config.hidden_dim), persistent=False)
        elif config.pos_embedding == "none":
            module.pos_embed = None
        else:
            raise ValueError(f"cnn_attn_ae supports pos_embedding modes ['learned', 'sinusoidal', 'none'] only; got '{config.pos_embedding}'.")


class CnnAttnEncoder(nn.Module):
    """Convolutional-stem transformer encoder over elevation profiles.

    Attributes:
        num_tokens : Number of patch tokens per profile.
        stem       : 1-D convolution and activation applied to the raw profile.
        tokenize   : Strided convolution turning the stem output into tokens.
        pos_embed  : Positional embedding of shape (1, num_tokens, hidden_dim), or None when disabled.
        blocks     : Stack of transformer blocks over the tokens.
        pool       : Token pooling to a single feature vector.
        head       : Linear projection of the pooled token to the latent dimension.
    """

    def __init__(self, config: CnnAttnAutoencoderConfig) -> None:
        """Builds the convolutional stem, tokeniser and attention stack.

        Args:
            config : Architecture configuration supplying profile length, patch size, channel widths, depth, head count and dropout.
        """
        super().__init__()
        pad             = config.seq_kernel_size // 2
        self.num_tokens = config.profile_length // config.patch_size

        self.stem = nn.Sequential(
            nn.Conv1d(1, config.seq_channels, config.seq_kernel_size, padding=pad),
            LayerFactory.build_activation(config.activation),
        )
        self.tokenize  = nn.Conv1d(config.seq_channels, config.hidden_dim, config.patch_size, stride=config.patch_size)
        CnnAttnPositional.build(self, config, self.num_tokens)
        self.blocks    = nn.Sequential(*[
            TransformerBlock(config.hidden_dim, config.num_heads, mlp_ratio=config.ffn_ratio, dropout=config.dropout, ffn_activation=config.activation)
            for _ in range(max(1, config.depth))
        ])
        self.pool      = ProfileAutoencoderBlocks.token_pool(config.pooling, config.hidden_dim, "cnn_attn_ae")
        self.head      = nn.Linear(config.hidden_dim, config.embedding_dim)

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        """Encodes profiles into latent codes.

        Args:
            curve : Profile map of shape (batch, profile_length, height, width).

        Returns:
            Latent map of shape (batch, embedding_dim, height, width).
        """
        seq, dims = ProfileAutoencoderBlocks.to_sequence(curve)
        feats  = self.stem(seq.unsqueeze(1))
        tokens = self.tokenize(feats).transpose(1, 2)
        if self.pos_embed is not None:
            tokens = tokens + self.pos_embed
        tokens = self.blocks(tokens)
        z      = self.head(self.pool(tokens))
        return ProfileAutoencoderBlocks.from_sequence(z, dims)


class CnnAttnDecoder(nn.Module):
    """Transformer decoder reassembling a profile from a latent code.

    Attributes:
        num_tokens : Number of patch tokens per profile.
        project    : Linear lift from the latent dimension to the hidden width.
        pos_embed  : Positional embedding of shape (1, num_tokens, hidden_dim), or None when disabled.
        blocks     : Stack of transformer blocks over the tokens.
        to_patches : Linear projection of each token to a profile patch.
        head       : Linear layer mapping the concatenated patches to the profile.
    """

    def __init__(self, config: CnnAttnAutoencoderConfig) -> None:
        """Builds the token expansion, attention stack and patch reassembly.

        Args:
            config : Architecture configuration supplying profile length, patch size, hidden and embedding dimensions, depth, head count and dropout.
        """
        super().__init__()
        self.num_tokens = config.profile_length // config.patch_size

        self.project    = nn.Linear(config.embedding_dim, config.hidden_dim)
        CnnAttnPositional.build(self, config, self.num_tokens)
        self.blocks     = nn.Sequential(*[
            TransformerBlock(config.hidden_dim, config.num_heads, mlp_ratio=config.ffn_ratio, dropout=config.dropout, ffn_activation=config.activation)
            for _ in range(max(1, config.depth))
        ])
        self.to_patches = nn.Linear(config.hidden_dim, config.patch_size)
        self.head       = nn.Linear(self.num_tokens * config.patch_size, config.profile_length)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes latent codes back into profiles.

        Args:
            z : Latent map of shape (batch, embedding_dim, height, width).

        Returns:
            Profile map of shape (batch, profile_length, height, width).
        """
        seq, dims = ProfileAutoencoderBlocks.to_sequence(z)
        tokens  = self.project(seq).unsqueeze(1).expand(-1, self.num_tokens, -1)
        if self.pos_embed is not None:
            tokens = tokens + self.pos_embed
        tokens  = self.blocks(tokens)
        patches = self.to_patches(tokens).reshape(seq.shape[0], -1)
        curve   = self.head(patches)
        return ProfileAutoencoderBlocks.from_sequence(curve, dims)


class CnnAttnAutoencoder(ProfileAutoencoderBase):
    """Profile autoencoder pairing the CNN-attention encoder and decoder."""

    def __init__(self, config: CnnAttnAutoencoderConfig | None = None) -> None:
        """Assembles the autoencoder, using default settings when config is None."""
        config = config if config is not None else CnnAttnAutoencoderConfig()
        super().__init__(config, CnnAttnEncoder(config), CnnAttnDecoder(config))
