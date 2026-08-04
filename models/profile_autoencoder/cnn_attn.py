from __future__ import annotations

import torch
import torch.nn as nn

from configuration.architectures     import CnnAttnAutoencoderConfig
from models.profile_autoencoder.base import ProfileAutoencoderBase, ProfileAutoencoderBlocks
from models.blocks                   import TransformerBlock, build_activation


class CnnAttnEncoder(nn.Module):
    def __init__(self, config: CnnAttnAutoencoderConfig) -> None:
        super().__init__()
        pad             = config.seq_kernel_size // 2
        self.num_tokens = config.profile_length // config.patch_size

        self.stem = nn.Sequential(
            nn.Conv1d(1, config.seq_channels, config.seq_kernel_size, padding=pad),
            build_activation(config.activation),
        )
        self.tokenize  = nn.Conv1d(config.seq_channels, config.hidden_dim, config.patch_size, stride=config.patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, config.hidden_dim))
        self.blocks    = nn.Sequential(*[
            TransformerBlock(config.hidden_dim, config.num_heads, dropout=config.dropout, ffn_activation=config.activation)
            for _ in range(max(1, config.depth))
        ])
        self.head      = nn.Linear(config.hidden_dim, config.embedding_dim)

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        seq, dims = ProfileAutoencoderBlocks.to_sequence(curve)
        feats  = self.stem(seq.unsqueeze(1))
        tokens = self.tokenize(feats).transpose(1, 2) + self.pos_embed
        tokens = self.blocks(tokens)
        z      = self.head(tokens.mean(dim=1))
        return ProfileAutoencoderBlocks.from_sequence(z, dims)


class CnnAttnDecoder(nn.Module):
    def __init__(self, config: CnnAttnAutoencoderConfig) -> None:
        super().__init__()
        self.num_tokens = config.profile_length // config.patch_size

        self.project    = nn.Linear(config.embedding_dim, config.hidden_dim)
        self.pos_embed  = nn.Parameter(torch.zeros(1, self.num_tokens, config.hidden_dim))
        self.blocks     = nn.Sequential(*[
            TransformerBlock(config.hidden_dim, config.num_heads, dropout=config.dropout, ffn_activation=config.activation)
            for _ in range(max(1, config.depth))
        ])
        self.to_patches = nn.Linear(config.hidden_dim, config.patch_size)
        self.head       = nn.Linear(self.num_tokens * config.patch_size, config.profile_length)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        seq, dims = ProfileAutoencoderBlocks.to_sequence(z)
        tokens  = self.project(seq).unsqueeze(1).expand(-1, self.num_tokens, -1) + self.pos_embed
        tokens  = self.blocks(tokens)
        patches = self.to_patches(tokens).reshape(seq.shape[0], -1)
        curve   = self.head(patches)
        return ProfileAutoencoderBlocks.from_sequence(curve, dims)


class CnnAttnAutoencoder(ProfileAutoencoderBase):
    def __init__(self, config: CnnAttnAutoencoderConfig | None = None) -> None:
        config = config if config is not None else CnnAttnAutoencoderConfig()
        super().__init__(config, CnnAttnEncoder(config), CnnAttnDecoder(config))
