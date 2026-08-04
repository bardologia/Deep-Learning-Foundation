from __future__ import annotations

import torch
import torch.nn as nn

from configuration.architectures   import ViTImageAutoencoderConfig
from models.image_autoencoder.base import ImageAutoencoderBase
from models.blocks                 import PatchEmbedding, TransformerBlock, tokens_to_feature_map


class ViTImageEncoder(nn.Module):
    def __init__(self, config: ViTImageAutoencoderConfig) -> None:
        super().__init__()
        self.patch_size   = config.patch_size
        self.patch_embed  = PatchEmbedding(config.in_channels, config.hidden_dim, config.patch_size)
        self.pos_conv     = nn.Conv2d(config.hidden_dim, config.hidden_dim, kernel_size=3, padding=1, groups=config.hidden_dim)
        self.blocks       = nn.ModuleList([
            TransformerBlock(config.hidden_dim, config.num_heads, config.mlp_ratio, dropout=config.dropout, ffn_activation=config.activation)
            for _ in range(max(1, config.depth))
        ])
        self.norm         = nn.LayerNorm(config.hidden_dim)
        self.to_embedding = nn.Conv2d(config.hidden_dim, config.embedding_dim, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2], image.shape[-1]
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(f"ViT autoencoder input {height}x{width} is not divisible by patch_size={self.patch_size}; the patch embedding would silently drop border rows and columns.")

        tokens, gh, gw = self.patch_embed(image)
        tokens = tokens + self.pos_conv(tokens_to_feature_map(tokens, gh, gw)).flatten(2).transpose(1, 2)

        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)
        return self.to_embedding(tokens_to_feature_map(tokens, gh, gw))


class ViTImageDecoder(nn.Module):
    def __init__(self, config: ViTImageAutoencoderConfig) -> None:
        super().__init__()
        self.from_embedding = nn.Conv2d(config.embedding_dim, config.hidden_dim, kernel_size=1)
        self.pos_conv       = nn.Conv2d(config.hidden_dim, config.hidden_dim, kernel_size=3, padding=1, groups=config.hidden_dim)
        self.blocks         = nn.ModuleList([
            TransformerBlock(config.hidden_dim, config.num_heads, config.mlp_ratio, dropout=config.dropout, ffn_activation=config.activation)
            for _ in range(max(1, config.depth))
        ])
        self.norm           = nn.LayerNorm(config.hidden_dim)
        self.unpatch        = nn.ConvTranspose2d(config.hidden_dim, config.in_channels, kernel_size=config.patch_size, stride=config.patch_size)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        feats  = self.from_embedding(z)
        gh, gw = feats.shape[-2], feats.shape[-1]
        tokens = feats.flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_conv(feats).flatten(2).transpose(1, 2)

        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)
        return self.unpatch(tokens_to_feature_map(tokens, gh, gw))


class ViTImageAutoencoder(ImageAutoencoderBase):
    def __init__(self, config: ViTImageAutoencoderConfig | None = None) -> None:
        config = config if config is not None else ViTImageAutoencoderConfig()
        if config.normalization != "layernorm":
            raise ValueError(f"vit_ae hardcodes LayerNorm and does not honor normalization='{config.normalization}'; leave the field at 'layernorm'.")

        super().__init__(config, ViTImageEncoder(config), ViTImageDecoder(config))
