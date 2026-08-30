"""Vision-transformer image autoencoder over tomographic slice stacks.

The encoder tokenises the input with a patch embedding, adds a depthwise
convolutional positional term, and projects the transformer output to a latent
feature map; the decoder mirrors it and unpatches back to image resolution.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from configuration.architectures   import ImageAutoencoderBaseConfig, ViTImageAutoencoderConfig
from models.image_autoencoder.base import ImageAutoencoderBase
from models.blocks                 import FeatureGeometry, PatchEmbedding, TransformerBlock


class ViTImageEncoder(nn.Module):
    """Patch-embedding transformer encoder producing a latent feature map.

    Attributes:
        patch_size   : Side length in pixels of the square token patches.
        patch_embed  : Convolutional patch tokeniser.
        pos_conv     : Depthwise convolution supplying positional information.
        blocks       : Stack of transformer blocks applied to the token sequence.
        norm         : Final token LayerNorm.
        to_embedding : 1x1 convolution mapping hidden channels to the latent dim.
    """

    def __init__(self, config: ViTImageAutoencoderConfig) -> None:
        """Builds the encoder stack from the autoencoder configuration.

        Args:
            config : ViT autoencoder configuration supplying patch size, hidden and embedding dimensions, depth, head count and dropout.
        """
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
        """Encodes an image batch into a latent feature map.

        Args:
            image : Input stack of shape (batch, in_channels, height, width); height and width must both be multiples of the patch size.

        Returns:
            Latent map of shape (batch, embedding_dim, height / patch_size,
            width / patch_size).

        Raises:
            ValueError : If the spatial dimensions are not divisible by the patch size, which would silently crop border rows and columns.
        """
        height, width = image.shape[-2], image.shape[-1]
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(f"ViT autoencoder input {height}x{width} is not divisible by patch_size={self.patch_size}; the patch embedding would silently drop border rows and columns.")

        tokens, gh, gw = self.patch_embed(image)
        tokens = tokens + self.pos_conv(FeatureGeometry.tokens_to_feature_map(tokens, gh, gw)).flatten(2).transpose(1, 2)

        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)
        return self.to_embedding(FeatureGeometry.tokens_to_feature_map(tokens, gh, gw))


class ViTImageDecoder(nn.Module):
    """Transformer decoder unpatching a latent map back to image resolution.

    Attributes:
        from_embedding : 1x1 convolution lifting the latent dim to hidden width.
        pos_conv       : Depthwise convolution supplying positional information.
        blocks         : Stack of transformer blocks applied to the token sequence.
        norm           : Final token LayerNorm.
        unpatch        : Transposed convolution expanding each token to a patch.
    """

    def __init__(self, config: ViTImageAutoencoderConfig) -> None:
        """Builds the decoder stack from the autoencoder configuration.

        Args:
            config : ViT autoencoder configuration supplying patch size, hidden and embedding dimensions, depth, head count and dropout.
        """
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
        """Decodes a latent feature map back to an image batch.

        Args:
            z : Latent map of shape (batch, embedding_dim, grid_h, grid_w).

        Returns:
            Reconstruction of shape (batch, in_channels, grid_h * patch_size,
            grid_w * patch_size).
        """
        feats  = self.from_embedding(z)
        gh, gw = feats.shape[-2], feats.shape[-1]
        tokens = feats.flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_conv(feats).flatten(2).transpose(1, 2)

        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)
        return self.unpatch(FeatureGeometry.tokens_to_feature_map(tokens, gh, gw))


class ViTImageAutoencoder(ImageAutoencoderBase):
    """Image autoencoder pairing the ViT encoder and decoder."""

    def __init__(self, config: ViTImageAutoencoderConfig | None = None) -> None:
        """Assembles the ViT autoencoder and rejects unsupported config fields.

        Args:
            config : ViT autoencoder configuration; defaults are used when None.

        Raises:
            ValueError : If normalization is not 'layernorm', or if downsample_factor or base_channels deviate from their base defaults, since this architecture never reads them.
        """
        config = config if config is not None else ViTImageAutoencoderConfig()
        if config.normalization != "layernorm":
            raise ValueError(f"vit_ae hardcodes LayerNorm and does not honor normalization='{config.normalization}'; leave the field at 'layernorm'.")
        if config.downsample_factor != ImageAutoencoderBaseConfig.downsample_factor:
            raise ValueError(f"vit_ae reduces the grid by patch_size={config.patch_size} and never reads downsample_factor; leave the field at {ImageAutoencoderBaseConfig.downsample_factor}, got {config.downsample_factor}.")
        if config.base_channels != ImageAutoencoderBaseConfig.base_channels:
            raise ValueError(f"vit_ae sizes itself through hidden_dim and never reads base_channels; leave the field at {ImageAutoencoderBaseConfig.base_channels}, got {config.base_channels}.")

        super().__init__(config, ViTImageEncoder(config), ViTImageDecoder(config))
