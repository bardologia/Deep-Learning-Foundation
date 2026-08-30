"""Common interface shared by every 2D image autoencoder in the model zoo."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import EmbeddingNorm, LayerFactory


class ImageAutoencoderBase(EmbeddingNorm, nn.Module):
    """Encoder-decoder pair with a normalized latent, shared by all image autoencoders.

    Subclasses only build the encoder and decoder; the embedding normalization, the
    weight initialization and the encode/decode interface live here.

    Attributes:
        config  : Autoencoder configuration carrying widths, ``embedding_norm`` and ``init_mode``.
        encoder : Module mapping an image stack to the latent feature map.
        decoder : Module mapping the latent feature map back to the image stack.
    """

    def __init__(self, config, encoder: nn.Module, decoder: nn.Module) -> None:
        """Stores the encoder and decoder and initializes their weights.

        Args:
            config  : Autoencoder configuration.
            encoder : Module producing the latent feature map.
            decoder : Module reconstructing the input from the latent feature map.

        Raises:
            ValueError : If ``config.embedding_norm`` is not one of ``EMBEDDING_NORMS``.
        """
        super().__init__()
        self.config = config

        if config.embedding_norm not in self.EMBEDDING_NORMS:
            raise ValueError(f"Unknown embedding_norm '{config.embedding_norm}'. Available: {list(self.EMBEDDING_NORMS)}")

        self.encoder = encoder
        self.decoder = decoder

        LayerFactory.initialize_weights(module=self, mode=config.init_mode)

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """Encodes an image stack into the normalized latent feature map.

        Args:
            image : Input stack of shape (batch, in_channels, height, width).

        Returns:
            Latent of shape (batch, embedding_dim, height / downsample, width / downsample).
        """
        return self.normalize_embedding(self.encoder(image))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstructs an image stack from a latent feature map.

        Args:
            z : Latent of shape (batch, embedding_dim, latent_height, latent_width).

        Returns:
            Reconstruction of shape (batch, in_channels, height, width).
        """
        return self.decoder(z)

    def encode_features(self, image: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
        """Encodes an image and resamples the latent onto a requested grid.

        Used when the latent feeds a downstream model that expects a specific
        spatial size.

        Args:
            image  : Input stack of shape (batch, in_channels, height, width).
            out_hw : Target (height, width) of the returned latent.

        Returns:
            Latent of shape (batch, embedding_dim, out_hw[0], out_hw[1]).
        """
        z = self.encode(image)
        if z.shape[-2:] != tuple(out_hw):
            z = F.interpolate(z, size=tuple(out_hw), mode="bilinear", align_corners=False)
        return z

    def reconstruct(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encodes and decodes an image, returning both the reconstruction and the latent.

        Args:
            image : Input stack of shape (batch, in_channels, height, width).

        Returns:
            Tuple of the reconstruction, shaped like ``image``, and the latent of
            shape (batch, embedding_dim, latent_height, latent_width).
        """
        z         = self.encode(image)
        image_hat = self.decode(z)
        return image_hat, z

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Returns the reconstruction of an image stack.

        Args:
            image : Input stack of shape (batch, in_channels, height, width).

        Returns:
            Reconstruction of the same shape as ``image``.
        """
        image_hat, _ = self.reconstruct(image)
        return image_hat
