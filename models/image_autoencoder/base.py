from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import EmbeddingNorm, initialize_weights


class ImageAutoencoderBase(EmbeddingNorm, nn.Module):
    def __init__(self, config, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self.config = config

        if config.embedding_norm not in self.EMBEDDING_NORMS:
            raise ValueError(f"Unknown embedding_norm '{config.embedding_norm}'. Available: {list(self.EMBEDDING_NORMS)}")

        self.encoder = encoder
        self.decoder = decoder

        initialize_weights(module=self, mode=config.init_mode)

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return self.normalize_embedding(self.encoder(image))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def encode_features(self, image: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
        z = self.encode(image)
        if z.shape[-2:] != tuple(out_hw):
            z = F.interpolate(z, size=tuple(out_hw), mode="bilinear", align_corners=False)
        return z

    def reconstruct(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z         = self.encode(image)
        image_hat = self.decode(z)
        return image_hat, z

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image_hat, _ = self.reconstruct(image)
        return image_hat
