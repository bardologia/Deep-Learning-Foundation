from __future__ import annotations

import torch
import torch.nn as nn

from models.blocks import ChannelLayerNorm, EmbeddingNorm, build_activation, initialize_weights


class ProfileAutoencoderBlocks:
    @staticmethod
    def conv1x1(in_ch: int, out_ch: int) -> nn.Conv2d:
        return nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)

    @staticmethod
    def mlp_stack(in_ch: int, hidden: int, out_ch: int, depth: int, activation: str, dropout: float) -> nn.Sequential:
        layers: list[nn.Module] = []
        prev = in_ch

        for _ in range(max(1, depth)):
            layers.append(ProfileAutoencoderBlocks.conv1x1(prev, hidden))
            layers.append(build_activation(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout2d(dropout))
            prev = hidden

        layers.append(ProfileAutoencoderBlocks.conv1x1(prev, out_ch))
        return nn.Sequential(*layers)

    @staticmethod
    def to_sequence(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        B, L, H, W = x.shape
        seq = x.permute(0, 2, 3, 1).reshape(B * H * W, L)
        return seq, (B, H, W)

    @staticmethod
    def from_sequence(seq: torch.Tensor, dims: tuple[int, int, int]) -> torch.Tensor:
        B, H, W = dims
        L_out   = seq.shape[1]
        return seq.reshape(B, H, W, L_out).permute(0, 3, 1, 2).contiguous()


class ProfileAutoencoderBase(EmbeddingNorm, nn.Module):
    def __init__(self, config, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self.config = config

        if config.embedding_norm not in self.EMBEDDING_NORMS:
            raise ValueError(f"Unknown embedding_norm '{config.embedding_norm}'. Available: {list(self.EMBEDDING_NORMS)}")

        self.encoder = encoder
        self.decoder = decoder

        self.embedding_layernorm = ChannelLayerNorm(config.embedding_dim) if config.embedding_norm == "layernorm" else None

        initialize_weights(module=self, mode=config.init_mode)

    def encode(self, curve: torch.Tensor) -> torch.Tensor:
        return self.normalize_embedding(self.encoder(curve))

    def normalize_embedding(self, z: torch.Tensor) -> torch.Tensor:
        if self.config.embedding_norm == "layernorm":
            return self.embedding_layernorm(z)
        return super().normalize_embedding(z)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def reconstruct(self, curve: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z         = self.encode(curve)
        curve_hat = self.decode(z)
        return curve_hat, z

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        curve_hat, _ = self.reconstruct(curve)
        return curve_hat
