"""Shared building blocks and base class for profile autoencoders.

Profile autoencoders compress the per-pixel z profile of a cube, so
every architecture works on tensors of shape (batch, profile_length, height,
width) and flattens the spatial axes into a pixel batch before the 1-D body.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.blocks import AttentionPool1d, ChannelLayerNorm, EmbeddingNorm, LayerFactory, TokenAttentionPool, TokenMaxPool, TokenMeanPool


class ProfileAutoencoderBlocks:
    """Layer constructors and profile/sequence layout conversions."""

    @staticmethod
    def conv1x1(in_ch: int, out_ch: int) -> nn.Conv2d:
        """Returns a 1x1 convolution acting as a per-pixel linear layer."""
        return nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)

    @staticmethod
    def norm2d(kind: str, channels: int, family: str, supported: tuple[str, ...]) -> nn.Module | None:
        """Returns the per-pixel normalization layer of a configuration name.

        Args:
            kind      : One of the modes in ``supported``; ``none`` returns None so the caller can omit the layer entirely.
            channels  : Channel width being normalized.
            family    : Model registry key used in the error message.
            supported : Modes this family honors.

        Returns:
            The normalization module, or None for ``none``.

        Raises:
            ValueError : If the mode is not supported by the family.
        """
        if kind not in supported:
            raise ValueError(f"{family} supports normalization modes {list(supported)} only; got '{kind}'.")

        if kind == "none":      return None
        if kind == "batchnorm": return nn.BatchNorm2d(channels)
        return ChannelLayerNorm(channels)

    @staticmethod
    def norm1d(kind: str, channels: int, family: str, supported: tuple[str, ...]) -> nn.Module | None:
        """Returns the sequence normalization layer of a configuration name.

        Args:
            kind      : One of the modes in ``supported``; ``none`` returns None so the caller can omit the layer entirely.
            channels  : Channel width being normalized.
            family    : Model registry key used in the error message.
            supported : Modes this family honors.

        Returns:
            The normalization module over (pixels, channels, length) features,
            or None for ``none``.

        Raises:
            ValueError : If the mode is not supported by the family.
        """
        if kind not in supported:
            raise ValueError(f"{family} supports normalization modes {list(supported)} only; got '{kind}'.")

        if kind == "none": return None
        return nn.BatchNorm1d(channels)

    @staticmethod
    def pool1d(kind: str, channels: int, family: str) -> nn.Module:
        """Returns the length-axis pooling module of a configuration name.

        Args:
            kind     : One of ``avg``, ``max`` or ``attention``.
            channels : Channel width of the pooled features.
            family   : Model registry key used in the error message.

        Returns:
            Module mapping (pixels, channels, length) to (pixels, channels, 1).

        Raises:
            ValueError : If the pooling kind is unknown.
        """
        if kind == "avg":       return nn.AdaptiveAvgPool1d(1)
        if kind == "max":       return nn.AdaptiveMaxPool1d(1)
        if kind == "attention": return AttentionPool1d(channels)
        raise ValueError(f"{family} supports pooling modes ['avg', 'max', 'attention'] only; got '{kind}'.")

    @staticmethod
    def token_pool(kind: str, dim: int, family: str) -> nn.Module:
        """Returns the token pooling module of a configuration name.

        Args:
            kind   : One of ``avg``, ``max`` or ``attention``.
            dim    : Feature width of the pooled tokens.
            family : Model registry key used in the error message.

        Returns:
            Module mapping (pixels, tokens, dim) to (pixels, dim).

        Raises:
            ValueError : If the pooling kind is unknown.
        """
        if kind == "avg":       return TokenMeanPool()
        if kind == "max":       return TokenMaxPool()
        if kind == "attention": return TokenAttentionPool(dim)
        raise ValueError(f"{family} supports pooling modes ['avg', 'max', 'attention'] only; got '{kind}'.")

    @staticmethod
    def hidden_widths(hidden_dims: list[int] | None, hidden_dim: int, depth: int) -> list[int]:
        """Returns the per-layer hidden widths of an MLP-style stack.

        Args:
            hidden_dims : Explicit per-layer widths; None or empty falls back to ``depth`` copies of ``hidden_dim``.
            hidden_dim  : Uniform width used by the fallback.
            depth       : Number of layers used by the fallback, clamped to one.

        Returns:
            The per-layer widths.
        """
        return list(hidden_dims) if hidden_dims else [hidden_dim] * max(1, depth)

    @staticmethod
    def mlp_stack(in_ch: int, widths: list[int], out_ch: int, activation: str, dropout: float, normalization: str = "none", family: str = "mlp_ae") -> nn.Sequential:
        """Builds a per-pixel MLP as a stack of 1x1 convolutions.

        Args:
            in_ch         : Number of input channels, i.e. the profile or latent length.
            widths        : Width of each hidden layer, one entry per layer.
            out_ch        : Number of output channels.
            activation    : Name of the activation resolved by ``LayerFactory``.
            dropout       : Dropout2d probability, applied only when positive.
            normalization : Per-pixel normalization inserted after each hidden layer, one of ``none``, ``batchnorm`` or ``layernorm``.
            family        : Model registry key used in error messages.

        Returns:
            Sequential module mapping (batch, in_ch, height, width) to
            (batch, out_ch, height, width).

        Raises:
            ValueError : If the normalization mode is unsupported.
        """
        layers: list[nn.Module] = []
        prev = in_ch

        for hidden in widths:
            layers.append(ProfileAutoencoderBlocks.conv1x1(prev, hidden))

            norm = ProfileAutoencoderBlocks.norm2d(normalization, hidden, family, ("none", "batchnorm", "layernorm"))
            if norm is not None:
                layers.append(norm)

            layers.append(LayerFactory.build_activation(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout2d(dropout))
            prev = hidden

        layers.append(ProfileAutoencoderBlocks.conv1x1(prev, out_ch))
        return nn.Sequential(*layers)

    @staticmethod
    def to_sequence(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        """Flattens a profile map into one sequence per pixel.

        Args:
            x : Profile map of shape (batch, length, height, width).

        Returns:
            Tuple of the sequence matrix of shape (batch * height * width,
            length) and the (batch, height, width) dimensions needed to restore
            the map layout.
        """
        B, L, H, W = x.shape
        seq = x.permute(0, 2, 3, 1).reshape(B * H * W, L)
        return seq, (B, H, W)

    @staticmethod
    def from_sequence(seq: torch.Tensor, dims: tuple[int, int, int]) -> torch.Tensor:
        """Restores the map layout from a per-pixel sequence matrix.

        Args:
            seq  : Sequence matrix of shape (batch * height * width, out_length).
            dims : The (batch, height, width) dimensions returned by ``to_sequence``.

        Returns:
            Map of shape (batch, out_length, height, width).
        """
        B, H, W = dims
        L_out   = seq.shape[1]
        return seq.reshape(B, H, W, L_out).permute(0, 3, 1, 2).contiguous()


class ProfileAutoencoderBase(EmbeddingNorm, nn.Module):
    """Common encode/decode plumbing shared by all profile autoencoders.

    Attributes:
        config              : Architecture configuration of the concrete autoencoder.
        encoder             : Module mapping a profile map to a latent map.
        decoder             : Module mapping a latent map back to a profile map.
        embedding_layernorm : Channel LayerNorm used when embedding_norm is 'layernorm', otherwise None.
    """

    def __init__(self, config, encoder: nn.Module, decoder: nn.Module) -> None:
        """Wires encoder and decoder and initialises the weights.

        Args:
            config  : Architecture configuration carrying embedding_dim, embedding_norm and init_mode.
            encoder : Encoder module to use.
            decoder : Decoder module to use.

        Raises:
            ValueError : If embedding_norm is not one of the supported modes.
        """
        super().__init__()
        self.config = config

        if config.embedding_norm not in self.EMBEDDING_NORMS:
            raise ValueError(f"Unknown embedding_norm '{config.embedding_norm}'. Available: {list(self.EMBEDDING_NORMS)}")

        self.encoder = encoder
        self.decoder = decoder

        self.embedding_layernorm = ChannelLayerNorm(config.embedding_dim) if config.embedding_norm == "layernorm" else None

        LayerFactory.initialize_weights(module=self, mode=config.init_mode)

    def encode(self, curve: torch.Tensor) -> torch.Tensor:
        """Encodes profiles into normalized latent codes.

        Args:
            curve : Profile map of shape (batch, profile_length, height, width).

        Returns:
            Latent map of shape (batch, embedding_dim, height, width).
        """
        return self.normalize_embedding(self.encoder(curve))

    def normalize_embedding(self, z: torch.Tensor) -> torch.Tensor:
        """Applies the configured normalization to a latent map.

        Args:
            z : Latent map of shape (batch, embedding_dim, height, width).

        Returns:
            Latent map of the same shape, channel-LayerNormed when
            embedding_norm is 'layernorm' and otherwise normalized by the
            ``EmbeddingNorm`` base implementation.
        """
        if self.config.embedding_norm == "layernorm":
            return self.embedding_layernorm(z)
        return super().normalize_embedding(z)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes a latent map of shape (batch, embedding_dim, height, width) into profiles."""
        return self.decoder(z)

    def reconstruct(self, curve: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encodes and decodes profiles, returning both reconstruction and code.

        Args:
            curve : Profile map of shape (batch, profile_length, height, width).

        Returns:
            Tuple of the reconstructed profile map with the input shape and the
            latent map of shape (batch, embedding_dim, height, width).
        """
        z         = self.encode(curve)
        curve_hat = self.decode(z)
        return curve_hat, z

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        """Returns the reconstructed profile map for the given profile map."""
        curve_hat, _ = self.reconstruct(curve)
        return curve_hat
