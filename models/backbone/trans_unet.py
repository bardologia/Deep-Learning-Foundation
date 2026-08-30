"""TransUNet backbone: a convolutional U-Net whose bottleneck is a transformer.

The CNN encoder reduces the input, the bottleneck feature map is tokenized with
non-overlapping patches and processed by transformer blocks, and the CNN decoder
restores resolution using the encoder skips.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from configuration.architectures import TransUNetConfig
from ..blocks                    import ConvBlock, FeatureGeometry, LayerFactory, OutputHeadsMixin, PatchEmbedding, TransformerBlock


class TransUNet(nn.Module, OutputHeadsMixin):
    """TransUNet backbone predicting per-pixel Gaussian mixture parameters.

    Attributes:
        config               : Architecture configuration driving widths, depth and head choice.
        expected_grid_size   : Token grid side the positional embedding was built for.
        positional_embedding : Learned positional embedding of shape (1, expected_grid_size ** 2, bottleneck_channels), resampled when the actual grid differs.
        embedding_channels   : Channel width of the decoded embedding fed to the output head.
    """

    def __init__(self, config: TransUNetConfig | None = None):
        """Builds the CNN encoder, the transformer bottleneck, the decoder and the head.

        Args:
            config : Architecture configuration; defaults to ``TransUNetConfig()``.

        Raises:
            ValueError : If ``cnn_features`` is empty or ``patch_size`` is not positive.
        """
        super().__init__()
        if config is None:
            config = TransUNetConfig()
        self.config = config

        if len(config.cnn_features) == 0:
            raise ValueError("cnn_features must contain at least one channel size")
        if config.patch_size <= 0:
            raise ValueError("patch_size must be a positive integer")

        cnn_features        = config.cnn_features
        bottleneck_channels = cnn_features[-1] * config.bottleneck_factor

        self.encoder_blocks    = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        channels = config.in_channels
        for feature_size in cnn_features:
            self.encoder_blocks.append(
                ConvBlock(
                    input_channels  = channels,
                    output_channels = feature_size,
                    dropout         = config.dropout,
                    activation      = config.activation,
                    normalization   = config.normalization,
                    bias            = config.conv_bias,
                )
            )
            self.downsample_layers.append(nn.MaxPool2d(kernel_size=2))
            channels = feature_size

        self.pre_transformer_conv = ConvBlock(
            input_channels  = cnn_features[-1],
            output_channels = bottleneck_channels,
            dropout         = config.dropout,
            activation      = config.activation,
            normalization   = config.normalization,
            bias            = config.conv_bias,
        )

        self.patch_embedding = PatchEmbedding(
            in_channels   = bottleneck_channels,
            embedding_dim = bottleneck_channels,
            patch_size    = config.patch_size,
        )

        drop_path_rates = [x.item() for x in torch.linspace(0, config.stochastic_depth_rate, config.transformer_layers)]
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim     = bottleneck_channels,
                num_heads         = config.transformer_heads,
                mlp_ratio         = config.transformer_mlp_ratio,
                dropout           = config.dropout,
                attention_dropout = config.attention_dropout,
                ffn_activation    = config.ffn_activation,
                drop_path_rate    = drop_path_rates[i],
            )
            for i in range(config.transformer_layers)
        ])
        self.transformer_norm = nn.LayerNorm(bottleneck_channels)

        cnn_downsample          = 2 ** len(cnn_features)
        self.expected_grid_size = config.image_size // cnn_downsample // config.patch_size
        num_patches             = self.expected_grid_size ** 2
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, num_patches, bottleneck_channels)
        )
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)

        reversed_features    = [bottleneck_channels] + cnn_features[::-1]
        self.upsample_layers = nn.ModuleList()
        self.decoder_blocks  = nn.ModuleList()
        for index in range(len(reversed_features) - 1):
            self.upsample_layers.append(
                LayerFactory.build_upsample(
                    mode         = config.upsample_mode,
                    in_channels  = reversed_features[index],
                    out_channels = reversed_features[index + 1],
                )
            )
            self.decoder_blocks.append(
                ConvBlock(
                    input_channels  = reversed_features[index + 1] * 2,
                    output_channels = reversed_features[index + 1],
                    dropout         = config.dropout,
                    activation      = config.activation,
                    normalization   = config.normalization,
                    bias            = config.conv_bias,
                )
            )

        self.embedding_channels = cnn_features[0]
        self._build_output_head()

        LayerFactory.initialize_weights(module=self, mode=config.init_mode)

    def encode_decode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the embedding at input resolution.

        Args:
            x : Input feature stack of shape (batch, in_channels, height, width).

        Returns:
            Embedding of shape (batch, embedding_channels, height, width).
        """
        skip_connections: list[torch.Tensor] = []
        for encoder_block, downsample in zip(self.encoder_blocks, self.downsample_layers):
            x = encoder_block(x)
            skip_connections.append(x)
            x = downsample(x)

        x = self.pre_transformer_conv(x)

        tokens, grid_height, grid_width = self.patch_embedding(x)

        tokens = tokens + FeatureGeometry.resize_positional_embedding(self.positional_embedding, self.expected_grid_size, grid_height, grid_width)

        for transformer_block in self.transformer_blocks:
            tokens = transformer_block(tokens)
        tokens = self.transformer_norm(tokens)
        x      = FeatureGeometry.tokens_to_feature_map(tokens, grid_height, grid_width)

        for upsample, decoder_block, skip in zip(
            self.upsample_layers, self.decoder_blocks, reversed(skip_connections)
        ):
            x = upsample(x)
            x = FeatureGeometry.match_spatial_size(source=x, reference=skip)
            x = torch.cat([skip, x], dim=1)
            x = decoder_block(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Maps the input stack to per-pixel outputs through the configured head.

        Args:
            x : Input feature stack of shape (batch, in_channels, height, width).

        Returns:
            Tensor of shape (batch, out_channels, height, width) holding the
            interleaved Gaussian parameters, or the raw embedding when the head
            is ``none``.
        """
        return self._head_forward(self.encode_decode(x))
