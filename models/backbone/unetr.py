from __future__ import annotations

import torch
import torch.nn as nn

from configuration.architectures import UNETRConfig
from ..blocks                    import ConvBlock, OutputHeadsMixin, PatchEmbedding, TransformerBlock, build_activation, build_norm2d, initialize_weights, match_spatial_size, resize_positional_embedding, tokens_to_feature_map


class ProgressiveProjectionHead(nn.Module):
    def __init__(
        self,
        input_channels:  int,
        output_channels: int,
        upsample_steps:  int,
        activation:      str   = "relu",
        normalization:   str   = "batch",
        bias:            bool  = False,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels  = input_channels,
                out_channels = output_channels,
                kernel_size  = 3,
                padding      = 1,
                bias         = bias,
            ),
            build_norm2d(normalization, output_channels),
            build_activation(activation),
        ]
        for _ in range(upsample_steps):
            layers.extend([
                nn.ConvTranspose2d(
                    in_channels  = output_channels,
                    out_channels = output_channels,
                    kernel_size  = 2,
                    stride       = 2,
                    bias         = bias,
                ),
                nn.Conv2d(
                    in_channels  = output_channels,
                    out_channels = output_channels,
                    kernel_size  = 3,
                    padding      = 1,
                    bias         = bias,
                ),
                build_norm2d(normalization, output_channels),
                build_activation(activation),
            ])
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UNETR(nn.Module, OutputHeadsMixin):
    def __init__(self, config: UNETRConfig | None = None):
        super().__init__()
        config      = config or UNETRConfig()
        self.config = config

        if config.patch_size <= 0:
            raise ValueError("patch_size must be a positive integer")
        if config.image_size % config.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if config.transformer_layers < 4:
            raise ValueError("transformer_layers must be at least 4 to collect four skip connections")
        if len(config.decoder_features) != 4:
            raise ValueError("decoder_features must contain exactly four feature sizes")

        self.patch_embedding = PatchEmbedding(
            in_channels   = config.in_channels,
            embedding_dim = config.embedding_dim,
            patch_size    = config.patch_size,
        )

        self.expected_grid_size   = config.image_size // config.patch_size
        self.positional_embedding = nn.Parameter(torch.zeros(1, self.expected_grid_size ** 2, config.embedding_dim))
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)

        drop_path_rates = [x.item() for x in torch.linspace(0, config.stochastic_depth_rate, config.transformer_layers)]
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim     = config.embedding_dim,
                num_heads         = config.transformer_heads,
                mlp_ratio         = config.transformer_mlp_ratio,
                dropout           = config.dropout,
                attention_dropout = config.attention_dropout,
                ffn_activation    = config.ffn_activation,
                drop_path_rate    = drop_path_rates[i],
            )
            for i in range(config.transformer_layers)
        ])
        self.transformer_norm = nn.LayerNorm(config.embedding_dim)

        total_layers = config.transformer_layers
        self.skip_layer_indices = [
            total_layers // 4 - 1,
            total_layers // 2 - 1,
            (3 * total_layers) // 4 - 1,
            total_layers - 1,
        ]

        decoder_features      = config.decoder_features
        self.transformer_skip_heads = nn.ModuleList([
            ProgressiveProjectionHead(
                input_channels  = config.embedding_dim,
                output_channels = decoder_features[-1],
                upsample_steps  = 3,
                activation      = config.activation,
                normalization   = config.normalization,
                bias            = config.conv_bias,
            ),
            ProgressiveProjectionHead(
                input_channels  = config.embedding_dim,
                output_channels = decoder_features[-2],
                upsample_steps  = 2,
                activation      = config.activation,
                normalization   = config.normalization,
                bias            = config.conv_bias,
            ),
            ProgressiveProjectionHead(
                input_channels  = config.embedding_dim,
                output_channels = decoder_features[-3],
                upsample_steps  = 1,
                activation      = config.activation,
                normalization   = config.normalization,
                bias            = config.conv_bias,
            ),
        ])
        self.bottleneck_projection = ProgressiveProjectionHead(
            input_channels  = config.embedding_dim,
            output_channels = decoder_features[0],
            upsample_steps  = 0,
            activation      = config.activation,
            normalization   = config.normalization,
            bias            = config.conv_bias,
        )

        self.input_skip_conv = ConvBlock(
            input_channels  = config.in_channels,
            output_channels = decoder_features[-1],
            dropout         = config.dropout,
            activation      = config.activation,
            normalization   = config.normalization,
            bias            = config.conv_bias,
        )
        
        self.upsample_layers = nn.ModuleList()
        self.decoder_blocks  = nn.ModuleList()

        upsample_output_channels = decoder_features[1:] + [decoder_features[-1]]
        skip_channels            = [decoder_features[1], decoder_features[2], decoder_features[3], decoder_features[3]]

        for index, output_channels in enumerate(upsample_output_channels):
            input_to_upsample = decoder_features[index]

            self.upsample_layers.append(
                nn.ConvTranspose2d(
                    in_channels  = input_to_upsample,
                    out_channels = output_channels,
                    kernel_size  = 2,
                    stride       = 2,
                )
            )

            self.decoder_blocks.append(
                ConvBlock(
                    input_channels  = output_channels + skip_channels[index],
                    output_channels = output_channels,
                    dropout         = config.dropout,
                    activation      = config.activation,
                    normalization   = config.normalization,
                    bias            = config.conv_bias,
                )
            )

        remaining_scale = config.patch_size // (2 ** len(decoder_features))
        if remaining_scale > 1:
            self.final_upsample = nn.ConvTranspose2d(
                in_channels  = decoder_features[-1],
                out_channels = decoder_features[-1],
                kernel_size  = remaining_scale,
                stride       = remaining_scale,
            )
        else:
            self.final_upsample = nn.Identity()
        self.embedding_channels = decoder_features[-1]
        self._build_output_head()

        initialize_weights(module=self, mode=config.init_mode)

    def encode_decode(self, x: torch.Tensor) -> torch.Tensor:
        original_input = x

        tokens, grid_height, grid_width = self.patch_embedding(x)
        tokens = tokens + resize_positional_embedding(self.positional_embedding, self.expected_grid_size, grid_height, grid_width)

        transformer_skip_maps = []
        for layer_index, transformer_block in enumerate(self.transformer_blocks):
            tokens = transformer_block(tokens)
            if layer_index in self.skip_layer_indices:
                skip_tokens = self.transformer_norm(tokens) if layer_index == len(self.transformer_blocks) - 1 else tokens
                transformer_skip_maps.append(tokens_to_feature_map(skip_tokens, grid_height, grid_width))

        progressively_upsampled_skips = [
            skip_head(feature_map)
            for skip_head, feature_map in zip(self.transformer_skip_heads, transformer_skip_maps[:-1])
        ]
        progressively_upsampled_skips = progressively_upsampled_skips[::-1]

        x = self.bottleneck_projection(transformer_skip_maps[-1])

        for index in range(len(self.upsample_layers)):
            x = self.upsample_layers[index](x)

            if index < len(progressively_upsampled_skips):
                skip = progressively_upsampled_skips[index]
            else:
                skip = self.input_skip_conv(original_input)

            x = match_spatial_size(source=x, reference=skip)
            x = torch.cat([x, skip], dim=1)
            x = self.decoder_blocks[index](x)

        x = self.final_upsample(x)
        return match_spatial_size(source=x, reference=original_input)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._head_forward(self.encode_decode(x))
