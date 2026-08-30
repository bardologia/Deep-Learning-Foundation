"""Lightweight HRNet backbone that keeps parallel multi-resolution branches alive.

Unlike an encoder-decoder, every branch runs at its own stride from the stem to the
output and exchanges information through repeated cross-resolution fusion, so the
highest-resolution stream never loses its spatial detail.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

from configuration.architectures import HRNetLiteConfig
from ..blocks                    import LayerFactory, OutputHeadsMixin, ResidualConvBlock


class BranchFusion(nn.Module):
    """Exchanges features between all resolution branches of an HRNet stage.

    Each target branch sums transformed copies of every source branch: strided
    convolutions for higher-to-lower resolution, 1x1 projections plus bilinear
    resampling for lower-to-higher, identity for the branch itself.

    Attributes:
        n_branches : Number of parallel resolution branches being fused.
        transforms : Grid of per-(target, source) transform modules.
        activation : Activation applied to each fused branch.
    """

    def __init__(self, branch_channels: list[int], activation: str, normalization: str, bias: bool):
        """Builds the cross-resolution transform grid.

        Args:
            branch_channels : Channel width of each branch, ordered from the highest spatial resolution (index 0) to the lowest.
            activation      : Activation name resolved by ``LayerFactory.build_activation``.
            normalization   : 2D normalization name resolved by ``LayerFactory.build_norm2d``.
            bias            : Whether the transform convolutions carry a bias term.
        """
        super().__init__()
        n_branches      = len(branch_channels)
        self.n_branches = n_branches

        self.transforms = nn.ModuleList()
        for target_index in range(n_branches):
            row = nn.ModuleList()
            for source_index in range(n_branches):
                if source_index == target_index:
                    row.append(nn.Identity())
                elif source_index < target_index:
                    downs    = []
                    channels = branch_channels[source_index]
                    n_steps  = target_index - source_index
                    for step in range(n_steps):
                        is_final     = step == n_steps - 1
                        out_channels = branch_channels[target_index] if is_final else channels
                        downs += [
                            nn.Conv2d(channels, out_channels, kernel_size=3, stride=2, padding=1, bias=bias),
                            LayerFactory.build_norm2d(normalization, out_channels),
                        ]

                        if not is_final:
                            downs.append(LayerFactory.build_activation(activation))

                        channels = out_channels
                    row.append(nn.Sequential(*downs))
                else:
                    row.append(nn.Sequential(
                        nn.Conv2d(branch_channels[source_index], branch_channels[target_index], kernel_size=1, bias=bias),
                        LayerFactory.build_norm2d(normalization, branch_channels[target_index]),
                    ))
            self.transforms.append(row)

        self.activation = LayerFactory.build_activation(activation)

    def forward(self, branch_inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Fuses every branch into every other branch.

        Args:
            branch_inputs : One feature map per branch, each of shape (batch, branch_channels[i], height_i, width_i).

        Returns:
            One fused feature map per branch, each keeping the shape of the
            corresponding entry of ``branch_inputs``.
        """
        outputs = []
        for target_index in range(self.n_branches):
            target_size = branch_inputs[target_index].shape[2:]

            fused = None
            for source_index in range(self.n_branches):
                transformed = self.transforms[target_index][source_index](branch_inputs[source_index])
                if transformed.shape[2:] != target_size:
                    transformed = functional.interpolate(transformed, size=target_size, mode="bilinear", align_corners=False)
                fused = transformed if fused is None else fused + transformed

            outputs.append(self.activation(fused))

        return outputs


class HRNetLite(nn.Module, OutputHeadsMixin):
    """HRNet-style backbone predicting per-pixel Gaussian mixture parameters.

    A stem produces the full-resolution branch; each further stage adds one
    half-resolution branch, runs residual blocks on every branch and fuses them.
    All branches are finally upsampled to the stem resolution and concatenated.

    Attributes:
        config             : Architecture configuration driving widths, depth and head choice.
        embedding_channels : Channel width of the fused embedding fed to the output head.
    """

    def __init__(self, config: HRNetLiteConfig | None = None):
        """Builds the stem, the branch ladder, the fusion modules and the output head.

        Args:
            config : Architecture configuration; defaults to ``HRNetLiteConfig()``.

        Raises:
            ValueError : If ``config.n_branches`` is smaller than two.
        """
        super().__init__()
        if config is None:
            config = HRNetLiteConfig()
        self.config = config

        if config.n_branches < 2:
            raise ValueError("n_branches must be at least 2")

        base            = config.base_channels
        branch_channels = [base * (2 ** index) for index in range(config.n_branches)]

        self.stem = ResidualConvBlock(
            input_channels  = config.in_channels,
            output_channels = base,
            dropout         = config.dropout,
            activation      = config.activation,
            normalization   = config.normalization,
            bias            = config.conv_bias,
            first_unit      = True,
        )

        self.transition_modules = nn.ModuleList()
        self.stage_modules      = nn.ModuleList()
        self.fuse_modules       = nn.ModuleList()

        for stage_index in range(1, config.n_branches):
            active = branch_channels[:stage_index + 1]

            self.transition_modules.append(nn.Sequential(
                nn.Conv2d(active[-2], active[-1], kernel_size=3, stride=2, padding=1, bias=config.conv_bias),
                LayerFactory.build_norm2d(config.normalization, active[-1]),
                LayerFactory.build_activation(config.activation),
            ))

            branch_blocks = nn.ModuleList()
            for channels in active:
                branch_blocks.append(nn.Sequential(*[
                    ResidualConvBlock(
                        input_channels  = channels,
                        output_channels = channels,
                        dropout         = config.dropout,
                        activation      = config.activation,
                        normalization   = config.normalization,
                        bias            = config.conv_bias,
                    )
                    for _ in range(config.blocks_per_stage)
                ]))
            self.stage_modules.append(branch_blocks)

            self.fuse_modules.append(BranchFusion(active, config.activation, config.normalization, config.conv_bias))

        concat_channels = sum(branch_channels)
        self.final_fuse = nn.Sequential(
            nn.Conv2d(concat_channels, base * 2, kernel_size=3, padding=1, bias=config.conv_bias),
            LayerFactory.build_norm2d(config.normalization, base * 2),
            LayerFactory.build_activation(config.activation),
        )

        self.embedding_channels = base * 2
        self._build_output_head()

        LayerFactory.initialize_weights(module=self, mode=config.init_mode)

    def encode_decode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the fused multi-resolution embedding at input resolution.

        Args:
            x : Input feature stack of shape (batch, in_channels, height, width).

        Returns:
            Embedding of shape (batch, embedding_channels, height, width).
        """
        branches = [self.stem(x)]

        for transition, branch_blocks, fusion in zip(self.transition_modules, self.stage_modules, self.fuse_modules):
            branches.append(transition(branches[-1]))
            branches = [blocks(branch) for blocks, branch in zip(branch_blocks, branches)]
            branches = fusion(branches)

        target_size = branches[0].shape[2:]
        upsampled   = [branches[0]]
        for branch in branches[1:]:
            upsampled.append(functional.interpolate(branch, size=target_size, mode="bilinear", align_corners=False))

        return self.final_fuse(torch.cat(upsampled, dim=1))

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
