from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

from configuration.architectures import HRNetLiteConfig
from ..blocks                    import build_activation, build_norm2d, initialize_weights
from ..blocks                    import OutputHeadsMixin, ResidualConvBlock


class BranchFusion(nn.Module):
    def __init__(self, branch_channels: list[int], activation: str, normalization: str, bias: bool):
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
                            build_norm2d(normalization, out_channels),
                        ]

                        if not is_final:
                            downs.append(build_activation(activation))

                        channels = out_channels
                    row.append(nn.Sequential(*downs))
                else:
                    row.append(nn.Sequential(
                        nn.Conv2d(branch_channels[source_index], branch_channels[target_index], kernel_size=1, bias=bias),
                        build_norm2d(normalization, branch_channels[target_index]),
                    ))
            self.transforms.append(row)

        self.activation = build_activation(activation)

    def forward(self, branch_inputs: list[torch.Tensor]) -> list[torch.Tensor]:
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
    def __init__(self, config: HRNetLiteConfig | None = None):
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
        )

        self.transition_modules = nn.ModuleList()
        self.stage_modules      = nn.ModuleList()
        self.fuse_modules       = nn.ModuleList()

        for stage_index in range(1, config.n_branches):
            active = branch_channels[:stage_index + 1]

            self.transition_modules.append(nn.Sequential(
                nn.Conv2d(active[-2], active[-1], kernel_size=3, stride=2, padding=1, bias=config.conv_bias),
                build_norm2d(config.normalization, active[-1]),
                build_activation(config.activation),
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
            build_norm2d(config.normalization, base * 2),
            build_activation(config.activation),
        )

        self.embedding_channels = base * 2
        self._build_output_head()

        initialize_weights(module=self, mode=config.init_mode)

    def encode_decode(self, x: torch.Tensor) -> torch.Tensor:
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
        return self._head_forward(self.encode_decode(x))
