from __future__ import annotations

import torch

from configuration.training.backbone import BackboneTrainerConfig
from models                          import get_backbone


TARGET_CHANNELS = ["a", "b"]


def tiny_model(in_channels: int = 2, out_channels: int = 2):
    return get_backbone(
        "resunet",
        in_channels       = in_channels,
        out_channels      = out_channels,
        features          = [8, 16],
        bottleneck_factor = 1,
        dropout           = 0.0,
        normalization     = "instance",
    )


def tiny_trainer_config(epochs: int = 1) -> BackboneTrainerConfig:
    cfg = BackboneTrainerConfig()

    cfg.training.epochs               = epochs
    cfg.training.validation_frequency = 1
    cfg.resources.enabled             = False

    return cfg


def batch_tensors(n: int = 6, in_channels: int = 2, out_channels: int = 2, hw: int = 16, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen     = torch.Generator().manual_seed(seed)
    inputs  = torch.randn(n, in_channels, hw, hw, generator=gen)
    targets = torch.randn(n, out_channels, hw, hw, generator=gen)
    weights = torch.ones(n, 1, hw, hw)

    return inputs, targets, weights
