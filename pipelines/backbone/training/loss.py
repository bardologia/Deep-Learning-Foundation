from __future__ import annotations

import torch
import torch.nn.functional as F

from configuration.training.general.loss import LossConfig, LossKind
from tools.monitoring.logger             import Logger
from tools.training.aggregation          import WeightedMean


class Loss:

    def __init__(self, config: LossConfig, target_channels: list[str], logger: Logger) -> None:
        self.config          = config
        self.target_channels = list(target_channels)
        self.channel_weights = list(config.channel_weights) if config.channel_weights else [1.0] * len(self.target_channels)

        logger.section("[Loss]")
        logger.kv_table({
            "Kind"            : config.kind.value,
            "Huber delta"     : config.huber_delta if config.kind is LossKind.HUBER else "unused",
            "Channel weights" : ", ".join(f"{name}={weight:g}" for name, weight in zip(self.target_channels, self.channel_weights)),
            "Space"           : "normalized targets",
            "Pixel weights"   : "padding 0; a validation or test pixel 1 over the number of patches covering it; a train pixel 1",
            "Pooling"         : "sum(weight * value) / sum(weight), over each batch and over a whole split",
        })

    def elementwise(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.config.kind is LossKind.MSE:
            return (predictions - targets) ** 2

        if self.config.kind is LossKind.L1:
            return (predictions - targets).abs()

        return F.huber_loss(predictions, targets, reduction="none", delta=self.config.huber_delta)

    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> dict:
        values   = self.elementwise(predictions.float(), targets.float())
        channels = {name: WeightedMean.of(values[:, index], weights[:, 0]) for index, name in enumerate(self.target_channels)}

        weight_sum = sum(self.channel_weights)
        total      = sum(weight * channels[name].total for name, weight in zip(self.target_channels, self.channel_weights)) / weight_sum
        pixels     = channels[self.target_channels[0]].weight

        return {"total_loss": WeightedMean(total, pixels), "channels": channels}
