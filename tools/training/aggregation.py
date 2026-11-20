from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class WeightedMean:
    FLOOR = 1e-12

    total  : torch.Tensor
    weight : torch.Tensor

    @classmethod
    def of(cls, values: torch.Tensor, weights: torch.Tensor) -> "WeightedMean":
        weights = torch.broadcast_to(weights, values.shape)

        return cls((weights * values).sum(), weights.sum())

    @property
    def value(self) -> torch.Tensor:
        return self.total / self.weight.clamp(min=self.FLOOR)


class MetricAggregator:

    def __init__(self) -> None:
        self.totals  = {}
        self.weights = {}

    def add(self, loss_dict: dict) -> None:
        for key, mean in {"total": loss_dict["total_loss"], **loss_dict["channels"]}.items():
            self.totals[key]  = self.totals.get(key, 0.0)  + mean.total.detach()
            self.weights[key] = self.weights.get(key, 0.0) + mean.weight.detach()

    def _reduce(self, key: str) -> float:
        return float(self.totals[key]) / max(float(self.weights[key]), WeightedMean.FLOOR)

    def reduce_total(self) -> float:
        return self._reduce("total")

    def reduce_channels(self) -> dict:
        return {key: self._reduce(key) for key in self.totals if key != "total"}


class ScalarsPage:

    STAGES = (("train", "training"), ("val", "validation"))

    def __init__(self) -> None:
        self.categories = {}

    def add(self, category: str, title: str, patterns: list) -> None:
        self.categories.setdefault(category, {})[title] = list(patterns)

    def add_by_stage(self, category: str, label: str, family: str) -> None:
        for stage, word in self.STAGES:
            self.add(category, f"{label}, {word}", [f"{family}/.+/{stage}$"])

    def loss(self) -> "ScalarsPage":
        self.add("Loss", "Total loss by stage", ["loss/train$", "loss/val$", "loss/test$"])
        self.add_by_stage("Loss", "Loss per target channel", "loss_channels")
        return self

    def optimiser(self) -> "ScalarsPage":
        self.add("Optimiser", "Learning rate per group", ["optim/lr/"])
        self.add("Optimiser", "Gradient norm, total and per group", ["optim/grad_norm$", "optim/grad_norm/"])
        return self

    def layout(self) -> dict:
        return {category: {title: ["Multiline", list(patterns)] for title, patterns in charts.items()} for category, charts in self.categories.items()}
