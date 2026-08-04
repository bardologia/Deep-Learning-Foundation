from __future__ import annotations

import torch

from tools.loss.curve_loss import CurveLoss


class ReconstructionLoss:
    def __init__(self, kind: str, component: str, huber_delta: float, charbonnier_eps: float) -> None:
        self.kind            = kind
        self.component       = component
        self.huber_delta     = huber_delta
        self.charbonnier_eps = charbonnier_eps

    @property
    def loss_generation(self) -> int:
        return 0

    def _value(self, diff: torch.Tensor) -> torch.Tensor:
        if   self.kind == "mse":         return CurveLoss.mse_diff(diff)
        elif self.kind == "l1":          return CurveLoss.l1_diff(diff)
        elif self.kind == "huber":       return CurveLoss.huber_diff(diff, self.huber_delta)
        elif self.kind == "charbonnier": return CurveLoss.charbonnier_diff(diff, self.charbonnier_eps)
        else:                            raise ValueError(f"Unknown reconstruction loss kind '{self.kind}'. Available: mse, l1, huber, charbonnier")

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> dict:
        value = self._value(prediction - target)

        return {
            "total_loss" : value,
            "components" : {self.component: value},
            "monitor"    : {},
            "occupancy"  : {},
            "physical"   : {},
        }
