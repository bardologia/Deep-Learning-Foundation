from __future__ import annotations

import torch


class Loss:
    KINDS = ("l1", "mse")

    def __init__(self, kind: str, x_axis: torch.Tensor) -> None:
        if kind not in self.KINDS:
            raise ValueError(f"Unknown curve_loss '{kind}'. Available: {self.KINDS}")

        self.kind   = kind
        self.x_axis = x_axis

    @property
    def loss_generation(self) -> int:
        return 0

    def _curve_error(self, prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        difference = prediction - target
        per_pixel  = difference.abs().mean(dim=1) if self.kind == "l1" else difference.pow(2).mean(dim=1)

        return (per_pixel * weights).sum() / weights.sum().clamp(min=1.0)

    @torch.no_grad()
    def _peak_error_m(self, prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        predicted_peak = self.x_axis[prediction.argmax(dim=1)]
        target_peak    = self.x_axis[target.argmax(dim=1)]

        return ((predicted_peak - target_peak).abs() * weights).sum() / weights.sum().clamp(min=1.0)

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict:
        weights = mask.to(prediction.dtype)
        value   = self._curve_error(prediction, target, weights)

        return {
            "total_loss" : value,
            "components" : {f"curve_{self.kind}": value},
            "monitor"    : {},
            "occupancy"  : {},
            "physical"   : {"peak_mae_m": self._peak_error_m(prediction, target, weights)},
        }
