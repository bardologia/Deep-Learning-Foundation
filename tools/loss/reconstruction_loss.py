
from __future__ import annotations

from dataclasses import dataclass

import torch

from tools.loss.curve_loss import CurveLoss


@dataclass(frozen=True)
class ReconstructionTerm:

    kind      : str
    component : str
    weight    : float


class ReconstructionLoss:

    KINDS = ("mse", "l1", "huber", "charbonnier", "cosine", "log_energy", "tv", "sobolev")

    def __init__(self, terms: tuple[ReconstructionTerm, ...], huber_delta: float, charbonnier_eps: float) -> None:
        self.terms           = tuple(terms)
        self.huber_delta     = huber_delta
        self.charbonnier_eps = charbonnier_eps

    @property
    def loss_generation(self) -> int:
        return 0

    def term_value(self, kind: str, prediction: torch.Tensor, target: torch.Tensor, diff: torch.Tensor) -> torch.Tensor:
        if   kind == "mse":         return CurveLoss.mse_diff(diff)
        elif kind == "l1":          return CurveLoss.l1_diff(diff)
        elif kind == "huber":       return CurveLoss.huber_diff(diff, self.huber_delta)
        elif kind == "charbonnier": return CurveLoss.charbonnier_diff(diff, self.charbonnier_eps)
        elif kind == "cosine":      return CurveLoss.cosine(prediction, target, axis=1)
        elif kind == "log_energy":  return CurveLoss.log_energy(prediction, target, axis=1)
        elif kind == "tv":          return CurveLoss.l1_diff(diff[:, 1:] - diff[:, :-1])

        return CurveLoss.mse_diff(CurveLoss.second_difference(diff))

    def evaluate(self, prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        diff       = prediction - target
        components = {}
        total      = torch.zeros((), dtype=prediction.dtype, device=prediction.device)

        for term in self.terms:
            value = self.term_value(term.kind, prediction, target, diff)

            components[term.component] = value
            total = total + term.weight * value

        return total, components

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> dict:
        total, components = self.evaluate(prediction, target)

        return {
            "total_loss" : total,
            "components" : components,
            "monitor"    : {},
            "occupancy"  : {},
            "physical"   : {},
        }
