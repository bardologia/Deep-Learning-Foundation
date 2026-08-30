"""Reconstruction objective for the autoencoder training pipelines.

Sums a configurable set of weighted penalties between a reconstruction and its
target curve or patch, and emits the component breakdown the trainers log. The
available term kinds cover the pointwise robust penalties, the cosine shape and
log-energy terms, and the first- and second-difference smoothness penalties on
the residual.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tools.loss.curve_loss import CurveLoss


@dataclass(frozen=True)
class ReconstructionTerm:
    """One weighted term of a reconstruction objective.

    Attributes:
        kind      : Penalty kind, one of the keys of ``ReconstructionLoss.KINDS``.
        component : Key under which the unweighted term value is reported.
        weight    : Weight of the term in the total.
    """

    kind      : str
    component : str
    weight    : float


class ReconstructionLoss:
    """Weighted sum of reconstruction penalties with a component breakdown.

    Attributes:
        KINDS           : Names of the supported penalty kinds.
        terms           : Enabled terms in evaluation order.
        huber_delta     : Transition magnitude of the Huber penalty.
        charbonnier_eps : Smoothing constant of the Charbonnier penalty.
    """

    KINDS = ("mse", "l1", "huber", "charbonnier", "cosine", "log_energy", "tv", "sobolev")

    def __init__(self, terms: tuple[ReconstructionTerm, ...], huber_delta: float, charbonnier_eps: float) -> None:
        """Validates and stores the enabled terms and robustness parameters.

        Args:
            terms           : Enabled terms in evaluation order.
            huber_delta     : Transition magnitude of the Huber penalty.
            charbonnier_eps : Smoothing constant of the Charbonnier penalty.

        Raises:
            ValueError : If no term is enabled or a term kind is unknown.
        """
        if not terms:
            raise ValueError("Reconstruction loss needs at least one enabled term")

        unknown = [term.kind for term in terms if term.kind not in self.KINDS]
        if unknown:
            raise ValueError(f"Unknown reconstruction loss kinds {unknown}. Available: {', '.join(self.KINDS)}")

        self.terms           = tuple(terms)
        self.huber_delta     = huber_delta
        self.charbonnier_eps = charbonnier_eps

    @property
    def loss_generation(self) -> int:
        """Schema generation of this loss, used by the trainers to route logging."""
        return 0

    def term_value(self, kind: str, prediction: torch.Tensor, target: torch.Tensor, diff: torch.Tensor) -> torch.Tensor:
        """Returns the unweighted value of one penalty kind.

        Args:
            kind       : Penalty kind, one of ``KINDS``.
            prediction : Reconstruction of shape (batch, elevation, ...).
            target     : Target of the same shape.
            diff       : Precomputed residual ``prediction - target``.

        Returns:
            Scalar penalty.

        Raises:
            ValueError : If the kind has no penalty branch.
        """
        if   kind == "mse":         return CurveLoss.mse_diff(diff)
        elif kind == "l1":          return CurveLoss.l1_diff(diff)
        elif kind == "huber":       return CurveLoss.huber_diff(diff, self.huber_delta)
        elif kind == "charbonnier": return CurveLoss.charbonnier_diff(diff, self.charbonnier_eps)
        elif kind == "cosine":      return CurveLoss.cosine(prediction, target, axis=1)
        elif kind == "log_energy":  return CurveLoss.log_energy(prediction, target, axis=1)
        elif kind == "tv":          return CurveLoss.l1_diff(diff[:, 1:] - diff[:, :-1])
        elif kind == "sobolev":     return CurveLoss.mse_diff(CurveLoss.second_difference(diff))

        raise ValueError(f"Unknown reconstruction loss kind '{kind}'. Available: {', '.join(self.KINDS)}")

    def evaluate(self, prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Returns the weighted total and the component breakdown.

        Args:
            prediction : Reconstruction of shape (batch, elevation, ...).
            target     : Target of the same shape.

        Returns:
            Tuple of the scalar total and a mapping of component key to the
            unweighted scalar value of each enabled term.
        """
        diff       = prediction - target
        components = {}
        total      = torch.zeros((), dtype=prediction.dtype, device=prediction.device)

        for term in self.terms:
            value = self.term_value(term.kind, prediction, target, diff)

            components[term.component] = value
            total = total + term.weight * value

        return total, components

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> dict:
        """Evaluates the objective in the payload shape the trainers consume.

        Args:
            prediction : Reconstruction of shape (batch, elevation, ...).
            target     : Target of the same shape.

        Returns:
            Mapping with the scalar "total_loss", the "components" breakdown,
            and empty "monitor", "occupancy" and "physical" sections.
        """
        total, components = self.evaluate(prediction, target)

        return {
            "total_loss" : total,
            "components" : components,
            "monitor"    : {},
            "occupancy"  : {},
            "physical"   : {},
        }
