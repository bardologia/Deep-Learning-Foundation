from __future__ import annotations

from tools.loss.reconstruction_loss import ReconstructionLoss


class Loss(ReconstructionLoss):
    def __init__(self, ae_cfg) -> None:
        super().__init__(ae_cfg.curve_kind, "curve_recon", ae_cfg.huber_delta, ae_cfg.charbonnier_eps)
