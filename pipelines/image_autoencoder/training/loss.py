from __future__ import annotations

from tools.loss.reconstruction_loss import ReconstructionLoss


class Loss(ReconstructionLoss):
    def __init__(self, ae_loss_cfg) -> None:
        super().__init__(ae_loss_cfg.recon_kind, "image_recon", ae_loss_cfg.huber_delta, ae_loss_cfg.charbonnier_eps)
