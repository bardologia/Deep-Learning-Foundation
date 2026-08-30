"""Reconstruction loss of the image autoencoder."""

from __future__ import annotations

from tools.loss.reconstruction_loss import ReconstructionLoss, ReconstructionTerm


class Loss(ReconstructionLoss):
    """Pixel-space reconstruction loss between an input patch stack and its reconstruction.

    Only the elementwise penalties of the shared objective are admissible here:
    the shape, energy and smoothness kinds reduce along axis 1, which is the
    channel axis of a (batch, channels, height, width) patch stack rather than
    an elevation axis, so they are rejected.

    Attributes:
        KINDS : Reconstruction penalty kinds the image autoencoder accepts.
    """

    KINDS = ("mse", "l1", "huber", "charbonnier")

    def __init__(self, ae_loss_cfg) -> None:
        """Configures the single pointwise term and its robustness parameters.

        Args:
            ae_loss_cfg : Autoencoder loss configuration holding `recon_kind`, `huber_delta` and `charbonnier_eps`.

        Raises:
            ValueError : If `recon_kind` is not one of mse, l1, huber or charbonnier.
        """
        term = ReconstructionTerm(ae_loss_cfg.recon_kind, "image_recon", 1.0)

        super().__init__((term,), ae_loss_cfg.huber_delta, ae_loss_cfg.charbonnier_eps)
