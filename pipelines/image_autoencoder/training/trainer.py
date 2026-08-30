"""Training loop of the image autoencoder."""

from __future__ import annotations

from pipelines.autoencoder_common.trainer      import AutoencoderTrainer
from pipelines.image_autoencoder.training.loss import Loss


class Trainer(AutoencoderTrainer):
    """Trains the image autoencoder to reconstruct its own normalized input patches."""

    stage_name    = "Image Autoencoder"
    section_title = "[Image Autoencoder Training]"

    def _build_criterion(self):
        """Returns the reconstruction loss and logs its configured settings."""
        cfg = self.config.ae_loss

        self.logger.section("[Loss Function]")
        self.logger.kv_table({
            "Reconstruction kind" : cfg.recon_kind,
            "Huber delta"         : f"{cfg.huber_delta:g} (used only when kind=huber)",
            "Charbonnier eps"     : f"{cfg.charbonnier_eps:g} (used only when kind=charbonnier)",
        })

        return Loss(cfg)

    def _compute_loss(self, batch):
        """Returns the reconstruction loss of one batch.

        Args:
            batch : Dataset batch whose first element holds the normalized input patches of shape (batch, channels, rows, cols).
        """
        image        = batch[0].to(self.device)
        image_hat, _ = self.model.reconstruct(image)
        return self.criterion(image_hat, image)
