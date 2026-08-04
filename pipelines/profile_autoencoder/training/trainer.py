from __future__ import annotations

from pipelines.autoencoder_common.trainer        import AutoencoderTrainer
from pipelines.profile_autoencoder.training.loss import Loss


class Trainer(AutoencoderTrainer):
    stage_name    = "Profile Autoencoder"
    section_title = "[Profile Autoencoder Training]"

    def _build_criterion(self):
        cfg = self.config.ae_loss

        self.logger.section("[Loss Function]")
        self.logger.kv_table({
            "Curve kind"      : cfg.curve_kind,
            "Huber delta"     : f"{cfg.huber_delta:g} (used only when kind=huber)",
            "Charbonnier eps" : f"{cfg.charbonnier_eps:g} (used only when kind=charbonnier)",
        })

        return Loss(cfg)

    def _compute_loss(self, batch):
        curve        = batch.to(self.device).unsqueeze(-1).unsqueeze(-1)
        curve_hat, _ = self.model.reconstruct(curve)
        return self.criterion(curve_hat, curve)
