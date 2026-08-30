"""Trainer specialisation for the 1-D profile autoencoder.

Wires the profile reconstruction loss into the shared autoencoder training loop
and adds the optional latent-noise consistency term.
"""

from __future__ import annotations

import torch

from pipelines.autoencoder_common.trainer        import AutoencoderTrainer
from pipelines.profile_autoencoder.training.loss import Loss


class Trainer(AutoencoderTrainer):
    """Trains a profile autoencoder on noisy/clean elevation-profile pairs.

    Attributes:
        stage_name    : Label used by the shared trainer for progress reporting.
        section_title : Header written to the log when training starts.
    """

    stage_name    = "Profile Autoencoder"
    section_title = "[Profile Autoencoder Training]"

    def _build_criterion(self):
        """Logs the enabled loss terms and returns the profile reconstruction criterion."""
        cfg       = self.config.ae_loss
        criterion = Loss(cfg)

        table = {term.component: f"weight={term.weight:g}" for term in criterion.terms}

        table["Huber delta"]     = f"{cfg.huber_delta:g} (used only by the huber term)"
        table["Charbonnier eps"] = f"{cfg.charbonnier_eps:g} (used only by the charbonnier term)"
        table["Latent noise"]    = f"std={cfg.latent_noise_std:g} weight={cfg.weight_latent_noise:g}" if cfg.use_latent_noise else "off"

        self.logger.section("[Loss Function]")
        self.logger.kv_table(table)

        return criterion

    def _compute_loss(self, batch):
        """Reconstructs a batch of profiles and returns its loss dictionary.

        Args:
            batch : Pair of tensors (noisy, clean), each of shape (batch, x_axis_length), holding the measured and ground-truth elevation profiles in normalized reflectivity units. Both are reshaped to (batch, x_axis_length, 1, 1) before the model call.

        Returns:
            Loss dictionary with key "total_loss" and a "components" sub-dict.
            When latent-noise regularisation is enabled, the decoded output of a
            jittered latent contributes extra "latent_*" components and its
            weighted value is added to the total.
        """
        noisy, clean = batch
        noisy        = noisy.to(self.device).unsqueeze(-1).unsqueeze(-1)
        clean        = clean.to(self.device).unsqueeze(-1).unsqueeze(-1)

        curve_hat, z = self.model.reconstruct(noisy)
        loss_dict    = self.criterion(curve_hat, clean)

        cfg = self.config.ae_loss
        if cfg.use_latent_noise:
            jittered     = self.model.decode(z + torch.randn_like(z) * cfg.latent_noise_std)
            value, terms = self.criterion.evaluate(jittered, clean)

            for name, term in terms.items():
                loss_dict["components"][f"latent_{name}"] = term

            loss_dict["total_loss"] = loss_dict["total_loss"] + cfg.weight_latent_noise * value

        return loss_dict
