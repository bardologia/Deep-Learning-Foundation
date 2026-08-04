from __future__ import annotations

import torch

from tools.data.gaussians  import GaussianCurve
from tools.loss.curve_loss import CurveLoss


class Loss:
    def __init__(self, autoencoder, target_provider, embedding_cfg, x_axis, norm_stats, params_per_gaussian: int, profile_normalizer) -> None:
        self.autoencoder        = autoencoder
        self.target_provider    = target_provider
        self.emb_cfg            = embedding_cfg
        self.x_axis             = x_axis
        self.norm_stats         = norm_stats
        self.ppg                = params_per_gaussian
        self.profile_normalizer = profile_normalizer

    @property
    def loss_generation(self) -> int:
        return 0

    def _embedding_terms(self, z_hat, z_star) -> tuple[torch.Tensor, dict]:
        cfg        = self.emb_cfg
        components = {}
        total      = torch.zeros((), dtype=z_hat.dtype, device=z_hat.device)

        if cfg.use_embedding_mse:
            val = CurveLoss.mse_diff(z_hat - z_star)
            components["embedding_mse"] = val
            total = total + cfg.weight_embedding_mse * val

        if cfg.use_embedding_cosine:
            val = CurveLoss.cosine(z_hat, z_star, axis=1)
            components["embedding_cosine"] = val
            total = total + cfg.weight_embedding_cosine * val

        if cfg.use_embedding_smoothl1:
            val = CurveLoss.smooth_l1_diff(z_hat - z_star, cfg.smoothl1_beta)
            components["embedding_smoothl1"] = val
            total = total + cfg.weight_embedding_smoothl1 * val

        return total, components

    def __call__(self, z_hat, gt_params_norm) -> dict:
        with torch.no_grad():
            gt_phys    = self.norm_stats.denormalize_output(gt_params_norm.float())
            gt_curve   = GaussianCurve.reconstruct(gt_phys, self.x_axis, self.ppg).to(z_hat.dtype)
            gt_curve_n = self.profile_normalizer.normalize(gt_curve)

        z_hat_n    = self.autoencoder.normalize_embedding(z_hat)
        z_star_raw = self.target_provider.target(self.autoencoder.encoder, gt_curve_n)
        z_star_n   = self.autoencoder.normalize_embedding(z_star_raw)

        total, components = self._embedding_terms(z_hat_n, z_star_n)

        if self.emb_cfg.use_curve_recon:
            cfg       = self.emb_cfg
            curve_hat = self.autoencoder.decode(z_hat_n)
            diff      = curve_hat - gt_curve_n

            if   cfg.curve_kind == "mse":         c_val = CurveLoss.mse_diff(diff)
            elif cfg.curve_kind == "l1":          c_val = CurveLoss.l1_diff(diff)
            elif cfg.curve_kind == "huber":       c_val = CurveLoss.huber_diff(diff, cfg.huber_delta)
            elif cfg.curve_kind == "charbonnier": c_val = CurveLoss.charbonnier_diff(diff, cfg.charbonnier_eps)
            else:                                 raise ValueError(f"Unknown curve loss kind '{cfg.curve_kind}'. Available: mse, l1, huber, charbonnier")

            components["curve_recon"] = c_val
            total = total + cfg.weight_curve_recon * c_val

        return {"total_loss": total, "components": components, "monitor": {}, "occupancy": {}, "physical": {}}
