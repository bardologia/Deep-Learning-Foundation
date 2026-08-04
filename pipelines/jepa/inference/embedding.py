from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from tools.data.gaussians    import GaussianReconstructor
from tools.monitoring.logger import Logger


class JepaEmbeddingEvaluator:
    def __init__(self, run, logger: Logger) -> None:
        self._run    = run
        self.logger  = logger
        self.adapter = run.model.module
        self.device  = torch.device(run.model.device)

    def _gt_curves(self, gt_params_batch: torch.Tensor, x: np.ndarray) -> torch.Tensor:
        run = self._run
        n_K = run.n_gaussians

        gt_params = gt_params_batch[:, :n_K * 3]
        gt_phys   = run.dataset.normalizer.denormalize_output(gt_params).cpu().numpy().astype(np.float32)

        B, _, H, W = gt_phys.shape
        gt_gauss   = gt_phys.reshape(B, n_K, 3, H, W)
        gt_curves  = GaussianReconstructor.reconstruct_batch(gt_gauss, x)

        return torch.from_numpy(gt_curves.astype(np.float32)).to(self.device)

    def _batch_terms(self, images: torch.Tensor, gt_curves: torch.Tensor) -> Dict[str, float]:
        jepa        = self.adapter.jepa
        autoencoder = jepa.profile_autoencoder
        normalizer  = self.adapter.profile_normalizer

        z_hat_n    = autoencoder.normalize_embedding(jepa(images))
        gt_curve_n = normalizer.normalize(gt_curves)
        z_star_n   = autoencoder.encode(gt_curve_n)

        decode_hat  = autoencoder.decode(z_hat_n)
        decode_star = autoencoder.decode(z_star_n)

        cosine = F.cosine_similarity(z_hat_n, z_star_n, dim=1)

        return {
            "embedding_sq" : float(((z_hat_n - z_star_n) ** 2).sum()),
            "embedding_n"  : float(z_hat_n.numel()),
            "cosine_sum"   : float(cosine.sum()),
            "cosine_n"     : float(cosine.numel()),
            "decode_sq"    : float(((decode_star - gt_curve_n) ** 2).sum()),
            "chain_sq"     : float(((decode_hat  - gt_curve_n) ** 2).sum()),
            "curve_n"      : float(gt_curve_n.numel()),
        }

    def _accumulate(self) -> Dict[str, float]:
        x    = np.asarray(self._run.x_axis, dtype=np.float32).reshape(1, 1, -1, 1, 1)
        sums = {}

        with torch.no_grad():
            for batch in self._run.loader:
                images    = batch[0].to(self.device)
                gt_curves = self._gt_curves(batch[1], x)

                terms = self._batch_terms(images, gt_curves)
                for key, value in terms.items():
                    sums[key] = sums.get(key, 0.0) + value

        if not sums or sums["embedding_n"] == 0.0:
            raise ValueError("JEPA embedding evaluation saw no samples; the inference loader is empty.")

        return sums

    def _report(self, metrics: Dict[str, float]) -> None:
        self.logger.section("[Inference: JEPA Embedding Diagnostics]")
        self.logger.kv_table({
            "Embedding MSE"           : f"{metrics['jepa_embedding_mse']:.6g}",
            "Embedding cosine"        : f"{metrics['jepa_embedding_cosine']:.4f}",
            "Decoder-only MSE (norm)" : f"{metrics['jepa_decode_mse_norm']:.6g}",
            "Full-chain MSE (norm)"   : f"{metrics['jepa_chain_mse_norm']:.6g}",
        })

    def run(self) -> Dict[str, float]:
        sums = self._accumulate()

        metrics = {
            "jepa_embedding_mse"    : sums["embedding_sq"] / sums["embedding_n"],
            "jepa_embedding_cosine" : sums["cosine_sum"]   / sums["cosine_n"],
            "jepa_decode_mse_norm"  : sums["decode_sq"]    / sums["curve_n"],
            "jepa_chain_mse_norm"   : sums["chain_sq"]     / sums["curve_n"],
        }

        self._report(metrics)

        return metrics
