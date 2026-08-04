from __future__ import annotations

import torch

from pipelines.autoencoder_common.inference.predictor import AeReconstructionPredictor, AeResult, BatchReconstruction


class ProfileAeResult(AeResult):
    pass


class ProfileAePredictor(AeReconstructionPredictor):
    SECTION      = "[Profile AE Inference: Reconstruct]"
    RESULT_CLASS = ProfileAeResult

    def _batch_input(self, batch):
        return batch

    def _reconstruct_batch(self, curve_n: torch.Tensor) -> BatchReconstruction:
        x = curve_n.to(self.device).unsqueeze(-1).unsqueeze(-1)

        with torch.no_grad():
            curve_hat_n, z = self.model.reconstruct(x)

        gt   = self.normalizer.denormalize(x).squeeze(-1).squeeze(-1)
        pred = self.normalizer.denormalize(curve_hat_n).squeeze(-1).squeeze(-1)
        emb  = z.squeeze(-1).squeeze(-1)

        return BatchReconstruction(
            gt     = gt.cpu().numpy(),
            pred   = pred.cpu().numpy(),
            emb    = emb.cpu().numpy(),
            gt_n   = x.squeeze(-1).squeeze(-1).cpu().numpy(),
            pred_n = curve_hat_n.squeeze(-1).squeeze(-1).cpu().numpy(),
        )

    def _summary(self, result) -> dict:
        return {
            "Curves"         : result.gt.shape[0],
            "Profile length" : result.gt.shape[1],
            "Embedding dim"  : result.embeddings.shape[1],
        }
