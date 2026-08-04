from __future__ import annotations

import torch

from pipelines.autoencoder_common.inference.predictor import AeReconstructionPredictor, AeResult, BatchReconstruction


class ImageAeResult(AeResult):
    pass


class ImageAePredictor(AeReconstructionPredictor):
    SECTION      = "[Image AE Inference: Reconstruct]"
    RESULT_CLASS = ImageAeResult

    def _batch_input(self, batch):
        return batch[0]

    def _reconstruct_batch(self, image_n: torch.Tensor) -> BatchReconstruction:
        image_n = image_n.to(self.device)

        with torch.no_grad():
            image_hat_n, z = self.model.reconstruct(image_n)

        gt_n   = image_n.cpu().numpy()
        pred_n = image_hat_n.cpu().numpy()
        emb    = z.mean(dim=tuple(range(2, z.ndim))) if z.ndim > 2 else z

        return BatchReconstruction(
            gt     = self.normalizer.denormalize_input(gt_n),
            pred   = self.normalizer.denormalize_input(pred_n),
            emb    = emb.cpu().numpy(),
            gt_n   = gt_n,
            pred_n = pred_n,
        )

    def _summary(self, result) -> dict:
        return {
            "Patches"       : result.gt.shape[0],
            "Channels"      : result.gt.shape[1],
            "Patch size"    : f"{result.gt.shape[2]} x {result.gt.shape[3]}",
            "Embedding dim" : result.embeddings.shape[1],
        }
