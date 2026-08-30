"""Batched reconstruction of image patches by a trained image autoencoder.

Runs the encoder-decoder over the split loader, denormalizes both the input and
its reconstruction, and reduces spatial latents to one embedding vector per
patch.
"""

from __future__ import annotations

import torch

from pipelines.autoencoder_common.inference.predictor import AeReconstructionPredictor, AeResult, BatchReconstruction


class ImageAeResult(AeResult):
    """Reconstruction result of an image-autoencoder split, patches stacked along axis 0."""

    pass


class ImageAePredictor(AeReconstructionPredictor):
    """Reconstructs every patch of a split and collects the latent embeddings."""

    SECTION      = "[Image AE Inference: Reconstruct]"
    RESULT_CLASS = ImageAeResult

    def _batch_input(self, batch):
        """Returns the normalized input patch tensor of a dataset batch."""
        return batch[0]

    def _reconstruct_batch(self, image_n: torch.Tensor) -> BatchReconstruction:
        """Reconstructs one batch of normalized patches and denormalizes the result.

        Args:
            image_n : Normalized input patches of shape (batch, channels, rows, cols).

        Returns:
            The batch reconstruction holding denormalized ground truth and
            prediction, the normalized pair, and per-patch embeddings of shape
            (batch, embedding_dim), spatially averaged when the latent is a map.
        """
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
        """Returns the patch count, channel count, patch extent and latent width for logging."""
        return {
            "Patches"       : result.gt.shape[0],
            "Channels"      : result.gt.shape[1],
            "Patch size"    : f"{result.gt.shape[2]} x {result.gt.shape[3]}",
            "Embedding dim" : result.embeddings.shape[1],
        }
