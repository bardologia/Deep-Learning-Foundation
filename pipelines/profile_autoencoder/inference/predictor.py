"""Batched reconstruction of elevation profiles by a trained profile autoencoder.

Feeds normalised curves through the autoencoder and returns both normalised and
denormalised ground truth and reconstructions together with the latent codes.
"""

from __future__ import annotations

import torch

from pipelines.autoencoder_common.inference.predictor import AeReconstructionPredictor, AeResult, BatchReconstruction


class ProfileAeResult(AeResult):
    """Reconstruction result of a profile-autoencoder run."""

    pass


class ProfileAePredictor(AeReconstructionPredictor):
    """Reconstructs a split of elevation profiles batch by batch."""

    SECTION      = "[Profile AE Inference: Reconstruct]"
    RESULT_CLASS = ProfileAeResult

    def _batch_input(self, batch):
        """Returns the noisy curve tensor of shape (batch, H) from a loader batch."""
        return batch[0]

    def _reconstruct_batch(self, curve_n: torch.Tensor) -> BatchReconstruction:
        """Reconstructs one batch of normalised curves.

        Args:
            curve_n : Normalised profiles of shape (batch, H).

        Returns:
            Ground truth and reconstruction in physical and normalised units,
            each of shape (batch, H), plus latent codes of shape
            (batch, embedding_dim).
        """
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
        """Returns the curve count, profile length and latent size for logging."""
        return {
            "Curves"         : result.gt.shape[0],
            "Profile length" : result.gt.shape[1],
            "Embedding dim"  : result.embeddings.shape[1],
        }
