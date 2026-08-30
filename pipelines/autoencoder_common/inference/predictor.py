"""Split-wide reconstruction loop shared by the autoencoder families.

Batches are reconstructed one at a time and written into pre-allocated arrays
sized to the split, while the normalized-domain error is accumulated on the fly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tools.monitoring.logger import Logger


@dataclass
class BatchReconstruction:
    """Reconstruction of one batch in both physical and normalized units.

    Attributes:
        gt     : Ground-truth targets of shape (batch, *sample_shape).
        pred   : Reconstructions of the same shape as gt.
        emb    : Latent codes of shape (batch, embedding_dim).
        gt_n   : Ground-truth targets in normalized units.
        pred_n : Reconstructions in normalized units.
    """

    gt     : np.ndarray
    pred   : np.ndarray
    emb    : np.ndarray
    gt_n   : np.ndarray
    pred_n : np.ndarray


@dataclass
class AeResult:
    """Reconstruction of a whole split.

    Attributes:
        gt                : Ground-truth targets of shape (n_samples, *sample_shape).
        pred              : Reconstructions of the same shape as gt.
        embeddings        : Latent codes of shape (n_samples, embedding_dim).
        normalized_errors : Mean squared and absolute error in normalized units.
    """

    gt                : np.ndarray
    pred              : np.ndarray
    embeddings        : np.ndarray
    normalized_errors : dict


class NormalizedErrorAccumulator:
    """Running sums of squared and absolute error in normalized units.

    Attributes:
        squared  : Sum of squared residuals over all elements seen.
        absolute : Sum of absolute residuals over all elements seen.
        count    : Number of elements accumulated.
    """

    def __init__(self) -> None:
        """Starts the accumulator at zero."""
        self.squared  = 0.0
        self.absolute = 0.0
        self.count    = 0

    def update(self, gt_n: np.ndarray, pred_n: np.ndarray) -> None:
        """Accumulates the residual of one batch.

        Args:
            gt_n   : Ground truth in normalized units, any shape.
            pred_n : Reconstruction in normalized units, same shape as gt_n.
        """
        diff = np.asarray(pred_n, dtype=np.float32) - np.asarray(gt_n, dtype=np.float32)

        self.squared  += float(np.sum(diff * diff,     dtype=np.float64))
        self.absolute += float(np.sum(np.abs(diff),    dtype=np.float64))
        self.count    += int(diff.size)

    def as_dict(self) -> dict:
        """Returns the mean normalized squared and absolute error.

        Returns:
            Mapping with keys 'mse_mean_normalized' and 'mae_mean_normalized'.

        Raises:
            ValueError : If no elements were accumulated.
        """
        if self.count == 0:
            raise ValueError("The reconstruction loader yielded no batches; the normalized reconstruction error is undefined")

        return {
            "mse_mean_normalized" : self.squared  / self.count,
            "mae_mean_normalized" : self.absolute / self.count,
        }


class AeReconstructionPredictor:
    """Reconstructs an entire split batch by batch into fixed result arrays.

    Attributes:
        run        : Loaded run bundle carrying the model, normalizer and split loader.
        device     : Torch device the model runs on.
        logger     : Logger used for the section header and summary table.
        model      : The trained autoencoder.
        normalizer : Normalizer used to map between physical and normalized units.
    """

    SECTION      = "[AE Inference: Reconstruct]"
    RESULT_CLASS = AeResult

    def __init__(self, run, device: str, logger: Logger) -> None:
        """Binds the loaded run and inference device.

        Args:
            run    : Loaded run bundle with model, normalizer and split loader.
            device : Torch device string, e.g. 'cuda:0'.
            logger : Logger used for reporting progress.
        """
        self.run        = run
        self.device     = device
        self.logger     = logger
        self.model      = run.model
        self.normalizer = run.normalizer

    def _batch_input(self, batch):
        """Returns the model input extracted from a loader batch; implemented by subclasses."""
        raise NotImplementedError

    def _reconstruct_batch(self, batch_input) -> BatchReconstruction:
        """Returns the reconstruction of one batch; implemented by subclasses."""
        raise NotImplementedError

    def _allocate(self, reconstruction: BatchReconstruction, n_samples: int):
        """Returns an empty result sized to the split from the first batch's shapes."""
        return self.RESULT_CLASS(
            gt                = np.empty((n_samples, *reconstruction.gt.shape[1:]),   dtype=np.float32),
            pred              = np.empty((n_samples, *reconstruction.pred.shape[1:]), dtype=np.float32),
            embeddings        = np.empty((n_samples, *reconstruction.emb.shape[1:]),  dtype=np.float32),
            normalized_errors = {},
        )

    def _summary(self, result) -> dict:
        """Returns the log table summarising the split result; implemented by subclasses."""
        raise NotImplementedError

    def run_inference(self):
        """Reconstructs every batch of the split into one result object.

        Returns:
            The split result carrying ground truth, reconstructions, latent
            embeddings and the normalized-domain errors.

        Raises:
            ValueError : If the loader yields no batches, yields more samples than the dataset holds, or covers only part of the split.
        """
        self.logger.section(self.SECTION)

        n_samples  = len(self.run.loader.dataset)
        normalized = NormalizedErrorAccumulator()
        result     = None
        filled     = 0

        for batch in self.run.loader:
            reconstruction = self._reconstruct_batch(self._batch_input(batch))
            size           = reconstruction.gt.shape[0]

            if result is None:
                result = self._allocate(reconstruction, n_samples)

            if filled + size > n_samples:
                raise ValueError(f"The reconstruction loader yielded more than the {n_samples} samples its dataset holds; the split cannot be assembled into a fixed result array.")

            result.gt        [filled:filled + size] = reconstruction.gt
            result.pred      [filled:filled + size] = reconstruction.pred
            result.embeddings[filled:filled + size] = reconstruction.emb

            normalized.update(reconstruction.gt_n, reconstruction.pred_n)
            filled += size

        if result is None:
            raise ValueError("The reconstruction loader yielded no batches; the reconstruction result is undefined")

        if filled != n_samples:
            raise ValueError(f"The reconstruction loader yielded {filled} of the {n_samples} samples its dataset holds; inference must cover the whole split, so the loader must not drop the last batch.")

        result.normalized_errors = normalized.as_dict()

        self.logger.kv_table(self._summary(result))

        return result
