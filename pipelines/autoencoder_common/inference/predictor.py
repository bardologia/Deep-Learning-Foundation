from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tools.monitoring.logger import Logger


@dataclass
class BatchReconstruction:
    gt     : np.ndarray
    pred   : np.ndarray
    emb    : np.ndarray
    gt_n   : np.ndarray
    pred_n : np.ndarray


@dataclass
class AeResult:
    gt                : np.ndarray
    pred              : np.ndarray
    embeddings        : np.ndarray
    normalized_errors : dict


class NormalizedErrorAccumulator:
    def __init__(self) -> None:
        self.squared  = 0.0
        self.absolute = 0.0
        self.count    = 0

    def update(self, gt_n: np.ndarray, pred_n: np.ndarray) -> None:
        diff = np.asarray(pred_n, dtype=np.float64) - np.asarray(gt_n, dtype=np.float64)

        self.squared  += float(np.sum(diff ** 2))
        self.absolute += float(np.sum(np.abs(diff)))
        self.count    += int(diff.size)

    def as_dict(self) -> dict:
        if self.count == 0:
            raise ValueError("The reconstruction loader yielded no batches; the normalized reconstruction error is undefined")

        return {
            "mse_mean_normalized" : self.squared  / self.count,
            "mae_mean_normalized" : self.absolute / self.count,
        }


class AeReconstructionPredictor:
    SECTION      = "[AE Inference: Reconstruct]"
    RESULT_CLASS = AeResult

    def __init__(self, run, device: str, logger: Logger) -> None:
        self.run        = run
        self.device     = device
        self.logger     = logger
        self.model      = run.model
        self.normalizer = run.normalizer

    def _batch_input(self, batch):
        raise NotImplementedError

    def _reconstruct_batch(self, batch_input) -> BatchReconstruction:
        raise NotImplementedError

    def _summary(self, result) -> dict:
        raise NotImplementedError

    def run_inference(self):
        self.logger.section(self.SECTION)

        gt_chunks, pred_chunks, emb_chunks = [], [], []
        normalized                         = NormalizedErrorAccumulator()

        for batch in self.run.loader:
            reconstruction = self._reconstruct_batch(self._batch_input(batch))

            gt_chunks.append(reconstruction.gt)
            pred_chunks.append(reconstruction.pred)
            emb_chunks.append(reconstruction.emb)

            normalized.update(reconstruction.gt_n, reconstruction.pred_n)

        result = self.RESULT_CLASS(
            gt                = np.concatenate(gt_chunks,   axis=0).astype(np.float32),
            pred              = np.concatenate(pred_chunks, axis=0).astype(np.float32),
            embeddings        = np.concatenate(emb_chunks,  axis=0).astype(np.float32),
            normalized_errors = normalized.as_dict(),
        )

        self.logger.kv_table(self._summary(result))

        return result
