from __future__ import annotations

import numpy as np

from pipelines.autoencoder_common.inference.metrics import AeMetricsBase


class ImageAeMetrics(AeMetricsBase):
    def __init__(self, result) -> None:
        super().__init__(result)

        self.n_patches  = self.gt.shape[0]
        self.n_channels = self.gt.shape[1]

    def per_patch_mse(self) -> np.ndarray:
        diff = self.pred - self.gt
        return np.mean(diff ** 2, axis=(1, 2, 3), dtype=np.float64)

    def _physical_errors(self) -> dict:
        diff = self.pred - self.gt
        mean = float(np.mean(self.gt, dtype=np.float64))

        sse = float(np.sum(diff ** 2, dtype=np.float64))
        sst = float(np.sum((self.gt - mean) ** 2, dtype=np.float64))

        data_range = float(self.gt.max() - self.gt.min())
        mse        = float(np.mean(diff ** 2, dtype=np.float64))
        psnr       = float(10.0 * np.log10((data_range ** 2) / (mse + self.EPS))) if data_range > 0 else float("nan")

        return {
            "mse_mean"           : mse,
            "mse_median"         : float(np.median(self.per_patch_mse())),
            "mae_mean"           : float(np.mean(np.abs(diff), dtype=np.float64)),
            "rmse"               : float(np.sqrt(mse)),
            "max_abs_error_mean" : float(np.mean(np.max(np.abs(diff), axis=(1, 2, 3)), dtype=np.float64)),
            "r2"                 : float(1.0 - sse / (sst + self.EPS)),
            "psnr"               : psnr,
        }

    def _per_channel_errors(self) -> dict:
        diff = self.pred - self.gt

        channel_mse = np.mean(diff ** 2,    axis=(0, 2, 3), dtype=np.float64)
        channel_mae = np.mean(np.abs(diff), axis=(0, 2, 3), dtype=np.float64)

        return {
            "channel_mse" : [float(v) for v in channel_mse],
            "channel_mae" : [float(v) for v in channel_mae],
        }

    def compute(self) -> dict:
        metrics = {
            "n_patches"     : int(self.n_patches),
            "n_channels"    : int(self.n_channels),
            "patch_height"  : int(self.gt.shape[2]),
            "patch_width"   : int(self.gt.shape[3]),
            "embedding_dim" : int(self.emb.shape[1]),
        }

        metrics.update(self._physical_errors())
        metrics.update(self._normalized_errors())
        metrics.update(self._per_channel_errors())
        metrics.update(self._embedding_stats())

        return metrics
