from __future__ import annotations

import numpy as np

from pipelines.autoencoder_common.inference.metrics import AeMetricsBase


class ProfileAeMetrics(AeMetricsBase):
    def __init__(self, result, x_axis: np.ndarray, amp_zero_thr: float) -> None:
        super().__init__(result)

        self.x_axis       = np.asarray(x_axis, dtype=np.float64)
        self.amp_zero_thr = amp_zero_thr
        self.active       = self.gt.max(axis=1) > self.amp_zero_thr

    def per_curve_mse(self) -> np.ndarray:
        return np.mean((self.pred - self.gt) ** 2, axis=1, dtype=np.float64)

    def _physical_errors(self) -> dict:
        diff = self.pred - self.gt
        mean = float(np.mean(self.gt, dtype=np.float64))

        sse = float(np.sum(diff ** 2, dtype=np.float64))
        sst = float(np.sum((self.gt - mean) ** 2, dtype=np.float64))

        return {
            "mse_mean"           : float(np.mean(diff ** 2, dtype=np.float64)),
            "mse_median"         : float(np.median(self.per_curve_mse())),
            "mae_mean"           : float(np.mean(np.abs(diff), dtype=np.float64)),
            "rmse"               : float(np.sqrt(np.mean(diff ** 2, dtype=np.float64))),
            "max_abs_error_mean" : float(np.mean(np.max(np.abs(diff), axis=1), dtype=np.float64)),
            "r2"                 : float(1.0 - sse / (sst + self.EPS)),
        }

    def _shape_errors(self) -> dict:
        gt   = self.gt[self.active]
        pred = self.pred[self.active]

        gt_c   = gt   - gt.mean(axis=1,   keepdims=True)
        pred_c = pred - pred.mean(axis=1, keepdims=True)

        denom    = np.sqrt(np.sum(gt_c ** 2, axis=1, dtype=np.float64) * np.sum(pred_c ** 2, axis=1, dtype=np.float64))
        pearson  = np.sum(gt_c * pred_c, axis=1, dtype=np.float64) / (denom + self.EPS)

        rel_l2 = np.linalg.norm(pred - gt, axis=1) / (np.linalg.norm(gt, axis=1) + self.EPS)

        return {
            "pearson_mean"       : float(np.mean(pearson)),
            "pearson_median"     : float(np.median(pearson)),
            "relative_l2_mean"   : float(np.mean(rel_l2, dtype=np.float64)),
            "relative_l2_median" : float(np.median(rel_l2)),
        }

    def _power_errors(self) -> dict:
        gt   = self.gt[self.active]
        pred = self.pred[self.active]

        power_gt   = np.trapezoid(gt,   self.x_axis, axis=1)
        power_pred = np.trapezoid(pred, self.x_axis, axis=1)

        rel = np.abs(power_pred - power_gt) / (np.abs(power_gt) + self.EPS)

        peak_gt   = self.x_axis[np.argmax(gt,   axis=1)]
        peak_pred = self.x_axis[np.argmax(pred, axis=1)]

        amp_gt   = np.max(gt,   axis=1)
        amp_pred = np.max(pred, axis=1)

        return {
            "power_rel_error_mean"        : float(np.mean(rel)),
            "power_rel_error_median"      : float(np.median(rel)),
            "peak_location_mae"           : float(np.mean(np.abs(peak_pred - peak_gt))),
            "peak_amplitude_rel_err_mean" : float(np.mean(np.abs(amp_pred - amp_gt) / (amp_gt + self.EPS), dtype=np.float64)),
        }

    def compute(self) -> dict:
        metrics = {
            "n_curves"         : int(self.gt.shape[0]),
            "n_active_curves"  : int(self.active.sum()),
            "profile_length"   : int(self.gt.shape[1]),
            "embedding_dim"    : int(self.emb.shape[1]),
        }

        metrics.update(self._physical_errors())
        metrics.update(self._normalized_errors())
        metrics.update(self._shape_errors())
        metrics.update(self._power_errors())
        metrics.update(self._embedding_stats())

        return metrics
