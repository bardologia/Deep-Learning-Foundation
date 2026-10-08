"""Reconstruction metrics for the standalone profile autoencoder.

Extends the shared autoencoder metrics with profile-specific shape, integrated
power and peak-location errors, restricted to pixels carrying components.
"""

from __future__ import annotations

import numpy as np

from pipelines.autoencoder_common.inference.metrics import AeMetricsBase
from tools.data.gaussians                           import GaussianActivity


class ProfileAeMetrics(AeMetricsBase):
    """Scores reconstructed z profiles against their ground truth.

    Attributes:
        z_axis             : Z axis of shape (H,), in metres.
        activity_threshold : Amplitude above which a curve counts as active, as recorded by the parameter run.
        active             : Boolean mask of shape (n_curves,) marking active curves.
    """

    def __init__(self, result, z_axis: np.ndarray, activity_threshold: float) -> None:
        """Stores the reconstruction result and derives the activity mask.

        Args:
            result             : Reconstruction result holding ground truth, predictions and embeddings of shape (n_curves, H) and (n_curves, embedding_dim).
            z_axis             : Z axis of shape (H,), in metres.
            activity_threshold : Amplitude above which a curve counts as active, as recorded by the parameter run.
        """
        super().__init__(result)

        self.z_axis             = np.asarray(z_axis, dtype=np.float64)
        self.activity_threshold = activity_threshold
        self.active             = GaussianActivity.is_active(self.gt.max(axis=1), self.activity_threshold)

    def _physical_errors(self) -> dict:
        """Returns MSE, MAE, RMSE, max absolute error and R2 in physical units."""
        diff = self.pred - self.gt
        mean = float(np.mean(self.gt, dtype=np.float64))

        sse = float(np.sum(diff ** 2, dtype=np.float64))
        sst = float(np.sum((self.gt - mean) ** 2, dtype=np.float64))

        return {
            "mse_mean"           : float(np.mean(diff ** 2, dtype=np.float64)),
            "mse_median"         : float(np.median(self.per_sample_mse())),
            "mae_mean"           : float(np.mean(np.abs(diff), dtype=np.float64)),
            "rmse"               : float(np.sqrt(np.mean(diff ** 2, dtype=np.float64))),
            "max_abs_error_mean" : float(np.mean(np.max(np.abs(diff), axis=1), dtype=np.float64)),
            "r2"                 : float(1.0 - sse / (sst + self.EPS)),
        }

    def _shape_errors(self) -> dict:
        """Returns Pearson correlation and relative L2 error over active curves."""
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
        """Returns integrated-power, peak-location and peak-amplitude errors.

        Power is integrated over the z axis in metres, so the peak
        location error is reported in metres.
        """
        gt   = self.gt[self.active]
        pred = self.pred[self.active]

        power_gt   = np.trapezoid(gt,   self.z_axis, axis=1)
        power_pred = np.trapezoid(pred, self.z_axis, axis=1)

        rel = np.abs(power_pred - power_gt) / (np.abs(power_gt) + self.EPS)

        peak_gt   = self.z_axis[np.argmax(gt,   axis=1)]
        peak_pred = self.z_axis[np.argmax(pred, axis=1)]

        amp_gt   = np.max(gt,   axis=1)
        amp_pred = np.max(pred, axis=1)

        return {
            "power_rel_error_mean"        : float(np.mean(rel)),
            "power_rel_error_median"      : float(np.median(rel)),
            "peak_location_mae"           : float(np.mean(np.abs(peak_pred - peak_gt))),
            "peak_amplitude_rel_err_mean" : float(np.mean(np.abs(amp_pred - amp_gt) / (amp_gt + self.EPS), dtype=np.float64)),
        }

    def compute(self) -> dict:
        """Returns the full metric dictionary for the reconstructed split."""
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
