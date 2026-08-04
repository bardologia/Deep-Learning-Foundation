from __future__ import annotations

import numpy as np

from pipelines.unrolled.inference.loader    import UnrolledRun
from pipelines.unrolled.inference.predictor import UnrolledPrediction


class UnrolledMetrics:
    def __init__(self, prediction: UnrolledPrediction, run: UnrolledRun) -> None:
        self.prediction = prediction
        self.run        = run

    def compute(self) -> dict:
        valid   = self.prediction.valid_mask
        n_valid = int(valid.sum())

        if n_valid == 0:
            raise ValueError(f"No pixel in split '{self.run.split_name}' exceeds the power floor {self.run.power_floor}; the region contains no usable ground truth.")

        l1   = self.prediction.curve_l1_map[valid].astype(np.float64)
        mse  = self.prediction.curve_mse_map[valid].astype(np.float64)
        peak = self.prediction.peak_error_map[valid].astype(np.float64)

        loss = l1.mean() if self.run.curve_loss == "l1" else mse.mean()

        return {
            "split"                 : self.run.split_name,
            "split_region"          : list(self.run.split_region.as_tuple()),
            "model_name"            : self.run.model_name,
            "n_iterations"          : int(self.run.model_config.n_iterations),
            "checkpoint"            : self.run.checkpoint_path.name,
            "curve_loss"            : self.run.curve_loss,
            "measurement_noise_std" : float(self.run.noise_std),
            "power_floor"           : float(self.run.power_floor),

            "n_pixels"       : int(valid.size),
            "n_valid_pixels" : n_valid,
            "valid_fraction" : float(n_valid / valid.size),

            "loss"         : float(loss),
            "curve_l1"     : float(l1.mean()),
            "curve_l1_p50" : float(np.percentile(l1, 50)),
            "curve_l1_p90" : float(np.percentile(l1, 90)),
            "curve_mse"    : float(mse.mean()),
            "curve_rmse"   : float(np.sqrt(mse.mean())),
            "peak_mae_m"   : float(peak.mean()),

            "best_val_loss"   : float(self.run.training_summary["best_val_loss"]),
            "train_test_loss" : float(self.run.training_summary["test"]["loss"]),
        }
