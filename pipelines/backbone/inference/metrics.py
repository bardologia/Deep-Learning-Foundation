from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.data.io import FileIO


class Metrics:

    @staticmethod
    def channel(predictions: np.ndarray, targets: np.ndarray) -> dict:
        valid  = np.isfinite(predictions) & np.isfinite(targets)
        error  = predictions[valid].astype(np.float64) - targets[valid].astype(np.float64)
        spread = float(((targets[valid] - targets[valid].mean()) ** 2).sum())

        return {
            "rmse" : float(np.sqrt((error ** 2).mean())),
            "mae"  : float(np.abs(error).mean()),
            "bias" : float(error.mean()),
            "r2"   : 1.0 - float((error ** 2).sum()) / spread if spread > 0.0 else float("nan"),
        }

    @staticmethod
    def compute(predictions: np.ndarray, targets: np.ndarray, names: list[str]) -> dict:
        channels = {name: Metrics.channel(predictions[index], targets[index]) for index, name in enumerate(names)}
        summary  = {f"{key}_mean": float(np.mean([values[key] for values in channels.values()])) for key in ("rmse", "mae", "r2")}

        return {**summary, "channels": channels, "n_pixels": int(targets[0].size)}

    @staticmethod
    def write_json(metrics: dict, path: Path) -> Path:
        return FileIO.save_json(metrics, path)
