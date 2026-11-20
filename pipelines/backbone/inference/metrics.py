from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.data.io import FileIO


class Metrics:

    CHANNEL_KEYS = ("rmse", "mae", "bias", "r2")
    MEAN_KEYS    = ("rmse", "mae", "r2")

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
    def key(metric: str, channel: str) -> str:
        return f"{metric}_{channel}"

    @classmethod
    def compute(cls, predictions: np.ndarray, targets: np.ndarray, names: list[str]) -> dict:
        channels = {name: cls.channel(predictions[index], targets[index]) for index, name in enumerate(names)}
        means    = {f"{metric}_mean": float(np.mean([values[metric] for values in channels.values()])) for metric in cls.MEAN_KEYS}
        per_key  = {cls.key(metric, name): values[metric] for name, values in channels.items() for metric in cls.CHANNEL_KEYS}

        return {**means, **per_key, "n_pixels": int(targets[0].size)}

    @classmethod
    def channel_values(cls, metrics: dict, name: str) -> dict:
        return {metric: metrics[cls.key(metric, name)] for metric in cls.CHANNEL_KEYS}

    @staticmethod
    def write_json(metrics: dict, path: Path) -> Path:
        return FileIO.save_json(metrics, path)
