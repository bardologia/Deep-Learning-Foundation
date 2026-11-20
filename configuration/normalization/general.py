from __future__ import annotations

from dataclasses import dataclass
from enum        import Enum

import numpy as np


class NormMethod(str, Enum):

    ZSCORE     = "zscore"
    ROBUST_IQR = "robust_iqr"
    MIN_MAX    = "min_max"

    def fit(self, values: np.ndarray) -> tuple[float, float]:
        data = np.asarray(values, dtype=np.float64)

        if self is NormMethod.ZSCORE:
            return float(data.mean()), max(float(data.std()), 1e-8)

        if self is NormMethod.ROBUST_IQR:
            lower, median, upper = np.percentile(data, (25.0, 50.0, 75.0))
            return float(median), max(float(upper - lower), 1e-8)

        low, high = np.percentile(data, (0.1, 99.9))
        return float(low), max(float(high - low), 1e-8)


@dataclass
class NormalizationConfig:

    inputs  : NormMethod = NormMethod.ZSCORE
    targets : NormMethod = NormMethod.ZSCORE

    def as_dict(self) -> dict:
        return {"inputs": self.inputs.value, "targets": self.targets.value}

    @classmethod
    def from_dict(cls, payload: dict) -> "NormalizationConfig":
        return cls(inputs=NormMethod(payload["inputs"]), targets=NormMethod(payload["targets"]))


@dataclass
class ChannelStats:

    names  : list[str]
    method : NormMethod
    loc    : list[float]
    scale  : list[float]

    @property
    def n_channels(self) -> int:
        return len(self.names)

    def as_dict(self) -> dict:
        return {"method": self.method.value, "channels": [{"name": name, "loc": loc, "scale": scale} for name, loc, scale in zip(self.names, self.loc, self.scale)]}

    @classmethod
    def from_dict(cls, payload: dict) -> "ChannelStats":
        channels = payload["channels"]

        return cls(
            names  = [entry["name"] for entry in channels],
            method = NormMethod(payload["method"]),
            loc    = [float(entry["loc"]) for entry in channels],
            scale  = [float(entry["scale"]) for entry in channels],
        )
