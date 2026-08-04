from __future__ import annotations

import math
import numbers
import re
from typing import Any

import numpy as np


class FiniteScalar:

    @staticmethod
    def is_finite_number(value: Any) -> bool:
        if isinstance(value, bool):
            return False
        if not isinstance(value, numbers.Real):
            return False
        return math.isfinite(float(value))

    @staticmethod
    def coerce(value: Any) -> float | None:
        if FiniteScalar.is_finite_number(value):
            return float(value)
        return None


class SeedAggregation:
    @staticmethod
    def mean_std(values: list[float]) -> tuple[float, float | None]:
        mean = float(np.mean(values))
        std  = float(np.std(values, ddof=1)) if len(values) > 1 else None

        return mean, std

    @staticmethod
    def aggregate(dicts: list[dict], keys: list[str]) -> tuple[dict, dict]:
        means, stds = {}, {}

        for key in keys:
            values = [FiniteScalar.coerce(d.get(key)) for d in dicts]
            values = [value for value in values if value is not None]

            if not values:
                continue

            means[key], stds[key] = SeedAggregation.mean_std(values)

        return means, stds


class R2:

    EPSILON = 1e-12

    @staticmethod
    def pixel_map(pred: np.ndarray, ref: np.ndarray, axis: int) -> np.ndarray:
        ref_mean = ref.mean(axis=axis, keepdims=True, dtype=np.float64)

        ss_res = ((pred - ref) ** 2).sum(axis=axis, dtype=np.float64)
        ss_tot = ((ref - ref_mean) ** 2).sum(axis=axis, dtype=np.float64)

        return (1.0 - ss_res / (ss_tot + R2.EPSILON)).astype(np.float32)


class MetricOrientation:

    _HIGHER_OVERRIDES = {"relative_mse_reduction", "fraction_pred_beats_reduced"}
    _NEUTRAL_PATTERN  = re.compile(r"^(n_pixels|n_elevation|x_axis_|gt_|pred_)|^matched_(n_pairs|tol)$|^active_count_\w+_mean$|^active_frac_(gt|pred)$|^slot_\d+_active_(gt|pred)_frac$")
    _LOWER_TOKENS     = ("distance", "_dist", "error", "_err", "loss", "_mse", "_mae", "_rmse")
    _HIGHER_TOKENS    = ("r2", "ssim", "psnr", "cosine", "precision", "recall", "f1", "count_acc", "count_exact")

    @classmethod
    def direction(cls, key: str) -> str | None:
        if key in cls._HIGHER_OVERRIDES:
            return "higher"
        if cls._NEUTRAL_PATTERN.search(key):
            return None
        if any(token in key for token in cls._LOWER_TOKENS):
            return "lower"
        if any(token in key for token in cls._HIGHER_TOKENS):
            return "higher"
        return "lower"

    @classmethod
    def higher_is_better(cls, key: str) -> bool | None:
        direction = cls.direction(key)
        return None if direction is None else direction == "higher"


class RelativeImprovement:

    @staticmethod
    def fraction(baseline: Any, model: Any, higher_is_better: bool = False) -> float:
        baseline_value = FiniteScalar.coerce(baseline)
        model_value    = FiniteScalar.coerce(model)

        if baseline_value is None or model_value is None or baseline_value == 0.0:
            return float("nan")

        delta = (model_value - baseline_value) if higher_is_better else (baseline_value - model_value)
        return delta / abs(baseline_value)

    @staticmethod
    def percent(baseline: Any, model: Any, higher_is_better: bool = False, empty: str = "n/a") -> str:
        value = RelativeImprovement.fraction(baseline, model, higher_is_better)
        if not np.isfinite(value):
            return empty
        return f"{value * 100.0:+.1f}%"
