"""Scalar coercion, aggregation and orientation helpers for reported metrics.

Provides the guards that keep non-numeric or non-finite entries out of the
metric tables, seed-level mean and standard deviation aggregation, coefficient
of determination and Spearman correlation, the rule deciding whether a metric
key is better high or low, and relative-improvement formatting.
"""

from __future__ import annotations

import math
import numbers
import re
from typing import Any

import numpy as np


class FiniteScalar:
    """Guards that admit only finite real numbers into metric computations."""

    @staticmethod
    def is_finite_number(value: Any) -> bool:
        """Returns whether a value is a finite real number, excluding booleans."""
        if isinstance(value, bool):
            return False
        if not isinstance(value, numbers.Real):
            return False
        return math.isfinite(float(value))

    @staticmethod
    def coerce(value: Any) -> float | None:
        """Returns the value as a float, or None when it is not a finite real number."""
        if FiniteScalar.is_finite_number(value):
            return float(value)
        return None


class SeedAggregation:
    """Aggregation of metrics across the seeds of one trial."""

    @staticmethod
    def mean_std(values: list[float]) -> tuple[float, float | None]:
        """Returns the mean and the sample standard deviation, the latter None for one value."""
        mean = float(np.mean(values))
        std  = float(np.std(values, ddof=1)) if len(values) > 1 else None

        return mean, std

    @staticmethod
    def confidence_interval(values: list[float], level: float = 0.95) -> tuple[float, float] | None:
        """Returns the two-sided t-based confidence interval of the mean.

        Args:
            values : Per-seed metric values.
            level  : Coverage of the interval, between zero and one.

        Returns:
            The (low, high) interval, or None for fewer than two values.
        """
        if len(values) < 2:
            return None

        from scipy import stats

        mean = float(np.mean(values))
        half = float(stats.t.ppf(0.5 + level / 2.0, len(values) - 1)) * float(np.std(values, ddof=1)) / math.sqrt(len(values))

        return (mean - half, mean + half)

    @staticmethod
    def aggregate(dicts: list[dict], keys: list[str]) -> tuple[dict, dict]:
        """Aggregates the requested metric keys across per-seed metric mappings.

        Args:
            dicts : One metric mapping per seed.
            keys  : Metric keys to aggregate.

        Returns:
            Tuple of the mean and standard-deviation mappings; keys no seed
            reports with a finite value are absent from both.
        """
        means, stds = {}, {}

        for key in keys:
            values = [FiniteScalar.coerce(d.get(key)) for d in dicts]
            values = [value for value in values if value is not None]

            if not values:
                continue

            means[key], stds[key] = SeedAggregation.mean_std(values)

        return means, stds

    @staticmethod
    def intervals(dicts: list[dict], keys: list[str], level: float = 0.95) -> tuple[dict, dict]:
        """Returns the confidence interval and finite-sample count per metric key.

        Args:
            dicts : One metric mapping per seed.
            keys  : Metric keys to aggregate.
            level : Coverage of the intervals, between zero and one.

        Returns:
            Tuple of the interval mapping (absent for keys with fewer than two finite values) and the finite-count mapping (absent for keys with none).
        """
        intervals, counts = {}, {}

        for key in keys:
            values = [FiniteScalar.coerce(d.get(key)) for d in dicts]
            values = [value for value in values if value is not None]

            if not values:
                continue

            counts[key] = len(values)
            interval    = SeedAggregation.confidence_interval(values, level)
            if interval is not None:
                intervals[key] = interval

        return intervals, counts


class R2:
    """Coefficient of determination computed along one axis."""

    EPSILON = 1e-12

    @staticmethod
    def pixel_map(pred: np.ndarray, ref: np.ndarray, axis: int) -> np.ndarray:
        """Returns the per-pixel coefficient of determination of a prediction.

        Args:
            pred : Predicted values with the reduced dimension along axis.
            ref  : Reference values of the same shape.
            axis : Dimension the residual and total sums are taken over, usually the z axis of a profile stack.

        Returns:
            Float32 array with axis removed, holding 1 - SS_res / SS_tot.
        """
        ref_mean = ref.mean(axis=axis, keepdims=True, dtype=np.float64)

        ss_res = ((pred - ref) ** 2).sum(axis=axis, dtype=np.float64)
        ss_tot = ((ref - ref_mean) ** 2).sum(axis=axis, dtype=np.float64)

        return (1.0 - ss_res / (ss_tot + R2.EPSILON)).astype(np.float32)


class RankCorrelation:
    """Tie-corrected Spearman rank correlation between two flattened arrays."""

    MIN_PAIRS = 3

    @staticmethod
    def ranks(values: np.ndarray) -> np.ndarray:
        """Returns the zero-based ranks of a flattened array, ties sharing their average rank."""
        flat                      = np.asarray(values, dtype=np.float64).reshape(-1)
        _uniques, inverse, counts = np.unique(flat, return_inverse=True, return_counts=True)

        starts   = np.cumsum(counts) - counts
        averaged = starts + (counts - 1) / 2.0

        return averaged[inverse]

    @classmethod
    def spearman(cls, a: np.ndarray, b: np.ndarray) -> float:
        """Returns the Spearman correlation over the pairs finite in both arrays.

        Args:
            a : First array, flattened before pairing.
            b : Second array with the same number of elements as a.

        Returns:
            Correlation coefficient, or NaN when fewer than MIN_PAIRS finite
            pairs remain or either rank vector is constant.

        Raises:
            ValueError : If the arrays differ in size or hold fewer than MIN_PAIRS elements.
        """
        flat_a = np.asarray(a, dtype=np.float64).reshape(-1)
        flat_b = np.asarray(b, dtype=np.float64).reshape(-1)

        if flat_a.size != flat_b.size or flat_a.size < cls.MIN_PAIRS:
            raise ValueError(f"Spearman needs two matched vectors of at least {cls.MIN_PAIRS} entries, got {flat_a.size} and {flat_b.size}")

        both = np.isfinite(flat_a) & np.isfinite(flat_b)
        if int(both.sum()) < cls.MIN_PAIRS:
            return float("nan")

        rank_a = cls.ranks(flat_a[both])
        rank_b = cls.ranks(flat_b[both])

        if rank_a.std() == 0.0 or rank_b.std() == 0.0:
            return float("nan")

        return float(np.corrcoef(rank_a, rank_b)[0, 1])


class MetricOrientation:
    """Decides from a metric key whether larger values are better, worse or neither."""

    _HIGHER_OVERRIDES = {"relative_mse_reduction", "fraction_pred_beats_reduced"}
    _HIGHER_PATTERN   = re.compile(r"^improvement_|^phase_agreement_|valid_fraction")
    _NEUTRAL_PATTERN  = re.compile(r"^(n_pixels|n_z|x_axis_|gt_|pred_)|^matched_(n_pairs|tol)$|^active_count_\w+_mean$|^active_frac_(gt|pred)$|^slot_\d+_active_(gt|pred)_frac$")
    _LOWER_TOKENS     = ("distance", "_dist", "error", "_err", "loss", "_mse", "_mae", "_rmse")
    _HIGHER_TOKENS    = ("r2", "ssim", "psnr", "cosine", "precision", "recall", "f1", "count_acc", "count_exact")

    @classmethod
    def direction(cls, key: str) -> str | None:
        """Returns "higher", "lower", or None for descriptive metrics that must not be ranked.

        Args:
            key : Metric key as it appears in the reported metric mappings.

        Returns:
            The orientation, resolved by explicit override, then neutral
            pattern, then substring tokens, defaulting to "lower".
        """
        if key in cls._HIGHER_OVERRIDES or cls._HIGHER_PATTERN.search(key):
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
        """Returns the orientation as a boolean, or None for descriptive metrics."""
        direction = cls.direction(key)
        return None if direction is None else direction == "higher"


class RelativeImprovement:
    """Improvement of a model metric over a baseline, as a fraction or a percentage string."""

    @staticmethod
    def fraction(baseline: Any, model: Any, higher_is_better: bool = False) -> float:
        """Returns the signed improvement of the model over the baseline.

        Args:
            baseline         : Baseline metric value.
            model            : Model metric value.
            higher_is_better : Whether larger values count as better.

        Returns:
            Improvement as a fraction of the absolute baseline, positive when
            the model is better; NaN when either value is not finite or the
            baseline is zero.
        """
        baseline_value = FiniteScalar.coerce(baseline)
        model_value    = FiniteScalar.coerce(model)

        if baseline_value is None or model_value is None or baseline_value == 0.0:
            return float("nan")

        delta = (model_value - baseline_value) if higher_is_better else (baseline_value - model_value)
        return delta / abs(baseline_value)

    @staticmethod
    def percent(baseline: Any, model: Any, higher_is_better: bool = False, empty: str = "n/a") -> str:
        """Returns the improvement as a signed percentage string, or the placeholder when undefined.

        Args:
            baseline         : Baseline metric value.
            model            : Model metric value.
            higher_is_better : Whether larger values count as better.
            empty            : Text returned when the improvement is undefined.

        Returns:
            Formatted percentage such as "+12.3%", or the placeholder.
        """
        value = RelativeImprovement.fraction(baseline, model, higher_is_better)
        if not np.isfinite(value):
            return empty
        return f"{value * 100.0:+.1f}%"
