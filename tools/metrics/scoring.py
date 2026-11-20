
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
    def confidence_interval(values: list[float], level: float = 0.95) -> tuple[float, float] | None:
        if len(values) < 2:
            return None

        from scipy import stats

        mean = float(np.mean(values))
        half = float(stats.t.ppf(0.5 + level / 2.0, len(values) - 1)) * float(np.std(values, ddof=1)) / math.sqrt(len(values))

        return (mean - half, mean + half)

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

    @staticmethod
    def contributors(dicts: list[dict], keys: list[str]) -> dict[str, int]:
        return {key: sum(1 for d in dicts if FiniteScalar.coerce(d.get(key)) is not None) for key in keys}

    @staticmethod
    def shortfall(dicts: list[dict], keys: list[str], names: list[str]) -> dict[str, list[str]]:
        missing = {}

        for key in keys:
            absent = [name for name, d in zip(names, dicts) if FiniteScalar.coerce(d.get(key)) is None]
            if absent:
                missing[key] = absent

        return missing

    @staticmethod
    def intervals(dicts: list[dict], keys: list[str], level: float = 0.95) -> tuple[dict, dict]:
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


class RankCorrelation:

    MIN_PAIRS = 3

    @staticmethod
    def ranks(values: np.ndarray) -> np.ndarray:
        flat                      = np.asarray(values, dtype=np.float64).reshape(-1)
        _uniques, inverse, counts = np.unique(flat, return_inverse=True, return_counts=True)

        starts   = np.cumsum(counts) - counts
        averaged = starts + (counts - 1) / 2.0

        return averaged[inverse]

    @classmethod
    def spearman(cls, a: np.ndarray, b: np.ndarray) -> float:
        flat_a = np.asarray(a, dtype=np.float64).reshape(-1)
        flat_b = np.asarray(b, dtype=np.float64).reshape(-1)

        both = np.isfinite(flat_a) & np.isfinite(flat_b)
        if int(both.sum()) < cls.MIN_PAIRS:
            return float("nan")

        rank_a = cls.ranks(flat_a[both])
        rank_b = cls.ranks(flat_b[both])

        if rank_a.std() == 0.0 or rank_b.std() == 0.0:
            return float("nan")

        return float(np.corrcoef(rank_a, rank_b)[0, 1])


class MetricOrientation:

    _NEUTRAL_PATTERN = re.compile(r"^(n_|bias_)|^(seed|checkpoint_epoch)$")
    _LOWER_PATTERN   = re.compile(r"^(rmse|mae|mse)_|loss")
    _HIGHER_PATTERN  = re.compile(r"^r2_")

    @classmethod
    def direction(cls, key: str) -> str | None:
        rules = (
            (None,     bool(cls._NEUTRAL_PATTERN.search(key))),
            ("lower",  bool(cls._LOWER_PATTERN.search(key))),
            ("higher", bool(cls._HIGHER_PATTERN.search(key))),
        )

        return next(direction for direction, matched in rules if matched)
