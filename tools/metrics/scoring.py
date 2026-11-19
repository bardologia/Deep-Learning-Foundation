
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


class R2:

    EPSILON = 1e-12

    @staticmethod
    def pixel_map(pred: np.ndarray, ref: np.ndarray, axis: int) -> np.ndarray:
        ref_mean = ref.mean(axis=axis, keepdims=True, dtype=np.float64)

        ss_res = ((pred - ref) ** 2).sum(axis=axis, dtype=np.float64)
        ss_tot = ((ref - ref_mean) ** 2).sum(axis=axis, dtype=np.float64)

        defined = ss_tot > R2.EPSILON
        r2      = np.full(ss_tot.shape, np.nan, dtype=np.float64)

        r2[defined] = 1.0 - ss_res[defined] / ss_tot[defined]

        return r2.astype(np.float32)

    @staticmethod
    def undefined_count(r2_map: np.ndarray) -> int:
        return int(np.count_nonzero(~np.isfinite(r2_map)))


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

    _HIGHER_OVERRIDES = {"relative_mse_reduction", "fraction_pred_beats_reduced", "failure_frac_ok", "presence_rank_corr", "slot_gt_alignment", "slot_mu_rank_diag", "slot_usage_entropy"}
    _HIGHER_PATTERN   = re.compile(r"^improvement_|valid_fraction")
    _NEUTRAL_PATTERN  = re.compile(r"^(n_|num_|x_axis_|gt_|pred_|embedding_)|_undefined_count$|(^|_)matched_(n_pairs|tol)$|^pixel_\w+_std$|^(seed|patch_height|patch_width|profile_length)$|^active_count_\w+_mean$|^active_frac_(gt|pred)$|^slot_\d+_active_(gt|pred)_frac$|^slot_\d+_(amp|mu|sig)_((active|inactive|global)_)?(gt|pred)_(mean|std)$|^norm_in_ch\d+_(mean|std)$|^strat_|^ssim_\w+_n_(scored|nonfinite)$|^jepa_(decode_amplification|decoder_floor_frac)$|^(clamp_n_active|presence_n_scored|activity_threshold|roughness_ratio|roughness_gt_mean|roughness_pred_mean|checkpoint_epoch)$")
    _LOWER_PATTERN    = re.compile(r"^(mae|mse|rmse)(_|$)|^relative_l2_|^curve_l1|^label_r2_frac_below_|^z_ce_|^pixel_peak_idx_d_|^count_(over|under)_frac$|^clamp_\w+_frac$|^failure_(frac_\w+|halluc_rate|miss_rate)$|^norm_in(_ch\d+)?_(overflow|nonfinite)_frac$|^norm_in_worst_|^roughness_excess_frac$|(^|_)nmse$")
    _LOWER_TOKENS     = ("distance", "_dist", "error", "_err", "loss", "_mse", "_mae", "_rmse")
    _HIGHER_TOKENS    = ("r2", "ssim", "psnr", "cosine", "pearson", "precision", "recall", "f1", "count_acc", "count_exact")

    @classmethod
    def direction(cls, key: str) -> str | None:
        if key in cls._HIGHER_OVERRIDES or cls._HIGHER_PATTERN.search(key):
            return "higher"
        if cls._NEUTRAL_PATTERN.search(key):
            return None
        if cls._LOWER_PATTERN.search(key):
            return "lower"
        if any(token in key for token in cls._LOWER_TOKENS):
            return "lower"
        if any(token in key for token in cls._HIGHER_TOKENS):
            return "higher"

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
