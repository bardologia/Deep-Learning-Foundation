
from __future__ import annotations

import math

import numpy as np
import pytest

from tools.metrics.scoring import FiniteScalar, MetricOrientation, RankCorrelation, SeedAggregation


def test_finite_scalar_accepts_floats_and_ints():
    assert FiniteScalar.is_finite_number(1.5)
    assert FiniteScalar.is_finite_number(3)
    assert FiniteScalar.coerce(2) == 2.0


def test_finite_scalar_rejects_bool():
    assert FiniteScalar.is_finite_number(True) is False
    assert FiniteScalar.coerce(True) is None


def test_finite_scalar_rejects_nan_inf_and_strings():
    assert FiniteScalar.is_finite_number(float("nan")) is False
    assert FiniteScalar.is_finite_number(float("inf")) is False
    assert FiniteScalar.is_finite_number("x") is False
    assert FiniteScalar.coerce(float("nan")) is None
    assert FiniteScalar.coerce("x") is None


def test_orientation_of_the_error_and_loss_keys_is_lower():
    for key in ("rmse_mean", "mae_mean", "rmse_pointwise", "mae_neighbourhood", "best_val_loss"):
        assert MetricOrientation.direction(key) == "lower", key

    assert MetricOrientation.higher_is_better("rmse_mean") is False


def test_orientation_of_the_r2_keys_is_higher():
    for key in ("r2_mean", "r2_pointwise"):
        assert MetricOrientation.direction(key) == "higher", key

    assert MetricOrientation.higher_is_better("r2_mean") is True


def test_orientation_of_counts_and_signed_bias_is_neutral():
    for key in ("n_pixels", "bias_pointwise", "checkpoint_epoch", "seed"):
        assert MetricOrientation.direction(key) is None, key

    assert MetricOrientation.higher_is_better("n_pixels") is None


def test_rank_correlation_averages_tied_ranks():
    assert RankCorrelation.ranks(np.array([1.0, 1.0, 2.0])).tolist() == [0.5, 0.5, 2.0]
    assert RankCorrelation.ranks(np.array([[3.0, 1.0], [1.0, 7.0]])).tolist() == [2.0, 0.5, 0.5, 3.0]


def test_rank_correlation_is_unaffected_by_tie_order():
    values = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
    errors = np.array([5.0, 4.0, 3.0, 2.0, 1.0])

    shuffled        = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
    reversed_errors = np.array([3.0, 4.0, 5.0, 2.0, 1.0])

    assert RankCorrelation.spearman(values, errors) == pytest.approx(RankCorrelation.spearman(shuffled, reversed_errors))


def test_rank_correlation_of_monotone_and_inverted_vectors():
    a = np.array([0.1, 0.4, 0.2, 0.9])

    assert RankCorrelation.spearman(a,  a) == pytest.approx(1.0)
    assert RankCorrelation.spearman(a, -a) == pytest.approx(-1.0)


def test_rank_correlation_ignores_non_finite_pairs():
    a = np.array([1.0, 2.0, 3.0, np.nan])
    b = np.array([1.0, 2.0, 3.0, 10.0])

    assert RankCorrelation.spearman(a, b) == pytest.approx(1.0)


def test_rank_correlation_is_nan_without_variance():
    constant = np.zeros(5)

    assert math.isnan(RankCorrelation.spearman(constant, np.arange(5.0)))
    assert math.isnan(RankCorrelation.spearman(np.array([1.0, np.nan, np.nan, 4.0, 5.0]), np.array([1.0, 2.0, 3.0, np.nan, np.nan])))


def test_seed_aggregation_counts_contributors_and_names_the_shortfall():
    seeds = [{"f1": 0.8, "mae": 0.1}, {"f1": float("nan"), "mae": 0.2}, {"mae": 0.3}]
    names = ["seed0", "seed1", "seed2"]

    assert SeedAggregation.contributors(seeds, ["f1", "mae", "absent"]) == {"f1": 1, "mae": 3, "absent": 0}
    assert SeedAggregation.shortfall(seeds, ["f1", "mae"], names) == {"f1": ["seed1", "seed2"]}


def test_unknown_metric_has_no_orientation():
    with pytest.raises(StopIteration):
        MetricOrientation.direction("zzz_unclassified_metric")
