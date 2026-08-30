"""Tests covering the multi-metric trial ranking and scoring machinery.

Exercises ordinal ranking with ties, the magnitude-aware min-max score, the composite
and grouped breakdowns of ``RankingComputer``, and the primitives in ``RankMath``.
"""

from __future__ import annotations

import math

from tools.metrics.ranking import RankingComputer, RankMath


HIGHER = [("overall_r2_gt", "R²")]
LOWER  = [("curve_rmse_gt", "RMSE")]


def _compute(metrics, trials):
    """Returns the ranking result for the given metric specs and trial metric dictionaries."""
    return RankingComputer(metrics, trials).compute()


def test_lower_metric_best_value_ranks_first():
    """Verifies the smallest value of a lower-is-better metric takes rank one."""
    result = _compute(LOWER, [("a", {"curve_rmse_gt": 0.1}), ("b", {"curve_rmse_gt": 0.9})])
    assert result.order()[0] == "a"
    assert result.metric_rank("a", "curve_rmse_gt") == 1.0
    assert result.metric_rank("b", "curve_rmse_gt") == 2.0


def test_higher_metric_best_value_ranks_first():
    """Verifies the largest value of a higher-is-better metric leads the order."""
    result = _compute(HIGHER, [("a", {"overall_r2_gt": 0.2}), ("b", {"overall_r2_gt": 0.8})])
    assert result.order()[0] == "b"


def test_ties_share_averaged_rank():
    """Verifies tied values receive the averaged rank and the next value continues the sequence."""
    result = _compute(LOWER, [("a", {"curve_rmse_gt": 0.5}), ("b", {"curve_rmse_gt": 0.5}), ("c", {"curve_rmse_gt": 0.9})])
    assert result.metric_rank("a", "curve_rmse_gt") == 1.5
    assert result.metric_rank("b", "curve_rmse_gt") == 1.5
    assert result.metric_rank("c", "curve_rmse_gt") == 3.0


def test_score_is_magnitude_aware():
    """Verifies the metric score is the min-max normalisation of the values, not their ordinal rank."""
    trials = [("a", {"curve_rmse_gt": 0.0}), ("b", {"curve_rmse_gt": 0.1}), ("c", {"curve_rmse_gt": 1.0})]
    result = _compute(LOWER, trials)

    assert math.isclose(result.scores["a"]["curve_rmse_gt"], 1.0)
    assert math.isclose(result.scores["c"]["curve_rmse_gt"], 0.0)
    assert math.isclose(result.scores["b"]["curve_rmse_gt"], 0.9)


def test_score_separates_what_ordinal_rank_cannot():
    """Verifies equal mean ranks still yield different composites when the value spacing differs."""
    near = _compute(LOWER, [("a", {"curve_rmse_gt": 0.0}), ("b", {"curve_rmse_gt": 0.1}), ("c", {"curve_rmse_gt": 10.0})])
    far  = _compute(LOWER, [("a", {"curve_rmse_gt": 0.0}), ("b", {"curve_rmse_gt": 9.9}), ("c", {"curve_rmse_gt": 10.0})])

    assert near.mean_rank["b"] == far.mean_rank["b"]
    assert near.composite["b"] > far.composite["b"]


def test_missing_metric_scores_zero_and_ranks_last():
    """Verifies a trial missing a metric scores nothing for it and is ranked behind every reported trial."""
    metrics = [("curve_rmse_gt", "RMSE"), ("overall_r2_gt", "R²")]
    trials  = [
        ("a", {"curve_rmse_gt": 0.0, "overall_r2_gt": 0.9}),
        ("b", {"curve_rmse_gt": 0.5}),
        ("c", {"curve_rmse_gt": 1.0, "overall_r2_gt": 0.1}),
    ]
    result = _compute(metrics, trials)

    assert result.scores["b"].get("overall_r2_gt") is None
    assert math.isclose(result.composite["b"], (0.5 + 0.0) / 2.0)
    assert result.mean_rank["b"] == (2.0 + 4.0) / 2.0


def test_wins_counts_best_metrics():
    """Verifies the win count equals the number of metrics on which a trial is best."""
    metrics = [("curve_rmse_gt", "RMSE"), ("overall_r2_gt", "R²")]
    trials  = [("a", {"curve_rmse_gt": 0.1, "overall_r2_gt": 0.9}), ("b", {"curve_rmse_gt": 0.2, "overall_r2_gt": 0.5})]
    result  = _compute(metrics, trials)

    assert result.wins["a"] == 2
    assert result.wins["b"] == 0


def test_grouped_score_dampens_correlated_metrics():
    """Verifies grouping correlated metrics rebalances the overall score away from the flat composite."""
    groups  = [("Curve error", ["curve_rmse_gt", "curve_mae_gt"]), ("Variance explained", ["overall_r2_gt"])]
    metrics = [(key, key) for _, keys in groups for key in keys]
    trials  = [
        ("a", {"curve_rmse_gt": 0.0, "curve_mae_gt": 0.0, "overall_r2_gt": 0.0}),
        ("b", {"curve_rmse_gt": 1.0, "curve_mae_gt": 1.0, "overall_r2_gt": 1.0}),
    ]
    result    = _compute(metrics, trials)
    breakdown = result.group_breakdown(groups)

    assert math.isclose(result.composite["a"], 2.0 / 3.0)
    assert math.isclose(breakdown.overall["a"], 0.5)
    assert math.isclose(breakdown.overall["b"], 0.5)


def test_group_breakdown_drops_empty_groups():
    """Verifies groups whose metrics no trial reported are omitted from the breakdown."""
    groups  = [("Curve error", ["curve_rmse_gt"]), ("Gain vs reference", ["relative_mse_reduction"])]
    metrics = [(key, key) for _, keys in groups for key in keys]
    trials  = [("a", {"curve_rmse_gt": 0.1}), ("b", {"curve_rmse_gt": 0.2})]
    result  = _compute(metrics, trials)

    breakdown = result.group_breakdown(groups)
    assert breakdown.labels == ["Curve error"]


def test_relative_mse_reduction_higher_ranks_first():
    """Verifies relative MSE reduction is treated as a higher-is-better metric."""
    metrics = [("relative_mse_reduction", "vs reference")]
    result  = _compute(metrics, [("a", {"relative_mse_reduction": 0.4}), ("b", {"relative_mse_reduction": 0.1})])

    assert result.order()[0] == "a"
    assert result.metric_rank("a", "relative_mse_reduction") == 1.0


def test_average_ranks_descending():
    """Verifies reversed averaging ranks the largest value first."""
    ranks = RankMath.average_ranks({"a": 3.0, "b": 1.0, "c": 2.0}, reverse=True)
    assert ranks == {"a": 1.0, "c": 2.0, "b": 3.0}


def test_average_ranks_ties_within_the_shared_tolerance():
    """Verifies values within the shared tie tolerance are ranked and led as ties."""
    values = {"a": 1.0, "b": 1.0 + RankMath.TIE_TOLERANCE / 2.0, "c": 2.0}
    ranks  = RankMath.average_ranks(values, reverse=False)

    assert ranks["a"] == ranks["b"] == 1.5
    assert ranks["c"] == 3.0
    assert RankMath.leaders({"a": {"m": 1.0}, "b": {"m": 1.0 + RankMath.TIE_TOLERANCE / 2.0}}, "m") == {"a", "b"}


def test_minmax_all_equal_scores_one():
    """Verifies min-max scoring assigns one to every entry when all values are equal."""
    scores = RankMath.minmax_scores({"a": 0.5, "b": 0.5}, higher=True)
    assert scores == {"a": 1.0, "b": 1.0}
