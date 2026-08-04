from __future__ import annotations

import math

from tools.metrics.ranking import RankingComputer, RankMath


HIGHER = [("overall_r2_gt", "R²")]
LOWER  = [("curve_rmse_gt", "RMSE")]


def _compute(metrics, trials):
    return RankingComputer(metrics, trials).compute()


def test_lower_metric_best_value_ranks_first():
    result = _compute(LOWER, [("a", {"curve_rmse_gt": 0.1}), ("b", {"curve_rmse_gt": 0.9})])
    assert result.order()[0] == "a"
    assert result.metric_rank("a", "curve_rmse_gt") == 1.0
    assert result.metric_rank("b", "curve_rmse_gt") == 2.0


def test_higher_metric_best_value_ranks_first():
    result = _compute(HIGHER, [("a", {"overall_r2_gt": 0.2}), ("b", {"overall_r2_gt": 0.8})])
    assert result.order()[0] == "b"


def test_ties_share_averaged_rank():
    result = _compute(LOWER, [("a", {"curve_rmse_gt": 0.5}), ("b", {"curve_rmse_gt": 0.5}), ("c", {"curve_rmse_gt": 0.9})])
    assert result.metric_rank("a", "curve_rmse_gt") == 1.5
    assert result.metric_rank("b", "curve_rmse_gt") == 1.5
    assert result.metric_rank("c", "curve_rmse_gt") == 3.0


def test_score_is_magnitude_aware():
    trials = [("a", {"curve_rmse_gt": 0.0}), ("b", {"curve_rmse_gt": 0.1}), ("c", {"curve_rmse_gt": 1.0})]
    result = _compute(LOWER, trials)

    assert math.isclose(result.metric_score("a", "curve_rmse_gt"), 1.0)
    assert math.isclose(result.metric_score("c", "curve_rmse_gt"), 0.0)
    assert math.isclose(result.metric_score("b", "curve_rmse_gt"), 0.9)


def test_score_separates_what_ordinal_rank_cannot():
    near = _compute(LOWER, [("a", {"curve_rmse_gt": 0.0}), ("b", {"curve_rmse_gt": 0.1}), ("c", {"curve_rmse_gt": 10.0})])
    far  = _compute(LOWER, [("a", {"curve_rmse_gt": 0.0}), ("b", {"curve_rmse_gt": 9.9}), ("c", {"curve_rmse_gt": 10.0})])

    assert near.mean_rank["b"] == far.mean_rank["b"]
    assert near.composite["b"] > far.composite["b"]


def test_missing_metric_scores_zero_and_ranks_last():
    metrics = [("curve_rmse_gt", "RMSE"), ("overall_r2_gt", "R²")]
    trials  = [
        ("a", {"curve_rmse_gt": 0.0, "overall_r2_gt": 0.9}),
        ("b", {"curve_rmse_gt": 0.5}),
        ("c", {"curve_rmse_gt": 1.0, "overall_r2_gt": 0.1}),
    ]
    result = _compute(metrics, trials)

    assert result.metric_score("b", "overall_r2_gt") is None
    assert math.isclose(result.composite["b"], (0.5 + 0.0) / 2.0)
    assert result.mean_rank["b"] == (2.0 + 4.0) / 2.0


def test_wins_counts_best_metrics():
    metrics = [("curve_rmse_gt", "RMSE"), ("overall_r2_gt", "R²")]
    trials  = [("a", {"curve_rmse_gt": 0.1, "overall_r2_gt": 0.9}), ("b", {"curve_rmse_gt": 0.2, "overall_r2_gt": 0.5})]
    result  = _compute(metrics, trials)

    assert result.wins["a"] == 2
    assert result.wins["b"] == 0


def test_grouped_score_dampens_correlated_metrics():
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
    groups  = [("Curve error", ["curve_rmse_gt"]), ("Gain vs Capon", ["relative_mse_reduction"])]
    metrics = [(key, key) for _, keys in groups for key in keys]
    trials  = [("a", {"curve_rmse_gt": 0.1}), ("b", {"curve_rmse_gt": 0.2})]
    result  = _compute(metrics, trials)

    breakdown = result.group_breakdown(groups)
    assert breakdown.labels == ["Curve error"]


def test_relative_mse_reduction_higher_ranks_first():
    metrics = [("relative_mse_reduction", "vs Capon")]
    result  = _compute(metrics, [("a", {"relative_mse_reduction": 0.4}), ("b", {"relative_mse_reduction": 0.1})])

    assert result.order()[0] == "a"
    assert result.metric_rank("a", "relative_mse_reduction") == 1.0


def test_average_ranks_descending():
    ranks = RankMath.average_ranks({"a": 3.0, "b": 1.0, "c": 2.0}, reverse=True)
    assert ranks == {"a": 1.0, "c": 2.0, "b": 3.0}


def test_minmax_all_equal_scores_one():
    scores = RankMath.minmax_scores({"a": 0.5, "b": 0.5}, higher=True)
    assert scores == {"a": 1.0, "b": 1.0}
