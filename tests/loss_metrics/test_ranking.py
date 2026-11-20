
from __future__ import annotations

import math

from tools.metrics.ranking import RankingComputer, RankMath


HIGHER = [("r2_mean", "R²")]
LOWER  = [("rmse_mean", "RMSE")]


def _compute(metrics, trials):
    return RankingComputer(metrics, trials).compute()


def test_lower_metric_best_value_ranks_first():
    result = _compute(LOWER, [("a", {"rmse_mean": 0.1}), ("b", {"rmse_mean": 0.9})])
    assert result.order()[0] == "a"
    assert result.metric_rank("a", "rmse_mean") == 1.0
    assert result.metric_rank("b", "rmse_mean") == 2.0


def test_higher_metric_best_value_ranks_first():
    result = _compute(HIGHER, [("a", {"r2_mean": 0.2}), ("b", {"r2_mean": 0.8})])
    assert result.order()[0] == "b"


def test_ties_share_averaged_rank():
    result = _compute(LOWER, [("a", {"rmse_mean": 0.5}), ("b", {"rmse_mean": 0.5}), ("c", {"rmse_mean": 0.9})])
    assert result.metric_rank("a", "rmse_mean") == 1.5
    assert result.metric_rank("b", "rmse_mean") == 1.5
    assert result.metric_rank("c", "rmse_mean") == 3.0


def test_score_is_magnitude_aware():
    trials = [("a", {"rmse_mean": 0.0}), ("b", {"rmse_mean": 0.1}), ("c", {"rmse_mean": 1.0})]
    result = _compute(LOWER, trials)

    assert math.isclose(result.scores["a"]["rmse_mean"], 1.0)
    assert math.isclose(result.scores["c"]["rmse_mean"], 0.0)
    assert math.isclose(result.scores["b"]["rmse_mean"], 0.9)


def test_score_separates_what_ordinal_rank_cannot():
    near = _compute(LOWER, [("a", {"rmse_mean": 0.0}), ("b", {"rmse_mean": 0.1}), ("c", {"rmse_mean": 10.0})])
    far  = _compute(LOWER, [("a", {"rmse_mean": 0.0}), ("b", {"rmse_mean": 9.9}), ("c", {"rmse_mean": 10.0})])

    assert near.mean_rank["b"] == far.mean_rank["b"]
    assert near.composite["b"] > far.composite["b"]


def test_missing_metric_scores_zero_and_ranks_last():
    metrics = [("rmse_mean", "RMSE"), ("r2_mean", "R²")]
    trials  = [
        ("a", {"rmse_mean": 0.0, "r2_mean": 0.9}),
        ("b", {"rmse_mean": 0.5}),
        ("c", {"rmse_mean": 1.0, "r2_mean": 0.1}),
    ]
    result = _compute(metrics, trials)

    assert result.scores["b"].get("r2_mean") is None
    assert math.isclose(result.composite["b"], (0.5 + 0.0) / 2.0)
    assert result.mean_rank["b"] == (2.0 + 4.0) / 2.0


def test_wins_counts_best_metrics():
    metrics = [("rmse_mean", "RMSE"), ("r2_mean", "R²")]
    trials  = [("a", {"rmse_mean": 0.1, "r2_mean": 0.9}), ("b", {"rmse_mean": 0.2, "r2_mean": 0.5})]
    result  = _compute(metrics, trials)

    assert result.wins["a"] == 2
    assert result.wins["b"] == 0


def test_average_ranks_descending():
    ranks = RankMath.average_ranks({"a": 3.0, "b": 1.0, "c": 2.0}, reverse=True)
    assert ranks == {"a": 1.0, "c": 2.0, "b": 3.0}


def test_average_ranks_ties_within_the_shared_tolerance():
    values = {"a": 1.0, "b": 1.0 + RankMath.TIE_TOLERANCE / 2.0, "c": 2.0}
    ranks  = RankMath.average_ranks(values, reverse=False)

    assert ranks["a"] == ranks["b"] == 1.5
    assert ranks["c"] == 3.0
    assert RankMath.leaders({"a": {"m": 1.0}, "b": {"m": 1.0 + RankMath.TIE_TOLERANCE / 2.0}}, "m") == {"a", "b"}


def test_minmax_all_equal_scores_one():
    scores = RankMath.minmax_scores({"a": 0.5, "b": 0.5}, higher=True)
    assert scores == {"a": 1.0, "b": 1.0}
