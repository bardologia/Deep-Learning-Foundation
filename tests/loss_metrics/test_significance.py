"""Tests covering the paired permutation test and its multiple-comparison correction.

Exercises ``PairedPermutationTest`` in exact and Monte Carlo regimes, the
``HolmBonferroni`` adjustment, and the leader-versus-rest sweep in
``SignificanceVsLeader``.
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.metrics.significance import HolmBonferroni, PairedPermutationTest, SignificanceVsLeader


def test_identical_samples_are_not_significant():
    """Verifies identical paired samples give a zero mean difference and a p-value of one."""
    outcome = PairedPermutationTest().test([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0])

    assert outcome["mean_diff"] == pytest.approx(0.0)
    assert outcome["p_value"]   == pytest.approx(1.0)
    assert outcome["n_pairs"]   == 5


def test_consistent_shift_is_maximally_significant_for_five_pairs():
    """Verifies a consistent shift over five pairs reaches the smallest attainable exact p-value."""
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [2.0, 3.1, 4.2, 5.05, 6.3]

    outcome = PairedPermutationTest().test(a, b)

    assert outcome["mean_diff"] < 0.0
    assert outcome["p_value"]   == pytest.approx(2.0 / 32.0)


def test_exact_and_montecarlo_agree_roughly():
    """Verifies the Monte Carlo regime detects a unit shift over twenty pairs at the one percent level."""
    rng = np.random.default_rng(0)
    a   = rng.normal(0.0, 1.0, size=20)
    b   = a + 1.0

    outcome = PairedPermutationTest(n_permutations=5000, seed=1).test(list(a), list(b))

    assert outcome["p_value"] < 0.01


def test_mismatched_lengths_raise():
    """Verifies samples of different lengths are rejected."""
    with pytest.raises(ValueError):
        PairedPermutationTest().test([1.0, 2.0], [1.0])


def test_single_pair_raises():
    """Verifies a single pair is rejected as too few observations."""
    with pytest.raises(ValueError):
        PairedPermutationTest().test([1.0], [2.0])


def test_holm_adjustment_is_monotone_and_bounded():
    """Verifies Holm adjustment scales each p-value by its step and stays non-decreasing."""
    adjusted = HolmBonferroni.adjust({"a": 0.01, "b": 0.04, "c": 0.5})

    assert adjusted["a"] == pytest.approx(0.03)
    assert adjusted["b"] == pytest.approx(0.08)
    assert adjusted["c"] == pytest.approx(0.5)
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"]


def test_significance_vs_leader_pairs_common_seeds_and_adjusts():
    """Verifies the leader sweep pairs on shared seeds, adjusts p-values, skips the leader and reports None when pairs are too few."""
    per_seed = {
        "leader" : {"seed0": 1.0, "seed1": 1.1, "seed2": 0.9, "seed3": 1.0, "seed4": 1.05},
        "worse"  : {"seed0": 2.0, "seed1": 2.2, "seed2": 1.9, "seed3": 2.1, "seed4": 2.05},
        "tied"   : {"seed0": 1.0, "seed1": 1.1, "seed2": 0.9, "seed3": 1.0, "seed4": 1.05},
        "sparse" : {"seed0": 3.0},
    }

    results = SignificanceVsLeader().compute(per_seed, "leader")

    assert results["worse"]["p_value"]     == pytest.approx(2.0 / 32.0)
    assert results["worse"]["p_adjusted"]  >= results["worse"]["p_value"]
    assert results["tied"]["p_value"]      == pytest.approx(1.0)
    assert results["sparse"]["p_value"]    is None
    assert results["sparse"]["n_pairs"]    == 1
    assert "leader" not in results


def test_significance_vs_leader_missing_leader_raises():
    """Verifies naming a leader absent from the per-seed data raises."""
    with pytest.raises(KeyError):
        SignificanceVsLeader().compute({"a": {"seed0": 1.0}}, "missing")
