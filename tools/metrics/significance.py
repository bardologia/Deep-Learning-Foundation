"""Significance testing of seed-paired metric differences between trials.

Compares every trial against the leading trial with a two-sided paired sign-flip
permutation test over shared seeds, and corrects the resulting p-values for
multiple comparisons with the Holm-Bonferroni step-down procedure.
"""

from __future__ import annotations

from itertools import product

import numpy as np


class PairedPermutationTest:
    """Two-sided paired permutation test on the mean of the seedwise differences.

    All 2^n sign assignments are enumerated for small samples; larger samples
    use a random subset with the add-one correction on the p-value.

    Attributes:
        n_permutations : Number of random sign draws used above EXACT_MAX_PAIRS.
        rng            : Generator supplying the random sign draws.
    """

    EXACT_MAX_PAIRS = 12

    def __init__(self, n_permutations: int = 10000, seed: int = 0) -> None:
        """Initialises the test.

        Args:
            n_permutations : Number of random sign draws for large samples.
            seed           : Seed of the random generator.
        """
        self.n_permutations = int(n_permutations)
        self.rng            = np.random.default_rng(seed)

    def smallest_p(self, n_pairs: int) -> float:
        """Returns the smallest two-sided p-value the test can report for a sample size.

        The all-plus and all-minus sign assignments always reproduce the observed
        statistic, so the exact enumeration never reports less than 2 / 2^n, and
        the add-one rule of the sampled regime never reports less than
        1 / (n_permutations + 1). With the standard five seeds the floor is
        0.0625, above the usual 0.05 threshold.

        Args:
            n_pairs : Number of paired observations.

        Returns:
            Smallest attainable p-value.
        """
        if n_pairs <= self.EXACT_MAX_PAIRS:
            return 2.0 / 2 ** n_pairs

        return 1.0 / (self.n_permutations + 1)

    def test(self, a: list[float], b: list[float]) -> dict:
        """Tests whether the mean paired difference between two samples differs from zero.

        Args:
            a : Metric values of the first trial, one per seed.
            b : Metric values of the second trial, in the same seed order.

        Returns:
            Mapping with the observed "mean_diff", the two-sided "p_value", the
            number of "n_pairs", "cohen_d", the paired effect size
            mean_diff / std(diffs), None when the differences have zero spread,
            and "p_min", the smallest p-value attainable with that many pairs.

        Raises:
            ValueError : If the samples are not one-dimensional and of equal length, hold fewer than two pairs or carry a non-finite value.
        """
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)

        if a.shape != b.shape or a.ndim != 1:
            raise ValueError(f"Paired test needs two equal-length 1-D samples, got {a.shape} and {b.shape}")
        if a.size < 2:
            raise ValueError(f"Paired test needs at least 2 pairs, got {a.size}")
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            raise ValueError("Paired test needs only finite values in both samples")

        diffs    = a - b
        observed = diffs.mean()
        spread   = float(diffs.std(ddof=1))
        cohen_d  = float(observed / spread) if spread > 0.0 else None

        if a.size <= self.EXACT_MAX_PAIRS:
            signs = np.asarray(list(product((-1.0, 1.0), repeat=a.size)))
            perms = (signs * diffs).mean(axis=1)
            p     = float((np.abs(perms) >= abs(observed) - 1e-15).mean())
        else:
            signs = self.rng.choice((-1.0, 1.0), size=(self.n_permutations, a.size))
            perms = (signs * diffs).mean(axis=1)
            p     = float(((np.abs(perms) >= abs(observed) - 1e-15).sum() + 1) / (self.n_permutations + 1))

        return {"mean_diff": float(observed), "p_value": p, "n_pairs": int(a.size), "cohen_d": cohen_d, "p_min": self.smallest_p(int(a.size))}


class HolmBonferroni:
    """Holm-Bonferroni step-down correction for multiple comparisons."""

    @staticmethod
    def adjust(p_values: dict[str, float]) -> dict[str, float]:
        """Returns monotone step-down adjusted p-values, capped at one.

        Args:
            p_values : Raw p-value keyed by comparison name.

        Returns:
            Adjusted p-value keyed by comparison name.
        """
        items = sorted(p_values.items(), key=lambda item: item[1])
        m     = len(items)

        adjusted = {}
        running  = 0.0
        for rank, (key, p) in enumerate(items):
            value         = min(1.0, (m - rank) * p)
            running       = max(running, value)
            adjusted[key] = running

        return adjusted


class SignificanceVsLeader:
    """Tests every trial against the leading trial on their shared seeds.

    Attributes:
        test : Paired permutation test used for each comparison.
    """

    def __init__(self, test: PairedPermutationTest | None = None) -> None:
        """Initialises the comparison.

        Args:
            test : Paired permutation test to use; a default instance is built when omitted.
        """
        self.test = test if test is not None else PairedPermutationTest()

    def compute(self, per_seed: dict[str, dict[str, float]], leader: str) -> dict[str, dict]:
        """Compares every non-leading trial with the leader and corrects the p-values.

        Args:
            per_seed : Metric value keyed by trial name then seed identifier.
            leader   : Name of the trial every other trial is compared against.

        Returns:
            Mapping keyed by trial name with "p_value", "p_adjusted",
            "n_pairs", "mean_diff", "cohen_d" and "p_min", the smallest p-value
            the test can reach with that many shared seeds; every entry but
            "n_pairs" is None when fewer than two shared seeds carry a value.

        Raises:
            KeyError : If the leader is absent from the per-seed values.
        """
        if leader not in per_seed:
            raise KeyError(f"Leader '{leader}' is missing from the per-seed values")

        leader_values = per_seed[leader]

        raw     = {}
        results = {}
        for name, values in per_seed.items():
            if name == leader:
                continue

            common = sorted(set(values) & set(leader_values))
            paired = [(values[seed], leader_values[seed]) for seed in common if values[seed] is not None and leader_values[seed] is not None]

            if len(paired) < 2:
                results[name] = {"p_value": None, "p_adjusted": None, "n_pairs": len(paired), "mean_diff": None, "cohen_d": None, "p_min": None}
                continue

            a, b          = zip(*paired)
            outcome       = self.test.test(list(a), list(b))
            raw[name]     = outcome["p_value"]
            results[name] = {"p_value": outcome["p_value"], "p_adjusted": None, "n_pairs": outcome["n_pairs"], "mean_diff": outcome["mean_diff"], "cohen_d": outcome["cohen_d"], "p_min": outcome["p_min"]}

        for name, p_adj in HolmBonferroni.adjust(raw).items():
            results[name]["p_adjusted"] = p_adj

        return results
