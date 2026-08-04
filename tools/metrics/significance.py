from __future__ import annotations

from itertools import product

import numpy as np


class PairedPermutationTest:

    EXACT_MAX_PAIRS = 12

    def __init__(self, n_permutations: int = 10000, seed: int = 0) -> None:
        self.n_permutations = int(n_permutations)
        self.rng            = np.random.default_rng(seed)

    def test(self, a: list[float], b: list[float]) -> dict:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)

        if a.shape != b.shape or a.ndim != 1:
            raise ValueError(f"Paired test needs two equal-length 1-D samples, got {a.shape} and {b.shape}")
        if a.size < 2:
            raise ValueError(f"Paired test needs at least 2 pairs, got {a.size}")

        diffs    = a - b
        observed = diffs.mean()

        if a.size <= self.EXACT_MAX_PAIRS:
            signs = np.asarray(list(product((-1.0, 1.0), repeat=a.size)))
            perms = (signs * diffs).mean(axis=1)
            p     = float((np.abs(perms) >= abs(observed) - 1e-15).mean())
        else:
            signs = self.rng.choice((-1.0, 1.0), size=(self.n_permutations, a.size))
            perms = (signs * diffs).mean(axis=1)
            p     = float(((np.abs(perms) >= abs(observed) - 1e-15).sum() + 1) / (self.n_permutations + 1))

        return {"mean_diff": float(observed), "p_value": p, "n_pairs": int(a.size)}


class HolmBonferroni:

    @staticmethod
    def adjust(p_values: dict[str, float]) -> dict[str, float]:
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

    def __init__(self, test: PairedPermutationTest | None = None) -> None:
        self.test = test if test is not None else PairedPermutationTest()

    def compute(self, per_seed: dict[str, dict[str, float]], leader: str) -> dict[str, dict]:
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
                results[name] = {"p_value": None, "p_adjusted": None, "n_pairs": len(paired), "mean_diff": None}
                continue

            a, b          = zip(*paired)
            outcome       = self.test.test(list(a), list(b))
            raw[name]     = outcome["p_value"]
            results[name] = {"p_value": outcome["p_value"], "p_adjusted": None, "n_pairs": outcome["n_pairs"], "mean_diff": outcome["mean_diff"]}

        for name, p_adj in HolmBonferroni.adjust(raw).items():
            results[name]["p_adjusted"] = p_adj

        return results
