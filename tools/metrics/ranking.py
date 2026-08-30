"""Cross-trial ranking of models over a metric set.

Ranks and min-max scores every trial per metric according to that metric's
orientation, then aggregates into a mean rank, a composite score and a win
count, with an optional breakdown by metric group.
"""

from __future__ import annotations

from dataclasses import dataclass

from tools.metrics.scoring import FiniteScalar, MetricOrientation


class RankMath:
    """Tie-aware ranking, scaling and leader selection primitives."""

    TIE_TOLERANCE = 1e-12

    @staticmethod
    def average_ranks(values: dict[str, float], reverse: bool) -> dict[str, float]:
        """Returns competition ranks with ties sharing their average rank.

        Args:
            values  : Metric value keyed by trial name.
            reverse : Whether larger values rank first.

        Returns:
            One-based rank keyed by trial name.
        """
        ordered = sorted(values.items(), key=lambda item: item[1], reverse=reverse)
        ranks: dict[str, float] = {}

        start = 0
        count = len(ordered)
        while start < count:
            stop = start
            while stop + 1 < count and abs(ordered[stop + 1][1] - ordered[start][1]) <= RankMath.TIE_TOLERANCE:
                stop += 1

            average = (start + 1 + stop + 1) / 2.0
            for index in range(start, stop + 1):
                ranks[ordered[index][0]] = average

            start = stop + 1

        return ranks

    @staticmethod
    def minmax_scores(values: dict[str, float], higher: bool) -> dict[str, float]:
        """Returns goodness scores in [0, 1], one for the best trial and zero for the worst.

        Args:
            values : Metric value keyed by trial name.
            higher : Whether larger values are better.

        Returns:
            Score keyed by trial name; every trial scores 1.0 when the values
            span less than the tie tolerance, and an empty input maps to an
            empty result.
        """
        if not values:
            return {}

        goodness = {name: (value if higher else -value) for name, value in values.items()}
        low      = min(goodness.values())
        high     = max(goodness.values())

        if high - low <= RankMath.TIE_TOLERANCE:
            return {name: 1.0 for name in values}

        span = high - low
        return {name: (goodness[name] - low) / span for name in values}

    @staticmethod
    def leaders(scores: dict[str, dict[str, float]], key: str) -> set[str]:
        """Returns the trial names holding the best score for one metric, ties included.

        Args:
            scores : Per-metric score mapping keyed by trial name.
            key    : Metric key to select the leaders of.

        Returns:
            Names of the leading trials, empty when no trial reports the metric.
        """
        present = {name: per_metric[key] for name, per_metric in scores.items() if key in per_metric}
        if not present:
            return set()

        best = max(present.values())
        return {name for name, value in present.items() if value >= best - RankMath.TIE_TOLERANCE}


@dataclass
class GroupBreakdown:
    """Ranking aggregated over named groups of metrics.

    Attributes:
        labels      : Group labels that had at least one reported metric.
        group_score : Mean group score per group label, keyed by trial name.
        group_rank  : Rank within each group, keyed by group label then trial name.
        overall     : Mean of the group scores, keyed by trial name.
    """
    labels      : list[str]
    group_score : dict[str, dict[str, float]]
    group_rank  : dict[str, dict[str, float]]
    overall     : dict[str, float]

    def order(self) -> list[str]:
        """Returns the trial names ordered by descending overall score, then by name."""
        return sorted(self.overall, key=lambda name: (-self.overall[name], name))

    def leader_overall(self) -> float:
        """Returns the best overall score, or zero when no trial is present."""
        return max(self.overall.values()) if self.overall else 0.0

    def group_leaders(self, label: str) -> set[str]:
        """Returns the trial names holding the best score in one group, ties included."""
        present = {name: self.group_score[name][label] for name in self.group_score}
        if not present:
            return set()

        best = max(present.values())
        return {name for name, value in present.items() if value >= best - RankMath.TIE_TOLERANCE}


@dataclass
class RankingResult:
    """Outcome of ranking a set of trials over a metric set.

    Attributes:
        names       : Trial names in input order.
        metric_keys : Metric keys the ranking covered.
        ranks       : Per-metric rank keyed by trial name then metric key.
        scores      : Per-metric min-max score keyed by trial name then metric key.
        mean_rank   : Mean rank across the metric set, keyed by trial name.
        composite   : Mean score across the metric set, keyed by trial name.
        wins        : Number of metrics a trial leads, keyed by trial name.
    """
    names       : list[str]
    metric_keys : list[str]
    ranks       : dict[str, dict[str, float]]
    scores      : dict[str, dict[str, float]]
    mean_rank   : dict[str, float]
    composite   : dict[str, float]
    wins        : dict[str, int]

    @staticmethod
    def format_rank(value: float | None) -> str:
        """Returns a rank rendered as an integer, a one-decimal tie, or an em dash for None."""
        if value is None:
            return "—"

        rounded = round(value)
        if abs(value - rounded) < 1e-9:
            return str(int(rounded))
        return f"{value:.1f}"

    def metric_rank(self, name: str, key: str) -> float | None:
        """Returns one trial's rank for a metric, or None when it did not report it."""
        return self.ranks[name].get(key)

    def metric_leaders(self, key: str) -> set[str]:
        """Returns the trial names leading one metric, ties included."""
        return RankMath.leaders(self.scores, key)

    def order(self) -> list[str]:
        """Returns the trial names ordered by composite score, then mean rank, then name."""
        return sorted(self.names, key=lambda name: (-self.composite[name], self.mean_rank[name], name))

    def leader_composite(self) -> float:
        """Returns the best composite score, or zero when no trial is present."""
        return max(self.composite.values()) if self.composite else 0.0

    def group_breakdown(self, groups: list[tuple[str, list[str]]]) -> GroupBreakdown:
        """Aggregates the per-metric scores over named metric groups.

        Args:
            groups : Tuples of (group label, metric keys); groups no trial reports are dropped, and missing metrics score zero.

        Returns:
            The group scores, within-group ranks and overall means.
        """
        kept = [
            (label, keys)
            for label, keys in groups
            if any(key in self.scores[name] for name in self.names for key in keys)
        ]

        group_score = {name: {} for name in self.names}
        for label, keys in kept:
            for name in self.names:
                members = [self.scores[name].get(key, 0.0) for key in keys]
                group_score[name][label] = sum(members) / len(keys)

        group_rank = {}
        for label, _ in kept:
            values            = {name: group_score[name][label] for name in self.names}
            group_rank[label] = RankMath.average_ranks(values, reverse=True)

        overall = {
            name: (sum(group_score[name][label] for label, _ in kept) / len(kept) if kept else 0.0)
            for name in self.names
        }

        return GroupBreakdown(
            labels      = [label for label, _ in kept],
            group_score = group_score,
            group_rank  = group_rank,
            overall     = overall,
        )


class RankingComputer:
    """Ranks a set of trials over a metric set.

    Attributes:
        metrics : Tuples of (metric key, display label) defining the metric set.
        trials  : Tuples of (trial name, metric mapping) to rank.
    """

    WORST_RANK_PADDING = 1

    def __init__(self, metrics: list[tuple[str, str]], trials: list[tuple[str, dict]]) -> None:
        """Initialises the computer.

        Args:
            metrics : Tuples of (metric key, display label).
            trials  : Tuples of (trial name, metric mapping).
        """
        self.metrics = metrics
        self.trials  = trials

    @property
    def names(self) -> list[str]:
        """Trial names in input order."""
        return [name for name, _ in self.trials]

    def _present_values(self, key: str) -> dict[str, float]:
        """Returns the finite values of one metric, keyed by the trials that report it."""
        values = {}
        for name, metrics in self.trials:
            value = FiniteScalar.coerce(metrics.get(key))
            if value is not None:
                values[name] = value
        return values

    def _rank_and_score(self) -> tuple[dict, dict]:
        """Ranks and min-max scores every trial per metric, skipping neutral metrics.

        Returns:
            Tuple of the rank and score mappings, each keyed by trial name then
            metric key.
        """
        ranks  = {name: {} for name in self.names}
        scores = {name: {} for name in self.names}

        for key, _ in self.metrics:
            direction = MetricOrientation.direction(key)
            if direction is None:
                continue

            values     = self._present_values(key)
            higher     = direction == "higher"
            key_ranks  = RankMath.average_ranks(values, reverse=higher)
            key_scores = RankMath.minmax_scores(values, higher=higher)

            for name in values:
                ranks[name][key]  = key_ranks[name]
                scores[name][key] = key_scores[name]

        return ranks, scores

    def _aggregate(self, ranks: dict, scores: dict) -> tuple[dict, dict, dict]:
        """Reduces the per-metric ranks and scores to per-trial aggregates.

        Missing ranks count as one worse than the last place, and missing
        scores count as zero.

        Args:
            ranks  : Per-metric ranks keyed by trial name then metric key.
            scores : Per-metric scores keyed by trial name then metric key.

        Returns:
            Tuple of the mean rank, composite score and win count mappings.
        """
        worst     = len(self.trials) + self.WORST_RANK_PADDING
        n_metrics = len(self.metrics)

        mean_rank = {
            name: sum(ranks[name].get(key, worst) for key, _ in self.metrics) / n_metrics
            for name in self.names
        }
        composite = {
            name: sum(scores[name].get(key, 0.0) for key, _ in self.metrics) / n_metrics
            for name in self.names
        }
        wins = {
            name: sum(1 for key, _ in self.metrics if name in RankMath.leaders(scores, key))
            for name in self.names
        }

        return mean_rank, composite, wins

    def compute(self) -> RankingResult:
        """Ranks, scores and aggregates the trials over the configured metric set.

        Returns:
            The populated ranking result.
        """
        ranks, scores              = self._rank_and_score()
        mean_rank, composite, wins = self._aggregate(ranks, scores)

        return RankingResult(
            names       = self.names,
            metric_keys = [key for key, _ in self.metrics],
            ranks       = ranks,
            scores      = scores,
            mean_rank   = mean_rank,
            composite   = composite,
            wins        = wins,
        )
