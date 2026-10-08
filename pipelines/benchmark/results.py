"""Collection of benchmark run results with per-model seed aggregation."""

from __future__ import annotations

from pathlib import Path

from pipelines.shared.comparison.trial_collection import SeedRunAggregator, TrialCollector
from tools.monitoring.logger                      import Logger


class BenchmarkSeedCollector(TrialCollector):
    """Collects benchmark runs and folds their per-seed repeats into one record each.

    Attributes:
        runs            : Per-run records as collected from the training directory, before seed aggregation; populated by `collect`.
        seed_dispersion : Per-metric spread across seeds, populated by `collect`; every entry also carries ``n_used``, the per-metric count of seeds whose values entered the model's means, and ``missing_seeds``, the seeds without metrics.
    """

    def __init__(self, run_dir: Path, logger: Logger, inference_split: str | None) -> None:
        """Stores the run directory to scan and the split whose metrics are read.

        Args:
            run_dir         : Benchmark run directory holding the per-unit run directories.
            logger          : Logger for the collection pass.
            inference_split : Dataset split whose inference metrics are collected, or None for the training metrics.
        """
        super().__init__(run_dir=run_dir, logger=logger, inference_split=inference_split)
        self.runs            = []
        self.seed_dispersion = {}

    def _log_seed_coverage(self) -> None:
        """Records, per seed-swept model, the seeds without metrics, and closes with the models whose means cover fewer seeds than they trained.

        The per-model warnings, naming the seeds without inference and the
        metrics averaged over fewer seeds, come from the seed aggregator; this
        step adds ``missing_seeds`` to each dispersion entry and the summary
        line across models.
        """
        short = []

        for name, entry in self.seed_dispersion.items():
            seed_metrics = entry["seed_metrics"]
            missing      = [seed for seed, metrics in seed_metrics.items() if not metrics]

            entry["missing_seeds"] = missing

            if missing or min(entry["n_used"].values(), default=0) < entry["n_seeds"]:
                short.append(name)

        if short:
            self.logger.warning(f"{len(short)} of {len(self.seed_dispersion)} seed-swept models average fewer seeds than they trained: {short}")

    def collect(self) -> list:
        """Collects the run records and aggregates the seed repeats of each model.

        Returns:
            One record per model unit, with the metrics averaged over its seeds.
        """
        aggregator = SeedRunAggregator(self.logger)
        self.runs  = super().collect()
        records    = aggregator.aggregate(self.runs)

        self.seed_dispersion = aggregator.seed_dispersion
        self._log_seed_coverage()

        return records
