from __future__ import annotations

from pathlib import Path

from pipelines.shared.comparison.trial_collection import SeedRunAggregator, TrialCollector
from tools.monitoring.logger                      import Logger


class BenchmarkSeedCollector(TrialCollector):
    def __init__(self, run_dir: Path, logger: Logger) -> None:
        super().__init__(run_dir=run_dir, logger=logger)
        self.seed_dispersion = {}

    def collect(self) -> list:
        aggregator = SeedRunAggregator()
        records    = aggregator.aggregate(super().collect())

        self.seed_dispersion = aggregator.seed_dispersion
        return records
