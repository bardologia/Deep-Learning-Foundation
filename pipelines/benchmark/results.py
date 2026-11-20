
from __future__ import annotations

from pathlib import Path

from pipelines.shared.comparison.trial_collection import SeedRunAggregator, TrainingResultsIndex, TrialCollector
from pipelines.shared.orchestration.unit_manifest import UnitResultsByRun
from tools.monitoring.logger                      import Logger


class BenchmarkSeedCollector(TrialCollector):

    def __init__(self, run_dir: Path, logger: Logger, inference_split: str | None) -> None:
        super().__init__(run_dir=run_dir, logger=logger, inference_split=inference_split)
        self.runs            = []
        self.seed_dispersion = {}

    def _training_results_index(self, runs: list[tuple[str, Path]]) -> dict:
        return UnitResultsByRun(self.training_dir, self.logger).build(TrainingResultsIndex(self.pipeline_dir, self.logger), runs)

    def _log_seed_coverage(self) -> None:
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
        aggregator = SeedRunAggregator(self.logger)
        self.runs  = super().collect()
        records    = aggregator.aggregate(self.runs)

        self.seed_dispersion = aggregator.seed_dispersion
        self._log_seed_coverage()

        return records
