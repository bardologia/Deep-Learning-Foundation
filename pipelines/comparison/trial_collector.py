
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib     import Path

from pipelines.shared.comparison.trial_collection import SeedRunAggregator, TrialArtifactReader
from pipelines.shared.comparison.trial_collection import TrialRecord as BaseTrialRecord
from tools.monitoring.logger import Logger


@dataclass
class TrialRecord(BaseTrialRecord):

    figures_dir: Path | None = None

    def figure_subdir(self, name: str) -> Path | None:
        if self.figures_dir is None:
            return None
        path = self.figures_dir / name
        return path if path.is_dir() else None


class TrialCollector(TrialArtifactReader):

    SEED_DIR_PATTERN = re.compile(r"seed\d+")

    def __init__(self, runs_dir: Path, run_tags: list[str], logger: Logger, inference_split: str) -> None:
        self.runs_dir        = runs_dir
        self.run_tags        = run_tags
        self.logger          = logger
        self.inference_split = inference_split
        self.seed_dispersion = {}

    def _attach_figures(self, record: TrialRecord, inference_dir: Path) -> None:
        figures_dir = inference_dir / "figures"
        if figures_dir.is_dir():
            record.figures_dir = figures_dir

    def _nested_seed_dirs(self, run_dir: Path) -> list[Path]:
        if not run_dir.is_dir() or self.SEED_DIR_PATTERN.fullmatch(run_dir.name):
            return []

        return sorted(d for d in run_dir.iterdir() if d.is_dir() and self.SEED_DIR_PATTERN.fullmatch(d.name))

    def _expand_tags(self) -> list[tuple[str, Path]]:
        expanded = []
        seen     = set()

        for tag in self.run_tags:
            run_dir   = self.runs_dir / tag
            seed_dirs = self._nested_seed_dirs(run_dir)

            if seed_dirs:
                self.logger.ok(f"'{tag}' expanded to {len(seed_dirs)} seed run(s)")

            pairs = [(f"{tag}/{seed_dir.name}", seed_dir) for seed_dir in seed_dirs] if seed_dirs else [(tag, run_dir)]
            for name, path in pairs:
                if str(path) in seen:
                    continue
                seen.add(str(path))
                expanded.append((name, path))

        return expanded

    def collect(self) -> list[TrialRecord]:
        self.logger.section("Collecting trials")

        expanded = self._expand_tags()

        records = []
        for tag, run_dir in expanded:
            record            = TrialRecord(name=tag, run_dir=run_dir)
            record.checkpoint = self._read_checkpoint(run_dir)
            self._attach_inference(record)
            self._warn_unfinished(record)
            self._warn_no_inference(record)
            self.logger.info(f"{record.name:<36} {self._status(record)}")

            records.append(record)

        aggregator = SeedRunAggregator(self.logger)
        aggregated = aggregator.aggregate(records)

        self.seed_dispersion = aggregator.seed_dispersion
        if self.seed_dispersion:
            n_seed_runs = sum(entry["n_seeds"] for entry in self.seed_dispersion.values())
            self.logger.ok(f"Aggregated {n_seed_runs} seed run(s) into {len(self.seed_dispersion)} trial(s), metrics reported as seed means")

        scored = sum(1 for record in aggregated if record.metrics)

        self.logger.ok(f"{len(aggregated)} trial(s) collected, {scored} with inference metrics on split '{self.inference_split}'")

        return aggregated
