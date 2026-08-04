from __future__ import annotations

from pathlib import Path

from pipelines.backbone.inference.loader import RunLoader
from tools.runtime.run_selector          import RunSelector


class AnalysisRun:
    def __init__(self, run_dir: Path, config, logger) -> None:
        self.run_dir = Path(run_dir)
        self.config  = config
        self.logger  = logger

        self.output_dir = self.run_dir / config.output_subdir

    def _load_run(self):
        return RunLoader(self.run_dir, logger=self.logger).load(
            split           = self.config.split,
            batch_size      = self._batch_size(),
            num_workers     = 0,
            device          = self.config.device,
            checkpoint_name = self.config.checkpoint_name,
        )

    def _batch_size(self) -> int:
        return self.config.batch_size


class RunBatch:
    SELECTOR_ACTION : str
    SECTION_TITLE   : str
    RUN_CLASS       : type[AnalysisRun]

    def __init__(self, config, logger) -> None:
        self.config = config
        self.logger = logger

    def _select_runs(self) -> list[Path]:
        selector = RunSelector(self.config.runs_dir, self.config.checkpoint_name, self.logger, action=self.SELECTOR_ACTION)
        return selector.resolve(self.config.run_filter)

    def run(self) -> list[dict]:
        self.logger.section(self.SECTION_TITLE)

        results = []
        for run_dir in self._select_runs():
            self.logger.subsection(f"Run: {run_dir}")
            results.append(self.RUN_CLASS(run_dir, self.config, self.logger).run())

        return results
