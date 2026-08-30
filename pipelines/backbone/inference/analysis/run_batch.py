"""Shared base classes for per-run backbone analyses and their batch drivers.

An analysis is written once as an ``AnalysisRun`` subclass operating on a single
training run directory; the matching ``RunBatch`` subclass resolves the run
filter into run directories and applies the analysis to each of them.
"""

from __future__ import annotations

from pathlib import Path

from pipelines.backbone.inference.loader import RunLoader
from tools.runtime.run_selector          import RunSelector


class AnalysisRun:
    """Base class for an analysis applied to a single backbone training run.

    Attributes:
        run_dir    : Directory of the training run under analysis.
        config     : Analysis configuration carrying split, device and output fields.
        logger     : Logger used for progress and result lines.
        output_dir : Directory inside the run where artifacts are written.
    """

    def __init__(self, run_dir: Path, config, logger) -> None:
        """Binds the analysis to one run directory and resolves its output dir.

        Args:
            run_dir : Directory of the training run to analyse.
            config  : Analysis configuration; must expose ``output_subdir``.
            logger  : Logger for progress reporting.
        """
        self.run_dir = Path(run_dir)
        self.config  = config
        self.logger  = logger

        self.output_dir = self.run_dir / config.output_subdir

    def _load_run(self):
        """Returns the loaded run (model, dataset and loader) for the configured split."""
        return RunLoader(self.run_dir, logger=self.logger).load(
            split           = self.config.split,
            batch_size      = self._batch_size(),
            num_workers     = 0,
            device          = self.config.device,
            checkpoint_name = self.config.checkpoint_name,
            load_tomogram   = False,
        )

    def _batch_size(self) -> int:
        """Returns the loader batch size this analysis needs."""
        return self.config.batch_size


class RunBatch:
    """Base class applying one ``AnalysisRun`` subclass to every selected run.

    Attributes:
        config : Analysis configuration carrying the runs directory and run filter.
        logger : Logger used for the section header and per-run subsections.
    """

    SELECTOR_ACTION : str
    SECTION_TITLE   : str
    RUN_CLASS       : type[AnalysisRun]

    def __init__(self, config, logger) -> None:
        """Stores the analysis configuration and logger for the batch.

        Args:
            config : Analysis configuration; must expose ``runs_dir``, ``run_filter`` and ``checkpoint_name``.
            logger : Logger for section and per-run reporting.
        """
        self.config = config
        self.logger = logger

    def _select_runs(self) -> list[Path]:
        """Returns the run directories matching the configured run filter."""
        selector = RunSelector(self.config.runs_dir, self.config.checkpoint_name, self.logger, action=self.SELECTOR_ACTION)
        return selector.resolve(self.config.run_filter)

    def run(self) -> list[dict]:
        """Runs the analysis on every selected run.

        Returns:
            One result dictionary per run, in selection order.
        """
        self.logger.section(self.SECTION_TITLE)

        results = []
        for run_dir in self._select_runs():
            self.logger.subsection(f"Run: {run_dir}")
            results.append(self.RUN_CLASS(run_dir, self.config, self.logger).run())

        return results
