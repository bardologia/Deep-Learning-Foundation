"""Shared base classes for per-run backbone analyses and their batch drivers.

An analysis is written once as an ``AnalysisRun`` subclass operating on a single
training run directory; the matching ``RunBatch`` subclass resolves the run
filter into run directories and applies the analysis to each of them. Every
pass records its resolved configuration, the checkpoint it loaded and the code
revision beside its outputs, and stamps a completion marker once they are
written, so an output directory states what produced it and whether it finished.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from pipelines.backbone.inference.loader import RunLoader
from tools.data.io                       import FileIO
from tools.reporting.plotting            import PlotBase
from tools.runtime.completion            import CompletionMarker
from tools.runtime.config_cli            import ConfigCli
from tools.runtime.git_state             import GitState
from tools.runtime.run_selector          import RunSelector
from tools.runtime.run_tag               import RunTag


@dataclass
class AnalysisOutcome:
    """Result of one per-run analysis, as logged on its result line.

    Attributes:
        payload     : Payload the analysis wrote as its summary JSON.
        headline    : One-line statement of the analysis's main result.
        report_path : Path of the written markdown report.
        status      : 'ok', 'warning' or 'critical'; selects the level of the result line.
        replaced    : Whether the output directory held an earlier result that this pass replaced.
    """

    payload     : dict
    headline    : str
    report_path : Path
    status      : str  = "ok"
    replaced    : bool = False


class AnalysisRun:
    """Base class for an analysis applied to a single backbone training run.

    The orchestrator loads the run, prepares the output directory, records the
    configuration, checkpoint and revision of the pass, runs the subclass's
    analysis, stamps the completion marker and logs the result line.

    Attributes:
        run_dir             : Directory of the training run under analysis.
        config              : Analysis configuration carrying split, device and output fields.
        logger              : Logger used for progress and result lines.
        output_dir          : Directory inside the run where artifacts are written.
        CONFIG_FILENAME     : File receiving the resolved analysis configuration.
        PROVENANCE_FILENAME : File receiving the checkpoint, split, device and revision of the pass.
        PLOTS_SUBDIR        : Figure subdirectory removed before a pass replaces an earlier result.
    """

    SUMMARY_FILENAME    : str
    CONFIG_FILENAME     = "analysis_config.json"
    PROVENANCE_FILENAME = "analysis_provenance.json"
    PLOTS_SUBDIR        = "plots"

    def __init__(self, run_dir: Path, config, logger) -> None:
        """Binds the analysis to one run directory and resolves its output dir.

        Args:
            run_dir : Directory of the training run to analyse.
            config  : Analysis configuration; must expose ``output_subdir``.
            logger  : Logger for progress reporting.

        Raises:
            ValueError : If the output subdirectory resolves to the run directory itself, whose completion marker the analysis would clear and overwrite.
        """
        self.run_dir = Path(run_dir)
        self.config  = config
        self.logger  = logger

        self.output_dir = self.run_dir / config.output_subdir

        if self.output_dir.resolve() == self.run_dir.resolve():
            raise ValueError(f"output_subdir '{config.output_subdir}' resolves to the run directory {self.run_dir}; an analysis clears and stamps the completion marker of its output directory, so it needs a subdirectory of its own")

    @property
    def stage(self) -> str:
        """Returns the analysis name recorded in the completion marker, the stem of its summary file."""
        return Path(self.SUMMARY_FILENAME).stem

    def _load_run(self):
        """Returns the loaded run (model, dataset and loader) for the configured split."""
        return RunLoader(self.run_dir, logger=self.logger).load(
            split           = self.config.split,
            batch_size      = self._batch_size(),
            num_workers     = 0,
            device          = self.config.device,
            checkpoint_name = self.config.checkpoint_name,
            load_cube   = False,
        )

    def _batch_size(self) -> int:
        """Returns the loader batch size this analysis needs."""
        return self.config.batch_size

    def _prepare_output(self) -> bool:
        """Clears the completion marker and figure subdirectory of an earlier pass, warning when one is replaced.

        An earlier pass is recognised by its completion marker, its summary JSON or
        its provenance file; without a marker it never finished.

        Returns:
            Whether the output directory held an earlier result.
        """
        marker   = CompletionMarker.path(self.output_dir)
        earlier  = [marker, self.output_dir / self.SUMMARY_FILENAME, self.output_dir / self.PROVENANCE_FILENAME]
        previous = any(path.is_file() for path in earlier)

        if previous and marker.is_file():
            stamp = CompletionMarker.payload(self.output_dir)
            self.logger.warning(f"{self.run_dir.name}: {self.output_dir} holds a complete {self.stage} result of checkpoint '{stamp['checkpoint']}' (epoch {stamp['epoch']}, split '{stamp['split']}', completed {stamp['completed_at']}); it is replaced and its {self.PLOTS_SUBDIR}/ figures are removed")
        elif previous:
            self.logger.warning(f"{self.run_dir.name}: {self.output_dir} holds an unfinished {self.stage} result without a completion marker; it is replaced and its {self.PLOTS_SUBDIR}/ figures are removed")

        CompletionMarker.clear(self.output_dir)

        plots_dir = self.output_dir / self.PLOTS_SUBDIR
        if plots_dir.is_dir():
            shutil.rmtree(plots_dir)

        FileIO.ensure_dirs(self.output_dir)
        return previous

    def _record_provenance(self, run) -> None:
        """Writes the resolved configuration and the checkpoint, split and revision of this pass.

        Args:
            run : Loaded run whose checkpoint metadata is recorded.
        """
        ConfigCli.save_resolved(self.config, self.output_dir / self.CONFIG_FILENAME)

        checkpoint = {"name": self.config.checkpoint_name, **run.checkpoint_meta}
        FileIO.save_json({
            "analysis"   : self.stage,
            "run_dir"    : str(self.run_dir),
            "split"      : self.config.split,
            "device"     : self.config.device,
            "checkpoint" : checkpoint,
            "git_commit" : GitState.commit(),
            "started_at" : RunTag.timestamp(),
        }, self.output_dir / self.PROVENANCE_FILENAME)

        self.logger.kv_table(checkpoint, title="Checkpoint")

    def _analyse(self, run) -> AnalysisOutcome:
        """Runs the analysis on the loaded run and writes its outputs.

        Args:
            run : Loaded run bundling the model, the dataloader and the dataset.

        Returns:
            The outcome of the analysis.

        Raises:
            NotImplementedError : Always; every analysis implements its own pass.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement _analyse")

    def _stamp_complete(self, run, outcome: AnalysisOutcome) -> None:
        """Stamps the completion marker naming the analysis, split, checkpoint and status.

        Args:
            run     : Loaded run whose checkpoint epoch is recorded.
            outcome : Outcome of the finished analysis.
        """
        CompletionMarker.stamp(self.output_dir, {
            "stage"      : self.stage,
            "split"      : self.config.split,
            "checkpoint" : self.config.checkpoint_name,
            "epoch"      : run.checkpoint_meta["epoch"],
            "status"     : outcome.status,
        })

    def _report(self, outcome: AnalysisOutcome) -> None:
        """Logs the result line at the level its status calls for.

        Args:
            outcome : Outcome of the finished analysis.

        Raises:
            KeyError : If the status is not 'ok', 'warning' or 'critical'.
        """
        levels = {"ok": self.logger.ok, "warning": self.logger.warning, "critical": self.logger.error}
        levels[outcome.status](f"{self.run_dir.name}: {outcome.headline} -> {outcome.report_path}")

    def run(self) -> AnalysisOutcome:
        """Loads the run, records the pass, analyses it and stamps the completion marker.

        Returns:
            The outcome of the analysis, flagged when it replaced an earlier result.
        """
        run      = self._load_run()
        replaced = self._prepare_output()

        PlotBase.use_style(self.config.figure_style)
        self._record_provenance(run)

        outcome          = self._analyse(run)
        outcome.replaced = replaced

        self._stamp_complete(run, outcome)
        self._report(outcome)

        return outcome


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

    def _analyse_run(self, run_dir: Path, index: int, total: int) -> tuple[AnalysisOutcome, float]:
        """Runs the analysis on one run under a timer that logs its duration on success and on failure.

        Args:
            run_dir : Run directory to analyse.
            index   : One-based position of the run in the batch.
            total   : Number of runs in the batch.

        Returns:
            A pair of the analysis outcome and its wall-clock duration in seconds.
        """
        self.logger.subsection(f"Run {index}/{total}: {run_dir}")

        start = time.perf_counter()
        with self.logger.timer(f"{self.SECTION_TITLE} [{index}/{total}] {run_dir.name}"):
            outcome = self.RUN_CLASS(run_dir, self.config, self.logger).run()

        return outcome, time.perf_counter() - start

    def _summarize(self, run_dirs: list[Path], outcomes: list[AnalysisOutcome], durations: list[float]) -> None:
        """Logs one table row per run with its status, result, duration and report, then the batch tally.

        Args:
            run_dirs  : Analysed run directories, in batch order.
            outcomes  : Outcome of each run.
            durations : Wall-clock duration of each run in seconds.
        """
        runs_dir = Path(self.config.runs_dir)
        rows     = []
        for run_dir, outcome, duration in zip(run_dirs, outcomes, durations):
            rows.append({
                "Run"      : str(run_dir.relative_to(runs_dir)),
                "Status"   : outcome.status,
                "Result"   : outcome.headline,
                "Duration" : f"{duration:.1f}s",
                "Report"   : str(outcome.report_path),
            })

        self.logger.metrics_table(rows, columns=["Run", "Status", "Result", "Duration", "Report"], title=f"{self.SECTION_TITLE}: summary")

        flagged  = [row["Run"] for row in rows if row["Status"] != "ok"]
        replaced = sum(outcome.replaced for outcome in outcomes)

        if flagged:
            self.logger.warning(f"{len(flagged)}/{len(rows)} run(s) flagged by the {self.SECTION_TITLE.lower()}: {', '.join(flagged)}")

        self.logger.ok(f"{len(rows)}/{len(run_dirs)} run(s) analysed in {sum(durations):.1f}s, {len(flagged)} flagged, {replaced} replaced an earlier result")

    def run(self) -> list[AnalysisOutcome]:
        """Runs the analysis on every selected run and closes with the batch summary.

        Returns:
            One outcome per run, in selection order.
        """
        self.logger.section(self.SECTION_TITLE)

        run_dirs  = self._select_runs()
        outcomes  = []
        durations = []

        for index, run_dir in enumerate(run_dirs, start=1):
            outcome, duration = self._analyse_run(run_dir, index, len(run_dirs))
            outcomes.append(outcome)
            durations.append(duration)

        self._summarize(run_dirs, outcomes, durations)

        return outcomes
