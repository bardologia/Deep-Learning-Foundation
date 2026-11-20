
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

    payload     : dict
    headline    : str
    report_path : Path
    status      : str  = "ok"
    replaced    : bool = False


class AnalysisConfigTable:

    PROFILE_TAG = "(machine profile)"

    @classmethod
    def log(cls, logger, config, profile_overrides: dict) -> None:
        rows = {}
        for path, value in ConfigCli.to_mapping(config).items():
            rows[path] = f"{value} {cls.PROFILE_TAG}" if path in profile_overrides else value

        logger.kv_table(rows, title="Configuration")


class AnalysisRun:

    SUMMARY_FILENAME    : str
    CONFIG_FILENAME     = "analysis_config.json"
    PROVENANCE_FILENAME = "analysis_provenance.json"
    PLOTS_SUBDIR        = "plots"

    def __init__(self, run_dir: Path, config, logger) -> None:
        self.run_dir = Path(run_dir)
        self.config  = config
        self.logger  = logger

        self.output_dir = self.run_dir / config.output_subdir

    @property
    def stage(self) -> str:
        return Path(self.SUMMARY_FILENAME).stem

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

    def _prepare_output(self) -> bool:
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

    def _stamp_complete(self, run, outcome: AnalysisOutcome) -> None:
        CompletionMarker.stamp(self.output_dir, {
            "stage"      : self.stage,
            "split"      : self.config.split,
            "checkpoint" : self.config.checkpoint_name,
            "epoch"      : run.checkpoint_meta["epoch"] + 1,
            "status"     : outcome.status,
        })

    def _report(self, outcome: AnalysisOutcome) -> None:
        levels = {"ok": self.logger.ok, "warning": self.logger.warning, "critical": self.logger.error}
        levels[outcome.status](f"{self.run_dir.name}: {outcome.headline} -> {outcome.report_path}")

    def run(self) -> AnalysisOutcome:
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

    SELECTOR_ACTION : str
    SECTION_TITLE   : str
    RUN_CLASS       : type[AnalysisRun]

    def __init__(self, config, logger, profile_overrides: dict) -> None:
        self.config            = config
        self.logger            = logger
        self.profile_overrides = profile_overrides

    def _select_runs(self) -> list[Path]:
        selector = RunSelector(self.config.runs_dir, self.config.checkpoint_name, self.logger, action=self.SELECTOR_ACTION)
        return selector.resolve(self.config.run_filter)

    def _analyse_run(self, run_dir: Path, index: int, total: int) -> tuple[AnalysisOutcome, float]:
        self.logger.subsection(f"Run {index}/{total}: {run_dir}")

        start = time.perf_counter()
        with self.logger.timer(f"{self.SECTION_TITLE} [{index}/{total}] {run_dir.name}"):
            outcome = self.RUN_CLASS(run_dir, self.config, self.logger).run()

        return outcome, time.perf_counter() - start

    def _summarize(self, run_dirs: list[Path], outcomes: list[AnalysisOutcome], durations: list[float]) -> None:
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
        self.logger.section(self.SECTION_TITLE)
        AnalysisConfigTable.log(self.logger, self.config, self.profile_overrides)

        run_dirs  = self._select_runs()
        outcomes  = []
        durations = []

        for index, run_dir in enumerate(run_dirs, start=1):
            outcome, duration = self._analyse_run(run_dir, index, len(run_dirs))
            outcomes.append(outcome)
            durations.append(duration)

        self._summarize(run_dirs, outcomes, durations)

        return outcomes
