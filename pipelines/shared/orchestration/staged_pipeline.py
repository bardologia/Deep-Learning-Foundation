"""Base class for multi-stage pipelines with on-disk stage bookkeeping.

Owns the run directory layout (``<log_base_dir>/<run_tag>/pipeline``), persists
the resolved configuration next to it, keeps every earlier configuration the
run was launched with, and records per-stage status in ``state.json`` so a
rerun can see what already finished. Its ``run`` wraps the stage sequence a
subclass implements, so an exception that ends the run is written to the
pipeline log with the stage it happened in and marked in the state, and a
run whose units partly failed ends with a unit-level summary and a non-zero
exit once its reports are written.
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.markup import escape

from pipelines.shared.orchestration.resumed_run import ResumedRun
from tools.data.io                              import FileIO
from tools.monitoring.logger                    import Logger
from tools.runtime.config_cli                   import ConfigCli
from tools.runtime.git_state                    import RunProvenance
from tools.runtime.run_tag                      import RunTag


class PartialRun(SystemExit):
    """Exit raised once a run wrote its reports with some of its units failed, so the shell and the console record a failure."""


class StagedPipeline:
    """Owns the run directory, logger and stage state of a staged pipeline.

    Attributes:
        config          : Entry configuration for the pipeline.
        entry_script    : Path of the entry script that launched the pipeline.
        resumed         : Description of the recorded run this launch continues, or None for a fresh run.
        run_tag         : Timestamp-style tag naming the run directory.
        run_dir         : Root directory of the run under the configured log base.
        pipeline_dir    : Subdirectory holding pipeline logs and state.
        history_dir     : Subdirectory keeping the resolved configuration of every earlier launch of this run.
        state_path      : Path of the JSON file recording per-stage status.
        launches_path   : JSON-lines file gaining one record per launch: start time, stage, resume source, archived recording and provenance.
        provenance      : Commit, host, interpreter and library versions of this launch, from RunProvenance.
        current_stage   : Name of the stage in progress, recorded as failed when an exception ends the run inside it.
        archived_config : Path the recording of the earlier launch was moved to, or None when this launch kept it.
        config_changes  : Recorded leaves this launch rewrote, mapped to their (recorded, current) values; empty for a first launch or an unchanged relaunch.
        logger          : Logger writing into the pipeline directory.
        state           : Stage bookkeeping loaded from disk or freshly initialised.
    """

    LOGGER_NAME = "staged_pipeline"
    ABSENT      = "<absent>"

    def __init__(self, config, entry_script: Path, resumed=None) -> None:
        """Prepares the run directory, saves the resolved config and loads state.

        The run tag the pipeline settles on is written back into the
        configuration, so the recording names the directory it lives in.

        Args:
            config       : Entry configuration carrying ``run_tag`` and path settings.
            entry_script : Path of the entry script that launched the pipeline.
            resumed      : Resumed-run description whose report is logged when the pipeline starts, brought up to date with the configuration as the entry finally set it, or None.
        """
        self.config        = config
        self.entry_script  = entry_script
        self.resumed       = resumed
        self.run_tag       = config.run_tag or RunTag.now()
        config.run_tag     = self.run_tag
        self.current_stage = "setup"

        self.run_dir       = Path(config.paths.log_base_dir) / self.run_tag
        self.pipeline_dir  = self.run_dir / "pipeline"
        self.history_dir   = self.pipeline_dir / "config_history"
        self.state_path    = self.pipeline_dir / "state.json"
        self.launches_path = self.pipeline_dir / "launches.jsonl"
        self.provenance    = RunProvenance.collect()

        FileIO.ensure_dir(self.pipeline_dir)
        self.archived_config = None
        self.config_changes  = self._archive_previous_config()
        ConfigCli.save_resolved(config, self.pipeline_dir / "resolved_config.json")

        if resumed is not None:
            resumed.refresh(config)

        self._record_launch()

        self.logger = Logger(log_dir=str(self.pipeline_dir), name=self.LOGGER_NAME)
        self.state  = self._recorded_state()

    def _archive_previous_config(self) -> dict:
        """Keeps the configuration an earlier launch of this run recorded, when this launch changes it.

        Returns:
            The leaves that differ between the recording and this launch, mapped to their (recorded, current) values; a leaf one side lacks reads as absent. Empty when there is no recording or it matches.
        """
        previous = self.pipeline_dir / "resolved_config.json"
        if not previous.is_file():
            return {}

        recorded = FileIO.load_json(previous)
        current  = ConfigCli.to_mapping(self.config)
        if recorded == current:
            return {}

        FileIO.ensure_dir(self.history_dir)
        self.archived_config = self.history_dir / f"{RunTag.now()}.json"
        previous.replace(self.archived_config)

        changes = {}
        for path in sorted(set(recorded) | set(current)):
            before = recorded[path] if path in recorded else self.ABSENT
            after  = current[path]  if path in current  else self.ABSENT
            if before != after:
                changes[path] = (before, after)

        return changes

    def _record_launch(self) -> None:
        """Appends this launch to the run's launch record, so every stage record and archived configuration can be traced to the code and host that produced it."""
        record = {
            "started"         : RunTag.timestamp(),
            "stage"           : self.config.stage,
            "resumed_from"    : str(self.resumed.root) if self.resumed is not None else None,
            "archived_config" : str(self.archived_config) if self.archived_config is not None else None,
            "provenance"      : self.provenance,
        }

        with open(self.launches_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def _recorded_state(self) -> dict:
        """Returns the stage state read from disk, or an empty state if absent."""
        if not self.state_path.is_file():
            return {"run_tag": self.run_tag, "stages": {}}

        return {"run_tag": self.run_tag, "stages": FileIO.load_json(self.state_path)["stages"]}

    def _log_config_changes(self) -> None:
        """Warns when a relaunch by run tag changed the recorded configuration beyond the launch-owned fields.

        The hardware, the stage and the resume path belong to each launch, so
        a compare job following a prepare job is not a change. Anything else
        means the completed units keep their recorded settings while the
        remaining ones run under the new ones.
        """
        changes = {path: values for path, values in self.config_changes.items() if path.split(".")[0] not in ResumedRun.LAUNCH_FIELDS}
        if not changes:
            return

        rows = [{"Field": path, "Recorded": repr(before), "Now": repr(after)} for path, (before, after) in changes.items()]

        self.logger.warning(f"{len(rows)} leaf(s) differ from the configuration this run recorded; the recording was archived to {self.archived_config}. Units already completed keep their recorded settings while the remaining ones run under the new ones; launch with resume_from to continue a run with exactly its recorded configuration")
        self.logger.metrics_table(rows, columns=["Field", "Recorded", "Now"], title=f"{len(rows)} leaf(s) changed since the recording")

    def _log_launch(self) -> None:
        """Logs the code and host this launch runs from, then where its configuration came from: the resumed run's report, or the changes a relaunch by run tag made to the recording."""
        self.logger.kv_table({
            "Commit"        : self.provenance["git_commit"],
            "Host"          : self.provenance["host"],
            "Conda env"     : self.provenance["conda_env"],
            "Launch record" : str(self.launches_path),
        }, title="Launch provenance")

        if self.resumed is not None:
            self.resumed.log(self.logger)
            return

        self._log_config_changes()

    def _begin_stage(self, stage_name: str) -> None:
        """Names the stage that is starting, so an exception escaping it is attributed to it."""
        self.current_stage = stage_name

    def _mark_stage(self, stage_name: str, status: str) -> None:
        """Records a stage's status with a timestamp and flushes state to disk."""
        self.state["stages"][stage_name] = {
            "status"    : status,
            "timestamp" : RunTag.timestamp(),
        }

        FileIO.save_json(self.state, self.state_path, indent=2)

    def _finish(self, training_results: list[dict], inference_results: list[dict], outputs: dict) -> None:
        """Marks the pipeline completed or partial from the unit results and logs the end summary.

        Args:
            training_results  : One training stage result per planned unit, each carrying ``name`` and ``status``.
            inference_results : One inference stage result per unit, empty when the inference stage was skipped, which the summary says instead of counting zero inferences.
            outputs           : Report locations appended to the summary table.

        Raises:
            PartialRun : If any unit failed training or inference, after every failed unit was logged; the reports are already written at this point.
        """
        failed_training  = [result["name"] for result in training_results if result["status"] != "DONE"]
        failed_inference = [result["name"] for result in inference_results if result["status"] == "FAILED"]
        failed           = len(failed_training) + len(failed_inference)
        planned          = len(training_results)
        title            = "Done" if not failed else f"Partial: {failed} of {planned} units failed"

        self._mark_stage("pipeline", "completed" if not failed else "partial")

        inference = {"Inference": "stage skipped"}
        if inference_results:
            inference = {
                "Inferred"          : sum(1 for result in inference_results if result["status"] == "DONE"),
                "Inference failed"  : len(failed_inference),
                "Inference skipped" : sum(1 for result in inference_results if result["status"] == "SKIPPED"),
            }

        self.logger.section("Pipeline summary")
        self.logger.kv_table({
            "Units planned"   : planned,
            "Trained"         : sum(1 for result in training_results if result["status"] == "DONE"),
            "Training failed" : len(failed_training),
            **inference,
            **outputs,
            "Pipeline logs"   : str(self.pipeline_dir),
        }, title=title)

        if not failed:
            return

        self.logger.error(title)
        for name in failed_training:
            self.logger.error(f"training failed: {name}")
        for name in failed_inference:
            self.logger.error(f"inference failed: {name}")

        raise PartialRun(f"{title} ({len(failed_training)} training, {len(failed_inference)} inference); the reports cover the units that finished, see {self.pipeline_dir}")

    def _record_abort(self, error: BaseException) -> None:
        """Logs the stage an escaping exception ended the run in and marks it and the pipeline failed.

        A SystemExit with a zero code is a deliberate clean exit and leaves the
        state as the run recorded it.

        Args:
            error : Exception escaping the stage sequence.
        """
        if not Logger.aborted(error):
            return

        self.logger.error(f"Aborted in stage '{self.current_stage}' by {type(error).__name__}: {escape(str(error))}")

        for stage_name in dict.fromkeys((self.current_stage, "pipeline")):
            self._mark_stage(stage_name, "failed")

    def _execute(self) -> None:
        """Runs the stage sequence of the pipeline; implemented by subclasses."""
        raise NotImplementedError

    def run(self) -> None:
        """Runs the stage sequence inside the pipeline logger, recording any exception that ends it.

        The logger closes with an ABORTED banner and the traceback when an
        exception escapes, and with the END banner otherwise; a partial run
        counts as ended, its exit is raised once the log is closed.

        Raises:
            PartialRun    : If the stage sequence finished with failed units.
            BaseException : Whatever else the stage sequence raised, after it was logged and marked in the state.
        """
        partial = None

        with self.logger:
            try:
                self._execute()
            except PartialRun as error:
                partial = error
            except BaseException as error:
                self._record_abort(error)
                raise

        if partial is not None:
            raise partial
