"""Base class for multi-stage pipelines with on-disk stage bookkeeping.

Owns the run directory layout (``<log_base_dir>/<run_tag>/pipeline``), persists
the resolved configuration next to it, keeps every earlier configuration the
run was launched with, and records per-stage status in ``state.json`` so a
rerun can see what already finished. Every launch appends its provenance to
``launches.jsonl`` and names the recorded settings a relaunch changed.
"""

from __future__ import annotations

import json
from pathlib import Path

from pipelines.shared.orchestration.resumed_run import ResumedRun
from tools.data.io                              import FileIO
from tools.monitoring.logger                    import Logger
from tools.runtime.config_cli                   import ConfigCli
from tools.runtime.git_state                    import RunProvenance
from tools.runtime.run_tag                      import RunTag


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

    def _mark_stage(self, stage_name: str, status: str) -> None:
        """Records a stage's status with a timestamp and flushes state to disk."""
        self.state["stages"][stage_name] = {
            "status"    : status,
            "timestamp" : RunTag.timestamp(),
        }

        FileIO.save_json(self.state, self.state_path, indent=2)
