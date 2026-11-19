
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
    pass


class StagedPipeline:

    LOGGER_NAME = "staged_pipeline"
    ABSENT      = "<absent>"

    def __init__(self, config, entry_script: Path, resumed=None) -> None:
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
        if not self.state_path.is_file():
            return {"run_tag": self.run_tag, "stages": {}}

        return {"run_tag": self.run_tag, "stages": FileIO.load_json(self.state_path)["stages"]}

    def _log_config_changes(self) -> None:
        changes = {path: values for path, values in self.config_changes.items() if path.split(".")[0] not in ResumedRun.LAUNCH_FIELDS}
        if not changes:
            return

        rows = [{"Field": path, "Recorded": repr(before), "Now": repr(after)} for path, (before, after) in changes.items()]

        self.logger.warning(f"{len(rows)} leaf(s) differ from the configuration this run recorded; the recording was archived to {self.archived_config}. Units already completed keep their recorded settings while the remaining ones run under the new ones; launch with resume_from to continue a run with exactly its recorded configuration")
        self.logger.metrics_table(rows, columns=["Field", "Recorded", "Now"], title=f"{len(rows)} leaf(s) changed since the recording")

    def _log_launch(self) -> None:
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
        self.current_stage = stage_name

    def _mark_stage(self, stage_name: str, status: str) -> None:
        self.state["stages"][stage_name] = {
            "status"    : status,
            "timestamp" : RunTag.timestamp(),
        }

        FileIO.save_json(self.state, self.state_path, indent=2)

    def _finish(self, training_results: list[dict], inference_results: list[dict], outputs: dict) -> None:
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
        if not Logger.aborted(error):
            return

        self.logger.error(f"Aborted in stage '{self.current_stage}' by {type(error).__name__}: {escape(str(error))}")

        for stage_name in dict.fromkeys((self.current_stage, "pipeline")):
            self._mark_stage(stage_name, "failed")

    def _execute(self) -> None:
        raise NotImplementedError

    def run(self) -> None:
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
