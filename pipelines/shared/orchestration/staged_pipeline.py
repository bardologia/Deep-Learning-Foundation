"""Base class for multi-stage pipelines with on-disk stage bookkeeping.

Owns the run directory layout (``<log_base_dir>/<run_tag>/pipeline``), persists
the resolved configuration next to it, keeps every earlier configuration the
run was launched with, and records per-stage status in ``state.json`` so a
rerun can see what already finished.
"""

from __future__ import annotations

from pathlib import Path

from tools.data.io            import FileIO
from tools.monitoring.logger  import Logger
from tools.runtime.config_cli import ConfigCli
from tools.runtime.run_tag    import RunTag


class StagedPipeline:
    """Owns the run directory, logger and stage state of a staged pipeline.

    Attributes:
        config       : Entry configuration for the pipeline.
        entry_script : Path of the entry script that launched the pipeline.
        resumed      : Description of the recorded run this launch continues, or None for a fresh run.
        run_tag      : Timestamp-style tag naming the run directory.
        run_dir      : Root directory of the run under the configured log base.
        pipeline_dir : Subdirectory holding pipeline logs and state.
        history_dir  : Subdirectory keeping the resolved configuration of every earlier launch of this run.
        state_path   : Path of the JSON file recording per-stage status.
        logger       : Logger writing into the pipeline directory.
        state        : Stage bookkeeping loaded from disk or freshly initialised.
    """

    LOGGER_NAME = "staged_pipeline"

    def __init__(self, config, entry_script: Path, resumed=None) -> None:
        """Prepares the run directory, saves the resolved config and loads state.

        The run tag the pipeline settles on is written back into the
        configuration, so the recording names the directory it lives in.

        Args:
            config       : Entry configuration carrying ``run_tag`` and path settings.
            entry_script : Path of the entry script that launched the pipeline.
            resumed      : Resumed-run description whose report is logged when the pipeline starts, or None.
        """
        self.config       = config
        self.entry_script = entry_script
        self.resumed      = resumed
        self.run_tag      = config.run_tag or RunTag.now()
        config.run_tag    = self.run_tag

        self.run_dir      = Path(config.paths.log_base_dir) / self.run_tag
        self.pipeline_dir = self.run_dir / "pipeline"
        self.history_dir  = self.pipeline_dir / "config_history"
        self.state_path   = self.pipeline_dir / "state.json"

        FileIO.ensure_dir(self.pipeline_dir)
        self._archive_previous_config()
        ConfigCli.save_resolved(config, self.pipeline_dir / "resolved_config.json")

        self.logger = Logger(log_dir=str(self.pipeline_dir), name=self.LOGGER_NAME)
        self.state  = self._recorded_state()

    def _archive_previous_config(self) -> None:
        """Keeps the configuration an earlier launch of this run recorded, when this launch changes it."""
        previous = self.pipeline_dir / "resolved_config.json"
        if not previous.is_file():
            return

        if FileIO.load_json(previous) == ConfigCli.to_mapping(self.config):
            return

        FileIO.ensure_dir(self.history_dir)
        previous.replace(self.history_dir / f"{RunTag.now()}.json")

    def _log_resumed(self) -> None:
        """Logs where a resumed run's configuration came from and what this launch changed."""
        if self.resumed is not None:
            self.resumed.log(self.logger)

    def _recorded_state(self) -> dict:
        """Returns the stage state read from disk, or an empty state if absent."""
        if not self.state_path.is_file():
            return {"run_tag": self.run_tag, "stages": {}}

        return {"run_tag": self.run_tag, "stages": FileIO.load_json(self.state_path)["stages"]}

    def _mark_stage(self, stage_name: str, status: str) -> None:
        """Records a stage's status with a timestamp and flushes state to disk."""
        self.state["stages"][stage_name] = {
            "status"    : status,
            "timestamp" : RunTag.timestamp(),
        }

        FileIO.save_json(self.state, self.state_path, indent=2)
