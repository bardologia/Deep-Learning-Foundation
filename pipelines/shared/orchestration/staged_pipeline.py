from __future__ import annotations

from pathlib import Path

from tools.data.io            import FileIO
from tools.monitoring.logger  import Logger
from tools.runtime.config_cli import ConfigCli
from tools.runtime.run_tag    import RunTag


class StagedPipeline:
    LOGGER_NAME = "staged_pipeline"

    def __init__(self, config, entry_script: Path) -> None:
        self.config       = config
        self.entry_script = entry_script
        self.run_tag      = config.run_tag or RunTag.now()

        self.run_dir      = Path(config.paths.log_base_dir) / self.run_tag
        self.pipeline_dir = self.run_dir / "pipeline"
        self.state_path   = self.pipeline_dir / "state.json"

        FileIO.ensure_dir(self.pipeline_dir)
        ConfigCli.save_resolved(config, self.pipeline_dir / "resolved_config.json")

        self.logger = Logger(log_dir=str(self.pipeline_dir), name=self.LOGGER_NAME)
        self.state  = {"run_tag": self.run_tag, "stages": {}}

    def _mark_stage(self, stage_name: str, status: str) -> None:
        self.state["stages"][stage_name] = {
            "status"    : status,
            "timestamp" : RunTag.timestamp(),
        }

        FileIO.save_json(self.state, self.state_path, indent=2)
