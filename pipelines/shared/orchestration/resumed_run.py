
from __future__ import annotations

from pathlib import Path

from tools.data.io            import FileIO
from tools.monitoring.logger  import Logger
from tools.runtime.config_cli import ConfigCli


class StagedRunLocator:

    PIPELINE_DIRNAME = "pipeline"
    CONFIG_FILENAME  = "resolved_config.json"

    @classmethod
    def config_path(cls, run_dir: Path) -> Path:
        return Path(run_dir) / cls.PIPELINE_DIRNAME / cls.CONFIG_FILENAME

    @classmethod
    def root(cls, resume_from: str) -> Path:
        target = Path(resume_from).expanduser()

        for candidate in (target, *target.parents):
            if cls.config_path(candidate).is_file():
                return candidate


class TuningRunLocator(StagedRunLocator):

    @classmethod
    def config_path(cls, run_dir: Path) -> Path:
        return Path(run_dir) / cls.CONFIG_FILENAME

    @classmethod
    def root(cls, resume_from: str) -> Path:
        target = Path(resume_from).expanduser()

        for candidate in (target, *target.parents):
            if cls.config_path(candidate).is_file():
                return candidate


class ResumedRun:

    LAUNCH_FIELDS   = ("gpus", "gpus_file", "poll_interval_s", "stage", "resume_from")
    LOCATION_FIELDS = ("run_tag", "paths.log_base_dir", "resume", "resume_from")
    LOCATOR         = StagedRunLocator

    def __init__(self, config_class, launch_config, cli_overrides: dict) -> None:
        self.config_class  = config_class
        self.launch_config = launch_config
        self.cli_overrides = dict(cli_overrides)
        self.root          = self.LOCATOR.root(launch_config.resume_from)
        self.config_path   = self.LOCATOR.config_path(self.root)
        self.recorded      = FileIO.load_json(self.config_path)
        self.baseline      = {}
        self.changed       = {}

    def _launch_overrides(self) -> dict:
        overrides = {field: getattr(self.launch_config, field) for field in self.LAUNCH_FIELDS}
        overrides.update({path: value for path, value in self.cli_overrides.items() if path.split(".")[0] != "resume_from"})
        overrides["resume_from"] = ""

        return overrides

    def load(self):
        config        = ConfigCli.load_resolved(self.config_class(), self.config_path)
        self.baseline = ConfigCli.to_mapping(config)

        overrides = self._launch_overrides()
        ConfigCli.apply_overrides(config, overrides)

        self._settle_location(config)

        current      = ConfigCli.to_mapping(config)
        self.changed = {path: (self.baseline[path], current[path]) for path in self.baseline if path not in self.LOCATION_FIELDS and current[path] != self.baseline[path]}

        return config

    def _settle_location(self, config) -> None:
        config.run_tag            = self.root.name
        config.paths.log_base_dir = self.root.parent
        config.resume             = True

    def log(self, logger: Logger) -> None:
        logger.section("Resumed run")
        logger.kv_table({
            "Run directory"       : str(self.root),
            "Recorded config"     : str(self.config_path),
            "Recorded leaves"     : len(self.recorded),
            "Launch-owned fields" : ", ".join(self.LAUNCH_FIELDS),
            "Typed overrides"     : ", ".join(sorted(path for path in self.cli_overrides if path.split(".")[0] != "resume_from")) or "none",
        }, title="Configuration source")

        if not self.changed:
            logger.ok("the run continues with exactly the configuration it recorded")
            return

        rows = [{"Field": path, "Recorded": repr(before), "Now": repr(after)} for path, (before, after) in sorted(self.changed.items())]
        logger.metrics_table(rows, columns=["Field", "Recorded", "Now"], title=f"{len(rows)} leaf(s) changed for this launch")


class TuningResumedRun(ResumedRun):

    LAUNCH_FIELDS   = ("gpus", "gpus_file", "stage", "resume_from")
    LOCATION_FIELDS = ("run_tag", "paths.log_base_dir", "resume_from")
    LOCATOR         = TuningRunLocator

    def _settle_location(self, config) -> None:
        config.run_tag            = self.root.name
        config.paths.log_base_dir = self.root.parent
