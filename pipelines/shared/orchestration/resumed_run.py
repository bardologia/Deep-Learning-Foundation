"""Resuming a staged run from any path inside it, with its recorded configuration.

A staged pipeline records the configuration it ran with under
``<run>/pipeline/resolved_config.json``. Resuming such a run means locating
that file from whatever path the operator pastes, loading it back as the
configuration of the new launch, and layering only what this launch decides on
top: the hardware it runs on, the stage it executes, and the overrides typed on
its command line.
"""

from __future__ import annotations

from pathlib import Path

from tools.data.io            import FileIO
from tools.monitoring.logger  import Logger
from tools.runtime.config_cli import ConfigCli


class StagedRunLocator:
    """Finds the root of a staged run from a path at or below it.

    Attributes:
        PIPELINE_DIRNAME : Directory inside a staged run holding its bookkeeping.
        CONFIG_FILENAME  : Name of the recorded configuration inside that directory.
    """

    PIPELINE_DIRNAME = "pipeline"
    CONFIG_FILENAME  = "resolved_config.json"

    @classmethod
    def config_path(cls, run_dir: Path) -> Path:
        """Returns the recorded configuration path of a run directory."""
        return Path(run_dir) / cls.PIPELINE_DIRNAME / cls.CONFIG_FILENAME

    @classmethod
    def root(cls, resume_from: str) -> Path:
        """Returns the staged run directory holding the pasted path.

        The path may be the run directory itself, its pipeline, folds or
        training directory, one unit run inside it, or any file in the tree;
        the walk climbs until it finds the recorded configuration.

        Args:
            resume_from : Path pasted by the operator.

        Raises:
            FileNotFoundError : If the path does not exist, or no directory at or above it holds the recorded configuration.
        """
        target = Path(resume_from).expanduser()

        if not target.exists():
            raise FileNotFoundError(f"resume_from points at '{target}', which does not exist; paste the run directory of the experiment to finish, or any path inside it")

        for candidate in (target, *target.parents):
            if cls.config_path(candidate).is_file():
                return candidate

        raise FileNotFoundError(f"resume_from '{target}' lies in no staged run: no directory at or above it holds {cls.PIPELINE_DIRNAME}/{cls.CONFIG_FILENAME}, which every cross-validation and benchmark launch writes first")


class TuningRunLocator(StagedRunLocator):
    """Finds the root of a tuning run, which records its configuration beside its study database."""

    @classmethod
    def config_path(cls, run_dir: Path) -> Path:
        """Returns the recorded configuration path of a tuning run directory."""
        return Path(run_dir) / cls.CONFIG_FILENAME

    @classmethod
    def root(cls, resume_from: str) -> Path:
        """Returns the tuning run directory holding the pasted path.

        Args:
            resume_from : Path pasted by the operator.

        Raises:
            FileNotFoundError : If the path does not exist, or no directory at or above it holds the recorded configuration.
        """
        target = Path(resume_from).expanduser()

        if not target.exists():
            raise FileNotFoundError(f"resume_from points at '{target}', which does not exist; paste the run directory of the tuning run to continue, or any path inside it")

        for candidate in (target, *target.parents):
            if cls.config_path(candidate).is_file():
                return candidate

        raise FileNotFoundError(f"resume_from '{target}' lies in no tuning run: no directory at or above it holds {cls.CONFIG_FILENAME}, which every tuning launch writes beside its study database")


class ResumedRun:
    """Rebuilds a staged run's configuration from its recording, with this launch's decisions on top.

    Attributes:
        LAUNCH_FIELDS   : Top-level fields that always come from the current launch rather than the recording: the hardware, the stage and the resume path itself.
        LOCATION_FIELDS : Leaves the resume itself sets from the run's location and its own nature, left out of the change report.
        LOCATOR         : Locator finding the run root and its recorded configuration.
        config_class    : Entry configuration class of the pipeline.
        launch_config   : Configuration of the current launch, already carrying the machine profile and the command-line overrides.
        cli_overrides   : Overrides typed on the current command line, keyed by dotted path.
        root            : Staged run directory being resumed.
        config_path     : Recorded configuration inside that run.
        recorded        : Flattened recorded configuration, keyed by dotted path.
        baseline        : Flattened recording as loaded into today's configuration class, the reference of the change report; filled by `load`.
        changed         : Recorded leaves this launch rewrote, mapped to their (recorded, new) values; filled by `load` and brought up to date by `refresh`.
    """

    LAUNCH_FIELDS   = ("gpus", "gpus_file", "poll_interval_s", "stage", "resume_from")
    LOCATION_FIELDS = ("run_tag", "paths.log_base_dir", "resume", "resume_from")
    LOCATOR         = StagedRunLocator

    def __init__(self, config_class, launch_config, cli_overrides: dict) -> None:
        """Locates the run and reads its recording.

        Args:
            config_class  : Entry configuration class of the pipeline.
            launch_config : Configuration of the current launch, whose ``resume_from`` names the run.
            cli_overrides : Overrides typed on the command line, keyed by dotted path.
        """
        self.config_class  = config_class
        self.launch_config = launch_config
        self.cli_overrides = dict(cli_overrides)
        self.root          = self.LOCATOR.root(launch_config.resume_from)
        self.config_path   = self.LOCATOR.config_path(self.root)
        self.recorded      = FileIO.load_json(self.config_path)
        self.baseline      = {}
        self.changed       = {}

    def _launch_overrides(self) -> dict:
        """Returns the leaves this launch decides: the launch fields and the typed overrides, minus the resume path."""
        overrides = {field: getattr(self.launch_config, field) for field in self.LAUNCH_FIELDS}
        overrides.update({path: value for path, value in self.cli_overrides.items() if path.split(".")[0] != "resume_from"})
        overrides["resume_from"] = ""

        return overrides

    def load(self):
        """Returns the recorded configuration with the launch's decisions applied.

        The run tag and the log base are taken from the run's location, so the
        pipeline writes back into the same directory, and resume is switched on.

        Raises:
            KeyError : If the recording does not match the configuration class of today's code; propagated from the strict loader.
        """
        config        = ConfigCli.load_resolved(self.config_class(), self.config_path)
        self.baseline = ConfigCli.to_mapping(config)

        overrides = self._launch_overrides()
        ConfigCli.apply_overrides(config, overrides)

        self._settle_location(config)
        self.refresh(config)

        return config

    def _settle_location(self, config) -> None:
        """Points the configuration at the run's directory and switches its unit resume on."""
        config.run_tag            = self.root.name
        config.paths.log_base_dir = self.root.parent
        config.resume             = True

    def refresh(self, config) -> None:
        """Recomputes the changed leaves against the configuration as it finally stands.

        An entry may derive further leaves after loading, such as the warmup
        curriculum fields inherited from a typed complete-stage override; the
        pipeline calls this with the configuration it runs, so those leaves
        reach the change report too.

        Args:
            config : Configuration the launch runs with.
        """
        current      = ConfigCli.to_mapping(config)
        self.changed = {path: (self.baseline[path], current[path]) for path in self.baseline if path not in self.LOCATION_FIELDS and current[path] != self.baseline[path]}

    def log(self, logger: Logger) -> None:
        """Logs where the configuration came from and every leaf this launch changed."""
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
    """Resumed tuning run: the recording sits beside the study database and the entry has no unit resume flag."""

    LAUNCH_FIELDS   = ("gpus", "gpus_file", "stage", "resume_from")
    LOCATION_FIELDS = ("run_tag", "paths.log_base_dir", "resume_from")
    LOCATOR         = TuningRunLocator

    def _settle_location(self, config) -> None:
        """Points the configuration at the tuning run's directory."""
        config.run_tag            = self.root.name
        config.paths.log_base_dir = self.root.parent
