"""Resuming a training run or sweep from a pasted path with the configuration it recorded.

Three kinds of directory can be resumed: one run, which recorded its entry
configuration under ``docs/``; a seed sweep, whose root holds one ``seed<N>``
run per seed and, since sweeps record themselves, ``batch_train_logs/
resolved_config.json``; and a trial sweep, whose root holds one run per
planned unit and the same recording. The resume locates the recording that
speaks for the pasted path, loads it as the configuration of the new launch,
and layers only what this launch decides on top: the hardware, the resume
flags, and the overrides typed on the command line. A unit launched by a sweep
scheduler is handed its own directory as the path, and reads the sweep's
recording when it has none of its own yet.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib     import Path

from pipelines.shared.config.resolved_entry_config import ResolvedEntryConfigWriter
from pipelines.shared.training.seed_sweep          import SeedSet
from tools.data.io                                 import FileIO
from tools.monitoring.logger                       import Logger
from tools.runtime.config_cli                      import ConfigCli
from tools.runtime.run_tag                         import RunTag


class SweepRecording:
    """Where a sweep scheduler records the configuration it launched with.

    Attributes:
        DIRNAME  : Log directory of a sweep inside its root.
        FILENAME : Name of the recording inside that directory.
    """

    DIRNAME  = "batch_train_logs"
    FILENAME = "resolved_config.json"

    @classmethod
    def path(cls, root: Path) -> Path:
        """Returns the recording path of a sweep root."""
        return Path(root) / cls.DIRNAME / cls.FILENAME

    @classmethod
    def write(cls, root: Path, config) -> Path:
        """Records a sweep's effective configuration, keeping the earlier one when it differs.

        The resume path is cleared in the recording, so the file describes the
        sweep rather than the launch that resumed it.

        Args:
            root   : Sweep root directory.
            config : Effective configuration of the sweep.

        Returns:
            Path of the written recording.
        """
        mapping                = ConfigCli.to_mapping(config)
        mapping["resume_from"] = ""

        path = cls.path(root)
        if path.is_file() and FileIO.load_json(path) != mapping:
            history = path.parent / "config_history"
            FileIO.ensure_dir(history)
            path.replace(history / f"{RunTag.now()}.json")

        return FileIO.save_json(mapping, path, indent=2)

    @classmethod
    def climb(cls, target: Path) -> Path | None:
        """Returns the recording of the sweep holding a path, or None when no directory at or above it has one."""
        for candidate in (target, *target.parents):
            if cls.path(candidate).is_file():
                return cls.path(candidate)

        return None


class TrainingResume:
    """Locates and loads the recording a training launch resumes from.

    Attributes:
        LAUNCH_FIELDS : Top-level fields that always come from the current launch: the hardware and the resume flags.
        UNIT_FIELDS   : Top-level fields a sweep scheduler rewrites per unit, so a single seed run's recording cannot speak for the whole sweep on them.
        RUN           : Kind of a pasted run directory.
        SEED_SWEEP    : Kind of a pasted seed sweep root or one of its seed runs.
        TRIAL_SWEEP   : Kind of a pasted trial sweep root or one of its units.
        config_class  : Entry configuration class of the training entry.
        launch_config : Configuration of the current launch, carrying the machine profile and the command-line overrides.
        cli_overrides : Overrides typed on the current command line, keyed by dotted path.
        target        : Pasted path.
        kind          : Which of the three directory kinds the path is.
        root          : Directory the launch resumes: the run, or the sweep root.
        recording     : Recording the configuration is loaded from, or None when the run or sweep predates recordings.
        changed       : Recorded leaves this launch rewrote, mapped to their (recorded, new) values; filled by `configure`.
    """

    LAUNCH_FIELDS = ("gpu", "gpus", "gpus_file", "poll_interval_s", "ddp", "resume", "resume_from")
    UNIT_FIELDS   = ("seed", "seeds", "run_name", "logdir", "infer_after", "infer_at_end")

    RUN         = "run"
    SEED_SWEEP  = "seed sweep"
    TRIAL_SWEEP = "trial sweep"

    def __init__(self, config_class, launch_config, cli_overrides: dict, trial_sweep: bool, unit: bool = False) -> None:
        """Classifies the pasted path and locates its recording.

        Args:
            config_class  : Entry configuration class of the training entry.
            launch_config : Configuration of the current launch, whose ``resume_from`` names the path.
            cli_overrides : Overrides typed on the command line, keyed by dotted path.
            trial_sweep   : Whether the launch schedules a trial sweep, which makes the path a trial sweep root or one of its units.
            unit          : Whether the launch is one unit a scheduler handed its exact directory, which is then a run whatever it is called.

        Raises:
            FileNotFoundError : If the path, or for a fresh unit its parent directory, does not exist.
        """
        self.config_class  = config_class
        self.launch_config = launch_config
        self.cli_overrides = dict(cli_overrides)
        self.target        = Path(launch_config.resume_from).expanduser()
        self.changed       = {}

        if not self.target.is_dir() and not self.target.parent.is_dir():
            raise FileNotFoundError(f"resume_from points at '{self.target}', which does not exist and has no parent directory to be created in; paste the run directory, the sweep folder or one of its runs")

        self.kind, self.root, self.recording = self._locate(trial_sweep, unit)

    def _seed_sweep_root(self) -> Path | None:
        """Returns the seed sweep root the path belongs to, or None when it is no seed sweep."""
        if SeedSet.SEED_DIR.fullmatch(self.target.name):
            return self.target.parent

        if self.target.is_dir() and any(child.is_dir() and SeedSet.SEED_DIR.fullmatch(child.name) for child in self.target.iterdir()):
            return self.target

        return None

    def _first_seed_recording(self, root: Path) -> Path | None:
        """Returns the recording of the first seed run under a sweep root, or None when none recorded one."""
        for child in sorted(root.iterdir()):
            if child.is_dir() and SeedSet.SEED_DIR.fullmatch(child.name) and ResolvedEntryConfigWriter.path_for(child).is_file():
                return ResolvedEntryConfigWriter.path_for(child)

        return None

    def _run_recording(self) -> Path | None:
        """Returns a run's own recording, or the recording of the sweep it sits in when it has none yet."""
        own = ResolvedEntryConfigWriter.path_for(self.target)
        if own.is_file():
            return own

        return SweepRecording.climb(self.target.parent)

    def _locate(self, trial_sweep: bool, unit: bool) -> tuple:
        """Returns the kind of the path, the directory resumed and the recording read.

        A trial sweep reads the sweep recording found at or above the path. A
        seed sweep reads its own recording, or the first seed run's when the
        sweep predates recordings. A path that is itself a sweep root, which
        its own recording proves, is a trial sweep even when the launch did not
        say so, so the sweep settings do not have to be retyped to resume it. A
        run, and any unit a scheduler handed its directory, reads its own
        recording, or the recording of the sweep it sits in when it has none
        yet.

        Args:
            trial_sweep : Whether the launch schedules a trial sweep.
            unit        : Whether the launch is one unit handed its exact directory.
        """
        if unit:
            return self.RUN, self.target, self._run_recording()

        if trial_sweep:
            recording = SweepRecording.climb(self.target)
            root      = recording.parent.parent if recording is not None else self.target
            return self.TRIAL_SWEEP, root, recording

        seed_root = self._seed_sweep_root()
        if seed_root is not None:
            recording = SweepRecording.path(seed_root) if SweepRecording.path(seed_root).is_file() else self._first_seed_recording(seed_root)
            return self.SEED_SWEEP, seed_root, recording

        if SweepRecording.path(self.target).is_file():
            return self.TRIAL_SWEEP, self.target, SweepRecording.path(self.target)

        return self.RUN, self.target, self._run_recording()

    def _launch_fields(self) -> list[str]:
        """Returns the fields taken from the launch rather than the recording, limited to the ones the entry has.

        A seed sweep resumed through one seed run's recording also takes the
        per-unit fields from the launch, since that recording names one seed
        only.
        """
        names  = {spec.name for spec in fields(self.config_class)}
        chosen = list(self.LAUNCH_FIELDS)

        if self.kind == self.SEED_SWEEP and self.recording is not None and self.recording.parent.name != SweepRecording.DIRNAME:
            chosen += list(self.UNIT_FIELDS)

        return [name for name in chosen if name in names]

    def configure(self):
        """Returns the configuration of the launch: the recording with this launch's decisions on top.

        Without a recording the launch configuration is returned as it is, so
        a run or sweep launched before recordings existed is still resumable
        from the settings typed on the command line.

        Raises:
            KeyError : If the recording does not match the configuration class of today's code; propagated from the strict loader.
        """
        if self.recording is None:
            return self.launch_config

        config   = ConfigCli.load_resolved(self.config_class(), self.recording)
        recorded = ConfigCli.to_mapping(config)

        overrides = {field: getattr(self.launch_config, field) for field in self._launch_fields()}
        overrides.update({path: value for path, value in self.cli_overrides.items() if path.split(".")[0] != "resume_from"})
        overrides["resume_from"] = self.launch_config.resume_from

        ConfigCli.apply_overrides(config, overrides)

        current      = ConfigCli.to_mapping(config)
        excluded     = set(self._launch_fields()) | set(self.UNIT_FIELDS)
        self.changed = {path: (recorded[path], current[path]) for path in recorded if path.split(".")[0] not in excluded and current[path] != recorded[path]}

        return config

    def log(self, logger: Logger) -> None:
        """Logs the kind of directory resumed, the recording used and every leaf this launch changed."""
        logger.section("Resume from")
        logger.kv_table({
            "Path"            : str(self.target),
            "Kind"            : self.kind,
            "Directory"       : str(self.root),
            "Recording"       : str(self.recording) if self.recording is not None else "none: no recorded configuration found, this launch's settings are used",
            "Typed overrides" : ", ".join(sorted(path for path in self.cli_overrides if path.split(".")[0] != "resume_from")) or "none",
        }, title="Configuration source")

        if self.recording is None:
            logger.warning("the run or sweep recorded no configuration; pass every setting it was launched with, or the defaults apply")
            return

        if not self.changed:
            logger.ok("the launch continues with exactly the configuration recorded")
            return

        rows = [{"Field": path, "Recorded": repr(before), "Now": repr(after)} for path, (before, after) in sorted(self.changed.items())]
        logger.metrics_table(rows, columns=["Field", "Recorded", "Now"], title=f"{len(rows)} leaf(s) changed for this launch")


class ResumeLauncher:
    """Applies the resume to a launch configuration and reports it into the resumed directory's logs."""

    @staticmethod
    def configure(config_class, config, cli_overrides: dict, trial_sweep: bool, unit: bool = False):
        """Returns the configuration to launch with, loaded from the recording when ``resume_from`` is set.

        Args:
            config_class  : Entry configuration class of the training entry.
            config        : Configuration parsed from the command line.
            cli_overrides : Overrides typed on the command line, keyed by dotted path.
            trial_sweep   : Whether the launch schedules a trial sweep.
            unit          : Whether the launch is one unit a scheduler handed its exact directory.

        Returns:
            The launch configuration unchanged without a resume path, else the
            resumed configuration.
        """
        if not config.resume_from:
            return config

        resume  = TrainingResume(config_class, config, cli_overrides, trial_sweep=trial_sweep, unit=unit)
        resumed = resume.configure()

        log_dir = resume.root / SweepRecording.DIRNAME if resume.kind != TrainingResume.RUN else resume.root / "logs"
        logger  = Logger(log_dir=str(log_dir), name="resume_from", file_mode="a")
        resume.log(logger)
        logger.close()

        return resumed
