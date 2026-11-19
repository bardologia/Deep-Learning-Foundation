
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

    DIRNAME  = "batch_train_logs"
    FILENAME = "resolved_config.json"

    @classmethod
    def path(cls, root: Path) -> Path:
        return Path(root) / cls.DIRNAME / cls.FILENAME

    @classmethod
    def write(cls, root: Path, config) -> Path:
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
        for candidate in (target, *target.parents):
            if cls.path(candidate).is_file():
                return cls.path(candidate)

        return None


class TrainingResume:

    LAUNCH_FIELDS = ("gpu", "gpus", "gpus_file", "poll_interval_s", "ddp", "resume", "resume_from")
    UNIT_FIELDS   = ("seed", "seeds", "run_name", "logdir", "infer_after", "infer_at_end")

    RUN         = "run"
    SEED_SWEEP  = "seed sweep"
    TRIAL_SWEEP = "trial sweep"

    def __init__(self, config_class, launch_config, cli_overrides: dict, trial_sweep: bool, unit: bool = False) -> None:
        self.config_class  = config_class
        self.launch_config = launch_config
        self.cli_overrides = dict(cli_overrides)
        self.target        = Path(launch_config.resume_from).expanduser()
        self.changed       = {}

        if not self.target.is_dir() and not self.target.parent.is_dir():
            raise FileNotFoundError(f"resume_from points at '{self.target}', which does not exist and has no parent directory to be created in; paste the run directory, the sweep folder or one of its runs")

        self.kind, self.root, self.recording = self._locate(trial_sweep, unit)

    def _seed_sweep_root(self) -> Path | None:
        if SeedSet.SEED_DIR.fullmatch(self.target.name):
            return self.target.parent

        if self.target.is_dir() and any(child.is_dir() and SeedSet.SEED_DIR.fullmatch(child.name) for child in self.target.iterdir()):
            return self.target

        return None

    def _first_seed_recording(self, root: Path) -> Path | None:
        for child in sorted(root.iterdir()):
            if child.is_dir() and SeedSet.SEED_DIR.fullmatch(child.name) and ResolvedEntryConfigWriter.path_for(child).is_file():
                return ResolvedEntryConfigWriter.path_for(child)

        return None

    def _run_recording(self) -> Path | None:
        own = ResolvedEntryConfigWriter.path_for(self.target)
        if own.is_file():
            return own

        return SweepRecording.climb(self.target.parent)

    def _locate(self, trial_sweep: bool, unit: bool) -> tuple:
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
        names  = {spec.name for spec in fields(self.config_class)}
        chosen = list(self.LAUNCH_FIELDS)

        if self.kind == self.SEED_SWEEP and self.recording is not None and self.recording.parent.name != SweepRecording.DIRNAME:
            chosen += list(self.UNIT_FIELDS)

        return [name for name in chosen if name in names]

    def configure(self):
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

    @staticmethod
    def configure(config_class, config, cli_overrides: dict, trial_sweep: bool, unit: bool = False):
        if not config.resume_from:
            return config

        resume  = TrainingResume(config_class, config, cli_overrides, trial_sweep=trial_sweep, unit=unit)
        resumed = resume.configure()

        log_dir = resume.root / SweepRecording.DIRNAME if resume.kind != TrainingResume.RUN else resume.root / "logs"
        logger  = Logger(log_dir=str(log_dir), name="resume_from", file_mode="a")
        resume.log(logger)
        logger.close()

        return resumed
