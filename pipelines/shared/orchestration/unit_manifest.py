
from __future__ import annotations

from pathlib import Path
from typing  import Callable

from pipelines.shared.inference.run_classifier import RunDirectoryWalk
from tools.data.io                             import FileIO
from tools.monitoring.logger                   import Logger
from tools.runtime.completion                  import CompletionMarker
from tools.runtime.run_tag                     import RunTag


class UnitManifest:

    META_DIRNAME = RunDirectoryWalk.MARKER
    FILENAME     = "unit.json"

    @classmethod
    def path(cls, run_directory: Path) -> Path:
        return Path(run_directory) / cls.META_DIRNAME / cls.FILENAME

    @classmethod
    def exists(cls, run_directory: Path) -> bool:
        return cls.path(run_directory).is_file()

    @classmethod
    def write(cls, run_directory: Path, kind: str, payload: dict) -> Path:
        return FileIO.save_json({"kind": kind, "written_at": RunTag.timestamp(), **payload}, cls.path(run_directory))

    @classmethod
    def read(cls, run_directory: Path) -> dict:
        path = cls.path(run_directory)
        if not path.is_file():
            raise FileNotFoundError(f"No unit manifest at {path}; the run was not written by a staged-experiment worker, so its fold, model or seed identity cannot be read back")

        return FileIO.load_json(path)


class UnitLedger:

    def __init__(self, stage_dir: Path, kind: str, identity: Callable[[dict], tuple]) -> None:
        self.stage_dir    = Path(stage_dir)
        self.kind         = kind
        self.identity     = identity
        self.by_identity  = {}
        self.unidentified = []

    def _index(self, run_dir: Path) -> None:
        if not UnitManifest.exists(run_dir):
            self.unidentified.append(run_dir)
            return

        manifest = UnitManifest.read(run_dir)
        if manifest["kind"] != self.kind:
            self.unidentified.append(run_dir)
            return

        key = self.identity(manifest)
        if key in self.by_identity:
            raise ValueError(f"Two run directories under {self.stage_dir} record the same unit identity {key}: {self.by_identity[key]} and {run_dir}; remove or move one of them before resuming")

        self.by_identity[key] = run_dir

    def directory(self, identity: tuple) -> Path | None:
        return self.by_identity.get(identity)

    def build(self) -> "UnitLedger":
        if self.stage_dir.is_dir():
            for run_dir in RunDirectoryWalk.walk(self.stage_dir):
                self._index(run_dir)

        return self


class UnidentifiedRuns:

    def __init__(self, stage_dir: Path, directories: list[Path], kind: str, planned: dict[str, Path]) -> None:
        self.stage_dir   = Path(stage_dir)
        self.directories = list(directories)
        self.kind        = kind
        self.planned     = {name: Path(path) for name, path in planned.items()}

    def _case(self, directory: Path) -> str:
        claimants = [name for name, path in self.planned.items() if path == directory]

        if not claimants:
            return "claimed by no planned unit, ignored by the stage; the comparison still collects it beside the planned units"
        if CompletionMarker.is_complete(directory):
            return f"sits at planned unit '{claimants[0]}'s own path and holds a completion marker, so the stage uses it as that unit's completed run"

        return f"sits at planned unit '{claimants[0]}'s own path without a completion marker, so the stage treats it as that unit's unfinished run"

    def log(self, logger: Logger, stage_label: str) -> None:
        if not self.directories:
            return

        logger.warning(f"{stage_label}: {len(self.directories)} run directory(ies) under {self.stage_dir} carry no {self.kind} manifest")

        for directory in self.directories:
            logger.warning(f"{directory.relative_to(self.stage_dir).as_posix()}: {self._case(directory)}")


class UnitResultsByRun:

    def __init__(self, stage_dir: Path, logger: Logger) -> None:
        self.stage_dir = Path(stage_dir)
        self.logger    = logger

    def _run_of(self, result: dict, names: list[str]) -> str | None:
        parts = Path(result["log_file"]).parent.parts

        for name in names:
            tail = (self.stage_dir.name, *Path(name).parts)
            if parts[-len(tail):] == tail:
                return name

        return None

    def rekey(self, results: dict, runs: list[tuple[str, Path]]) -> dict:
        names    = [name for name, _ in runs]
        keyed    = {}
        renamed  = []
        orphaned = []

        for unit, result in results.items():
            name = self._run_of(result, names)
            if name is None:
                orphaned.append(unit)
                continue

            keyed[name] = result
            if name != unit:
                renamed.append(f"{unit} -> {name}")

        unmatched = [name for name in names if name not in keyed]

        if renamed:
            self.logger.info(f"{len(renamed)} unit result(s) matched to the renamed run directory holding their worker log: {renamed}")
        if orphaned:
            self.logger.warning(f"{len(orphaned)} unit result(s) name a worker log outside every collected run directory, so their training status and duration reach no report: {orphaned}")
        if unmatched:
            self.logger.warning(f"{len(unmatched)} collected run(s) have no stage result, so the reports show their training status and duration as unknown: {unmatched}")

        return keyed

    def build(self, index, runs: list[tuple[str, Path]]) -> dict:
        results = index.build(runs)
        if not index.path.is_file():
            return results

        return self.rekey(results, runs)
