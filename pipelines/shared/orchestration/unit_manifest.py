"""Unit manifests: the identity a fan-out worker records inside its run directory.

A staged experiment names its unit directories descriptively, and nothing that
runs later may depend on those names. The worker therefore writes
``meta/unit.json`` before anything else, and the passes that come after it
(resume reconciliation, the generic inference entries, report collection) read
the unit's identity from that file: which experiment it belongs to, which fold
or model and seed it is, and which regions it scores. Run directories without
a manifest are reported by every stage that reconciles, with what the stage
does to each of them, and the per-unit stage results are matched back to the
run directories the units ran in when report collection reads them.
"""

from __future__ import annotations

from pathlib import Path
from typing  import Callable

from pipelines.shared.inference.run_classifier import RunDirectoryWalk
from tools.data.io                             import FileIO
from tools.monitoring.logger                   import Logger
from tools.runtime.completion                  import CompletionMarker
from tools.runtime.run_tag                     import RunTag


class UnitManifest:
    """Reads and writes the ``meta/unit.json`` manifest of one unit run directory.

    Attributes:
        META_DIRNAME : Directory inside a run holding its machine-readable metadata.
        FILENAME     : Name of the manifest file inside that directory.
    """

    META_DIRNAME = RunDirectoryWalk.MARKER
    FILENAME     = "unit.json"

    @classmethod
    def path(cls, run_directory: Path) -> Path:
        """Returns the manifest path of a run directory."""
        return Path(run_directory) / cls.META_DIRNAME / cls.FILENAME

    @classmethod
    def exists(cls, run_directory: Path) -> bool:
        """Returns whether the run directory carries a manifest."""
        return cls.path(run_directory).is_file()

    @classmethod
    def write(cls, run_directory: Path, kind: str, payload: dict) -> Path:
        """Writes the manifest, stamping the unit kind and the writing time into it.

        Args:
            run_directory : Run directory of the unit; its meta directory is created.
            kind          : Experiment kind the unit belongs to, e.g. ``cross_validation``.
            payload       : Identity fields of the unit, kept as they are.

        Returns:
            Path of the written manifest.
        """
        return FileIO.save_json({"kind": kind, "written_at": RunTag.timestamp(), **payload}, cls.path(run_directory))

    @classmethod
    def read(cls, run_directory: Path) -> dict:
        """Returns the manifest payload of a run directory.

        Args:
            run_directory : Run directory of the unit.

        Raises:
            FileNotFoundError : If the run carries no manifest, which means it was not written by a fan-out worker of this version.
        """
        path = cls.path(run_directory)
        if not path.is_file():
            raise FileNotFoundError(f"No unit manifest at {path}; the run was not written by a staged-experiment worker, so its fold, model or seed identity cannot be read back")

        return FileIO.load_json(path)


class UnitLedger:
    """Existing unit run directories of one stage, keyed by the identity their manifests record.

    Walks the stage directory for run directories, keeps the ones whose manifest
    is of the requested kind, and indexes them by the identity tuple a caller
    derives from each manifest. Run directories without a manifest are listed
    as unidentified so the caller can report them instead of guessing.

    Attributes:
        stage_dir    : Directory holding the unit run directories of the stage.
        kind         : Manifest kind the ledger indexes.
        identity     : Callable turning a manifest payload into the hashable identity of its unit.
        by_identity  : Run directory per identity, filled by `build`.
        unidentified : Run directories without a manifest of the kind, filled by `build`.
    """

    def __init__(self, stage_dir: Path, kind: str, identity: Callable[[dict], tuple]) -> None:
        """Binds the ledger to a stage directory, a manifest kind and an identity rule."""
        self.stage_dir    = Path(stage_dir)
        self.kind         = kind
        self.identity     = identity
        self.by_identity  = {}
        self.unidentified = []

    def _index(self, run_dir: Path) -> None:
        """Files one run directory under its identity, or as unidentified.

        Args:
            run_dir : Run directory found under the stage directory.

        Raises:
            ValueError : If another run directory already claims the same identity, which leaves the unit ambiguous until one of them is removed.
        """
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
        """Returns the run directory recorded for an identity, or None when the unit has none yet."""
        return self.by_identity.get(identity)

    def build(self) -> "UnitLedger":
        """Walks the stage directory and indexes every run directory, returning the ledger itself."""
        if self.stage_dir.is_dir():
            for run_dir in RunDirectoryWalk.walk(self.stage_dir):
                self._index(run_dir)

        return self


class UnidentifiedRuns:
    """Report of the run directories under a stage directory that no planned unit's manifest claims.

    A directory that sits at a planned unit's own path is used by the stage as
    that unit's directory, reused when it holds a completion marker and treated
    as unfinished otherwise; any other unidentified directory is ignored by the
    stage but still collected by the comparison beside the planned units.

    Attributes:
        stage_dir   : Stage directory the paths are reported relative to.
        directories : Run directories without a manifest of the stage's kind.
        kind        : Manifest kind the stage indexes, named in the messages.
        planned     : Directory each planned unit lives or will live in, keyed by unit name.
    """

    BACKFILL_HINT = "run scripts/backfill_unit_manifests.py to write their manifests"

    def __init__(self, stage_dir: Path, directories: list[Path], kind: str, planned: dict[str, Path]) -> None:
        """Binds the report to a stage directory, its unidentified directories and the planned unit directories."""
        self.stage_dir   = Path(stage_dir)
        self.directories = list(directories)
        self.kind        = kind
        self.planned     = {name: Path(path) for name, path in planned.items()}

    def _case(self, directory: Path) -> str:
        """Returns what the stage does with one unidentified directory."""
        claimants = [name for name, path in self.planned.items() if path == directory]

        if not claimants:
            return "claimed by no planned unit, ignored by the stage; the comparison still collects it beside the planned units"
        if CompletionMarker.is_complete(directory):
            return f"sits at planned unit '{claimants[0]}'s own path and holds a completion marker, so the stage uses it as that unit's completed run"

        return f"sits at planned unit '{claimants[0]}'s own path without a completion marker, so the stage treats it as that unit's unfinished run"

    def log(self, logger: Logger, stage_label: str) -> None:
        """Warns once with the count and once per directory with what the stage does to it.

        Args:
            logger      : Logger receiving the warnings.
            stage_label : Name of the stage reporting, so a full run tells its training and inference reports apart.
        """
        if not self.directories:
            return

        logger.warning(f"{stage_label}: {len(self.directories)} run directory(ies) under {self.stage_dir} carry no {self.kind} manifest; {self.BACKFILL_HINT}")

        for directory in self.directories:
            logger.warning(f"{directory.relative_to(self.stage_dir).as_posix()}: {self._case(directory)}")


class UnitResultsByRun:
    """Re-keys a queued stage's per-unit results from planned unit names to the run directories the units ran in.

    A queued stage keys its results by planned unit name, while report
    collection keys every run by its directory relative to the stage
    directory; the two differ for every unit a resume matched to a renamed
    directory through its manifest. Each stored result names the worker log
    the stage wrote into the unit's own directory, and the trailing part of
    that path, from the stage directory's name down, identifies the run, so a
    run tree moved to another machine still matches.

    Attributes:
        stage_dir : Stage directory the collected runs live under.
        logger    : Logger receiving the matching report.
    """

    def __init__(self, stage_dir: Path, logger: Logger) -> None:
        """Binds the matcher to the stage directory of the collected runs and to a logger."""
        self.stage_dir = Path(stage_dir)
        self.logger    = logger

    def _run_of(self, result: dict, names: list[str]) -> str | None:
        """Returns the collected run whose directory holds a result's worker log, or None when no collected run does.

        Args:
            result : Stage result naming its worker log under ``log_file``.
            names  : Collected run names, each the run directory relative to the stage directory.
        """
        parts = Path(result["log_file"]).parent.parts

        for name in names:
            tail = (self.stage_dir.name, *Path(name).parts)
            if parts[-len(tail):] == tail:
                return name

        return None

    def rekey(self, results: dict, runs: list[tuple[str, Path]]) -> dict:
        """Returns the stored results keyed by collected run name, reporting what could not be matched.

        Args:
            results : Stored stage results keyed by planned unit name.
            runs    : (run name, run directory) pairs the collector found under the stage directory.

        Returns:
            The results keyed by the name of the run each one belongs to.
        """
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
        """Returns the training results of the collected runs keyed by run name.

        Results the stage stored are re-keyed through their worker logs;
        results the index rebuilt from completion markers, when the stage
        stored none, are keyed by run name already.

        Args:
            index : Training results index of the run's pipeline directory, exposing ``path`` and ``build(runs)``.
            runs  : (run name, run directory) pairs the collector found under the stage directory.
        """
        results = index.build(runs)
        if not index.path.is_file():
            return results

        return self.rekey(results, runs)
