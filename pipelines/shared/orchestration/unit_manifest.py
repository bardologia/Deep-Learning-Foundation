"""Unit manifests: the identity a fan-out worker records inside its run directory.

A staged experiment names its unit directories descriptively, and nothing that
runs later may depend on those names. The worker therefore writes
``meta/unit.json`` before anything else, and the passes that come after it
(resume reconciliation, the generic inference entries, report collection) read
the unit's identity from that file: which experiment it belongs to, which fold
or model and seed it is, and which regions it scores.
"""

from __future__ import annotations

from pathlib import Path
from typing  import Callable

from pipelines.shared.inference.run_classifier import RunDirectoryWalk
from tools.data.io                             import FileIO
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
