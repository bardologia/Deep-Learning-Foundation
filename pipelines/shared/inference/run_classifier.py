"""Discovery and family classification of training run directories.

A run's family is read off the architecture config it persisted under meta/, so
classification never depends on directory naming.
"""

from __future__ import annotations

from pathlib import Path


class RunType:
    """Names of the model families a run can belong to."""

    BACKBONE   = "backbone"
    PROFILE_AE = "profile_ae"
    IMAGE_AE   = "image_ae"
    DUAL       = "dual"


class RunArtifacts:
    """Filenames of the per-family architecture configs stored under meta/."""

    BACKBONE_CONFIG   = "model_config.json"
    PROFILE_AE_CONFIG = "profile_autoencoder_config.json"
    IMAGE_AE_CONFIG   = "image_autoencoder_config.json"
    DUAL_CONFIG       = "dual_model_config.json"


class InferenceLayout:
    """Where each run family writes its inference stamps inside a run directory.

    Attributes:
        DIRNAME        : Directory inside a run holding its inference outputs.
        TARGET_DIRNAME : Directory inside a run holding the stamps scored on a foreign dataset, one subdirectory per target dataset.
        SUBDIRS        : Family sub-directory under it, for the families that nest their stamps.
    """

    DIRNAME        = "inference"
    TARGET_DIRNAME = "cross_dataset"
    SUBDIRS        = {
        RunType.PROFILE_AE : RunType.PROFILE_AE,
        RunType.IMAGE_AE   : RunType.IMAGE_AE,
    }

    @classmethod
    def base(cls, run_directory: Path, run_type: str) -> Path:
        """Returns the directory a family's inference stamps are written under.

        Args:
            run_directory : Training run directory.
            run_type      : Run family, one of the RunType names.
        """
        base = Path(run_directory) / cls.DIRNAME

        return base / cls.SUBDIRS[run_type] if run_type in cls.SUBDIRS else base

    @classmethod
    def target_base(cls, run_directory: Path, run_type: str, dataset_name: str) -> Path:
        """Returns the directory a family's stamps scored on a foreign dataset are written under.

        These stamps sit outside the run's inference directory, so the leaderboards,
        cube views, comparisons, report collectors and resume checks that read a run's
        own inference never mix in a score measured on another scene.

        Args:
            run_directory : Training run directory.
            run_type      : Run family, one of the RunType names.
            dataset_name  : Directory name of the target dataset.
        """
        base = Path(run_directory) / cls.TARGET_DIRNAME / dataset_name

        return base / cls.SUBDIRS[run_type] if run_type in cls.SUBDIRS else base


class RunDirectoryWalk:
    """Finds run directories nested anywhere under a runs root.

    Attributes:
        MARKER    : Sub-directory whose presence marks a directory as a run.
        MAX_DEPTH : Deepest level still descended into while searching.
    """

    MARKER    = "meta"
    MAX_DEPTH = 6

    @classmethod
    def walk(cls, root: Path, depth: int = 0):
        """Yields every run directory under a root, in sorted order.

        Args:
            root  : Directory to search.
            depth : Current recursion depth; descent stops past MAX_DEPTH.

        Yields:
            Directories containing a meta/ sub-directory. Hidden directories are
            skipped, and a run is never descended into.
        """
        for entry in sorted(Path(root).iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue

            if (entry / cls.MARKER).is_dir():
                yield entry
            elif depth < cls.MAX_DEPTH:
                yield from cls.walk(entry, depth + 1)


class RunClassifier:
    """Determines a run's model family from the configs it persisted.

    Attributes:
        TYPE_ARTIFACTS : (run type, config filename) pairs probed in order, so a run carrying several configs is classified by the first match.
    """

    TYPE_ARTIFACTS = (
        (RunType.BACKBONE,   RunArtifacts.BACKBONE_CONFIG),
        (RunType.PROFILE_AE, RunArtifacts.PROFILE_AE_CONFIG),
        (RunType.IMAGE_AE,   RunArtifacts.IMAGE_AE_CONFIG),
        (RunType.DUAL,       RunArtifacts.DUAL_CONFIG),
    )

    @classmethod
    def classify(cls, run_directory: Path) -> str:
        """Returns the model family of a run directory.

        Args:
            run_directory : Training run whose meta/ directory is probed.

        Raises:
            ValueError : The run carries no recognized architecture config.
        """
        meta = Path(run_directory) / "meta"

        for run_type, filename in cls.TYPE_ARTIFACTS:
            if (meta / filename).is_file():
                return run_type

        raise ValueError(f"Run '{run_directory}' has no recognized model config under meta/ (no backbone, profile-autoencoder, image-autoencoder, or dual config); cannot infer.")
