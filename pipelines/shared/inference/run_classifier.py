from __future__ import annotations

from pathlib import Path


class RunArtifacts:

    BACKBONE_CONFIG = "model_config.json"

    @classmethod
    def is_run(cls, run_directory: Path) -> bool:
        return (Path(run_directory) / "meta" / cls.BACKBONE_CONFIG).is_file()


class InferenceLayout:

    DIRNAME        = "inference"
    TARGET_DIRNAME = "cross_dataset"

    @classmethod
    def base(cls, run_directory: Path) -> Path:
        return Path(run_directory) / cls.DIRNAME

    @classmethod
    def target_base(cls, run_directory: Path, dataset_name: str) -> Path:
        return Path(run_directory) / cls.TARGET_DIRNAME / dataset_name


class RunDirectoryWalk:

    MARKER    = "meta"
    MAX_DEPTH = 6

    @classmethod
    def walk(cls, root: Path, depth: int = 0):
        for entry in sorted(Path(root).iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue

            if (entry / cls.MARKER).is_dir():
                yield entry
            elif depth < cls.MAX_DEPTH:
                yield from cls.walk(entry, depth + 1)
