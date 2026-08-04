from __future__ import annotations

from pathlib import Path


class DatasetQueueResolver:
    def __init__(self, base_path: Path, dataset_filter: list) -> None:
        self.base_path      = base_path
        self.dataset_filter = dataset_filter

    def resolve(self) -> list[Path]:
        if not isinstance(self.dataset_filter, (list, tuple)):
            raise TypeError(f"dataset_filter must be a list of dataset names, got {type(self.dataset_filter).__name__}: {self.dataset_filter!r}")

        if not self.base_path.is_dir():
            raise NotADirectoryError(f"dataset_base_path does not exist: {self.base_path}")

        dataset_dirs = sorted(
            [d for d in self.base_path.iterdir() if d.is_dir()]
            if not self.dataset_filter
            else [self.base_path / str(name) for name in self.dataset_filter]
        )

        invalid = [d for d in dataset_dirs if not (d / "data").is_dir()]
        if invalid:
            names = ", ".join(d.name for d in invalid)
            raise NotADirectoryError(f"Queue entries without a data/ directory under {self.base_path}: {names}")

        return dataset_dirs
