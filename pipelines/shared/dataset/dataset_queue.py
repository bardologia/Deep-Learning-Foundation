
from __future__ import annotations

from pathlib import Path


class DatasetQueueResolver:

    def __init__(self, base_path: Path, dataset_filter: list) -> None:
        self.base_path      = base_path
        self.dataset_filter = dataset_filter

    def resolve(self) -> list[Path]:
        dataset_dirs = sorted(
            [d for d in self.base_path.iterdir() if d.is_dir()]
            if not self.dataset_filter
            else [self.base_path / str(name) for name in self.dataset_filter]
        )

        return dataset_dirs
