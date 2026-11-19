
from __future__ import annotations

from pathlib import Path


class RunDirectorySettledMarker:

    RELATIVE_PATH = Path("docs") / "run_directory_settled.marker"

    @classmethod
    def path_for(cls, run_directory) -> Path:
        return Path(run_directory) / cls.RELATIVE_PATH

    @classmethod
    def write(cls, run_directory) -> Path:
        marker = cls.path_for(run_directory)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("settled\n", encoding="utf-8")

        return marker
