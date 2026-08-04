from __future__ import annotations

from pathlib import Path

from tools.data.io         import FileIO
from tools.runtime.run_tag import RunTag


class CompletionMarker:

    FILENAME = "complete.json"

    @staticmethod
    def path(directory: Path) -> Path:
        return Path(directory) / CompletionMarker.FILENAME

    @staticmethod
    def is_complete(directory: Path) -> bool:
        return CompletionMarker.path(directory).is_file()

    @staticmethod
    def payload(directory: Path) -> dict:
        return FileIO.load_json(CompletionMarker.path(directory))

    @staticmethod
    def clear(directory: Path) -> None:
        CompletionMarker.path(directory).unlink(missing_ok=True)

    @staticmethod
    def stamp(directory: Path, payload: dict) -> Path:
        path = CompletionMarker.path(directory)
        FileIO.save_json({"completed_at": RunTag.timestamp(), **payload}, path)
        return path
