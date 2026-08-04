from __future__ import annotations

import json
from pathlib import Path


class RepoMapLibrary:

    DATA_FILE = "repomap_data.json"

    def __init__(self) -> None:
        self.data = self._load()

    def _load(self) -> dict:
        path = Path(__file__).resolve().parent / self.DATA_FILE
        return json.loads(path.read_text(encoding="utf-8"))

    def collect(self) -> list[dict]:
        return self.data["folders"]
