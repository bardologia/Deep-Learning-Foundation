from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


class FileIO:
    @staticmethod
    def ensure_dir(path: Path) -> Path:
        Path(path).mkdir(parents=True, exist_ok=True)
        return Path(path)

    @staticmethod
    def ensure_dirs(*paths: Path) -> None:
        for path in paths:
            Path(path).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def save_json(payload: dict, path: Path, indent: int = 4, atomic: bool = False) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        target = path.with_name(f"{path.name}.{os.getpid()}.tmp") if atomic else path
        with open(target, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=indent, default=str)

        if atomic:
            os.replace(target, path)

        return path

    @staticmethod
    def load_json(path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


class DatasetFingerprint:

    CHUNK_BYTES = 1 << 22

    @classmethod
    def digest(cls, path: Path) -> dict:
        hasher = hashlib.sha256()
        size   = 0

        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(cls.CHUNK_BYTES)
                if not chunk:
                    break
                hasher.update(chunk)
                size += len(chunk)

        return {"sha256": hasher.hexdigest(), "bytes": size}

    @classmethod
    def compute(cls, root: Path, files: list[Path]) -> dict:
        root    = Path(root)
        entries = {}

        for path in files:
            path = Path(path)
            try:
                key = str(path.relative_to(root))
            except ValueError:
                key = str(path)
            entries[key] = cls.digest(path)

        return entries
