from __future__ import annotations

import re
from pathlib import Path
from typing  import Callable

import numpy as np


class TrackReader:
    MIN_ROWS = 4

    def __init__(self, reader: Callable | None = None):
        self._reader = reader

    def read(self, path: str | Path) -> np.ndarray:
        reader = self._reader if self._reader is not None else self._default_reader()
        data   = np.asarray(reader(str(path)))

        if data.ndim != 2 or data.shape[0] < self.MIN_ROWS:
            raise ValueError(f"Track file {path} has shape {data.shape}, expected (>= {self.MIN_ROWS}, n_azimuth)")

        return data

    @staticmethod
    def _default_reader() -> Callable:
        from STEtools.ste_io import rrat
        return rrat


class PassProductResolver:
    PRODUCT_SUBDIR    = Path(".")
    PRODUCT_PATTERNS  = ()
    TRACK_DIR_PATTERN = re.compile(r"[Tt]\d+\w*")

    def label(self, pass_directory: str | Path) -> str:
        node   = self._pass_node(pass_directory)
        flight = node.parent.name
        return f"{flight}_{node.name}" if flight else node.name

    def _pass_node(self, pass_directory: str | Path) -> Path:
        path = Path(pass_directory)
        return path.parent if self.TRACK_DIR_PATTERN.fullmatch(path.name) else path

    def resolve(self, pass_directory: str | Path) -> Path:
        directory = Path(pass_directory) / self.PRODUCT_SUBDIR

        for pattern in self.PRODUCT_PATTERNS:
            candidates = sorted(directory.glob(pattern))
            if candidates:
                return candidates[0]

        raise FileNotFoundError(f"No file matching {self.PRODUCT_PATTERNS} under {directory}")

    def resolve_passes(self, pass_directories: list) -> dict:
        mapping = {}

        for directory in pass_directories:
            label = self.label(directory)
            if label in mapping:
                raise ValueError(f"Pass directory {directory} resolves to label '{label}' which is already taken; every pass must map to a unique label or tracks would be silently dropped.")

            mapping[label] = self.resolve(directory)

        return mapping


class TrackFileResolver(PassProductResolver):
    PRODUCT_SUBDIR   = Path("INF") / "INF-TRACK"
    PRODUCT_PATTERNS = ("track_sar_resa_*.rat", "track_*.rat")
