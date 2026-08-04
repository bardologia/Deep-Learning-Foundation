from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib     import Path

from tools.data.rat          import RatCube
from tools.data.regions      import CropRegion
from tools.monitoring.logger import Logger


@dataclass
class ExternalSource:
    path   : Path
    window : CropRegion
    cube   : RatCube

    @property
    def n_channels(self) -> int:
        return self.cube.n_channels

    @property
    def n_gaussians(self) -> int:
        return self.cube.n_channels // 3


class WindowParser:

    PATTERN = re.compile(r"(\d+)a(\d+)a(\d+)a(\d+)")

    def from_text(self, text: str) -> CropRegion:
        match = self.PATTERN.fullmatch(str(text).strip())

        if match is None:
            raise ValueError(f"Window '{text}' is not an azimuth/range identifier; expected '<az_start>a<az_end>a<rg_start>a<rg_end>', e.g. '12400a16000a500a4000'")

        return CropRegion(*(int(group) for group in match.groups()))

    def from_filename(self, path: Path) -> CropRegion:
        matches = self.PATTERN.findall(Path(path).stem)

        if len(matches) != 1:
            raise ValueError(f"Filename {Path(path).name} carries {len(matches)} azimuth/range identifiers; exactly one '<az_start>a<az_end>a<rg_start>a<rg_end>' token is required, or give the window explicitly through source_windows")

        return CropRegion(*(int(group) for group in matches[0]))


class ExternalSourceResolver:

    def __init__(self, source_files: list, source_windows: list, logger: Logger) -> None:
        self.source_files   = [Path(path) for path in source_files]
        self.source_windows = [str(text) for text in source_windows]
        self.logger         = logger
        self.parser         = WindowParser()

    def _validate_inputs(self) -> None:
        if not self.source_files:
            raise ValueError("source_files is empty; name at least one external parameter file to inject")

        if self.source_windows and len(self.source_windows) != len(self.source_files):
            raise ValueError(f"source_windows holds {len(self.source_windows)} entries for {len(self.source_files)} source files; give one window per file or leave it empty to read each window from its filename")

    def _window(self, index: int, path: Path) -> CropRegion:
        if self.source_windows:
            return self.parser.from_text(self.source_windows[index])

        return self.parser.from_filename(path)

    def _build(self, index: int, path: Path) -> ExternalSource:
        cube   = RatCube(path)
        window = self._window(index, path)

        if (cube.azimuth_size, cube.range_size) != (window.azimuth_size, window.range_size):
            raise ValueError(f"External source {path.name} holds a {cube.azimuth_size} x {cube.range_size} (azimuth, range) grid but its window {window.as_identifier_string()} spans {window.azimuth_size} x {window.range_size}; the file and its declared extent disagree")

        if cube.n_channels % 3 != 0:
            raise ValueError(f"External source {path.name} has {cube.n_channels} channels, which is not divisible by the 3 parameters (amplitude, mean, sigma) every Gaussian carries")

        return ExternalSource(path=path, window=window, cube=cube)

    def _validate_consistency(self, sources: list) -> None:
        channel_counts = {source.n_channels for source in sources}

        if len(channel_counts) != 1:
            details = ", ".join(f"{source.path.name}={source.n_channels}" for source in sources)
            raise ValueError(f"External sources disagree on channel count ({details}); every file must describe the same number of Gaussians")

    def _log(self, sources: list) -> None:
        rows = [
            {
                "Source"    : source.path.name,
                "Window"    : source.window.as_labeled_string(),
                "Grid"      : f"{source.cube.azimuth_size} x {source.cube.range_size}",
                "Channels"  : source.n_channels,
                "Gaussians" : source.n_gaussians,
                "Type"      : str(source.cube.spec.dtype),
            }
            for source in sources
        ]

        self.logger.section("[External Sources Resolved]")
        self.logger.metrics_table(rows, ["Source", "Window", "Grid", "Channels", "Gaussians", "Type"])

    def resolve(self) -> list[ExternalSource]:
        self._validate_inputs()

        sources = [self._build(index, path) for index, path in enumerate(self.source_files)]

        self._validate_consistency(sources)
        self._log(sources)

        return sources
