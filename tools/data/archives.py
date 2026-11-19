
from __future__ import annotations

import zipfile

from pathlib import Path
from typing  import List, Tuple


class ArchivePacker:

    SUFFIX = ".zip"

    def __init__(self, entries: List[Tuple[Path, str]], output_dir: Path, stem: str, max_bytes: int, logger) -> None:
        self.entries    = list(entries)
        self.output_dir = Path(output_dir)
        self.stem       = stem
        self.max_bytes  = int(max_bytes)
        self.logger     = logger

    def _clear(self) -> None:
        for path in sorted(self.output_dir.glob(f"{self.stem}*{self.SUFFIX}")):
            path.unlink()

    def _plan(self) -> List[List[Tuple[Path, str]]]:
        if not self.entries:
            raise ValueError(f"Nothing to pack into {self.output_dir / self.stem}{self.SUFFIX}: the file set is empty")

        volumes = []
        filled  = 0

        for source, arcname in self.entries:
            if not source.is_file():
                raise FileNotFoundError(f"Cannot pack {source}: the file vanished after it was selected")

            size = source.stat().st_size

            if size > self.max_bytes:
                raise ValueError(f"Cannot pack {source}: it is {size / (1024 * 1024):,.1f} MB on its own, above the {self.max_bytes / (1024 * 1024):,.1f} MB cap; raise max_attachment_mb or drop it from the gathered paths")

            if not volumes or filled + size > self.max_bytes:
                volumes.append([])
                filled = 0

            volumes[-1].append((source, arcname))
            filled += size

        return volumes

    def _volume_name(self, index: int, total: int) -> str:
        if total == 1:
            return f"{self.stem}{self.SUFFIX}"

        return f"{self.stem}_{index:02d}of{total:02d}{self.SUFFIX}"

    def _write(self, volume: List[Tuple[Path, str]], index: int, total: int) -> Path:
        path = self.output_dir / self._volume_name(index, total)

        with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for source, arcname in volume:
                archive.write(source, arcname)

        self.logger.ok(f"Packed {path.name}: {len(volume)} file(s), {path.stat().st_size / (1024 * 1024):,.1f} MB")
        return path

    def pack(self) -> List[Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._clear()

        volumes = self._plan()
        return [self._write(volume, index, len(volumes)) for index, volume in enumerate(volumes, start=1)]
