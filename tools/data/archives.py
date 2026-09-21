"""Packing of file sets into size-capped zip archives.

ArchivePacker splits a file set into as many deflate archives as the size cap
requires, planning the volumes from the on-disk sizes before writing any of
them so every archive name can carry its position in the set. The cap is
measured on the uncompressed sources, which is an upper bound for the written
archive and leaves room for the base64 expansion an email attachment adds.
"""

from __future__ import annotations

import zipfile

from pathlib import Path
from typing  import List, Tuple


class ArchivePacker:
    """Packs a set of files into zip volumes that each stay under a size cap.

    Attributes:
        entries    : Files to pack, as (source path, path inside the archive) pairs.
        output_dir : Directory receiving the written volumes.
        stem       : Filename stem shared by every written volume.
        max_bytes  : Upper bound of one volume, measured on the uncompressed sources.
        logger     : Logger for the per-volume progress.
        SUFFIX     : Extension of every written volume.
    """

    SUFFIX = ".zip"

    def __init__(self, entries: List[Tuple[Path, str]], output_dir: Path, stem: str, max_bytes: int, logger) -> None:
        """Initializes the packer with its file set, destination and size cap.

        Args:
            entries    : Files to pack, as (source path, path inside the archive) pairs, in the order they are written.
            output_dir : Directory receiving the written volumes, created if absent.
            stem       : Filename stem shared by every written volume.
            max_bytes  : Upper bound of one volume in bytes, measured on the uncompressed sources.
            logger     : Logger for the per-volume progress.
        """
        self.entries    = list(entries)
        self.output_dir = Path(output_dir)
        self.stem       = stem
        self.max_bytes  = int(max_bytes)
        self.logger     = logger

    def _clear(self) -> None:
        """Deletes the volumes of a previous packing so a shorter set leaves none behind."""
        for path in sorted(self.output_dir.glob(f"{self.stem}*{self.SUFFIX}")):
            path.unlink()

    def _plan(self) -> List[List[Tuple[Path, str]]]:
        """Splits the file set into the volumes it needs, filling each one in entry order.

        Returns:
            One list of (source path, archive path) pairs per volume, in write order.

        Raises:
            ValueError        : If the file set is empty, or one file alone exceeds the cap.
            FileNotFoundError : If a planned source file disappeared between selection and packing.
        """
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
        """Returns the filename of one volume, numbered only when the set is split.

        Args:
            index : Position of the volume in the set, starting at one.
            total : Number of volumes in the set.

        Returns:
            The bare stem for a single volume, otherwise the stem with its position appended.
        """
        if total == 1:
            return f"{self.stem}{self.SUFFIX}"

        return f"{self.stem}_{index:02d}of{total:02d}{self.SUFFIX}"

    def _write(self, volume: List[Tuple[Path, str]], index: int, total: int) -> Path:
        """Writes one planned volume as a deflate archive and reports its size.

        Args:
            volume : Files of this volume, as (source path, archive path) pairs.
            index  : Position of the volume in the set, starting at one.
            total  : Number of volumes in the set.

        Returns:
            Path of the written archive.
        """
        path = self.output_dir / self._volume_name(index, total)

        with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for source, arcname in volume:
                archive.write(source, arcname)

        self.logger.ok(f"Packed {path.name}: {len(volume)} file(s), {path.stat().st_size / (1024 * 1024):,.1f} MB")
        return path

    def pack(self) -> List[Path]:
        """Plans the volumes and writes each of them into the output directory.

        Returns:
            The written archive paths, in volume order.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._clear()

        volumes = self._plan()
        return [self._write(volume, index, len(volumes)) for index, volume in enumerate(volumes, start=1)]
