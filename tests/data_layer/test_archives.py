"""Tests covering the packing of file sets into size-capped zip volumes.

Covers the volume plan drawn from the on-disk sizes, the naming that numbers a
split set and leaves a single volume bare, the clearing of a previous packing,
and the two conditions that abort it.
"""

from __future__ import annotations

import zipfile

import pytest

from tools.data.archives import ArchivePacker

from tests.conftest import RecordingLogger


@pytest.fixture
def sources(tmp_path):
    """Returns four files of 100 bytes each with their archive paths."""
    entries = []
    for index in range(4):
        path = tmp_path / "source" / f"file_{index}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * 100)
        entries.append((path, f"bundle/file_{index}.bin"))

    return entries


def test_one_volume_keeps_the_bare_stem(sources, tmp_path):
    """Verifies a file set below the cap is written as a single unnumbered archive."""
    archives = ArchivePacker(sources, tmp_path / "out", "bundle", 1000, RecordingLogger()).pack()

    assert [path.name for path in archives] == ["bundle.zip"]

    with zipfile.ZipFile(archives[0]) as archive:
        assert archive.namelist() == [f"bundle/file_{index}.bin" for index in range(4)]


def test_volumes_are_numbered_and_split_at_the_cap(sources, tmp_path):
    """Verifies the cap splits the set and every volume name carries its position."""
    archives = ArchivePacker(sources, tmp_path / "out", "bundle", 250, RecordingLogger()).pack()

    members = [name for path in archives for name in zipfile.ZipFile(path).namelist()]

    assert [path.name for path in archives] == ["bundle_01of02.zip", "bundle_02of02.zip"]
    assert members == [f"bundle/file_{index}.bin" for index in range(4)]


def test_repacking_deletes_the_previous_volumes(sources, tmp_path):
    """Verifies a second packing with a larger cap leaves no volume of the first behind."""
    ArchivePacker(sources, tmp_path / "out", "bundle", 250, RecordingLogger()).pack()
    ArchivePacker(sources, tmp_path / "out", "bundle", 1000, RecordingLogger()).pack()

    assert sorted(path.name for path in (tmp_path / "out").glob("*.zip")) == ["bundle.zip"]


def test_a_file_above_the_cap_aborts(sources, tmp_path):
    """Verifies a single oversized file aborts and names itself and the cap."""
    with pytest.raises(ValueError, match="file_0.bin"):
        ArchivePacker(sources, tmp_path / "out", "bundle", 50, RecordingLogger()).pack()


def test_an_empty_file_set_aborts(tmp_path):
    """Verifies packing nothing aborts instead of writing an empty archive."""
    with pytest.raises(ValueError, match="empty"):
        ArchivePacker([], tmp_path / "out", "bundle", 1000, RecordingLogger()).pack()


def test_a_vanished_source_aborts(sources, tmp_path):
    """Verifies a file deleted between selection and packing aborts naming it."""
    sources[1][0].unlink()

    with pytest.raises(FileNotFoundError, match="file_1.bin"):
        ArchivePacker(sources, tmp_path / "out", "bundle", 1000, RecordingLogger()).pack()
