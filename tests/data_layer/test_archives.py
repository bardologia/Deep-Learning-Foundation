
from __future__ import annotations

import zipfile

import pytest

from tools.data.archives import ArchivePacker

from tests.conftest import RecordingLogger


@pytest.fixture
def sources(tmp_path):
    entries = []
    for index in range(4):
        path = tmp_path / "source" / f"file_{index}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * 100)
        entries.append((path, f"bundle/file_{index}.bin"))

    return entries


def test_one_volume_keeps_the_bare_stem(sources, tmp_path):
    archives = ArchivePacker(sources, tmp_path / "out", "bundle", 1000, RecordingLogger()).pack()

    assert [path.name for path in archives] == ["bundle.zip"]

    with zipfile.ZipFile(archives[0]) as archive:
        assert archive.namelist() == [f"bundle/file_{index}.bin" for index in range(4)]


def test_volumes_are_numbered_and_split_at_the_cap(sources, tmp_path):
    archives = ArchivePacker(sources, tmp_path / "out", "bundle", 250, RecordingLogger()).pack()

    members = [name for path in archives for name in zipfile.ZipFile(path).namelist()]

    assert [path.name for path in archives] == ["bundle_01of02.zip", "bundle_02of02.zip"]
    assert members == [f"bundle/file_{index}.bin" for index in range(4)]


def test_repacking_deletes_the_previous_volumes(sources, tmp_path):
    ArchivePacker(sources, tmp_path / "out", "bundle", 250, RecordingLogger()).pack()
    ArchivePacker(sources, tmp_path / "out", "bundle", 1000, RecordingLogger()).pack()

    assert sorted(path.name for path in (tmp_path / "out").glob("*.zip")) == ["bundle.zip"]

