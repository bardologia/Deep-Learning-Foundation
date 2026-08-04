from __future__ import annotations

import json
import os
from pathlib  import Path
from unittest import mock

import pytest

import tools.data.io as io_module
from tools.data.io import FileIO


def test_ensure_dir_creates_and_returns(tmp_path):
    target = tmp_path / "a" / "b" / "c"
    result = FileIO.ensure_dir(target)

    assert result == target
    assert target.is_dir()


def test_ensure_dir_idempotent(tmp_path):
    target = tmp_path / "x"
    FileIO.ensure_dir(target)
    FileIO.ensure_dir(target)

    assert target.is_dir()


def test_ensure_dirs_creates_all(tmp_path):
    a = tmp_path / "one"
    b = tmp_path / "two" / "nested"
    c = tmp_path / "three"

    FileIO.ensure_dirs(a, b, c)

    assert a.is_dir()
    assert b.is_dir()
    assert c.is_dir()


def test_save_json_roundtrip(tmp_path):
    payload = {"k": 1, "nested": {"a": [1, 2, 3]}, "flag": True}
    path    = tmp_path / "out.json"

    returned = FileIO.save_json(payload, path)

    assert returned == path
    assert json.loads(path.read_text()) == payload


def test_save_json_creates_parent_dirs(tmp_path):
    path = tmp_path / "deep" / "nested" / "file.json"
    FileIO.save_json({"x": 1}, path)

    assert path.is_file()


def test_save_json_load_json_roundtrip(tmp_path):
    payload = {"alpha": 0.5, "beta": [True, False, None]}
    path    = tmp_path / "rt.json"

    FileIO.save_json(payload, path)
    loaded = FileIO.load_json(path)

    assert loaded == payload


def test_save_json_atomic_leaves_no_tmp(tmp_path):
    path = tmp_path / "atomic.json"
    FileIO.save_json({"v": 7}, path, atomic=True)

    assert path.is_file()
    assert list(tmp_path.glob("atomic.json.*.tmp")) == []
    assert FileIO.load_json(path) == {"v": 7}


def test_save_json_atomic_sidecar_is_process_private(tmp_path):
    path    = tmp_path / "atomic.json"
    written = []

    real_replace = os.replace

    def _capture(source, destination):
        written.append(Path(source).name)
        real_replace(source, destination)

    with mock.patch.object(io_module.os, "replace", _capture):
        FileIO.save_json({"v": 7}, path, atomic=True)

    assert written == [f"atomic.json.{os.getpid()}.tmp"]


def test_save_json_indent_applied(tmp_path):
    path = tmp_path / "indent.json"
    FileIO.save_json({"a": 1}, path, indent=2)

    text = path.read_text()
    assert "\n  " in text


def test_save_json_serializes_nonjson_via_default(tmp_path):
    path = tmp_path / "p.json"
    FileIO.save_json({"path": Path("/x/y")}, path)

    loaded = FileIO.load_json(path)
    assert loaded["path"] == "/x/y"


def test_save_text_metadata_format(tmp_path):
    path = tmp_path / "meta.txt"
    FileIO.save_text_metadata({"name": "foo", "count": 3}, path)

    lines = path.read_text().splitlines()
    assert lines == ["name: foo", "count: 3"]


def test_save_text_metadata_creates_parent(tmp_path):
    path = tmp_path / "sub" / "meta.txt"
    FileIO.save_text_metadata({"k": "v"}, path)

    assert path.is_file()


@pytest.mark.real_data
def test_load_json_reads_real_config_state(meta_dir):
    loaded = FileIO.load_json(meta_dir / "config_state.json")

    assert "crop" in loaded
    assert loaded["crop"]["azimuth_start"] == 1000


@pytest.mark.real_data
def test_save_json_roundtrip_real_dataset(dataset_json, tmp_path):
    path = tmp_path / "dataset_copy.json"
    FileIO.save_json(dataset_json, path)

    assert FileIO.load_json(path) == dataset_json
