"""Tests the migration of the legacy backbone_name key in stored model configs.

Covers the dry run leaving files untouched, the apply pass rewriting the key,
an unrecognised config aborting before any write, the discovery walk, and the
refusal of a missing root directory."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT   = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from fix_config_names import ConfigNameKeyFixer


def _legacy_config(run_dir: Path) -> Path:
    """Writes a model config using the legacy backbone_name key.

    Args:
        run_dir : Run directory that receives meta/model_config.json.

    Returns:
        Path of the written config file.
    """
    path = run_dir / "meta" / "model_config.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"backbone_name": "resunet", "config": {"base_channels": 32}}), encoding="utf-8")

    return path


def test_dry_run_reports_without_rewriting(tmp_path, monkeypatch):
    """Verifies the default dry run leaves the legacy config byte-identical."""
    monkeypatch.chdir(tmp_path)
    path = _legacy_config(tmp_path / "run")

    ConfigNameKeyFixer(tmp_path).run()

    assert json.loads(path.read_text(encoding="utf-8")) == {"backbone_name": "resunet", "config": {"base_channels": 32}}


def test_apply_rewrites_the_legacy_key(tmp_path, monkeypatch):
    """Verifies applying renames backbone_name to model_name and keeps the key order."""
    monkeypatch.chdir(tmp_path)
    path = _legacy_config(tmp_path / "run")

    ConfigNameKeyFixer(tmp_path, apply=True).run()

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert list(payload) == ["model_name", "config"]
    assert payload["model_name"] == "resunet"


def test_unrecognized_config_aborts_before_any_rewrite(tmp_path, monkeypatch):
    """Verifies an unrecognised config aborts the run before any file is rewritten."""
    monkeypatch.chdir(tmp_path)

    legacy  = _legacy_config(tmp_path / "aaa_run")
    foreign = tmp_path / "zzz_run" / "meta" / "image_autoencoder_config.json"
    foreign.parent.mkdir(parents=True)
    foreign.write_text(json.dumps({"unrelated": 1}), encoding="utf-8")

    with pytest.raises(KeyError, match="Nothing was rewritten"):
        ConfigNameKeyFixer(tmp_path, apply=True).run()

    assert json.loads(legacy.read_text(encoding="utf-8")) == {"backbone_name": "resunet", "config": {"base_channels": 32}}


def test_every_config_filename_is_found_in_one_walk(tmp_path, monkeypatch):
    """Verifies discovery finds every declared config filename in a single walk."""
    monkeypatch.chdir(tmp_path)

    for index, name in enumerate(ConfigNameKeyFixer.FILENAMES):
        path = tmp_path / f"run_{index}" / "meta" / name
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"model_name": "resunet"}), encoding="utf-8")

    assert [path.name for path in ConfigNameKeyFixer(tmp_path).discover()] == list(ConfigNameKeyFixer.FILENAMES)


def test_missing_root_is_refused(tmp_path, monkeypatch):
    """Verifies constructing the fixer on a missing root directory is rejected."""
    monkeypatch.chdir(tmp_path)

    with pytest.raises(NotADirectoryError):
        ConfigNameKeyFixer(tmp_path / "absent", apply=True)
