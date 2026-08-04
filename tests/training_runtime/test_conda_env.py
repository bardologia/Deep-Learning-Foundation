from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tools.runtime.conda_env import CondaEnv


def test_candidates_returns_paths():
    candidates = CondaEnv._candidates("SomeEnv")

    assert all(isinstance(c, Path) for c in candidates)
    assert candidates


def test_candidates_include_home_install_layouts():
    candidates = CondaEnv._candidates("MyEnv")
    rendered   = [str(c) for c in candidates]

    assert any(str(Path.home() / "miniconda3" / "envs" / "MyEnv" / "bin" / "python") == r for r in rendered)
    assert any(str(Path.home() / "anaconda3" / "envs" / "MyEnv" / "bin" / "python") == r for r in rendered)


def test_candidates_target_env_name_python():
    candidates = CondaEnv._candidates("TargetEnv")
    assert all(c.name == "python" for c in candidates)
    assert all("TargetEnv" in str(c) for c in candidates)


def test_interpreter_missing_env_raises_filenotfound():
    with pytest.raises(FileNotFoundError):
        CondaEnv.interpreter("definitely_not_a_real_env_xyz_123")


def test_interpreter_returns_existing_candidate(monkeypatch):
    fake = Path("/fake/envs/Probe/bin/python")
    monkeypatch.setattr(CondaEnv, "_candidates", staticmethod(lambda name: [fake]))
    monkeypatch.setattr(Path, "exists", lambda self: self == fake)

    assert CondaEnv.interpreter("Probe") == fake


def test_interpreter_skips_nonexistent_first_candidate(monkeypatch):
    missing = Path("/missing/python")
    present = Path("/present/python")
    monkeypatch.setattr(CondaEnv, "_candidates", staticmethod(lambda name: [missing, present]))
    monkeypatch.setattr(Path, "exists", lambda self: self == present)

    assert CondaEnv.interpreter("Probe") == present
