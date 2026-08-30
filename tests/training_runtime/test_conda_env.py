"""Tests for conda environment interpreter discovery.

Covers how CondaEnv enumerates candidate interpreter paths for a named
environment and how it selects the first one that exists on disk.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tools.runtime.conda_env import CondaEnv


def test_candidates_returns_paths():
    """Verifies that candidate enumeration yields a non-empty list of Path objects."""
    candidates = CondaEnv._candidates("SomeEnv")

    assert all(isinstance(c, Path) for c in candidates)
    assert candidates


def test_candidates_include_home_install_layouts():
    """Verifies that both miniconda3 and anaconda3 home layouts are among the candidates."""
    candidates = CondaEnv._candidates("MyEnv")
    rendered   = [str(c) for c in candidates]

    assert any(str(Path.home() / "miniconda3" / "envs" / "MyEnv" / "bin" / "python") == r for r in rendered)
    assert any(str(Path.home() / "anaconda3" / "envs" / "MyEnv" / "bin" / "python") == r for r in rendered)


def test_candidates_target_env_name_python():
    """Verifies that every candidate points at the `python` binary of the requested env."""
    candidates = CondaEnv._candidates("TargetEnv")
    assert all(c.name == "python" for c in candidates)
    assert all("TargetEnv" in str(c) for c in candidates)


def test_interpreter_missing_env_raises_filenotfound():
    """Verifies that an environment with no existing interpreter raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        CondaEnv.interpreter("definitely_not_a_real_env_xyz_123")


def test_interpreter_returns_existing_candidate(monkeypatch):
    """Verifies that the single existing candidate is returned."""
    fake = Path("/fake/envs/Probe/bin/python")
    monkeypatch.setattr(CondaEnv, "_candidates", staticmethod(lambda name: [fake]))
    monkeypatch.setattr(Path, "exists", lambda self: self == fake)

    assert CondaEnv.interpreter("Probe") == fake


def test_interpreter_skips_nonexistent_first_candidate(monkeypatch):
    """Verifies that a missing first candidate is skipped in favour of an existing later one."""
    missing = Path("/missing/python")
    present = Path("/present/python")
    monkeypatch.setattr(CondaEnv, "_candidates", staticmethod(lambda name: [missing, present]))
    monkeypatch.setattr(Path, "exists", lambda self: self == present)

    assert CondaEnv.interpreter("Probe") == present
