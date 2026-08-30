"""Tests for the detacher that re-execs an entry point as a background process."""

from __future__ import annotations

import pytest

import tools.runtime.detacher as module
from tools.runtime.detacher import Detacher


class _Process:
    """Popen stub exposing only the pid the detacher reports."""

    pid = 4242


def _ensure_with_captured_env(tmp_path, monkeypatch) -> dict:
    """Runs Detacher.ensure past the re-exec and returns the child environment it built."""
    captured = {}

    def _popen(command, env=None, **kwargs):
        captured["env"] = env
        return _Process()

    monkeypatch.setattr(module.subprocess, "Popen", _popen)
    monkeypatch.setattr(module.sys, "argv", ["train_backbone.py", "--gpu", "1", "--detach"])
    monkeypatch.delenv(Detacher.ENV_FLAG, raising=False)

    with pytest.raises(SystemExit):
        Detacher(log_dir=str(tmp_path)).ensure()

    return captured["env"]


def test_ensure_drops_the_pin_when_the_parent_inherited_no_mask(tmp_path, monkeypatch):
    """The re-exec child of a bare-machine parent sees no CUDA mask, so --gpu addresses physical devices."""
    monkeypatch.setenv("TOMOSAR_INHERITED_CUDA_MASK", "")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")

    env = _ensure_with_captured_env(tmp_path, monkeypatch)

    assert "CUDA_VISIBLE_DEVICES" not in env
    assert "TOMOSAR_INHERITED_CUDA_MASK" not in env
    assert env[Detacher.ENV_FLAG] == "1"


def test_ensure_restores_an_inherited_allocation_mask(tmp_path, monkeypatch):
    """The re-exec child of a scheduler-allocated parent sees the allocation's full mask."""
    monkeypatch.setenv("TOMOSAR_INHERITED_CUDA_MASK", "2,3")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")

    env = _ensure_with_captured_env(tmp_path, monkeypatch)

    assert env["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert "TOMOSAR_INHERITED_CUDA_MASK" not in env
