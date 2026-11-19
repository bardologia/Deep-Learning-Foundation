
from __future__ import annotations

import subprocess

import pytest

from system_monitor import GpuProbe
from web_logger     import WebLogger


class FakeRun:

    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout     = stdout
        self.stderr     = stderr


DEVICE_LINE = "0, A100, 50, 8000, 40000, 45, 100, 300, GPU-0"


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/nvidia-smi")
    return GpuProbe(WebLogger())


def test_a_working_probe_returns_the_parsed_rows(probe, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: FakeRun(0, DEVICE_LINE))

    assert probe.devices() == [DEVICE_LINE.split(", ")]
    assert probe.failed is False


def test_a_probe_that_never_answers_reports_unknown_not_empty(probe, monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=GpuProbe.TIMEOUT_S)

    monkeypatch.setattr(subprocess, "run", timeout)

    assert probe.devices() is None
    assert probe.failed is True


def test_a_failing_exit_code_reports_unknown_not_empty(probe, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: FakeRun(9, "", "driver mismatch"))

    assert probe.devices() is None
    assert probe.apps()    is None


def test_a_machine_without_the_tool_has_no_devices_rather_than_a_failure(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    probe = GpuProbe(WebLogger())

    assert probe.devices() == []
    assert probe.failed is False


def test_the_probe_recovers_once_nvidia_smi_answers(probe, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: FakeRun(9, "", "driver mismatch"))
    assert probe.devices() is None

    probe.cache.clear()
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: FakeRun(0, DEVICE_LINE))

    assert probe.devices() == [DEVICE_LINE.split(", ")]
    assert probe.failed is False


def test_a_failed_probe_makes_the_snapshot_declare_the_gpus_unknown(monitor, monkeypatch):
    monkeypatch.setattr(monitor, "gpu_occupancy", lambda: None)
    snapshot = monitor.snapshot()

    assert snapshot["gpus_known"] is False
    assert snapshot["gpus"]       == []


def test_a_working_probe_declares_the_gpus_known(monitor, monkeypatch):
    monkeypatch.setattr(monitor, "gpu_occupancy", lambda: [])
    snapshot = monitor.snapshot()

    assert snapshot["gpus_known"] is True
    assert snapshot["gpus"]       == []
