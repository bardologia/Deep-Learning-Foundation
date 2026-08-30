"""Tests for the per-job GPU energy meter.

Covers the memory-proportional split of device power across processes, the
attribution of shares to console jobs, the integration over the elapsed
interval, and the reset when GPU occupancy becomes unknown.
"""

from __future__ import annotations

import pytest

from energy_meter import JobEnergyMeter
from web_logger   import WebLogger


class StubSystem:
    """Stand-in system monitor serving a fixed GPU occupancy.

    Attributes:
        user      : Login name of the console user.
        occupancy : Device records returned by gpu_occupancy, or None for unknown.
    """

    def __init__(self) -> None:
        """Starts with no devices."""
        self.user      = "me"
        self.occupancy = []

    def gpu_occupancy(self) -> list[dict] | None:
        """Returns the scripted occupancy listing."""
        return self.occupancy


class StubProcesses:
    """Stand-in process manager recording booked energy.

    Attributes:
        jobs   : Job id keyed by pid.
        booked : Accumulated watt-hours keyed by job id.
    """

    def __init__(self) -> None:
        """Starts with no pid mapping and nothing booked."""
        self.jobs   = {}
        self.booked = {}

    def job_for_pid(self, pid: int) -> str | None:
        """Returns the job id owning a pid, or None when untracked."""
        return self.jobs.get(pid)

    def add_energy(self, job_id: str, wh: float) -> None:
        """Accumulates one energy slice for a job."""
        self.booked[job_id] = self.booked.get(job_id, 0.0) + wh


def _device(power: float | None, procs: list[dict]) -> dict:
    """Returns one occupancy record with the given power draw and processes."""
    return {"index": 0, "name": "A100", "util": 90.0, "mem_used": 30000, "mem_total": 40000, "temp": 60, "power": power, "power_limit": 300.0, "uuid": "GPU-0", "procs": procs}


@pytest.fixture
def meter():
    """Returns a meter over fresh system and process stubs."""
    return JobEnergyMeter(StubSystem(), StubProcesses(), WebLogger())


def test_power_splits_by_gpu_memory_share(meter):
    """Two of this user's processes split the device power by their GPU memory."""
    meter.system.occupancy = [_device(200.0, [
        {"pid": 10, "mem": 3000.0, "owner": "me", "cmd": "a"},
        {"pid": 20, "mem": 1000.0, "owner": "me", "cmd": "b"},
    ])]
    meter.processes.jobs = {10: "job-a", 20: "job-b"}

    watts = meter._job_watts(meter.system.occupancy)

    assert watts["job-a"] == pytest.approx(150.0)
    assert watts["job-b"] == pytest.approx(50.0)


def test_foreign_and_untracked_processes_get_nothing(meter):
    """Shares of other users' processes and untracked pids are not booked."""
    meter.system.occupancy = [_device(200.0, [
        {"pid": 10, "mem": 1000.0, "owner": "me", "cmd": "a"},
        {"pid": 30, "mem": 1000.0, "owner": "them", "cmd": "c"},
        {"pid": 40, "mem": 2000.0, "owner": "me", "cmd": "d"},
    ])]
    meter.processes.jobs = {10: "job-a"}

    watts = meter._job_watts(meter.system.occupancy)

    assert watts == {"job-a": pytest.approx(50.0)}


def test_sample_integrates_watts_over_the_elapsed_interval(meter, monkeypatch):
    """Two samples one hour apart book the attributed watts as watt-hours."""
    meter.system.occupancy = [_device(120.0, [{"pid": 10, "mem": 1000.0, "owner": "me", "cmd": "a"}])]
    meter.processes.jobs   = {10: "job-a"}

    clock = iter([0.0, 3600.0])
    monkeypatch.setattr("energy_meter.time.monotonic", lambda: next(clock))

    meter.sample()
    meter.sample()

    assert meter.processes.booked["job-a"] == pytest.approx(120.0)


def test_unknown_occupancy_resets_the_integration_window(meter, monkeypatch):
    """A gap with unknown occupancy is not integrated when the probe recovers."""
    meter.system.occupancy = [_device(120.0, [{"pid": 10, "mem": 1000.0, "owner": "me", "cmd": "a"}])]
    meter.processes.jobs   = {10: "job-a"}

    clock = iter([0.0, 7200.0])
    monkeypatch.setattr("energy_meter.time.monotonic", lambda: next(clock))

    meter.sample()
    occupancy              = meter.system.occupancy
    meter.system.occupancy = None
    meter.sample()

    meter.system.occupancy = occupancy
    monkeypatch.setattr("energy_meter.time.monotonic", lambda: 7215.0)
    meter.sample()

    assert meter.processes.booked == {}


def test_a_device_without_power_reading_is_skipped(meter):
    """A device reporting no power draw contributes nothing."""
    meter.system.occupancy = [_device(None, [{"pid": 10, "mem": 1000.0, "owner": "me", "cmd": "a"}])]
    meter.processes.jobs   = {10: "job-a"}

    assert meter._job_watts(meter.system.occupancy) == {}
