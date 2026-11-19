
from __future__ import annotations

import pytest

from energy_meter import JobEnergyMeter
from web_logger   import WebLogger


class StubSystem:

    def __init__(self) -> None:
        self.user      = "me"
        self.occupancy = []

    def gpu_occupancy(self) -> list[dict] | None:
        return self.occupancy


class StubProcesses:

    def __init__(self) -> None:
        self.jobs   = {}
        self.booked = {}

    def job_for_pid(self, pid: int) -> str | None:
        return self.jobs.get(pid)

    def add_energy(self, job_id: str, wh: float) -> None:
        self.booked[job_id] = self.booked.get(job_id, 0.0) + wh


def _device(power: float | None, procs: list[dict]) -> dict:
    return {"index": 0, "name": "A100", "util": 90.0, "mem_used": 30000, "mem_total": 40000, "temp": 60, "power": power, "power_limit": 300.0, "uuid": "GPU-0", "procs": procs}


@pytest.fixture
def meter():
    return JobEnergyMeter(StubSystem(), StubProcesses(), WebLogger())


def test_power_splits_by_gpu_memory_share(meter):
    meter.system.occupancy = [_device(200.0, [
        {"pid": 10, "mem": 3000.0, "owner": "me", "cmd": "a"},
        {"pid": 20, "mem": 1000.0, "owner": "me", "cmd": "b"},
    ])]
    meter.processes.jobs = {10: "job-a", 20: "job-b"}

    watts = meter._job_watts(meter.system.occupancy)

    assert watts["job-a"] == pytest.approx(150.0)
    assert watts["job-b"] == pytest.approx(50.0)


def test_foreign_and_untracked_processes_get_nothing(meter):
    meter.system.occupancy = [_device(200.0, [
        {"pid": 10, "mem": 1000.0, "owner": "me", "cmd": "a"},
        {"pid": 30, "mem": 1000.0, "owner": "them", "cmd": "c"},
        {"pid": 40, "mem": 2000.0, "owner": "me", "cmd": "d"},
    ])]
    meter.processes.jobs = {10: "job-a"}

    watts = meter._job_watts(meter.system.occupancy)

    assert watts == {"job-a": pytest.approx(50.0)}


def test_sample_integrates_watts_over_the_elapsed_interval(meter, monkeypatch):
    meter.system.occupancy = [_device(120.0, [{"pid": 10, "mem": 1000.0, "owner": "me", "cmd": "a"}])]
    meter.processes.jobs   = {10: "job-a"}

    clock = iter([0.0, 3600.0])
    monkeypatch.setattr("energy_meter.time.monotonic", lambda: next(clock))

    meter.sample()
    meter.sample()

    assert meter.processes.booked["job-a"] == pytest.approx(120.0)


def test_unknown_occupancy_resets_the_integration_window(meter, monkeypatch):
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
    meter.system.occupancy = [_device(None, [{"pid": 10, "mem": 1000.0, "owner": "me", "cmd": "a"}])]
    meter.processes.jobs   = {10: "job-a"}

    assert meter._job_watts(meter.system.occupancy) == {}
