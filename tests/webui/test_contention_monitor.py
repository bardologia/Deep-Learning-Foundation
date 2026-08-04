from __future__ import annotations

import time

import pytest

from contention_monitor import ContentionMonitor
from web_logger         import WebLogger

from tests.webui.conftest import StubPaths


class StubNuke:

    def __init__(self) -> None:
        self.fired = 0

    def nuke(self) -> dict:
        self.fired += 1
        return {"ok": True, "signalled": 0, "killed": 0}


@pytest.fixture
def monitor(tmp_path):
    return ContentionMonitor(StubPaths(tmp_path), WebLogger(), StubNuke())


def _signals(io_full: float) -> dict:
    return {
        "psi"  : {"cpu": {"some": 0.0, "full": 0.0}, "mem": {"some": 0.0, "full": 0.0}, "io": {"some": 99.0, "full": io_full}},
        "swap" : {"in_mbs": 0.0, "out_mbs": 0.0},
        "mem"  : {"used_pct": 10.0, "mine_gb": 1.0, "mine_share": 0.1},
        "io"   : {"device": "sda", "util": 99.0, "await_ms": 1.0, "read_mbs": 1.0, "write_mbs": 1.0, "total_mbs": 2.0, "mine_mbs": 2.0, "mine_share": 1.0},
        "cpu"  : {"load_ratio": 0.1, "mine_pct": 1.0, "mine_share": 0.1},
        "mine" : {"nproc": 1, "mem_gb": 1.0},
        "leak" : None,
        "top"  : None,
    }


def _alarm(monitor: ContentionMonitor, kind: str) -> dict | None:
    return next((entry for entry in monitor.state()["active"] if entry["kind"] == kind), None)


def test_an_alarm_that_becomes_mine_is_recorded_and_logged(monitor):
    monitor._raise("mem", "warn", False, "neighbour holds the RAM")

    assert monitor.state()["events"] == []

    monitor._raise("mem", "warn", True, "you hold the RAM")
    events = monitor.state()["events"]

    assert len(events)         == 1
    assert events[0]["kind"]   == "mem"
    assert _alarm(monitor, "mem")["mine"] is True


def test_an_escalation_to_danger_is_recorded_once(monitor):
    monitor._raise("mem", "warn", True, "you hold the RAM")
    monitor._raise("mem", "warn", True, "you hold the RAM")
    monitor._raise("mem", "danger", True, "you hold the RAM and the machine is stalling")
    monitor._raise("mem", "danger", True, "you hold the RAM and the machine is stalling")

    levels = [event["level"] for event in monitor.state()["events"]]
    assert levels == ["warn", "danger"]


def test_an_alarm_raised_for_a_neighbour_clears_without_a_log(monitor):
    monitor._raise("io", "warn", False, "neighbour drives the disk")
    monitor._drop("io")

    assert monitor.state()["active"] == []
    assert monitor.state()["events"] == []


def test_io_escalation_uses_its_own_full_threshold(monitor, monkeypatch):
    for _ in range(ContentionMonitor.SUSTAIN):
        monitor._evaluate(_signals(io_full=4.0))

    assert _alarm(monitor, "io")["level"] == "danger"

    monkeypatch.setattr(ContentionMonitor, "IO_PSI_FULL", 50.0)
    monitor._evaluate(_signals(io_full=4.0))

    assert _alarm(monitor, "io")["level"] == "warn"


def test_an_exiting_process_does_not_zero_the_cpu_rate(monitor):
    monitor._sample()
    monitor.prev_mine[10 ** 8] = (10 ** 9, 10 ** 12)

    deadline = time.monotonic() + 0.3
    while time.monotonic() < deadline:
        pass

    signals = monitor._sample()

    assert signals["cpu"]["mine_pct"] > 0.0
