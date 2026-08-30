"""Tests for the webui contention monitor that flags machine-wide resource pressure.

Covers alarm bookkeeping (only alarms attributed to the current user are logged,
escalations are logged once, neighbour alarms clear silently), the separate PSI
full-stall threshold used for I/O, and the CPU rate accounting that must survive
a process disappearing between samples.
"""

from __future__ import annotations

import time

import pytest

from contention_monitor import ContentionMonitor
from web_logger         import WebLogger

from tests.webui.conftest import StubNotifier, StubPaths


class StubNuke:
    """Stand-in for the process nuker, counting how often it is triggered.

    Attributes:
        fired : Number of nuke invocations.
    """

    def __init__(self) -> None:
        """Initialises the trigger counter at zero."""
        self.fired = 0

    def nuke(self) -> dict:
        """Records a trigger and returns a successful no-op result."""
        self.fired += 1
        return {"ok": True, "signalled": 0, "killed": 0}


@pytest.fixture
def monitor(tmp_path):
    """Returns a contention monitor over stub paths and a stub nuker."""
    return ContentionMonitor(StubPaths(tmp_path), WebLogger(), StubNotifier(), StubNuke())


def _signals(io_full: float) -> dict:
    """Builds a saturated-disk signal sample.

    Args:
        io_full : PSI full-stall percentage for the I/O domain.

    Returns:
        Signal dictionary with quiet CPU and memory but 99 percent disk
        utilisation entirely attributed to the current user.
    """
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
    """Returns the active alarm of the given kind, or None if it is not raised."""
    return next((entry for entry in monitor.state()["active"] if entry["kind"] == kind), None)


def test_an_alarm_that_becomes_mine_is_recorded_and_logged(monitor):
    """Checks an alarm is logged only once it is attributed to the current user."""
    monitor._raise("mem", "warn", False, "neighbour holds the RAM")

    assert monitor.state()["events"] == []

    monitor._raise("mem", "warn", True, "you hold the RAM")
    events = monitor.state()["events"]

    assert len(events)         == 1
    assert events[0]["kind"]   == "mem"
    assert _alarm(monitor, "mem")["mine"] is True


def test_an_escalation_to_danger_is_recorded_once(monitor):
    """Checks repeated raises at the same level log nothing and only the escalation is added."""
    monitor._raise("mem", "warn", True, "you hold the RAM")
    monitor._raise("mem", "warn", True, "you hold the RAM")
    monitor._raise("mem", "danger", True, "you hold the RAM and the machine is stalling")
    monitor._raise("mem", "danger", True, "you hold the RAM and the machine is stalling")

    levels = [event["level"] for event in monitor.state()["events"]]
    assert levels == ["warn", "danger"]


def test_an_alarm_raised_for_a_neighbour_clears_without_a_log(monitor):
    """Checks a neighbour-owned alarm leaves no event behind when it drops."""
    monitor._raise("io", "warn", False, "neighbour drives the disk")
    monitor._drop("io")

    assert monitor.state()["active"] == []
    assert monitor.state()["events"] == []


def test_io_escalation_uses_its_own_full_threshold(monitor, monkeypatch):
    """Checks the I/O alarm level follows IO_PSI_FULL rather than the shared threshold."""
    for _ in range(ContentionMonitor.SUSTAIN):
        monitor._evaluate(_signals(io_full=4.0))

    assert _alarm(monitor, "io")["level"] == "danger"

    monkeypatch.setattr(ContentionMonitor, "IO_PSI_FULL", 50.0)
    monitor._evaluate(_signals(io_full=4.0))

    assert _alarm(monitor, "io")["level"] == "warn"


def test_an_exiting_process_does_not_zero_the_cpu_rate(monitor):
    """Checks a stale process entry from a previous sample does not drive the measured CPU share to zero."""
    monitor._sample()
    monitor.prev_mine[10 ** 8] = (10 ** 9, 10 ** 12)

    deadline = time.monotonic() + 0.3
    while time.monotonic() < deadline:
        pass

    signals = monitor._sample()

    assert signals["cpu"]["mine_pct"] > 0.0


def test_an_announced_alert_is_pushed_through_the_notifier(monitor):
    """An alert attributed to this user is pushed once when it is announced."""
    for _ in range(ContentionMonitor.SUSTAIN):
        monitor._track("io", "warn", True, False, True, "disk contention caused by you")

    assert monitor.notifier.alerts == [("neighbour impact: io", "disk contention caused by you", "default")]


def test_a_neighbours_alert_is_not_pushed(monitor):
    """An alert another user causes stays out of the push channel."""
    for _ in range(ContentionMonitor.SUSTAIN):
        monitor._track("io", "warn", True, False, False, "disk contention caused by them")

    assert monitor.notifier.alerts == []


def test_the_auto_nuke_pushes_an_urgent_notification(monitor):
    """A fired auto-nuke pushes one urgent notification."""
    monitor.auto_nuke = True
    for _ in range(ContentionMonitor.SUSTAIN):
        monitor._track("mem", "danger", True, False, True, "memory contention caused by you")
    for _ in range(ContentionMonitor.NUKE_SUSTAIN):
        monitor._check_nuke()

    urgent = [alert for alert in monitor.notifier.alerts if alert[2] == "urgent"]

    assert monitor.nuke.fired == 1
    assert len(urgent) == 1
    assert urgent[0][0] == "auto-nuke fired"
