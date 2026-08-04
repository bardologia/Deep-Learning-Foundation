from __future__ import annotations

import time

import pytest

from proc_stats        import CpuCounters
from resource_watchdog import ResourceWatchdog
from web_logger        import WebLogger


class StubProcesses:

    def __init__(self, running: int = 2) -> None:
        self.running = running
        self.calls   = 0

    def stop_all(self) -> int:
        self.calls  += 1
        stopped      = self.running
        self.running = 0
        return stopped


@pytest.fixture
def watchdog():
    return ResourceWatchdog(StubProcesses(), WebLogger())


def _alert(watchdog: ResourceWatchdog, kind: str) -> dict | None:
    return next((entry for entry in watchdog.state()["active"] if entry["kind"] == kind), None)


def _breach(watchdog: ResourceWatchdog, kind: str, times: int, level: str = "warn") -> None:
    for _ in range(times):
        watchdog._track(kind, level, True, False, f"{kind} is over the line")


def test_an_alert_needs_a_sustained_breach(watchdog):
    _breach(watchdog, "cpu", ResourceWatchdog.SUSTAIN["cpu"] - 1)
    assert _alert(watchdog, "cpu") is None

    _breach(watchdog, "cpu", 1)
    assert _alert(watchdog, "cpu")["message"] == "cpu is over the line"


def test_a_single_calm_sample_resets_the_streak(watchdog):
    _breach(watchdog, "cpu", ResourceWatchdog.SUSTAIN["cpu"] - 1)
    watchdog._track("cpu", "warn", False, False, "calm")
    _breach(watchdog, "cpu", ResourceWatchdog.SUSTAIN["cpu"] - 1)

    assert _alert(watchdog, "cpu") is None


def test_an_active_alert_is_updated_in_place(watchdog):
    _breach(watchdog, "ram", ResourceWatchdog.SUSTAIN["ram"])
    since = _alert(watchdog, "ram")["since"]

    watchdog._track("ram", "danger", True, False, "ram is critical")
    updated = _alert(watchdog, "ram")

    assert len(watchdog.state()["active"]) == 1
    assert updated["level"]   == "danger"
    assert updated["message"] == "ram is critical"
    assert updated["since"]   == since


def test_an_alert_survives_until_the_clear_margin_is_crossed(watchdog):
    _breach(watchdog, "ram", ResourceWatchdog.SUSTAIN["ram"])

    watchdog._track("ram", "warn", False, False, "still elevated")
    assert _alert(watchdog, "ram") is not None

    watchdog._track("ram", "warn", False, True, "back to normal")
    assert _alert(watchdog, "ram") is None


def test_the_ram_kill_needs_the_sustain_before_it_fires(watchdog):
    watchdog._check_kill(ResourceWatchdog.RAM_KILL + 1.0)

    assert watchdog.processes.calls == 0
    assert watchdog.state()["events"] == []

    watchdog._check_kill(ResourceWatchdog.RAM_KILL + 1.0)
    events = watchdog.state()["events"]

    assert watchdog.processes.calls == 1
    assert len(events)       == 1
    assert events[0]["kind"] == "ram_kill"
    assert "terminated 2 running job(s)" in events[0]["message"]


def test_ram_dropping_below_the_kill_line_resets_the_streak(watchdog):
    watchdog._check_kill(ResourceWatchdog.RAM_KILL + 1.0)
    watchdog._check_kill(ResourceWatchdog.RAM_KILL - 10.0)
    watchdog._check_kill(ResourceWatchdog.RAM_KILL + 1.0)

    assert watchdog.processes.calls == 0


def test_a_second_kill_waits_for_the_cooldown(watchdog):
    for _ in range(ResourceWatchdog.SUSTAIN["ram_kill"] + 4):
        watchdog._check_kill(ResourceWatchdog.RAM_KILL + 1.0)

    assert watchdog.processes.calls == 1

    watchdog.last_kill = time.monotonic() - ResourceWatchdog.KILL_COOLDOWN - 1.0
    watchdog._check_kill(ResourceWatchdog.RAM_KILL + 1.0)

    assert watchdog.processes.calls == 2


def test_a_kill_with_nothing_running_still_records_the_event(watchdog):
    watchdog.processes.running = 0

    for _ in range(ResourceWatchdog.SUSTAIN["ram_kill"]):
        watchdog._check_kill(ResourceWatchdog.RAM_KILL + 1.0)

    assert "no console jobs were running" in watchdog.state()["events"][0]["message"]


def test_cpu_percent_is_a_delta_between_two_samples(watchdog, monkeypatch):
    samples = [{"cpu": (100, 200)}, {"cpu": (150, 300)}]
    monkeypatch.setattr(CpuCounters, "read", staticmethod(lambda: samples.pop(0)))

    assert watchdog._cpu_percent() is None
    assert watchdog._cpu_percent() == pytest.approx(50.0)


def test_cpu_percent_ignores_a_stalled_counter(watchdog, monkeypatch):
    monkeypatch.setattr(CpuCounters, "read", staticmethod(lambda: {"cpu": (100, 200)}))

    assert watchdog._cpu_percent() is None
    assert watchdog._cpu_percent() is None


def test_evaluate_escalates_the_ram_alert_and_kills(watchdog, monkeypatch):
    monkeypatch.setattr(ResourceWatchdog, "_cpu_percent", lambda self: 10.0)
    monkeypatch.setattr(ResourceWatchdog, "_ram_percent", lambda self: ResourceWatchdog.RAM_WARN + 1.0)

    for _ in range(ResourceWatchdog.SUSTAIN["ram"]):
        watchdog._evaluate()

    assert _alert(watchdog, "ram")["level"] == "warn"
    assert _alert(watchdog, "cpu") is None
    assert watchdog.processes.calls == 0

    monkeypatch.setattr(ResourceWatchdog, "_ram_percent", lambda self: ResourceWatchdog.RAM_KILL + 1.0)
    for _ in range(ResourceWatchdog.SUSTAIN["ram_kill"]):
        watchdog._evaluate()

    assert _alert(watchdog, "ram")["level"] == "danger"
    assert watchdog.processes.calls == 1


def test_state_reports_the_limits_it_enforces(watchdog):
    limits = watchdog.state()["limits"]

    assert watchdog.state()["armed"] is False
    assert limits["ram_kill"]  == ResourceWatchdog.RAM_KILL
    assert limits["ram_warn"]  == ResourceWatchdog.RAM_WARN
    assert limits["cpu_alert"] == ResourceWatchdog.CPU_ALERT
    assert limits["cooldown"]  == ResourceWatchdog.KILL_COOLDOWN


def test_the_event_log_is_bounded(watchdog):
    for _ in range(60):
        watchdog.last_kill = 0.0
        watchdog.streaks["ram_kill"] = ResourceWatchdog.SUSTAIN["ram_kill"] - 1
        watchdog._check_kill(ResourceWatchdog.RAM_KILL + 1.0)

    assert len(watchdog.state()["events"]) == 20
