
from __future__ import annotations

import os
import threading

import pytest

from contention_monitor import ContentionMonitor
from politeness         import JobPoliteness
from proc_stats         import UserSlices
from web_logger         import WebLogger

from tests.webui.conftest import StubNotifier, StubPaths


class StubNuke:

    def __init__(self) -> None:
        self.fired = 0

    def nuke(self) -> dict:
        self.fired += 1
        return {"ok": True, "signalled": 0, "killed": 0}


class StubThrottle:

    def __init__(self, ok: bool = True) -> None:
        self.calls  = 0
        self.result = {"ok": ok, "processes": 3, "renice": 0 if ok else 1, "ionice": 0, "error": "" if ok else "renice: failed to set priority"}

    def apply(self) -> dict:
        self.calls += 1
        return dict(self.result)


class FakeHost:

    def __init__(self, root) -> None:
        self.psi    = root / "pressure"
        self.slices = root / "user.slice"
        self.psi.mkdir()
        self.slices.mkdir()
        for name in ("cpu", "memory", "io"):
            (self.psi / name).write_text(self.pressure(0.0, 0.0))

    @staticmethod
    def pressure(some: float, full: float) -> str:
        return f"some avg10={some:.2f} avg60=0.00 avg300=0.00 total=0\nfull avg10={full:.2f} avg60=0.00 avg300=0.00 total=0\n"

    def host(self, resource: str, some: float, full: float = 0.0) -> None:
        (self.psi / resource).write_text(self.pressure(some, full))

    def user(self, uid: int, cpu_usec: int, anon: int = 0, cpu: tuple = (0.0, 0.0), mem: tuple = (0.0, 0.0), io: tuple = (0.0, 0.0)) -> None:
        base = self.slices / f"user-{uid}.slice"
        base.mkdir(exist_ok=True)
        (base / "cpu.stat").write_text(f"usage_usec {cpu_usec}\nuser_usec {cpu_usec}\nsystem_usec 0\n")
        (base / "memory.stat").write_text(f"anon {anon}\nfile 0\nshmem 0\n")
        (base / "cpu.pressure").write_text(self.pressure(*cpu))
        (base / "memory.pressure").write_text(self.pressure(*mem))
        (base / "io.pressure").write_text(self.pressure(*io))


ME       = os.getuid()
NEIGHBOUR = 424242


@pytest.fixture
def host(tmp_path, monkeypatch):
    fake = FakeHost(tmp_path)
    monkeypatch.setattr(UserSlices, "PSI_ROOT", str(fake.psi))
    monkeypatch.setattr(UserSlices, "ROOT", str(fake.slices))
    return fake


def _monitor(tmp_path, throttle: StubThrottle | None = None) -> ContentionMonitor:
    paths  = StubPaths(tmp_path / "repo")
    logger = WebLogger()
    return ContentionMonitor(paths, logger, StubNotifier(), StubNuke(), throttle or StubThrottle(), JobPoliteness(paths, logger))


@pytest.fixture
def monitor(tmp_path, host):
    return _monitor(tmp_path)


def _signals(io_full: float) -> dict:
    quiet = {"stall": 0.0, "full": 0.0, "user": None, "n": 0}
    return {
        "psi"    : {"cpu": {"some": 0.0, "full": 0.0}, "mem": {"some": 0.0, "full": 0.0}, "io": {"some": 99.0, "full": io_full}},
        "swap"   : {"in_mbs": 0.0, "out_mbs": 0.0},
        "mem"    : {"used_pct": 10.0, "mine_gb": 1.0, "mine_share": 0.1, "dominant": False},
        "io"     : {"device": "sda", "util": 99.0, "await_ms": 1.0, "read_mbs": 1.0, "write_mbs": 1.0, "total_mbs": 2.0, "mine_mbs": 2.0, "mine_share": 1.0, "dominant": True},
        "cpu"    : {"load_ratio": 0.1, "mine_pct": 1.0, "mine_share": 0.1, "dominant": False},
        "mine"   : {"nproc": 1, "mem_gb": 1.0, "stall": {"cpu": 0.0, "mem": 0.0, "io": 0.0}},
        "leak"   : None,
        "top"    : None,
        "users"  : [],
        "others" : {"cpu": dict(quiet), "mem": dict(quiet), "io": {"stall": 99.0, "full": io_full, "user": "neighbour", "n": 1}},
    }


def _alarm(monitor: ContentionMonitor, kind: str) -> dict | None:
    return next((entry for entry in monitor.state()["active"] if entry["kind"] == kind), None)


def _alarm_events(monitor: ContentionMonitor) -> list[dict]:
    return [event for event in monitor.state()["events"] if event["kind"] != "throttle"]


def _advance(monitor: ContentionMonitor, host: FakeHost, me_usec: int, other_usec: int, **pressure) -> dict:
    host.user(ME, me_usec, anon=4 << 30)
    host.user(NEIGHBOUR, other_usec, anon=1 << 30, **pressure)
    return monitor._sample()


def test_slice_rows_carry_cpu_memory_and_stall(host):
    host.user(NEIGHBOUR, 5_000_000, anon=3 << 30, cpu=(12.5, 1.0), io=(7.0, 0.0))

    rows = UserSlices.rows()

    assert len(rows) == 1
    assert rows[0]["uid"]      == NEIGHBOUR
    assert rows[0]["cpu_usec"] == 5_000_000
    assert rows[0]["mem"]      == 3 << 30
    assert rows[0]["stall"]    == {"cpu": {"some": 12.5, "full": 1.0}, "mem": {"some": 0.0, "full": 0.0}, "io": {"some": 7.0, "full": 0.0}}


def test_cpu_share_comes_from_the_slice_counters(monitor, host):
    _advance(monitor, host, 0, 0)
    sample = _advance(monitor, host, 6_000_000, 2_000_000)

    assert sample["cpu"]["mine_share"] == pytest.approx(0.75)
    assert sample["cpu"]["dominant"]  is True
    assert sample["others"]["cpu"]["user"] is not None


def test_the_worst_stalled_neighbour_is_reported_per_resource(monitor, host):
    _advance(monitor, host, 0, 0)
    sample = _advance(monitor, host, 1, 1, cpu=(45.0, 6.0), mem=(2.0, 0.0))

    assert sample["others"]["cpu"] == {"stall": 45.0, "full": 6.0, "user": sample["others"]["cpu"]["user"], "n": 1}
    assert sample["others"]["mem"]["n"] == 0
    assert sample["mine"]["stall"]      == {"cpu": 0.0, "mem": 0.0, "io": 0.0}


def test_a_stalled_neighbour_with_me_dominant_is_my_alarm(monitor, host):
    _advance(monitor, host, 0, 0)
    for step in range(1, ContentionMonitor.SUSTAIN + 1):
        monitor._evaluate(_advance(monitor, host, step * 9_000_000, step * 1_000_000, cpu=(50.0, 0.0)))

    alarm = _alarm(monitor, "cpu")

    assert alarm is not None
    assert alarm["mine"]  is True
    assert alarm["level"] == "warn"
    assert "waited for a core 50%" in alarm["message"]


def test_a_stalled_neighbour_with_another_dominant_is_not_mine(monitor, host):
    _advance(monitor, host, 0, 0)
    for step in range(1, ContentionMonitor.SUSTAIN + 1):
        monitor._evaluate(_advance(monitor, host, step * 1_000_000, step * 9_000_000, cpu=(50.0, 0.0)))

    alarm = _alarm(monitor, "cpu")

    assert alarm is not None
    assert alarm["mine"] is False
    assert monitor.state()["events"] == []


def test_host_pressure_without_a_stalled_neighbour_raises_nothing(monitor, host):
    host.host("cpu", 90.0)
    host.host("io", 90.0, 20.0)
    _advance(monitor, host, 0, 0)
    for step in range(1, ContentionMonitor.SUSTAIN + 2):
        monitor._evaluate(_advance(monitor, host, step * 9_000_000, step * 1_000_000))

    assert monitor.state()["active"] == []


def test_a_fully_stalled_neighbour_escalates_to_danger(monitor, host):
    _advance(monitor, host, 0, 0)
    for step in range(1, ContentionMonitor.SUSTAIN + 1):
        monitor._evaluate(_advance(monitor, host, step * 9_000_000, step * 1_000_000, cpu=(60.0, 10.0)))

    assert _alarm(monitor, "cpu")["level"] == "danger"


def test_missing_pressure_files_disable_the_monitor(tmp_path, host, monkeypatch):
    monkeypatch.setattr(UserSlices, "PSI_ROOT", str(tmp_path / "absent"))
    monitor = _monitor(tmp_path)

    monitor.start()

    assert monitor.state()["sources"]["psi"] is False
    assert monitor.state()["signals"] == {}
    assert "ContentionMonitor" not in [thread.name for thread in threading.enumerate()]


def test_an_alarm_that_becomes_mine_is_recorded_and_logged(monitor):
    monitor._raise("mem", "warn", False, "neighbour holds the RAM")

    assert monitor.state()["events"] == []

    monitor._raise("mem", "warn", True, "you hold the RAM")
    events = _alarm_events(monitor)

    assert len(events)         == 1
    assert events[0]["kind"]   == "mem"
    assert _alarm(monitor, "mem")["mine"] is True


def test_an_escalation_to_danger_is_recorded_once(monitor):
    monitor._raise("mem", "warn", True, "you hold the RAM")
    monitor._raise("mem", "warn", True, "you hold the RAM")
    monitor._raise("mem", "danger", True, "you hold the RAM and the neighbours are stalling")
    monitor._raise("mem", "danger", True, "you hold the RAM and the neighbours are stalling")

    levels = [event["level"] for event in _alarm_events(monitor)]
    assert levels == ["warn", "danger"]


def test_an_alarm_raised_for_a_neighbour_clears_without_a_log(monitor):
    monitor._raise("io", "warn", False, "neighbour drives the disk")
    monitor._drop("io")

    assert monitor.state()["active"] == []
    assert monitor.state()["events"] == []


def test_io_escalation_uses_its_own_full_threshold(monitor, monkeypatch):
    for _ in range(ContentionMonitor.SUSTAIN):
        monitor._evaluate(_signals(io_full=6.0))

    assert _alarm(monitor, "io")["level"] == "danger"

    monkeypatch.setattr(ContentionMonitor, "IO_FULL", 50.0)
    monitor._evaluate(_signals(io_full=6.0))

    assert _alarm(monitor, "io")["level"] == "warn"


def test_an_announced_alert_is_pushed_through_the_notifier(monitor):
    for _ in range(ContentionMonitor.SUSTAIN):
        monitor._track("io", "warn", True, False, True, "disk stall caused by you")

    pushed = [alert for alert in monitor.notifier.alerts if alert[0].startswith("neighbour impact")]

    assert pushed == [("neighbour impact: io", "disk stall caused by you", "default")]


def test_a_neighbours_alert_is_not_pushed(monitor):
    for _ in range(ContentionMonitor.SUSTAIN):
        monitor._track("io", "warn", True, False, False, "disk stall caused by them")

    assert monitor.notifier.alerts == []


def test_announcing_my_alarm_throttles_once(monitor):
    for _ in range(ContentionMonitor.SUSTAIN):
        monitor._track("cpu", "warn", True, False, True, "cpu stall caused by you")
    monitor._track("cpu", "warn", True, False, True, "cpu stall caused by you")
    monitor._track("cpu", "danger", True, False, True, "cpu stall caused by you, worse")

    events = [event for event in monitor.state()["events"] if event["kind"] == "throttle"]

    assert monitor.throttle.calls == 1
    assert len(events) == 1
    assert "lowered 3 processes" in events[0]["message"]
    assert monitor.state()["throttled"] == ["cpu"]
    assert ("auto-throttle", events[0]["message"], "default") in monitor.notifier.alerts


def test_a_dropped_alarm_can_throttle_again_when_it_returns(monitor):
    for _ in range(ContentionMonitor.SUSTAIN):
        monitor._track("mem", "warn", True, False, True, "memory stall caused by you")
    monitor._drop("mem")
    for _ in range(ContentionMonitor.SUSTAIN):
        monitor._track("mem", "warn", True, False, True, "memory stall caused by you")

    assert monitor.throttle.calls == 2
    assert monitor.state()["throttled"] == ["mem"]


def test_a_neighbours_alarm_never_throttles(monitor):
    for _ in range(ContentionMonitor.SUSTAIN):
        monitor._track("io", "danger", True, False, False, "disk stall caused by them")

    assert monitor.throttle.calls == 0
    assert monitor.state()["throttled"] == []


def test_a_leak_alarm_never_throttles(monitor):
    for _ in range(ContentionMonitor.SUSTAIN):
        monitor._track("leak", "warn", True, False, True, "process climbing")

    assert monitor.throttle.calls == 0


def test_auto_throttle_off_leaves_the_jobs_alone(monitor):
    monitor.politeness.configure({"auto_throttle": False})
    for _ in range(ContentionMonitor.SUSTAIN):
        monitor._track("cpu", "danger", True, False, True, "cpu stall caused by you")

    assert monitor.throttle.calls == 0
    assert _alarm(monitor, "cpu")["mine"] is True


def test_an_alarm_that_becomes_mine_throttles_at_the_flip(monitor):
    for _ in range(ContentionMonitor.SUSTAIN):
        monitor._track("cpu", "warn", True, False, False, "cpu stall caused by them")

    assert monitor.throttle.calls == 0

    monitor._track("cpu", "warn", True, False, True, "cpu stall caused by you")

    assert monitor.throttle.calls == 1


def test_a_failed_throttle_is_recorded_as_danger(tmp_path, host):
    monitor = _monitor(tmp_path, StubThrottle(ok=False))
    for _ in range(ContentionMonitor.SUSTAIN):
        monitor._track("cpu", "warn", True, False, True, "cpu stall caused by you")

    events = [event for event in monitor.state()["events"] if event["kind"] == "throttle"]

    assert len(events) == 1
    assert events[0]["level"] == "danger"
    assert "throttle FAILED" in events[0]["message"]
    assert ("auto-throttle", events[0]["message"], "high") in monitor.notifier.alerts


def test_the_state_carries_the_politeness_settings(monitor):
    state = monitor.state()["politeness"]

    assert state["level"]         == "polite"
    assert state["auto_throttle"] is True
    assert state["prefix"]        == "nice -n 10 ionice -c 2 -n 7"


def test_the_auto_nuke_pushes_an_urgent_notification(monitor):
    monitor.auto_nuke = True
    for _ in range(ContentionMonitor.SUSTAIN):
        monitor._track("mem", "danger", True, False, True, "memory stall caused by you")
    for _ in range(ContentionMonitor.NUKE_SUSTAIN):
        monitor._check_nuke()

    urgent = [alert for alert in monitor.notifier.alerts if alert[2] == "urgent"]

    assert monitor.nuke.fired == 1
    assert len(urgent) == 1
    assert urgent[0][0] == "auto-nuke fired"
