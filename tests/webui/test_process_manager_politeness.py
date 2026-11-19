
from __future__ import annotations

import os
import sys
import time

import pytest


SLEEPER = "import time\ntime.sleep(20)\n"


def _nice_of(pid: int) -> int:
    raw = open(f"/proc/{pid}/stat").read()
    return int(raw[raw.rindex(")") + 2:].split()[16])


def _raised(increment: int) -> int:
    return min(19, _nice_of(os.getpid()) + increment)


def _comm_of(pid: int) -> str:
    raw = open(f"/proc/{pid}/stat").read()
    return raw[raw.index("(") + 1:raw.rindex(")")]


def _launched(manager, level: str | None = None) -> dict:
    if level is not None:
        manager.politeness.configure({"level": level})

    job = manager.launch("sleeper", sys.executable)
    assert job["ok"] is True, job

    with manager.lock:
        record = dict(manager.jobs[job["job_id"]])
    assert record["pid"], record
    _wait_for_interpreter(record["pid"])
    return record


def _wait_for_interpreter(pid: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _comm_of(pid).startswith("python"):
            return
        time.sleep(0.02)
    raise AssertionError(f"pid {pid} is still {_comm_of(pid)!r} after {timeout} s")


@pytest.fixture
def manager(make_manager):
    return make_manager({"sleeper": SLEEPER})


def test_the_rendered_command_carries_the_prefix(manager):
    record = manager._make_record("sleeper", sys.executable, {"a": "1"})

    assert record["command"].startswith("nice -n 10 ionice -c 2 -n 7 ")
    assert record["command"].endswith(f"{sys.executable} -u main/analysis/sleeper.py --a 1")


def test_the_rendered_command_has_no_prefix_when_off(manager):
    manager.politeness.configure({"level": "off"})
    record = manager._make_record("sleeper", sys.executable, {})

    assert record["command"] == f"{sys.executable} -u main/analysis/sleeper.py"


@pytest.mark.parametrize("level, nice", [("polite", 10), ("background", 19)])
def test_a_launched_child_runs_at_the_levels_nice_value(manager, level, nice):
    record = _launched(manager, level)

    assert _nice_of(record["pid"]) == _raised(nice)
    assert _comm_of(record["pid"]).startswith("python")


def test_a_launched_child_inherits_the_console_nice_when_off(manager):
    record = _launched(manager, "off")

    assert _nice_of(record["pid"]) == _nice_of(os.getpid())


def test_changing_the_level_affects_only_later_launches(manager):
    first  = _launched(manager)
    second = _launched(manager, "background")

    assert _nice_of(first["pid"])  == _raised(10)
    assert _nice_of(second["pid"]) == _raised(19)
