from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.webui.conftest import job_record

DETACHER = "import os, subprocess, sys\nchild = os.path.join(os.getcwd(), 'child.py')\nlog = os.path.join(os.getcwd(), 'detached.out')\nsink = open(log, 'ab')\nproc = subprocess.Popen([sys.executable, '-u', child], stdout=sink, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)\nprint(f'detached fake as pid {proc.pid}, immune to hangup')\nprint(f'output: {log}')\n"
TICKER   = "import sys, time\nfor index in range(12):\n    sys.stdout.write(f'tick {index}\\n')\n    sys.stdout.flush()\n    time.sleep(0.15)\n"
CRASHER  = "raise SystemExit('boom, the run died on startup')\n"
SLEEPER  = "import time\ntime.sleep(120)\n"


@pytest.fixture
def manager(make_manager, tmp_path):
    (tmp_path / "child.py").write_text(TICKER)
    instance                  = make_manager({"fake_detach": DETACHER})
    instance.ORPHAN_MIN_AGE_S = 0.0
    instance.ORPHAN_RESCAN_S  = 0.0
    return instance


@pytest.fixture
def spawned():
    procs = []
    yield procs
    for proc in procs:
        proc.kill()
        proc.wait()


def stream_text(manager, job_id: str) -> str:
    stream = manager.streams[job_id]
    with stream.lock:
        events = list(stream.buffer)
    return "".join(event["data"] for event in events if event.get("type") == "chunk")


def wait_for_text(manager, job_id: str, needle: str, timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if needle in stream_text(manager, job_id):
            return True
        time.sleep(0.1)
    return False


def wait_for_log_path(manager, job_id: str, timeout: float = 20.0) -> str | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        path = job_record(manager, job_id).get("log_path")
        if path:
            return path
        time.sleep(0.1)
    return None


def test_detached_run_streams_its_log_into_the_job_stream(manager, tmp_path):
    job_id = manager.launch("fake_detach", sys.executable, detach=True)["job_id"]

    assert wait_for_log_path(manager, job_id) == str(tmp_path / "detached.out")
    assert wait_for_text(manager, job_id, "tick 0")
    assert wait_for_text(manager, job_id, "tick 11")


def test_detached_stream_ends_only_after_the_log_is_drained(manager):
    job_id = manager.launch("fake_detach", sys.executable, detach=True)["job_id"]

    assert wait_for_log_path(manager, job_id)

    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        if job_record(manager, job_id)["status"] == "finished":
            break
        time.sleep(0.1)

    assert job_record(manager, job_id)["status"] == "finished"
    assert "tick 11" in stream_text(manager, job_id)


def test_detached_run_that_dies_immediately_still_shows_its_output(manager, tmp_path):
    (tmp_path / "child.py").write_text(CRASHER)

    job_id = manager.launch("fake_detach", sys.executable, detach=True)["job_id"]

    assert wait_for_log_path(manager, job_id) == str(tmp_path / "detached.out")
    assert wait_for_text(manager, job_id, "boom, the run died on startup")


def test_adopted_orphan_streams_its_redirected_log(manager, spawned, tmp_path):
    script = tmp_path / "main" / "analysis" / "orphan_tick.py"
    script.write_text(TICKER)
    log = tmp_path / "orphan.out"

    with open(log, "ab") as sink:
        proc = subprocess.Popen([sys.executable, "-u", str(script)], stdout=sink, stderr=subprocess.STDOUT, start_new_session=True)
    spawned.append(proc)
    time.sleep(0.3)

    assert manager.adopt_orphans() == 1

    job_id = manager.list_jobs()[0]["job_id"]
    assert job_record(manager, job_id)["log_path"] == str(log)
    assert wait_for_text(manager, job_id, "tick 11")


def test_orphan_on_a_terminal_or_pipe_reports_no_log(manager, spawned, tmp_path):
    script = tmp_path / "main" / "analysis" / "orphan_sleep.py"
    script.write_text(SLEEPER)
    proc = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, start_new_session=True)
    spawned.append(proc)
    time.sleep(0.3)

    assert manager.adopt_orphans() == 1

    job_id = manager.list_jobs()[0]["job_id"]
    assert job_record(manager, job_id)["log_path"] is None
    assert manager._pid_log_path(proc.pid) is None


def test_catchup_on_a_large_existing_log_is_bounded(manager, spawned, tmp_path):
    from process_manager import LogTailer

    script = tmp_path / "main" / "analysis" / "orphan_tick.py"
    script.write_text(TICKER)
    log = tmp_path / "big.out"
    log.write_bytes(b"stale\n" * 400000)

    assert log.stat().st_size > 2 * LogTailer.CATCHUP_MAX

    with open(log, "ab") as sink:
        proc = subprocess.Popen([sys.executable, "-u", str(script)], stdout=sink, stderr=subprocess.STDOUT, start_new_session=True)
    spawned.append(proc)
    time.sleep(0.3)

    assert manager.adopt_orphans() == 1

    job_id = manager.list_jobs()[0]["job_id"]
    assert wait_for_text(manager, job_id, "tick 11")
    assert len(stream_text(manager, job_id)) < 2 * LogTailer.CATCHUP_MAX


def test_job_log_returns_the_flattened_transcript(manager, tmp_path):
    job_id = manager.launch("fake_detach", sys.executable, detach=True)["job_id"]

    assert wait_for_log_path(manager, job_id)
    assert wait_for_text(manager, job_id, "tick 11")

    payload = manager.job_log(job_id)

    assert payload["ok"] and payload["available"]
    assert payload["path"] == str(tmp_path / "detached.out")
    assert payload["truncated"] is False
    assert "tick 0" in payload["text"] and "tick 11" in payload["text"]
    assert "\x1b" not in payload["text"]


def test_job_log_reports_when_there_is_no_file(manager, spawned, tmp_path):
    script = tmp_path / "main" / "analysis" / "orphan_sleep.py"
    script.write_text(SLEEPER)
    proc = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, start_new_session=True)
    spawned.append(proc)
    time.sleep(0.3)

    assert manager.adopt_orphans() == 1

    payload = manager.job_log(manager.list_jobs()[0]["job_id"])

    assert payload["ok"] is True
    assert payload["available"] is False
    assert "pipe" in payload["reason"]


def test_job_log_serves_the_tail_of_a_large_file(manager, tmp_path):
    job_id = manager.launch("fake_detach", sys.executable, detach=True)["job_id"]

    assert wait_for_log_path(manager, job_id)
    assert wait_for_text(manager, job_id, "tick 11")

    log = tmp_path / "detached.out"
    with open(log, "ab") as sink:
        sink.write(b"THE_TAIL_MARKER\n")
        sink.write(b"filler line\n" * 400000)
        sink.write(b"THE_LAST_MARKER\n")

    payload = manager.job_log(job_id)

    assert payload["truncated"] is True
    assert payload["shown"] == manager.LOG_VIEW_BYTES
    assert "THE_LAST_MARKER" in payload["text"]
    assert "THE_TAIL_MARKER" not in payload["text"]


def test_job_log_rejects_an_unknown_job(manager):
    assert manager.job_log("nope") == {"ok": False, "error": "unknown job"}


def test_log_path_resolves_only_regular_files(manager, tmp_path):
    log = tmp_path / "resolve.out"

    with open(log, "ab") as sink:
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(2)"], stdout=sink, start_new_session=True)

    assert manager._pid_log_path(proc.pid) == Path(log)

    proc.kill()
    proc.wait()

    assert manager._pid_log_path(proc.pid) is None
