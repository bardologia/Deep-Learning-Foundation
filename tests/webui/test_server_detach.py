
from __future__ import annotations

from unittest.mock import Mock
import json
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

from process_manager import ServerDetacher
from web_logger      import WebLogger

from tests.webui.conftest import WEBUI_ROOT, StubPaths


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _get(port: int, route: str) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{route}", timeout=5) as response:
        return json.loads(response.read())


def _post(port: int, route: str) -> dict:
    request = urllib.request.Request(f"http://127.0.0.1:{port}{route}", data=b"{}", method="POST")
    with urllib.request.urlopen(request, timeout=40) as response:
        return json.loads(response.read())


def _make_root(path: Path) -> Path:
    for name in ("main", "configuration", "tools"):
        (path / name).mkdir(parents=True)
    return path


def _launch(root: Path) -> tuple[int, subprocess.Popen]:
    port = _free_port()
    proc = subprocess.Popen([sys.executable, str(WEBUI_ROOT / "serve.py"), "--port", str(port), "--root", str(root)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        try:
            _get(port, "/api/jobs")
            return port, proc
        except OSError:
            time.sleep(0.5)

    proc.kill()
    pytest.fail("server did not come up in time")


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def _wait_exit(proc: subprocess.Popen, timeout: float = 15.0) -> int | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and proc.poll() is None:
        time.sleep(0.2)
    return proc.poll()


@pytest.fixture
def server(tmp_path):
    root       = _make_root(tmp_path / "project_root")
    port, proc = _launch(root)

    yield port, proc, root
    _stop(proc)


def test_server_root_isolates_the_console_from_the_live_repository(server):
    port, _, root = server

    assert _get(port, "/api/jobs")["jobs"] == []

    system = _get(port, "/api/system")
    assert Path(system["server"]["log_path"]).parent == root / "logs"


def test_shutdown_stops_the_front_end_process(server):
    port, proc, _ = server

    result = _post(port, "/api/system/shutdown")
    assert result["ok"] is True
    assert result["pid"] == proc.pid
    assert result["siblings"] == []

    assert _wait_exit(proc) == 0


def test_shutdown_kills_every_other_console_on_the_same_root(server):
    port, proc, root = server
    _, other         = _launch(root)

    try:
        result = _post(port, "/api/system/shutdown")
        assert result["ok"] is True
        assert result["siblings"] == [other.pid]

        assert _wait_exit(other) == -signal.SIGKILL
        assert _wait_exit(proc)  == 0
    finally:
        _stop(other)


def test_shutdown_spares_consoles_on_other_roots(server, tmp_path):
    port, proc, _     = server
    other_port, other = _launch(_make_root(tmp_path / "other_root"))

    try:
        result = _post(port, "/api/system/shutdown")
        assert result["siblings"] == []
        assert _wait_exit(proc) == 0

        assert other.poll() is None
        assert _get(other_port, "/api/jobs")["jobs"] == []
    finally:
        _stop(other)


def test_detach_survives_termination_signals(server):
    port, proc, root = server

    before = _get(port, "/api/system")
    assert before["server"]["detached"] is False
    assert before["server"]["scope"] == "" and before["server"]["shield_error"] == ""

    result = _post(port, "/api/system/detach")
    assert result["ok"] is True
    assert result["detached"] is True
    assert Path(result["log_path"]) == root / "logs" / "webui_server.log"
    assert Path(result["log_path"]).exists()
    assert "scope" in result and "shield_error" in result

    if not result["shield_error"]:
        assert result["scope"] == f"ml-template-webui-{proc.pid}.scope"
        cgroup = Path(f"/proc/{proc.pid}/cgroup").read_text()
        assert result["scope"] in cgroup

    for signum in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT):
        proc.send_signal(signum)

    deadline = time.monotonic() + 5.0
    log_text = Path(result["log_path"]).read_text()
    while "ignored by the keep-alive shield" not in log_text and time.monotonic() < deadline:
        time.sleep(0.1)
        log_text = Path(result["log_path"]).read_text()

    after = _get(port, "/api/system")
    assert after["server"]["detached"] is True
    assert proc.poll() is None
    assert "ignored by the keep-alive shield" in log_text

    again = _post(port, "/api/system/detach")
    assert again["ok"] is True and again["detached"] is True


def test_unreadable_cgroup_file_lands_in_the_shield_error(tmp_path, monkeypatch):
    detacher = ServerDetacher(StubPaths(tmp_path), WebLogger())

    def unreadable(self, *args, **kwargs):
        Mock(side_effect=OSError("procfs unavailable"))()

    with monkeypatch.context() as patch:
        patch.setattr(subprocess, "run", lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, "", ""))
        patch.setattr(Path, "read_text", unreadable)
        detacher._escape_session()

    assert detacher.shield_error == "failed to read /proc/self/cgroup: procfs unavailable"
    assert detacher.scope        == ""


class CountingEvent:

    def __init__(self, detacher: ServerDetacher, rounds: int) -> None:
        self.detacher = detacher
        self.timeouts = []
        self.rounds   = rounds

    def wait(self, timeout: float | None = None) -> bool:
        self.timeouts.append(timeout)
        if len(self.timeouts) <= self.rounds:
            return False

        self.detacher.stopping.set()
        return True

    def clear(self) -> None:
        return None


def test_the_wait_loop_wakes_on_a_timeout_until_a_request_arrives(tmp_path):
    detacher           = ServerDetacher(StubPaths(tmp_path), WebLogger())
    detacher.requested = CountingEvent(detacher, rounds=3)

    detacher.wait_loop()

    assert detacher.requested.timeouts == [ServerDetacher.WAKE_S] * 4
    assert not detacher.applied.is_set()
