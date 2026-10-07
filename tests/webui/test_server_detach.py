"""Tests for the console server's root isolation, shutdown and detach behaviour.

The server tests drive a real ``serve.py`` process over HTTP: they check that the
``--root`` option keeps logs out of the live repository, that shutdown exits the
front-end process together with every other console on the same root while sparing
consoles on other roots, and that a detached server shields itself, surviving SIGHUP,
SIGTERM and SIGINT while reporting its log file, holding scope and any shield failure.
"""

from __future__ import annotations

import json
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

from tests.webui.conftest import WEBUI_ROOT


def _free_port() -> int:
    """Returns a loopback TCP port the operating system currently reports as free."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _get(port: int, route: str) -> dict:
    """Returns the decoded JSON body of a GET against the local console."""
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{route}", timeout=5) as response:
        return json.loads(response.read())


def _post(port: int, route: str) -> dict:
    """Returns the decoded JSON body of an empty POST against the local console."""
    request = urllib.request.Request(f"http://127.0.0.1:{port}{route}", data=b"{}", method="POST")
    with urllib.request.urlopen(request, timeout=40) as response:
        return json.loads(response.read())


def _make_root(path: Path) -> Path:
    """Creates the directories ProjectPaths requires under a temporary root and returns it."""
    for name in ("main", "configuration", "tools"):
        (path / name).mkdir(parents=True)
    return path


def _launch(root: Path) -> tuple[int, subprocess.Popen]:
    """Starts serve.py on a free port under the given root and returns (port, process) once it answers."""
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
    """Terminates a server process, escalating to SIGKILL when it does not exit in time."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def _wait_exit(proc: subprocess.Popen, timeout: float = 15.0) -> int | None:
    """Returns the process's exit code once it exits within the timeout, or None when it is still running."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and proc.poll() is None:
        time.sleep(0.2)
    return proc.poll()


@pytest.fixture
def server(tmp_path):
    """Starts serve.py on a free port under a temporary root and yields (port, process, root)."""
    root       = _make_root(tmp_path / "project_root")
    port, proc = _launch(root)

    yield port, proc, root
    _stop(proc)


def test_server_root_isolates_the_console_from_the_live_repository(server):
    """A server started with --root reports no jobs and writes its log under that root."""
    port, _, root = server

    assert _get(port, "/api/jobs")["jobs"] == []

    system = _get(port, "/api/system")
    assert Path(system["server"]["log_path"]).parent == root / "logs"


def test_shutdown_stops_the_front_end_process(server):
    """The shutdown route names the server pid, reports no siblings when it runs alone, and exits cleanly."""
    port, proc, _ = server

    result = _post(port, "/api/system/shutdown")
    assert result["ok"] is True
    assert result["pid"] == proc.pid
    assert result["siblings"] == []

    assert _wait_exit(proc) == 0


def test_shutdown_kills_every_other_console_on_the_same_root(server):
    """The shutdown route force-kills the other console servers on the same root and reports their pids."""
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
    """A console serving a different root is not a sibling and keeps running after the shutdown."""
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
    """After detaching, the server shields itself against SIGHUP, SIGTERM and SIGINT and detaching again is idempotent."""
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
        assert result["scope"] == f"tomosar-webui-{proc.pid}.scope"
        cgroup = Path(f"/proc/{proc.pid}/cgroup").read_text()
        assert result["scope"] in cgroup

    for signum in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT):
        proc.send_signal(signum)
    time.sleep(1.0)

    after = _get(port, "/api/system")
    assert after["server"]["detached"] is True
    assert proc.poll() is None

    log_text = Path(result["log_path"]).read_text()
    assert "ignored by the keep-alive shield" in log_text

    again = _post(port, "/api/system/detach")
    assert again["ok"] is True and again["detached"] is True
