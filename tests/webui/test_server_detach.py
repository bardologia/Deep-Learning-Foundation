"""Tests for the console server's root isolation, shutdown and detach behaviour.

Each test drives a real ``serve.py`` process over HTTP: it checks that the ``--root``
option keeps logs out of the live repository, that shutdown exits only the front-end
process, and that a detached server shields itself, surviving SIGHUP, SIGTERM and
SIGINT while reporting its log file, holding scope and any shield failure.
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


@pytest.fixture
def server(tmp_path):
    """Starts serve.py on a free port under a temporary root and yields (port, process, root)."""
    port = _free_port()
    root = tmp_path / "project_root"
    for name in ("main", "configuration", "tools"):
        (root / name).mkdir(parents=True)
    proc = subprocess.Popen([sys.executable, str(WEBUI_ROOT / "serve.py"), "--port", str(port), "--root", str(root)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        try:
            _get(port, "/api/jobs")
            break
        except OSError:
            time.sleep(0.5)
    else:
        proc.kill()
        pytest.fail("server did not come up in time")

    yield port, proc, root
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def test_server_root_isolates_the_console_from_the_live_repository(server):
    """A server started with --root reports no jobs and writes its log under that root."""
    port, _, root = server

    assert _get(port, "/api/jobs")["jobs"] == []

    system = _get(port, "/api/system")
    assert Path(system["server"]["log_path"]).parent == root / "logs"


def test_shutdown_stops_only_the_front_end_process(server):
    """The shutdown route names the server pid and exits that process cleanly."""
    port, proc, _ = server

    result = _post(port, "/api/system/shutdown")
    assert result["ok"] is True
    assert result["pid"] == proc.pid

    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline and proc.poll() is None:
        time.sleep(0.2)

    assert proc.poll() == 0


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
