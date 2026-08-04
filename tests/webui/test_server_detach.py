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
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _get(port: int, route: str) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{route}", timeout=5) as response:
        return json.loads(response.read())


def _post(port: int, route: str) -> dict:
    request = urllib.request.Request(f"http://127.0.0.1:{port}{route}", data=b"{}", method="POST")
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


@pytest.fixture
def server():
    port = _free_port()
    proc = subprocess.Popen([sys.executable, str(WEBUI_ROOT / "serve.py"), "--port", str(port)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

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

    yield port, proc
    proc.terminate()
    proc.wait(timeout=10)


def test_shutdown_stops_only_the_front_end_process(server):
    port, proc = server

    result = _post(port, "/api/system/shutdown")
    assert result["ok"] is True
    assert result["pid"] == proc.pid

    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline and proc.poll() is None:
        time.sleep(0.2)

    assert proc.poll() == 0


def test_detach_survives_hangup(server):
    port, proc = server

    before = _get(port, "/api/system")
    assert before["server"]["detached"] is False

    result = _post(port, "/api/system/detach")
    assert result["ok"] is True
    assert result["detached"] is True
    assert Path(result["log_path"]).exists()

    proc.send_signal(signal.SIGHUP)
    time.sleep(1.0)

    after = _get(port, "/api/system")
    assert after["server"]["detached"] is True
    assert proc.poll() is None

    again = _post(port, "/api/system/detach")
    assert again["ok"] is True and again["detached"] is True
