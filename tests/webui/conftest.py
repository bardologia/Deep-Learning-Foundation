
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import pytest

from configuration.reference_layout import ReferenceLayout
from tools.runtime.machine_profile  import MachineProfile

REPO_ROOT  = Path(__file__).resolve().parents[2]
WEBUI_ROOT = REPO_ROOT / "webui"

if str(WEBUI_ROOT) not in sys.path:
    sys.path.insert(0, str(WEBUI_ROOT))

from gpu_activity    import GpuActivityLog
from host_history    import HostHistoryLog
from notifier        import JobNotifier
from politeness      import JobPoliteness
from process_manager import ProcessManager
from system_monitor  import ActiveUsers, SystemHistory, SystemMonitor
from web_logger      import WebLogger

SLEEP_LONG = "import time\ntime.sleep(30)\n"
ARGS_DUMP  = "import pathlib, sys\npathlib.Path('argv.txt').write_text(' '.join(sys.argv[1:]))\n"


class FakeHandler:

    def __init__(self, command: str, path: str, body: bytes = b"", headers: dict | None = None) -> None:
        self.command = command
        self.path    = path
        self.headers = {"Content-Length": str(len(body)), **(headers or {})}
        self.rfile   = io.BytesIO(body)
        self.wfile   = io.BytesIO()
        self.status  = 0
        self.sent    = {}

    def send_response(self, status: int) -> None:
        self.status = status

    def send_header(self, name: str, value: str) -> None:
        self.sent[name] = value

    def end_headers(self) -> None:
        return

    def payload(self) -> dict:
        return json.loads(self.wfile.getvalue().decode("utf-8"))


class StubPaths:

    def __init__(self, root: Path) -> None:
        self.profile        = MachineProfile(None)
        self.repo_root      = root
        self.main_dir       = root / "main"
        self.runs_dir       = root / ReferenceLayout.RUNS_ROOT
        self.data_root      = root / ReferenceLayout.DATA_ROOT
        self.logs_dir       = root / "logs"
        self.gpu_guard_dir  = root / "logs" / "gpu_guard"
        self.gpu_pools_dir  = root / "logs" / "gpu_pools"
        self.saved_runs_dir = root / "logs" / "saved_runs"
        self.sequences_dir  = root / "logs" / "sequences"

    def has_script(self, key: str) -> bool:
        return (self.main_dir / "analysis" / f"{key}.py").exists()

    def script_entry(self, key: str) -> dict:
        path = self.main_dir / "analysis" / f"{key}.py"
        return {"path": path, "rel": f"main/analysis/{key}.py"}


class StubNotifier:

    def __init__(self) -> None:
        self.alerts = []

    def system_alert(self, title: str, body: str, priority: str = "high") -> None:
        self.alerts.append((title, body, priority))


class StubDescriber:

    def describe(self, key: str, interpreter: str, overrides: dict | None) -> str:
        return f"stub description for {key}"


def job_record(manager: ProcessManager, job_id: str) -> dict:
    with manager.lock:
        return dict(manager.jobs[job_id])


def wait_for_status(manager: ProcessManager, job_id: str, status: str, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if job_record(manager, job_id)["status"] == status:
            return True
        time.sleep(0.05)
    return False


def wait_until_finished(manager: ProcessManager, job_id: str, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if job_record(manager, job_id)["status"] != "running":
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def monitor(tmp_path, monkeypatch):
    monkeypatch.setattr(SystemMonitor, "_du_loop", lambda self: None)
    monkeypatch.setattr(SystemHistory, "sample_loop", lambda self: None)
    monkeypatch.setattr(ActiveUsers, "sample_loop", lambda self: None)
    monkeypatch.setattr(GpuActivityLog, "sample_loop", lambda self: None)
    monkeypatch.setattr(HostHistoryLog, "sample_loop", lambda self: None)
    return SystemMonitor(StubPaths(tmp_path), WebLogger())


@pytest.fixture
def make_manager(tmp_path):
    built = []

    def build(scripts: dict) -> ProcessManager:
        directory = tmp_path / "main" / "analysis"
        directory.mkdir(parents=True)

        for name, body in scripts.items():
            (directory / f"{name}.py").write_text(body)

        paths   = StubPaths(tmp_path)
        logger  = WebLogger()
        manager = ProcessManager(paths, logger, JobNotifier(paths, logger), StubDescriber(), JobPoliteness(paths, logger))
        built.append(manager)

        return manager

    yield build

    for manager in built:
        manager.stop_all()
