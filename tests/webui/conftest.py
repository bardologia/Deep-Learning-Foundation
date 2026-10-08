"""Shared fixtures and builders for the webui test suite.

Puts the ``webui`` package on ``sys.path`` and provides HTTP-handler and path
stubs, synthetic preprocessing runs, inference stamps and cross-validation fold
trees on disk, plus helpers that drive CubeExplorer loading and that build a
ProcessManager over throwaway scripts.
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from configuration.reference_layout import ReferenceLayout
from tools.runtime.machine_profile  import MachineProfile

REPO_ROOT  = Path(__file__).resolve().parents[2]
WEBUI_ROOT = REPO_ROOT / "webui"

if str(WEBUI_ROOT) not in sys.path:
    sys.path.insert(0, str(WEBUI_ROOT))

from cube_explorer   import CubeExplorer
from gpu_activity    import GpuActivityLog
from host_history    import HostHistoryLog
from notifier        import JobNotifier
from process_manager import ProcessManager
from system_monitor  import ActiveUsers, SystemHistory, SystemMonitor
from web_logger      import WebLogger

N_Z, N_ROWS, N_COLS, N_SLOTS = 5, 8, 6, 2
ACTIVITY_THRESHOLD           = 1e-4

SLEEP_LONG = "import time\ntime.sleep(30)\n"
ARGS_DUMP  = "import pathlib, sys\npathlib.Path('argv.txt').write_text(' '.join(sys.argv[1:]))\n"


class FakeHandler:
    """Stand-in for a BaseHTTPRequestHandler capturing the response it is sent.

    Attributes:
        command : HTTP method of the simulated request.
        path    : Request path including any query string.
        headers : Request headers, always carrying Content-Length.
        rfile   : Readable stream over the request body.
        wfile   : Writable stream collecting the response body.
        status  : Status code passed to send_response.
        sent    : Response headers passed to send_header, keyed by name.
    """

    def __init__(self, command: str, path: str, body: bytes = b"", headers: dict | None = None) -> None:
        """Builds a simulated request.

        Args:
            command : HTTP method, e.g. ``"GET"`` or ``"POST"``.
            path    : Request path.
            body    : Raw request body.
            headers : Extra request headers merged over Content-Length.
        """
        self.command = command
        self.path    = path
        self.headers = {"Content-Length": str(len(body)), **(headers or {})}
        self.rfile   = io.BytesIO(body)
        self.wfile   = io.BytesIO()
        self.status  = 0
        self.sent    = {}

    def send_response(self, status: int) -> None:
        """Records the response status code."""
        self.status = status

    def send_header(self, name: str, value: str) -> None:
        """Records a response header."""
        self.sent[name] = value

    def end_headers(self) -> None:
        """Ends the header block, which the fake does not need to act on."""
        return

    def payload(self) -> dict:
        """Returns the response body decoded as JSON."""
        return json.loads(self.wfile.getvalue().decode("utf-8"))


class StubPaths:
    """Minimal stand-in for the webui path registry rooted at a temporary directory.

    Attributes:
        profile        : Absent machine profile, so no root or hardware rewrite applies.
        repo_root      : Root of the fake repository.
        main_dir       : Directory holding entry-point scripts.
        runs_dir       : Root of the run-output tree.
        data_root      : Dataset root of the fake repository.
        logs_dir       : Root of the log tree.
        gpu_guard_dir  : Directory for GPU guard state files.
        gpu_pools_dir  : Directory for GPU pool state files.
        saved_runs_dir : Directory for saved run definitions.
        sequences_dir  : Directory for experiment sequence definitions.
    """

    def __init__(self, root: Path) -> None:
        """Derives the standard subdirectories from a root path."""
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
        """Returns whether an analysis script exists for the given catalog key."""
        return (self.main_dir / "analysis" / f"{key}.py").exists()

    def script_entry(self, key: str) -> dict:
        """Returns the absolute path and repo-relative path of an analysis script."""
        path = self.main_dir / "analysis" / f"{key}.py"
        return {"path": path, "rel": f"main/analysis/{key}.py"}


class StubNotifier:
    """Notifier stand-in recording every pushed system alert.

    Attributes:
        alerts : Recorded (title, body, priority) tuples, oldest first.
    """

    def __init__(self) -> None:
        """Starts with no recorded alerts."""
        self.alerts = []

    def system_alert(self, title: str, body: str, priority: str = "high") -> None:
        """Records one pushed alert."""
        self.alerts.append((title, body, priority))


class StubDescriber:
    """Job describer stand-in returning a fixed sentence per catalog key."""

    def describe(self, key: str, interpreter: str, overrides: dict | None) -> str:
        """Returns a constant description mentioning the catalog key."""
        return f"stub description for {key}"


def job_record(manager: ProcessManager, job_id: str) -> dict:
    """Returns a snapshot copy of a job record taken under the manager lock."""
    with manager.lock:
        return dict(manager.jobs[job_id])


def wait_for_status(manager: ProcessManager, job_id: str, status: str, timeout: float = 10.0) -> bool:
    """Polls until a job reaches a given status.

    Args:
        manager : Process manager owning the job.
        job_id  : Identifier of the job to poll.
        status  : Status value being waited for.
        timeout : Maximum wait in seconds.

    Returns:
        True if the status was observed before the timeout.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if job_record(manager, job_id)["status"] == status:
            return True
        time.sleep(0.05)
    return False


def wait_until_finished(manager: ProcessManager, job_id: str, timeout: float = 10.0) -> bool:
    """Polls until a job leaves the running state.

    Args:
        manager : Process manager owning the job.
        job_id  : Identifier of the job to poll.
        timeout : Maximum wait in seconds.

    Returns:
        True if the job stopped running before the timeout.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if job_record(manager, job_id)["status"] != "running":
            return True
        time.sleep(0.05)
    return False


def make_preproc(base: Path, rng: np.random.Generator | None = None, with_full: bool = False) -> Path:
    rng     = np.random.default_rng(0) if rng is None else rng
    preproc = base / "preproc"
    (preproc / "data").mkdir(parents=True)

    primary = rng.normal(size=(N_ROWS, N_COLS)) + 1j * rng.normal(size=(N_ROWS, N_COLS))
    np.save(preproc / "data" / "primary.npy", primary)

    artifacts = {"primary": "primary.npy"}

    if with_full:
        full          = np.zeros((N_Z, N_ROWS, N_COLS), dtype=np.complex64)
        full[2, 3, 4] = 2.0
        full[4, 5, 1] = 1.0
        np.save(preproc / "data" / "cube_full.npy", full)
        artifacts["cube_full"] = "cube_full.npy"

    layout = {"global_crop": [0, N_ROWS, 0, N_COLS], "artifacts": artifacts}
    (preproc / "data" / "dataset.json").write_text(json.dumps(layout))

    return preproc


def make_stamp(run: Path, preproc: Path, rng: np.random.Generator, sources: tuple = ("pred", "gt"), x_min: float = -10.0, shape: tuple = (N_Z, N_ROWS, N_COLS), name: str = "stamp_1", region: list | None = None) -> Path:
    """Writes an inference stamp with random curve cubes and its metrics file.

    Args:
        run     : Run directory receiving ``inference/<name>`` and ``meta``; several stamps may share it.
        preproc : Preprocessing run the dataset config points back to.
        rng     : Generator for the random curve cubes.
        sources : Cube prefixes to write, each as ``<source>_curves.npy``.
        x_min   : Lowest z value of the stamp's z axis.
        shape   : Cube shape as (z, rows, cols).
        name    : Stamp directory name.
        region  : Split region as [row_start, row_end, col_start, col_end]; the whole scene when None.

    Returns:
        Path of the created stamp directory.
    """
    stamp = run / "inference" / name
    (stamp / "cubes").mkdir(parents=True)
    (run / "meta").mkdir(parents=True, exist_ok=True)

    region = [0, N_ROWS, 0, N_COLS] if region is None else region

    (run / "meta" / "dataset_creation_config.json").write_text(json.dumps({"preprocessing_run_directory": str(preproc)}))
    (stamp / "metrics.json").write_text(json.dumps({"x_axis_min": x_min, "x_axis_max": 30.0, "split_region": region, "activity_threshold": ACTIVITY_THRESHOLD}))

    for source in sources:
        np.save(stamp / "cubes" / f"{source}_curves.npy", rng.random(shape).astype(np.float32))

    return stamp


def make_cube_run(base: Path, sources: tuple = ("pred", "gt"), with_full: bool = False, with_params: tuple = (), with_metrics: bool = False) -> Path:
    rng     = np.random.default_rng(0)
    preproc = make_preproc(base, rng, with_full=with_full)
    stamp   = make_stamp(base / "group" / "run_a", preproc, rng, sources)

    for source in with_params:
        block    = rng.random((3 * N_SLOTS, N_ROWS, N_COLS)).astype(np.float32)
        block[3] = 0.0
        np.save(stamp / "cubes" / f"params_{source}.npy", block)

    if with_metrics:
        r2       = rng.random((N_ROWS, N_COLS)).astype(np.float32)
        r2[0, 0] = np.nan
        np.save(stamp / "cubes" / "pixel_r2.npy", r2)
        np.save(stamp / "cubes" / "label_r2.npy", rng.random((N_ROWS, N_COLS)).astype(np.float32))
        np.save(stamp / "cubes" / "misshaped.npy", rng.random((3, 3)).astype(np.float32))

    return stamp


def open_explorer(base: Path, expected: int | None = None) -> tuple[CubeExplorer, list[str]]:
    explorer = CubeExplorer(WebLogger())
    listing  = explorer.list_cubes(str(base))

    assert listing["ok"], listing
    if expected is not None:
        assert len(listing["cubes"]) == expected, listing

    return explorer, [cube["id"] for cube in listing["cubes"]]


def load_cube(explorer: CubeExplorer, cube_id: str, timeout: float = 30.0) -> dict:
    assert explorer.start_load(cube_id)["ok"]

    deadline = time.time() + timeout
    while explorer.load_status()["state"] == "loading" and time.time() < deadline:
        time.sleep(0.05)

    status = explorer.load_status()
    assert status["state"] == "ready", status
    return status


def loaded_cube(base: Path, **kwargs) -> tuple[CubeExplorer, str]:
    make_cube_run(base, **kwargs)
    explorer, cube_ids = open_explorer(base, expected=1)
    load_cube(explorer, cube_ids[0])

    return explorer, cube_ids[0]


@pytest.fixture
def monitor(tmp_path, monkeypatch):
    """Yields a SystemMonitor with its disk-usage and sampling loops disabled."""
    monkeypatch.setattr(SystemMonitor, "_du_loop", lambda self: None)
    monkeypatch.setattr(SystemHistory, "sample_loop", lambda self: None)
    monkeypatch.setattr(ActiveUsers, "sample_loop", lambda self: None)
    monkeypatch.setattr(GpuActivityLog, "sample_loop", lambda self: None)
    monkeypatch.setattr(HostHistoryLog, "sample_loop", lambda self: None)
    return SystemMonitor(StubPaths(tmp_path), WebLogger())


@pytest.fixture
def make_manager(tmp_path):
    """Yields a factory building a ProcessManager over named throwaway scripts, stopping all jobs on teardown."""
    built = []

    def build(scripts: dict) -> ProcessManager:
        directory = tmp_path / "main" / "analysis"
        directory.mkdir(parents=True)

        for name, body in scripts.items():
            (directory / f"{name}.py").write_text(body)

        paths   = StubPaths(tmp_path)
        logger  = WebLogger()
        manager = ProcessManager(paths, logger, JobNotifier(paths, logger), StubDescriber())
        built.append(manager)

        return manager

    yield build

    for manager in built:
        manager.stop_all()
