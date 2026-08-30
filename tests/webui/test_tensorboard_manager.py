"""Tests for the manager that starts, reuses and stops TensorBoard instances for the console.

Covers the mapping from training entry points to their logdir configuration keys, the
one-instance-per-logdir lifecycle (start, reuse, replace a dead process, stop with a
terminate-then-kill escalation), the run-directory listing, and the small helpers for
stderr tails, status transitions and free ports.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types   import SimpleNamespace

import pytest

import tensorboard_manager
from project_paths       import ProjectPaths
from tensorboard_manager import TensorboardManager
from web_logger          import WebLogger

from tests.webui.conftest import StubPaths


class FakeProcess:
    """In-memory stand-in for a TensorBoard child process.

    Attributes:
        argv     : Command line the process was started with.
        cwd      : Working directory the process was started in.
        pid      : Synthetic process id.
        exit     : Exit status, or None while the process is considered alive.
        signals  : Number of terminate calls received.
        kills    : Number of kill calls received.
        waits    : Number of wait calls received.
        stubborn : Whether the process ignores terminate and only dies on kill.
    """

    def __init__(self, argv: list, cwd: str, stubborn: bool = False) -> None:
        """Records the launch arguments and starts the process in a live state."""
        self.argv     = argv
        self.cwd      = cwd
        self.pid      = 1000 + len(argv)
        self.exit     = None
        self.signals  = 0
        self.kills    = 0
        self.waits    = 0
        self.stubborn = stubborn

    def poll(self) -> int | None:
        """Returns the exit status, or None while the process is alive."""
        return self.exit

    def terminate(self) -> None:
        """Counts the terminate and exits unless the process is stubborn."""
        self.signals += 1
        if not self.stubborn:
            self.exit = -15

    def kill(self) -> None:
        """Counts the kill and exits with the SIGKILL status."""
        self.kills += 1
        self.exit   = -9

    def wait(self, timeout=None) -> int:
        """Returns the exit status, raising TimeoutExpired while the process is still alive.

        Args:
            timeout : Grace period reported in the raised timeout.

        Returns:
            The recorded exit status.

        Raises:
            subprocess.TimeoutExpired : If the process has not exited.
        """
        self.waits += 1
        if self.exit is None:
            raise subprocess.TimeoutExpired(cmd="tensorboard", timeout=timeout)
        return self.exit


class FakeLauncher:
    """Popen stand-in producing FakeProcess objects, or failing on demand.

    Attributes:
        started  : Processes handed out so far, in launch order.
        fail     : Whether every launch raises OSError instead of starting a process.
        stubborn : Whether launched processes ignore terminate.
    """

    def __init__(self, fail: bool = False, stubborn: bool = False) -> None:
        """Configures the failure and stubbornness of the processes this launcher hands out."""
        self.started  = []
        self.fail     = fail
        self.stubborn = stubborn

    def __call__(self, argv, cwd=None, stdout=None, stderr=None):
        """Returns a new FakeProcess for the command line, or raises OSError when configured to fail."""
        if self.fail:
            raise OSError("tensorboard binary is missing")

        process = FakeProcess(argv, cwd, self.stubborn)
        self.started.append(process)
        return process


@pytest.fixture
def launcher(monkeypatch):
    """Returns a FakeLauncher patched over subprocess.Popen, with the readiness probe disabled."""
    fake = FakeLauncher()
    monkeypatch.setattr(tensorboard_manager, "subprocess", SimpleNamespace(Popen=fake, TimeoutExpired=subprocess.TimeoutExpired))
    monkeypatch.setattr(tensorboard_manager, "shutil",     SimpleNamespace(which=lambda name: None))
    monkeypatch.setattr(TensorboardManager, "_probe",      lambda self, record: None)
    return fake


@pytest.fixture
def manager(tmp_path):
    """Returns a manager rooted at a temporary project directory."""
    return TensorboardManager(StubPaths(tmp_path), WebLogger())


def _events(directory: Path, count: int) -> None:
    """Writes ``count`` placeholder tfevents files into the directory."""
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (directory / f"events.out.tfevents.{index}").write_bytes(b"\x00")


def test_the_training_logdir_map_only_names_real_entry_points():
    """Every key in the logdir map corresponds to a main/ entry point that exists."""
    paths = ProjectPaths()

    missing = [key for key in TensorboardManager.TRAINING_LOGDIRS if not paths.has_script(key)]

    assert not missing, f"tensorboard logdir keys with no main/ entry point: {missing}"


def test_every_training_entry_point_is_mapped_to_a_logdir():
    """No train_* entry point is left without a logdir configuration key."""
    paths    = ProjectPaths()
    training = [key for key in ProjectPaths.SCRIPT_DIRS if key.startswith("train_") and paths.has_script(key)]

    unmapped = [key for key in training if key not in TensorboardManager.TRAINING_LOGDIRS]

    assert not unmapped, f"training entry points with no tensorboard logdir: {unmapped}"


def test_logdir_keys_are_only_offered_for_mapped_scripts(manager):
    """Mapped scripts expose their logdir config keys and unmapped ones expose none."""
    assert manager.logdir_keys("train_backbone") == ("logdir",)
    assert manager.logdir_keys("benchmark")      == ("paths.log_base_dir",)
    assert manager.logdir_keys("infer_backbone") is None


def test_ensure_starts_one_instance_and_publishes_its_url(manager, launcher, tmp_path):
    """Ensuring a logdir starts tensorboard.main with the logdir, path prefix and port it publishes."""
    result = manager.ensure(str(tmp_path / "runs"), "/usr/bin/python")

    assert result["ok"] is True
    assert result["logdir"] == str(tmp_path / "runs")
    assert result["status"] == "starting"
    assert result["url"]    == f"/tb/{result['id']}/"
    assert result["pid"]    == launcher.started[0].pid
    assert "stderr_path" not in result

    argv = launcher.started[0].argv
    assert argv[:3] == ["/usr/bin/python", "-m", "tensorboard.main"]
    assert argv[argv.index("--logdir") + 1]      == str(tmp_path / "runs")
    assert argv[argv.index("--path_prefix") + 1] == f"/tb/{result['id']}"
    assert argv[argv.index("--port") + 1]        == str(result["port"])


def test_a_second_request_for_a_live_logdir_reuses_it(manager, launcher, tmp_path):
    """A repeat request for a live logdir returns the same instance without launching another process."""
    first  = manager.ensure(str(tmp_path / "runs"), "python")
    second = manager.ensure(str(tmp_path / "runs"), "python")

    assert second["id"] == first["id"]
    assert len(launcher.started) == 1
    assert len(manager.list_instances()) == 1


def test_a_different_logdir_gets_its_own_instance(manager, launcher, tmp_path):
    """Two logdirs get separate instances on separate ports."""
    first  = manager.ensure(str(tmp_path / "a"), "python")
    second = manager.ensure(str(tmp_path / "b"), "python")

    assert first["id"] != second["id"]
    assert first["port"] != second["port"]
    assert len(launcher.started) == 2


def test_ensure_spec_serves_exactly_the_given_runs(manager, launcher, tmp_path):
    """A run spec starts TensorBoard with --logdir_spec over the sorted label:path pairs."""
    result = manager.ensure_spec({"b": str(tmp_path / "b"), "a": str(tmp_path / "a")}, "python")
    joined = f"a:{tmp_path / 'a'},b:{tmp_path / 'b'}"

    argv = launcher.started[0].argv
    assert result["ok"] is True
    assert result["logdir"] == joined
    assert argv[argv.index("--logdir_spec") + 1] == joined
    assert "--logdir" not in argv


def test_ensure_spec_reuses_a_live_instance_regardless_of_entry_order(manager, launcher, tmp_path):
    """The same run set in a different order reuses the live instance."""
    first  = manager.ensure_spec({"a": str(tmp_path / "a"), "b": str(tmp_path / "b")}, "python")
    second = manager.ensure_spec({"b": str(tmp_path / "b"), "a": str(tmp_path / "a")}, "python")

    assert second["id"] == first["id"]
    assert len(launcher.started) == 1


def test_ensure_spec_refuses_labels_or_paths_that_break_the_spec(manager, launcher, tmp_path):
    """Labels with separators and paths with commas are refused instead of silently mangled."""
    assert manager.ensure_spec({"a:b": str(tmp_path)}, "python")["ok"]          is False
    assert manager.ensure_spec({"a,b": str(tmp_path)}, "python")["ok"]          is False
    assert manager.ensure_spec({"a": str(tmp_path) + ",x"}, "python")["ok"]     is False
    assert launcher.started == []


def test_a_dead_instance_is_replaced_rather_than_reused(manager, launcher, tmp_path):
    """A logdir whose process has exited is served by a freshly launched instance."""
    first = manager.ensure(str(tmp_path / "runs"), "python")
    launcher.started[0].exit = 1

    second = manager.ensure(str(tmp_path / "runs"), "python")

    assert second["id"] != first["id"]
    assert len(launcher.started) == 2


def test_a_launch_failure_leaves_no_orphan_record(manager, monkeypatch, tmp_path):
    """A failed launch reports the error and registers no instance."""
    monkeypatch.setattr(tensorboard_manager, "subprocess", SimpleNamespace(Popen=FakeLauncher(fail=True)))
    monkeypatch.setattr(tensorboard_manager, "shutil",     SimpleNamespace(which=lambda name: None))

    result = manager.ensure(str(tmp_path / "runs"), "python")

    assert result["ok"] is False
    assert "missing" in result["error"]
    assert manager.list_instances() == []


def test_running_logdir_answers_only_while_the_process_lives(manager, launcher, tmp_path):
    """The logdir lookup returns an instance id only for a started, not yet stopped, logdir."""
    started = manager.ensure(str(tmp_path / "runs"), "python")

    assert manager.running_logdir(str(tmp_path / "runs")) == started["id"]
    assert manager.running_logdir(str(tmp_path / "other")) is None

    manager.stop(started["id"])
    assert manager.running_logdir(str(tmp_path / "runs")) is None


def test_listing_drops_a_process_that_died_behind_our_back(manager, launcher, tmp_path):
    """Listing forgets an instance whose process exited without going through stop."""
    started = manager.ensure(str(tmp_path / "runs"), "python")
    manager.get(started["id"])["status"] = "running"

    assert manager.list_instances()[0]["status"] == "running"

    launcher.started[0].exit = 0
    assert manager.list_instances() == []
    assert manager.get(started["id"]) is None


def test_stopping_an_unknown_instance_is_refused(manager):
    """Stopping an unknown instance id reports an error."""
    assert manager.stop("nope") == {"ok": False, "error": "unknown tensorboard instance"}


def test_a_stopped_instance_is_waited_on_and_forgotten(manager, launcher, tmp_path):
    """Stopping terminates the process once, waits for it, and drops the record without a kill."""
    started = manager.ensure(str(tmp_path / "runs"), "python")

    assert manager.stop(started["id"]) == {"ok": True}
    assert launcher.started[0].signals == 1
    assert launcher.started[0].waits   == 1
    assert launcher.started[0].kills   == 0
    assert manager.list_instances() == []
    assert manager.get(started["id"]) is None


def test_a_child_that_ignores_the_terminate_is_killed(manager, monkeypatch, tmp_path):
    """A process still alive after the stop grace period is killed."""
    stubborn = FakeLauncher(stubborn=True)
    monkeypatch.setattr(tensorboard_manager, "subprocess", SimpleNamespace(Popen=stubborn, TimeoutExpired=subprocess.TimeoutExpired))
    monkeypatch.setattr(tensorboard_manager, "shutil",     SimpleNamespace(which=lambda name: None))
    monkeypatch.setattr(TensorboardManager, "_probe",      lambda self, record: None)
    monkeypatch.setattr(TensorboardManager, "STOP_GRACE_S", 0.01)

    started = manager.ensure(str(tmp_path / "runs"), "python")
    manager.stop(started["id"])

    assert stubborn.started[0].signals == 1
    assert stubborn.started[0].kills   == 1
    assert stubborn.started[0].exit    == -9


def test_stop_all_terminates_every_live_process(manager, launcher, tmp_path):
    """Stopping everything signals only the live processes and clears the registry."""
    manager.ensure(str(tmp_path / "a"), "python")
    manager.ensure(str(tmp_path / "b"), "python")
    launcher.started[1].exit = 0

    manager.stop_all()

    assert launcher.started[0].signals == 1
    assert launcher.started[0].waits   == 1
    assert launcher.started[1].signals == 0
    assert manager.list_instances() == []


def test_listing_logdirs_counts_distinct_event_directories(manager, launcher, tmp_path):
    """Listing reports each run directory, its number of event directories and whether it is being served."""
    root = tmp_path / "runs"
    _events(root / "run_a" / "tensorboard", 2)
    _events(root / "run_b" / "tensorboard" / "train", 1)
    _events(root / "run_b" / "tensorboard" / "val", 1)
    (root / "run_c").mkdir(parents=True)
    (root / ".hidden").mkdir()
    (root / "notes.md").write_text("x")

    manager.ensure(str(root / "run_a"), "python")
    payload = manager.list_logdirs(str(root))
    entries = {entry["name"]: entry for entry in payload["logdirs"]}

    assert payload["ok"] is True
    assert sorted(entries) == ["run_a", "run_b", "run_c"]
    assert entries["run_a"]["run_count"] == 1
    assert entries["run_b"]["run_count"] == 2
    assert entries["run_c"]["run_count"] == 0
    assert entries["run_a"]["running"] is not None
    assert entries["run_b"]["running"] is None


def test_run_dirs_for_pid_matches_writer_pid_and_time(manager, tmp_path):
    """The scan returns the run directories whose event files carry the pid and are recent enough."""
    root = tmp_path / "runs"
    for run, name in [
        ("run_a", "events.out.tfevents.1000.host.42.0"),
        ("run_b", "events.out.tfevents.1000.host.43.0"),
        ("run_c", "events.out.tfevents.100.host.42.0"),
        ("run_d", "events.out.tfevents.1000.my.host.42.1.v2"),
        ("run_e", "events.out.tfevents.garbled.name"),
    ]:
        directory = root / run / "tensorboard"
        directory.mkdir(parents=True)
        (directory / name).write_bytes(b"\x00")

    matched = manager.run_dirs_for_pid(root, 42, since=900.0)

    assert sorted(matched) == [root / "run_a", root / "run_d"]


def test_an_event_file_directly_in_the_root_maps_to_the_root(manager, tmp_path):
    """An event file sitting directly in the scanned root yields the root itself."""
    root = tmp_path / "runs"
    root.mkdir()
    (root / "events.out.tfevents.1000.host.42.0").write_bytes(b"\x00")

    assert manager.run_dirs_for_pid(root, 42, since=900.0) == [root]


def test_listing_logdirs_refuses_a_missing_root(manager, tmp_path):
    """A non-existent runs root is refused with an explanatory error."""
    payload = manager.list_logdirs(str(tmp_path / "nowhere"))

    assert payload["ok"] is False
    assert "runs root not found" in payload["error"]


def test_the_stderr_tail_is_bounded_and_survives_a_missing_file(manager, tmp_path):
    """The stderr tail is truncated to the requested length and a missing file yields a placeholder."""
    log = tmp_path / "tb.log"
    log.write_text("HEAD" + "x" * 900 + "TAIL")

    tail = manager._stderr_tail({"stderr_path": log}, max_chars=100)

    assert len(tail)          == 100
    assert tail.endswith("TAIL")
    assert manager._stderr_tail({"stderr_path": tmp_path / "gone.log"}) == "no stderr captured"


def test_a_status_transition_is_applied_once(manager):
    """A status transition applies only while the record still holds the expected source status."""
    record = {"status": "starting"}

    assert manager._mark(record, "starting", "running") is True
    assert manager._mark(record, "starting", "failed")  is False
    assert record["status"] == "running"


def test_free_ports_are_bindable_and_distinct(manager):
    """Allocated ports fall in the unprivileged range."""
    ports = {TensorboardManager._free_port() for _ in range(5)}

    assert all(1024 < port < 65536 for port in ports)
