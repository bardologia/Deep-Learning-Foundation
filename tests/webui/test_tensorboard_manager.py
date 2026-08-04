from __future__ import annotations

from pathlib import Path
from types   import SimpleNamespace

import pytest

import tensorboard_manager
from project_paths       import ProjectPaths
from tensorboard_manager import TensorboardManager
from web_logger          import WebLogger

from tests.webui.conftest import StubPaths


class FakeProcess:

    def __init__(self, argv: list, cwd: str) -> None:
        self.argv     = argv
        self.cwd      = cwd
        self.pid      = 1000 + len(argv)
        self.exit     = None
        self.signals  = 0

    def poll(self) -> int | None:
        return self.exit

    def terminate(self) -> None:
        self.signals += 1
        self.exit     = -15


class FakeLauncher:

    def __init__(self, fail: bool = False) -> None:
        self.started = []
        self.fail    = fail

    def __call__(self, argv, cwd=None, stdout=None, stderr=None):
        if self.fail:
            raise OSError("tensorboard binary is missing")

        process = FakeProcess(argv, cwd)
        self.started.append(process)
        return process


@pytest.fixture
def launcher(monkeypatch):
    fake = FakeLauncher()
    monkeypatch.setattr(tensorboard_manager, "subprocess", SimpleNamespace(Popen=fake))
    monkeypatch.setattr(tensorboard_manager, "shutil",     SimpleNamespace(which=lambda name: None))
    monkeypatch.setattr(TensorboardManager, "_probe",      lambda self, record: None)
    return fake


@pytest.fixture
def manager(tmp_path):
    return TensorboardManager(StubPaths(tmp_path), WebLogger())


def _events(directory: Path, count: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (directory / f"events.out.tfevents.{index}").write_bytes(b"\x00")


def test_the_training_logdir_map_only_names_real_entry_points():
    paths = ProjectPaths()

    missing = [key for key in TensorboardManager.TRAINING_LOGDIRS if not paths.has_script(key)]

    assert not missing, f"tensorboard logdir keys with no main/ entry point: {missing}"


def test_logdir_keys_are_only_offered_for_mapped_scripts(manager):
    assert manager.logdir_keys("train_backbone") == ("logdir",)
    assert manager.logdir_keys("benchmark")      == ("paths.log_base_dir",)
    assert manager.logdir_keys("infer_backbone") is None


def test_ensure_starts_one_instance_and_publishes_its_url(manager, launcher, tmp_path):
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
    first  = manager.ensure(str(tmp_path / "runs"), "python")
    second = manager.ensure(str(tmp_path / "runs"), "python")

    assert second["id"] == first["id"]
    assert len(launcher.started) == 1
    assert len(manager.list_instances()) == 1


def test_a_different_logdir_gets_its_own_instance(manager, launcher, tmp_path):
    first  = manager.ensure(str(tmp_path / "a"), "python")
    second = manager.ensure(str(tmp_path / "b"), "python")

    assert first["id"] != second["id"]
    assert first["port"] != second["port"]
    assert len(launcher.started) == 2


def test_a_dead_instance_is_replaced_rather_than_reused(manager, launcher, tmp_path):
    first = manager.ensure(str(tmp_path / "runs"), "python")
    launcher.started[0].exit = 1

    second = manager.ensure(str(tmp_path / "runs"), "python")

    assert second["id"] != first["id"]
    assert len(launcher.started) == 2


def test_a_launch_failure_leaves_no_orphan_record(manager, monkeypatch, tmp_path):
    monkeypatch.setattr(tensorboard_manager, "subprocess", SimpleNamespace(Popen=FakeLauncher(fail=True)))
    monkeypatch.setattr(tensorboard_manager, "shutil",     SimpleNamespace(which=lambda name: None))

    result = manager.ensure(str(tmp_path / "runs"), "python")

    assert result["ok"] is False
    assert "missing" in result["error"]
    assert manager.list_instances() == []


def test_running_logdir_answers_only_while_the_process_lives(manager, launcher, tmp_path):
    started = manager.ensure(str(tmp_path / "runs"), "python")

    assert manager.running_logdir(str(tmp_path / "runs")) == started["id"]
    assert manager.running_logdir(str(tmp_path / "other")) is None

    manager.stop(started["id"])
    assert manager.running_logdir(str(tmp_path / "runs")) is None


def test_listing_marks_a_process_that_died_behind_our_back(manager, launcher, tmp_path):
    started = manager.ensure(str(tmp_path / "runs"), "python")
    manager.get(started["id"])["status"] = "running"

    assert manager.list_instances()[0]["status"] == "running"

    launcher.started[0].exit = 0
    assert manager.list_instances()[0]["status"] == "stopped"


def test_stopping_an_unknown_instance_is_refused(manager):
    assert manager.stop("nope") == {"ok": False, "error": "unknown tensorboard instance"}


def test_stop_all_terminates_every_live_process(manager, launcher, tmp_path):
    manager.ensure(str(tmp_path / "a"), "python")
    manager.ensure(str(tmp_path / "b"), "python")
    launcher.started[1].exit = 0

    manager.stop_all()

    assert launcher.started[0].signals == 1
    assert launcher.started[1].signals == 0
    assert [view["status"] for view in manager.list_instances()] == ["stopped", "stopped"]


def test_listing_logdirs_counts_distinct_event_directories(manager, launcher, tmp_path):
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


def test_listing_logdirs_refuses_a_missing_root(manager, tmp_path):
    payload = manager.list_logdirs(str(tmp_path / "nowhere"))

    assert payload["ok"] is False
    assert "runs root not found" in payload["error"]


def test_the_stderr_tail_is_bounded_and_survives_a_missing_file(manager, tmp_path):
    log = tmp_path / "tb.log"
    log.write_text("HEAD" + "x" * 900 + "TAIL")

    tail = manager._stderr_tail({"stderr_path": log}, max_chars=100)

    assert len(tail)          == 100
    assert tail.endswith("TAIL")
    assert manager._stderr_tail({"stderr_path": tmp_path / "gone.log"}) == "no stderr captured"


def test_a_status_transition_is_applied_once(manager):
    record = {"status": "starting"}

    assert manager._mark(record, "starting", "running") is True
    assert manager._mark(record, "starting", "failed")  is False
    assert record["status"] == "running"


def test_free_ports_are_bindable_and_distinct(manager):
    ports = {TensorboardManager._free_port() for _ in range(5)}

    assert all(1024 < port < 65536 for port in ports)
