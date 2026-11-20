from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib  import Path

import pytest

from configuration.diagnostics import ReportMailEntryConfig
from finalize_chain            import FinalizeChain
from run_launcher              import RunLauncher
from web_logger                import WebLogger


INTERPRETERS = [
    {"path": "/envs/ml-template/bin/python", "name": "ml-template"},
    {"path": "/envs/base/bin/python", "name": "base"},
]


class StubPaths:

    repo_root = Path("/repo")

    def discover_interpreters(self) -> list[dict]:
        return list(INTERPRETERS)

    def preferred_interpreter(self, interpreters: list[dict], script_key: str = "") -> str:
        return f"{interpreters[0]['path']}::{script_key}"


class StubResolver:

    def __init__(self, leaves: dict | None = None, ok: bool = True) -> None:
        self.leaves = leaves or {}
        self.ok     = ok
        self.calls  = []

    def resolve(self, key: str, interpreter: str) -> dict:
        self.calls.append((key, interpreter))
        if not self.ok:
            return {"ok": False, "error": "config import failed"}
        return {"ok": True, "leaves": [{"path": path, "value": value} for path, value in self.leaves.items()]}


class StubTensorboard:

    LOGDIRS          = {"train_backbone": ("logdir",), "benchmark": ("paths.log_base_dir",)}
    RUN_EVENT_FOLDER = "tensorboard"

    def __init__(self) -> None:
        self.ensured      = []
        self.spec_ensured = []
        self.event_dirs   = set()
        self.pid_run_dirs = []
        self.pid_queries  = []

    def logdir_keys(self, key: str) -> tuple | None:
        return self.LOGDIRS.get(key)

    def has_event_files(self, directory) -> bool:
        return str(directory) in self.event_dirs

    def ensure(self, logdir: str, interpreter: str) -> dict:
        self.ensured.append((logdir, interpreter))
        return {"ok": True}

    def ensure_spec(self, spec: dict, interpreter: str) -> dict:
        self.spec_ensured.append((spec, interpreter))
        return {"ok": True}

    def run_dirs_for_pid(self, root, pid: int, since: float) -> list:
        self.pid_queries.append((str(root), pid, since))
        return list(self.pid_run_dirs)


class StubProcesses:

    def __init__(self, ok: bool = True, units: list | None = None) -> None:
        self.launched = []
        self.queued   = []
        self.attached = []
        self.ok       = ok
        self.units    = units
        self.jobs     = {}
        self.lock     = threading.Lock()

    def progress(self, job_id) -> dict:
        if self.units is None:
            return {"ok": True, "supported": False, "live": False}
        return {"ok": True, "supported": True, "live": True, "progress": {"units": self.units}}

    def launch(self, key, interpreter, overrides, follow_up, detach) -> dict:
        self.launched.append((key, interpreter, overrides, follow_up, detach))
        job_id = f"job-{len(self.launched)}"
        with self.lock:
            self.jobs[job_id] = {"status": "running"}
        return {"ok": self.ok, "job_id": job_id}

    def enqueue(self, key, interpreter, overrides, follow_up, detach) -> dict:
        self.queued.append((key, interpreter, overrides, follow_up, detach))
        job_id = f"queued-{len(self.queued)}"
        with self.lock:
            self.jobs[job_id] = {"status": "queued"}
        return {"ok": self.ok, "job_id": job_id, "queued": True}

    def attach_finalize(self, job_id: str, flags: dict) -> None:
        self.attached.append((job_id, flags))

    def finish(self, job_id: str, status: str = "finished") -> None:
        with self.lock:
            self.jobs[job_id]["status"] = status


class StubNotifier:

    def __init__(self) -> None:
        self.alerts = []

    def system_alert(self, title: str, body: str, priority: str = "high") -> None:
        self.alerts.append((title, body))


def _launcher(resolver: StubResolver | None = None, processes: StubProcesses | None = None, notifier: StubNotifier | None = None) -> RunLauncher:
    return RunLauncher(StubPaths(), WebLogger(), resolver or StubResolver(), processes or StubProcesses(), StubTensorboard(), notifier or StubNotifier())


def _wait_for(items: list, count: int, timeout: float = 5.0) -> list:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and len(items) < count:
        time.sleep(0.01)
    return items


def test_the_preferred_interpreter_is_chosen_per_script():
    assert _launcher().preferred_interpreter("train_backbone") == "/envs/ml-template/bin/python::train_backbone"


def test_only_a_discovered_interpreter_is_accepted():
    launcher = _launcher()

    assert launcher.interpreter_error("/envs/base/bin/python") == ""
    assert "unknown interpreter '/usr/bin/python3'" in launcher.interpreter_error("/usr/bin/python3")


def test_an_override_wins_over_the_resolved_config():
    resolver = StubResolver({"logdir": "/data/from_config"})
    launcher = _launcher(resolver)

    logdir = launcher.training_logdir("train_backbone", {"logdir": "/data/from_form"}, "python")

    assert logdir       == "/data/from_form"
    assert resolver.calls == []


def test_the_resolved_config_supplies_the_logdir_when_no_override_is_given():
    resolver = StubResolver({"logdir": "/data/runs/backbone"})
    launcher = _launcher(resolver)

    assert launcher.training_logdir("train_backbone", {}, "python") == "/data/runs/backbone"
    assert resolver.calls == [("train_backbone", "python")]


def test_a_nested_logdir_leaf_is_read_by_its_full_path():
    resolver = StubResolver({"paths.log_base_dir": "/data/benchmarks"})

    assert _launcher(resolver).training_logdir("benchmark", {}, "python") == "/data/benchmarks"


def test_a_script_without_tensorboard_has_no_logdir():
    resolver = StubResolver({"logdir": "/data/runs"})

    assert _launcher(resolver).training_logdir("infer_backbone", {}, "python") is None
    assert resolver.calls == []


def test_an_empty_or_unresolvable_config_yields_no_logdir():
    assert _launcher(StubResolver({"logdir": ""})).training_logdir("train_backbone", {}, "python")  is None
    assert _launcher(StubResolver({})).training_logdir("train_backbone", {}, "python")              is None
    assert _launcher(StubResolver(ok=False)).training_logdir("train_backbone", {}, "python")        is None


def test_the_runs_root_is_the_parent_of_the_logdir():
    launcher = _launcher(StubResolver({"logdir": "/data/runs/group/run_a"}))

    assert launcher.runs_root("train_backbone", "python") == "/data/runs/group"
    assert launcher.runs_root("infer_backbone", "python") is None


def test_an_unknown_interpreter_is_refused_before_anything_is_started():
    processes = StubProcesses()
    result    = _launcher(processes=processes).execute("train_backbone", "/usr/bin/python3", {}, None, False, False)

    assert result["ok"] is False
    assert processes.launched == []
    assert processes.queued   == []


def test_execute_launches_and_forwards_every_argument():
    processes = StubProcesses()
    launcher  = _launcher(StubResolver({"logdir": "/data/runs"}), processes)

    result = launcher.execute("train_backbone", "/envs/ml-template/bin/python", {"epochs": 3}, "infer_backbone", True, False)

    assert result["job_id"] == "job-1"
    assert processes.queued == []
    assert processes.launched == [("train_backbone", "/envs/ml-template/bin/python", {"epochs": 3}, "infer_backbone", True)]


def test_execute_queues_instead_of_launching_when_asked():
    processes = StubProcesses()
    launcher  = _launcher(StubResolver({"logdir": "/data/runs"}), processes)

    result = launcher.execute("train_backbone", "/envs/ml-template/bin/python", {}, None, False, True)

    assert result["queued"] is True
    assert processes.launched == []
    assert processes.queued   == [("train_backbone", "/envs/ml-template/bin/python", {}, None, False)]


def test_a_launch_with_finalize_switches_registers_its_chain(monkeypatch):
    monkeypatch.setenv(ReportMailEntryConfig.password_env, "pw")
    processes = StubProcesses()
    launcher  = _launcher(StubResolver({"logdir": "/data/runs", "trials_mode": "augmentation"}), processes)

    result  = launcher.execute("train_backbone", "/envs/ml-template/bin/python", {}, None, True, False, FinalizeChain.flags({"compare_trials": True}))
    pending = launcher.finalizer.pending()

    assert result["ok"] is True
    assert len(processes.launched) == 1
    assert pending[0]["job_id"]    == result["job_id"]
    assert pending[0]["steps"]     == ["compare_trials"]
    assert pending[0]["experiment_dir"] == "/data/runs/augmentation"
    assert processes.attached == [(result["job_id"], FinalizeChain.flags({"compare_trials": True}))]


def test_a_launch_without_switches_stores_all_finalize_flags_off():
    processes = StubProcesses()
    launcher  = _launcher(StubResolver({"logdir": "/data/runs"}), processes)

    result = launcher.execute("train_backbone", "/envs/ml-template/bin/python", {}, None, False, True)

    assert processes.attached == [(result["job_id"], FinalizeChain.flags(None))]
    assert not any(processes.attached[0][1].values())


def test_a_refused_launch_stores_no_finalize_flags():
    processes = StubProcesses(ok=False)
    launcher  = _launcher(StubResolver({"logdir": "/data/runs"}), processes)

    launcher.execute("train_backbone", "/envs/ml-template/bin/python", {}, None, False, False)

    assert processes.attached == []


def test_a_launch_whose_chain_cannot_run_is_refused_before_it_starts():
    processes = StubProcesses()
    launcher  = _launcher(StubResolver({"logdir": "/data/runs"}), processes)

    result = launcher.execute("train_backbone", "/envs/ml-template/bin/python", {}, None, True, False, FinalizeChain.flags({"compare_seeds": True}))

    assert result["ok"] is False
    assert "trials_mode" in result["error"]
    assert processes.launched          == []
    assert launcher.finalizer.pending() == []


def test_a_launch_without_finalize_switches_registers_no_chain():
    launcher = _launcher(StubResolver({"logdir": "/data/runs"}))

    launcher.execute("infer_backbone", "/envs/ml-template/bin/python", {}, None, False, False)

    assert launcher.finalizer.pending() == []


def test_a_training_launch_autostarts_tensorboard_on_its_logdir():
    launcher = _launcher(StubResolver({"logdir": "/data/runs/backbone"}))

    launcher.execute("train_backbone", "/envs/ml-template/bin/python", {}, None, False, False)
    ensured = _wait_for(launcher.tensorboard.ensured, 1)

    assert ensured == [("/data/runs/backbone", "/envs/ml-template/bin/python")]


def test_a_non_training_launch_never_touches_tensorboard():
    launcher = _launcher(StubResolver({"logdir": "/data/runs"}))

    launcher.execute("infer_backbone", "/envs/ml-template/bin/python", {}, None, False, False)
    time.sleep(0.2)

    assert launcher.tensorboard.ensured == []


def test_a_refused_launch_never_starts_tensorboard():
    launcher = _launcher(StubResolver({"logdir": "/data/runs"}), StubProcesses(ok=False))

    launcher.execute("train_backbone", "/envs/ml-template/bin/python", {}, None, False, False)
    time.sleep(0.2)

    assert launcher.tensorboard.ensured == []


def test_a_broken_resolver_does_not_break_the_launch():
    processes = StubProcesses()
    launcher  = _launcher(StubResolver(ok=False), processes)

    result = launcher.execute("train_backbone", "/envs/ml-template/bin/python", {}, None, False, False)
    time.sleep(0.2)

    assert result["ok"] is True
    assert len(processes.launched)      == 1
    assert launcher.tensorboard.ensured == []


def test_a_raising_tensorboard_is_swallowed_by_the_autostart(monkeypatch):
    launcher = _launcher(StubResolver({"logdir": "/data/runs"}))
    monkeypatch.setattr(launcher.tensorboard, "ensure", lambda logdir, interpreter: (_ for _ in ()).throw(RuntimeError("boom")))

    launcher._autostart_tensorboard("train_backbone", {}, "python")


def _record(script: str = "train_backbone", overrides: dict | None = None, pid: int | None = None) -> dict:
    return {"job_id": "job-1", "script": script, "overrides": overrides or {}, "interpreter": "/envs/ml-template/bin/python", "pid": pid, "started": "2026-08-20T10:00:00"}


def _unit(name: str, log: str) -> dict:
    return {"name": name, "log": log, "status": "running"}


def _job_launcher(root: Path, units: list) -> RunLauncher:
    return _launcher(StubResolver({"logdir": str(root)}), StubProcesses(units=units))


def _run_dir(root: Path, name: str, launcher: RunLauncher | None = None) -> Path:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    if launcher is not None:
        launcher.tensorboard.event_dirs.add(str(directory))
    return directory


def test_only_entries_with_logdir_keys_are_tensorboard_capable():
    launcher = _launcher()

    assert launcher.tensorboard_capable("train_backbone") is True
    assert launcher.tensorboard_capable("infer_backbone") is False


def test_a_job_without_units_is_served_the_whole_logdir(tmp_path):
    launcher = _launcher(StubResolver({"logdir": str(tmp_path)}))

    result = launcher.job_tensorboard(_record())

    assert result["ok"] is True
    assert launcher.tensorboard.ensured      == [(str(tmp_path), "/envs/ml-template/bin/python")]
    assert launcher.tensorboard.spec_ensured == []


def test_a_flat_fanout_is_served_exactly_its_named_run_directories(tmp_path):
    units    = [_unit("run_a", str(tmp_path / "batch_train_logs" / "run_a.log")), _unit("run_b", str(tmp_path / "batch_train_logs" / "run_b.log"))]
    launcher = _job_launcher(tmp_path, units)

    _run_dir(tmp_path, "run_a", launcher)
    _run_dir(tmp_path, "run_b", launcher)
    _run_dir(tmp_path, "run_c", launcher)

    launcher.job_tensorboard(_record())

    assert launcher.tensorboard.ensured == []
    assert launcher.tensorboard.spec_ensured == [({"run_a": str(tmp_path / "run_a"), "run_b": str(tmp_path / "run_b")}, "/envs/ml-template/bin/python")]


def test_a_training_and_inference_twin_collapse_into_one_run(tmp_path):
    units    = [_unit("run_a", str(tmp_path / "batch_train_logs" / "run_a.log")), _unit("run_a", str(tmp_path / "batch_train_logs" / "run_a_infer.log"))]
    launcher = _job_launcher(tmp_path, units)

    _run_dir(tmp_path, "run_a", launcher)

    launcher.job_tensorboard(_record())

    assert launcher.tensorboard.spec_ensured == [({"run_a": str(tmp_path / "run_a")}, "/envs/ml-template/bin/python")]


def test_a_seed_sweep_run_name_with_slashes_resolves_and_labels_by_relative_path(tmp_path):
    units    = [_unit("base/seed0", str(tmp_path / "batch_train_logs" / "seed0.log"))]
    launcher = _job_launcher(tmp_path, units)

    _run_dir(tmp_path, "base/seed0", launcher)

    launcher.job_tensorboard(_record())

    assert launcher.tensorboard.spec_ensured == [({"base/seed0": str(tmp_path / "base" / "seed0")}, "/envs/ml-template/bin/python")]


def test_a_nested_fanout_is_served_the_directories_holding_its_logs(tmp_path):
    units    = [_unit("resunet", str(tmp_path / "bench" / "training" / "resunet" / "worker.log"))]
    launcher = _job_launcher(tmp_path, units)

    _run_dir(tmp_path, "bench/training/resunet", launcher)

    launcher.job_tensorboard(_record())

    assert launcher.tensorboard.spec_ensured == [({"bench/training/resunet": str(tmp_path / "bench" / "training" / "resunet")}, "/envs/ml-template/bin/python")]


def test_units_without_event_files_fall_back_to_the_whole_logdir(tmp_path):
    units    = [_unit("run_a", str(tmp_path / "batch_train_logs" / "run_a.log"))]
    launcher = _job_launcher(tmp_path, units)

    _run_dir(tmp_path, "run_a")

    launcher.job_tensorboard(_record())

    assert launcher.tensorboard.ensured      == [(str(tmp_path), "/envs/ml-template/bin/python")]
    assert launcher.tensorboard.spec_ensured == []


def test_a_unit_resolving_to_the_logdir_itself_is_not_wrapped_in_a_spec(tmp_path):
    units    = [_unit("run_a", str(tmp_path / "run_a.log"))]
    launcher = _job_launcher(tmp_path, units)

    launcher.tensorboard.event_dirs.add(str(tmp_path))

    launcher.job_tensorboard(_record())

    assert launcher.tensorboard.ensured      == [(str(tmp_path), "/envs/ml-template/bin/python")]
    assert launcher.tensorboard.spec_ensured == []


def test_a_single_run_is_served_the_directory_its_pid_wrote(tmp_path):
    launcher = _launcher(StubResolver({"logdir": str(tmp_path)}))
    launcher.tensorboard.pid_run_dirs = [tmp_path / "run_a"]

    launcher.job_tensorboard(_record(pid=42))

    assert launcher.tensorboard.ensured     == [(str(tmp_path / "run_a"), "/envs/ml-template/bin/python")]
    assert launcher.tensorboard.pid_queries == [(str(tmp_path), 42, datetime.fromisoformat("2026-08-20T10:00:00").timestamp() - 60.0)]


def test_an_ambiguous_pid_match_falls_back_to_the_whole_logdir(tmp_path):
    launcher = _launcher(StubResolver({"logdir": str(tmp_path)}))
    launcher.tensorboard.pid_run_dirs = [tmp_path / "run_a", tmp_path / "run_b"]

    launcher.job_tensorboard(_record(pid=42))

    assert launcher.tensorboard.ensured == [(str(tmp_path), "/envs/ml-template/bin/python")]


def test_a_pid_match_on_the_root_itself_is_not_narrowed(tmp_path):
    launcher = _launcher(StubResolver({"logdir": str(tmp_path)}))
    launcher.tensorboard.pid_run_dirs = [tmp_path]

    launcher.job_tensorboard(_record(pid=42))

    assert launcher.tensorboard.ensured == [(str(tmp_path), "/envs/ml-template/bin/python")]


def test_a_job_without_a_pid_never_scans_for_event_writers(tmp_path):
    launcher = _launcher(StubResolver({"logdir": str(tmp_path)}))

    launcher.job_tensorboard(_record())

    assert launcher.tensorboard.ensured     == [(str(tmp_path), "/envs/ml-template/bin/python")]
    assert launcher.tensorboard.pid_queries == []


def test_resolved_units_win_over_the_pid_scan(tmp_path):
    units    = [_unit("run_a", str(tmp_path / "batch_train_logs" / "run_a.log"))]
    launcher = _launcher(StubResolver({"logdir": str(tmp_path)}), StubProcesses(units=units))

    _run_dir(tmp_path, "run_a", launcher)
    launcher.tensorboard.pid_run_dirs = [tmp_path / "run_b"]

    launcher.job_tensorboard(_record(pid=42))

    assert launcher.tensorboard.spec_ensured == [({"run_a": str(tmp_path / "run_a")}, "/envs/ml-template/bin/python")]
    assert launcher.tensorboard.pid_queries  == []


def test_job_tensorboard_refuses_entries_without_event_files():
    result = _launcher().job_tensorboard(_record(script="infer_backbone"))

    assert result["ok"] is False
    assert "writes no TensorBoard event files" in result["error"]


def test_job_tensorboard_fails_when_the_logdir_cannot_be_resolved():
    result = _launcher(StubResolver(ok=False)).job_tensorboard(_record())

    assert result["ok"] is False
    assert "log directory" in result["error"]
