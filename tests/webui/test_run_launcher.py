from __future__ import annotations

import time

from run_launcher import RunLauncher
from web_logger   import WebLogger


INTERPRETERS = [
    {"path": "/envs/Dune/bin/python", "name": "Dune"},
    {"path": "/envs/base/bin/python", "name": "base"},
]


class StubPaths:

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

    LOGDIRS = {"train_backbone": ("logdir",), "benchmark": ("paths.log_base_dir",)}

    def __init__(self) -> None:
        self.ensured = []

    def logdir_keys(self, key: str) -> tuple | None:
        return self.LOGDIRS.get(key)

    def ensure(self, logdir: str, interpreter: str) -> dict:
        self.ensured.append((logdir, interpreter))
        return {"ok": True}


class StubProcesses:

    def __init__(self, ok: bool = True) -> None:
        self.launched = []
        self.queued   = []
        self.ok       = ok

    def launch(self, key, interpreter, overrides, follow_up, detach) -> dict:
        self.launched.append((key, interpreter, overrides, follow_up, detach))
        return {"ok": self.ok, "job_id": "job-1"}

    def enqueue(self, key, interpreter, overrides, follow_up, detach) -> dict:
        self.queued.append((key, interpreter, overrides, follow_up, detach))
        return {"ok": self.ok, "queued": True}


def _launcher(resolver: StubResolver | None = None, processes: StubProcesses | None = None) -> RunLauncher:
    return RunLauncher(StubPaths(), WebLogger(), resolver or StubResolver(), processes or StubProcesses(), StubTensorboard())


def _wait_for(items: list, count: int, timeout: float = 5.0) -> list:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and len(items) < count:
        time.sleep(0.01)
    return items


def test_the_preferred_interpreter_is_chosen_per_script():
    assert _launcher().preferred_interpreter("train_backbone") == "/envs/Dune/bin/python::train_backbone"


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

    result = launcher.execute("train_backbone", "/envs/Dune/bin/python", {"epochs": 3}, "infer_backbone", True, False)

    assert result["job_id"] == "job-1"
    assert processes.queued == []
    assert processes.launched == [("train_backbone", "/envs/Dune/bin/python", {"epochs": 3}, "infer_backbone", True)]


def test_execute_queues_instead_of_launching_when_asked():
    processes = StubProcesses()
    launcher  = _launcher(StubResolver({"logdir": "/data/runs"}), processes)

    result = launcher.execute("train_backbone", "/envs/Dune/bin/python", {}, None, False, True)

    assert result["queued"] is True
    assert processes.launched == []
    assert processes.queued   == [("train_backbone", "/envs/Dune/bin/python", {}, None, False)]


def test_a_training_launch_autostarts_tensorboard_on_its_logdir():
    launcher = _launcher(StubResolver({"logdir": "/data/runs/backbone"}))

    launcher.execute("train_backbone", "/envs/Dune/bin/python", {}, None, False, False)
    ensured = _wait_for(launcher.tensorboard.ensured, 1)

    assert ensured == [("/data/runs/backbone", "/envs/Dune/bin/python")]


def test_a_non_training_launch_never_touches_tensorboard():
    launcher = _launcher(StubResolver({"logdir": "/data/runs"}))

    launcher.execute("infer_backbone", "/envs/Dune/bin/python", {}, None, False, False)
    time.sleep(0.2)

    assert launcher.tensorboard.ensured == []


def test_a_refused_launch_never_starts_tensorboard():
    launcher = _launcher(StubResolver({"logdir": "/data/runs"}), StubProcesses(ok=False))

    launcher.execute("train_backbone", "/envs/Dune/bin/python", {}, None, False, False)
    time.sleep(0.2)

    assert launcher.tensorboard.ensured == []


def test_a_broken_resolver_does_not_break_the_launch():
    processes = StubProcesses()
    launcher  = _launcher(StubResolver(ok=False), processes)

    result = launcher.execute("train_backbone", "/envs/Dune/bin/python", {}, None, False, False)
    time.sleep(0.2)

    assert result["ok"] is True
    assert len(processes.launched)      == 1
    assert launcher.tensorboard.ensured == []


def test_a_raising_tensorboard_is_swallowed_by_the_autostart(monkeypatch):
    launcher = _launcher(StubResolver({"logdir": "/data/runs"}))
    monkeypatch.setattr(launcher.tensorboard, "ensure", lambda logdir, interpreter: (_ for _ in ()).throw(RuntimeError("boom")))

    launcher._autostart_tensorboard("train_backbone", {}, "python")
