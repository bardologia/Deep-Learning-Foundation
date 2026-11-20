
from __future__ import annotations

import threading
from pathlib import Path

import pytest

from configuration.diagnostics import ReportMailEntryConfig
from finalize_chain            import FinalizeChain, RunFinalizer
from web_logger                import WebLogger


LEAVES = {
    "train_backbone" : {"trials_mode": "augmentation", "trials_enabled": "True", "logdir": "/runs/backbone"},
    "infer_backbone" : {"runs_dir": "/runs/backbone"},
    "cross_validate" : {"run_tag": "None", "paths.log_base_dir": "/runs/cross_validation"},
    "benchmark"      : {"run_tag": "None", "paths.log_base_dir": "/runs/benchmark"},
    "compare_seeds"  : {},
    "compare_trials" : {},
}


class FakeResolver:

    def __init__(self, leaves_by_key: dict) -> None:
        self.leaves_by_key = leaves_by_key

    def resolve(self, key: str, interpreter: str) -> dict:
        leaves = self.leaves_by_key.get(key)
        if leaves is None:
            return {"ok": False, "error": f"no config for '{key}'"}
        return {"ok": True, "leaves": [{"path": path, "value": value} for path, value in leaves.items()]}


class FakeProcesses:

    def __init__(self) -> None:
        self.jobs = {}
        self.lock = threading.Lock()

    def register(self, job_id: str) -> None:
        with self.lock:
            self.jobs[job_id] = {"status": "running"}

    def finish(self, job_id: str, status: str = "finished") -> None:
        with self.lock:
            self.jobs[job_id]["status"] = status

    def drop(self, job_id: str) -> None:
        with self.lock:
            self.jobs.pop(job_id)


class FakeLauncher:

    def __init__(self, processes: FakeProcesses, ok: bool = True) -> None:
        self.processes = processes
        self.calls     = []
        self.ok        = ok

    def preferred_interpreter(self, key: str) -> str:
        return "/usr/bin/python3"

    def execute(self, key: str, interpreter: str, overrides: dict, follow_up, detach: bool, queue: bool) -> dict:
        self.calls.append({"key": key, "overrides": overrides, "detach": detach, "queue": queue})
        if not self.ok:
            return {"ok": False, "error": "spawn refused"}

        job_id = f"step-{len(self.calls)}"
        self.processes.register(job_id)
        return {"ok": True, "job_id": job_id}


class FakeNotifier:

    def __init__(self) -> None:
        self.alerts = []

    def system_alert(self, title: str, body: str, priority: str = "high") -> None:
        self.alerts.append((title, body))


@pytest.fixture
def rig(monkeypatch):
    monkeypatch.setenv(ReportMailEntryConfig.password_env, "pw")

    processes = FakeProcesses()
    launcher  = FakeLauncher(processes)
    notifier  = FakeNotifier()
    finalizer = RunFinalizer(launcher, processes, FakeResolver(LEAVES), notifier, WebLogger())

    return processes, launcher, notifier, finalizer


def _flags(*steps: str) -> dict:
    return {step: step in steps for step in FinalizeChain.STEPS}


def _sweep_tree(root: Path, trials: tuple = ("rung_a", "rung_b")) -> Path:
    for trial in trials:
        (root / trial / "seed0").mkdir(parents=True)
    (root / "batch_train_logs").mkdir()
    return root


def test_the_sweep_of_a_training_entry_is_its_trials_mode_directory():
    chain = FinalizeChain()

    assert chain.experiment_dir("train_backbone", {}, LEAVES["train_backbone"]) == "/runs/backbone/augmentation"
    assert chain.experiment_dir("train_backbone", {"trials_mode": "patch", "logdir": "/other"}, LEAVES["train_backbone"]) == "/other/patch"


def test_the_sweep_of_a_run_tagged_entry_sits_under_its_log_base_dir():
    chain = FinalizeChain()

    assert chain.experiment_dir("cross_validate", {"run_tag": "cv1"}, LEAVES["cross_validate"]) == "/runs/cross_validation/cv1/folds"
    assert chain.experiment_dir("benchmark", {"run_tag": "arch"}, LEAVES["benchmark"]) == "/runs/benchmark/arch/training"


def test_a_run_tagged_entry_without_a_fixed_tag_is_refused():
    with pytest.raises(ValueError, match="run_tag"):
        FinalizeChain().experiment_dir("benchmark", {}, LEAVES["benchmark"])


def test_an_entry_without_a_sweep_is_refused():
    with pytest.raises(ValueError, match="trials_mode"):
        FinalizeChain().experiment_dir("infer_backbone", {}, LEAVES["infer_backbone"])


def test_a_run_launched_on_its_own_stages_its_mail_where_the_entry_points(tmp_path):
    experiment = _sweep_tree(tmp_path / "augmentation")

    overrides = FinalizeChain().step_overrides("mail_experiment_reports", str(experiment))

    assert overrides == {"experiments_dir": str(experiment)}


def test_the_mail_step_refuses_a_sweep_without_unit_runs(tmp_path):
    (tmp_path / "augmentation").mkdir()

    with pytest.raises(FileNotFoundError, match="no trial directories"):
        FinalizeChain().step_overrides("mail_experiment_reports", str(tmp_path / "augmentation"))


def test_planning_refuses_a_mail_step_without_the_smtp_password(rig, monkeypatch):
    processes, launcher, notifier, finalizer = rig
    monkeypatch.delenv(ReportMailEntryConfig.password_env)

    plan, error = finalizer.plan("train_backbone", "/usr/bin/python3", {}, _flags("mail_experiment_reports"))

    assert plan is None
    assert ReportMailEntryConfig.password_env in error


def test_planning_a_launch_without_any_step_asks_for_no_chain(rig):
    processes, launcher, notifier, finalizer = rig

    assert finalizer.plan("train_backbone", "/usr/bin/python3", {}, None) == (None, "")
    assert finalizer.plan("train_backbone", "/usr/bin/python3", {}, _flags()) == (None, "")


def test_planning_refuses_an_entry_that_writes_no_sweep(rig):
    processes, launcher, notifier, finalizer = rig

    plan, error = finalizer.plan("infer_backbone", "/usr/bin/python3", {}, _flags("compare_trials"))

    assert plan is None
    assert "trials_mode" in error


def test_the_chain_walks_its_steps_after_the_run_finished(rig, tmp_path):
    processes, launcher, notifier, finalizer = rig
    experiment  = _sweep_tree(tmp_path / "augmentation")
    plan, error = finalizer.plan("train_backbone", "/usr/bin/python3", {"logdir": str(tmp_path)}, _flags(*FinalizeChain.STEPS))

    assert error == ""
    processes.register("run-1")
    finalizer.watch("run-1", plan)

    finalizer.scan()
    assert launcher.calls == []

    processes.finish("run-1")
    finalizer.scan()
    assert launcher.calls[0] == {"key": "compare_seeds", "overrides": {"runs_dir": str(experiment), "group_tags": "rung_a, rung_b"}, "detach": False, "queue": False}

    finalizer.scan()
    assert len(launcher.calls) == 1

    processes.finish("step-1")
    finalizer.scan()
    assert launcher.calls[1]["key"] == "compare_trials"
    assert launcher.calls[1]["overrides"] == {"runs_dir": str(experiment), "run_tags": "rung_a, rung_b"}

    processes.finish("step-2")
    finalizer.scan()
    assert launcher.calls[2]["key"] == "mail_experiment_reports"
    assert launcher.calls[2]["overrides"] == {"experiments_dir": str(tmp_path / "augmentation")}

    processes.finish("step-3")
    finalizer.scan()
    assert finalizer.pending() == []


def test_only_the_switched_on_steps_run(rig, tmp_path):
    processes, launcher, notifier, finalizer = rig
    _sweep_tree(tmp_path / "augmentation")
    plan, _ = finalizer.plan("train_backbone", "/usr/bin/python3", {"logdir": str(tmp_path)}, _flags("compare_trials"))

    processes.register("run-1")
    finalizer.watch("run-1", plan)
    processes.finish("run-1")
    finalizer.scan()

    assert [call["key"] for call in launcher.calls] == ["compare_trials"]


def test_a_failed_run_drops_its_chain(rig, tmp_path):
    processes, launcher, notifier, finalizer = rig
    _sweep_tree(tmp_path / "augmentation")
    plan, _ = finalizer.plan("train_backbone", "/usr/bin/python3", {"logdir": str(tmp_path)}, _flags(*FinalizeChain.STEPS))

    processes.register("run-1")
    finalizer.watch("run-1", plan)
    processes.finish("run-1", "failed")
    finalizer.scan()

    assert launcher.calls      == []
    assert finalizer.pending() == []


def test_a_run_gone_from_the_job_table_drops_its_chain(rig, tmp_path):
    processes, launcher, notifier, finalizer = rig
    _sweep_tree(tmp_path / "augmentation")
    plan, _ = finalizer.plan("train_backbone", "/usr/bin/python3", {"logdir": str(tmp_path)}, _flags("compare_seeds"))

    processes.register("run-1")
    finalizer.watch("run-1", plan)
    processes.drop("run-1")
    finalizer.scan()

    assert launcher.calls      == []
    assert finalizer.pending() == []


def test_a_failed_step_drops_the_rest_of_the_chain(rig, tmp_path):
    processes, launcher, notifier, finalizer = rig
    _sweep_tree(tmp_path / "augmentation")
    plan, _ = finalizer.plan("train_backbone", "/usr/bin/python3", {"logdir": str(tmp_path)}, _flags(*FinalizeChain.STEPS))

    processes.register("run-1")
    finalizer.watch("run-1", plan)
    processes.finish("run-1")
    finalizer.scan()
    processes.finish("step-1", "failed")
    finalizer.scan()

    assert [call["key"] for call in launcher.calls] == ["compare_seeds"]
    assert finalizer.pending() == []
    assert "compare_seeds" in notifier.alerts[0][0]


def test_an_empty_sweep_directory_drops_the_chain(rig, tmp_path):
    processes, launcher, notifier, finalizer = rig
    plan, _ = finalizer.plan("train_backbone", "/usr/bin/python3", {"logdir": str(tmp_path)}, _flags("compare_seeds"))

    processes.register("run-1")
    finalizer.watch("run-1", plan)
    processes.finish("run-1")
    finalizer.scan()

    assert launcher.calls      == []
    assert finalizer.pending() == []
    assert "experiment directory not found" in notifier.alerts[0][1]


def test_the_pending_listing_reports_the_steps_still_queued(rig, tmp_path):
    processes, launcher, notifier, finalizer = rig
    _sweep_tree(tmp_path / "augmentation")
    plan, _ = finalizer.plan("train_backbone", "/usr/bin/python3", {"logdir": str(tmp_path)}, _flags("compare_seeds", "compare_trials"))

    processes.register("run-1")
    finalizer.watch("run-1", plan)

    pending = finalizer.pending()
    assert pending[0]["job_id"] == "run-1"
    assert pending[0]["steps"][pending[0]["index"]:] == ["compare_seeds", "compare_trials"]

    processes.finish("run-1")
    finalizer.scan()
    processes.finish("step-1")
    finalizer.scan()

    pending = finalizer.pending()
    assert pending[0]["steps"][pending[0]["index"]:] == ["compare_trials"]
