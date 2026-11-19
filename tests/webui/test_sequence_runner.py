
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from finalize_chain  import FinalizeChain
from sequence_runner import SequenceRunner

from configuration.diagnostics import ReportMailEntryConfig
from sequence_store     import SequenceStore
from web_logger         import WebLogger

from tests.webui.test_sequence_store import LEAVES, FakeResolver, SequencePaths, _block, _definition, decision_engine


class FakeProcesses:

    def __init__(self) -> None:
        self.jobs    = {}
        self.lock    = threading.Lock()
        self.stopped = []

    def finish(self, job_id: str, status: str = "finished") -> None:
        with self.lock:
            self.jobs[job_id]["status"] = status

    def stop(self, job_id: str) -> dict:
        self.stopped.append(job_id)
        self.finish(job_id, "cancelled")
        return {"ok": True}


class FakeLauncher:

    def __init__(self, processes: FakeProcesses) -> None:
        self.processes = processes
        self.calls     = []

    def preferred_interpreter(self, key: str) -> str:
        return "/usr/bin/python3"

    def execute(self, key: str, interpreter: str, overrides: dict, follow_up, detach: bool, queue: bool) -> dict:
        job_id = f"job-{len(self.calls)}"
        self.calls.append({"key": key, "interpreter": interpreter, "overrides": overrides, "detach": detach})
        with self.processes.lock:
            self.processes.jobs[job_id] = {"status": "running"}
        return {"ok": True, "job_id": job_id}








class FakeNotifier:

    def __init__(self) -> None:
        self.alerts = []

    def system_alert(self, title: str, body: str, priority: str = "high") -> None:
        self.alerts.append((title, body, priority))


class LockProbingLauncher(FakeLauncher):

    def __init__(self, processes: FakeProcesses) -> None:
        super().__init__(processes)
        self.runner    = None
        self.lock_free = []

    def execute(self, key: str, interpreter: str, overrides: dict, follow_up, detach: bool, queue: bool) -> dict:
        acquired = self.runner.lock.acquire(blocking=False)
        self.lock_free.append(acquired)
        if acquired:
            self.runner.lock.release()
        return super().execute(key, interpreter, overrides, follow_up, detach, queue)


class StoppingLauncher(FakeLauncher):

    def __init__(self, processes: FakeProcesses) -> None:
        super().__init__(processes)
        self.runner      = None
        self.sequence_id = ""

    def execute(self, key: str, interpreter: str, overrides: dict, follow_up, detach: bool, queue: bool) -> dict:
        self.runner.stop_sequence(self.sequence_id)
        return super().execute(key, interpreter, overrides, follow_up, detach, queue)


@pytest.fixture
def rig(tmp_path, monkeypatch):
    monkeypatch.setenv(ReportMailEntryConfig.password_env, "pw")
    store     = SequenceStore(SequencePaths(tmp_path), WebLogger(), decision_engine())
    processes = FakeProcesses()
    launcher  = FakeLauncher(processes)
    runner    = SequenceRunner(store, processes, launcher, FakeResolver(LEAVES), FakeNotifier(), WebLogger())
    return store, processes, launcher, runner


def _saved(store: SequenceStore, blocks: list, **extra) -> str:
    sequence_id = store.create("seq")["entry"]["sequence_id"]
    result      = store.save(sequence_id, _definition(blocks, **extra))
    assert result["ok"], result
    return sequence_id


def _block_state(runner: SequenceRunner, sequence_id: str, block_id: str) -> dict:
    return runner.state(sequence_id)["blocks"][block_id]


def _finalize(*steps: str) -> dict:
    return {step: step in steps for step in FinalizeChain.STEPS}


def test_start_launches_only_the_blocks_without_predecessors(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1"), _block("b2", predecessors=["a1"])])

    runner.start_sequence(sequence_id)

    assert [call["key"] for call in launcher.calls] == ["train_backbone"]
    assert _block_state(runner, sequence_id, "a1")["status"] == "running"
    assert _block_state(runner, sequence_id, "b2")["status"] == "pending"


def test_a_block_launches_when_its_predecessor_finishes(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1"), _block("b2", script="infer_backbone", predecessors=["a1"])])

    runner.start_sequence(sequence_id)
    processes.finish("job-0")
    runner.scan()

    assert [call["key"] for call in launcher.calls] == ["train_backbone", "infer_backbone"]
    assert _block_state(runner, sequence_id, "a1")["status"] == "done"
    assert _block_state(runner, sequence_id, "b2")["status"] == "running"


def test_independent_blocks_launch_in_parallel(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1"), _block("b2", predecessors=["a1"]), _block("c3", predecessors=["a1"])])

    runner.start_sequence(sequence_id)
    processes.finish("job-0")
    runner.scan()

    assert len(launcher.calls) == 3
    assert _block_state(runner, sequence_id, "b2")["status"] == "running"
    assert _block_state(runner, sequence_id, "c3")["status"] == "running"


def test_a_checkpoint_block_holds_until_released(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1"), _block("b2", predecessors=["a1"], checkpoint=True)])

    runner.start_sequence(sequence_id)
    processes.finish("job-0")
    runner.scan()

    assert len(launcher.calls) == 1
    assert _block_state(runner, sequence_id, "b2")["status"] == "held"
    assert runner.state(sequence_id)["status"] == "held"

    runner.release(sequence_id, "b2")

    assert len(launcher.calls) == 2
    assert _block_state(runner, sequence_id, "b2")["status"] == "running"


def test_a_held_block_is_announced_once(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1", checkpoint=True)])

    runner.start_sequence(sequence_id)
    runner.scan()
    runner.scan()

    held = [alert for alert in runner.notifier.alerts if alert[0].startswith("sequence block held")]
    assert len(held) == 1
    assert "release it in the Sequences tab" in held[0][1]
    assert held[0][2] == "default"


def test_release_refuses_a_block_that_is_not_held(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1"), _block("b2", predecessors=["a1"])])

    runner.start_sequence(sequence_id)

    assert not runner.release(sequence_id, "b2")["ok"]
    assert not runner.release(sequence_id, "ghost")["ok"]


def test_a_failed_block_skips_its_descendants(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1"), _block("b2", predecessors=["a1"]), _block("c3", predecessors=["b2"])])

    runner.start_sequence(sequence_id)
    processes.finish("job-0", "failed")
    runner.scan()

    assert len(launcher.calls) == 1
    assert _block_state(runner, sequence_id, "a1")["status"] == "failed"
    assert _block_state(runner, sequence_id, "b2")["status"] == "skipped"
    assert _block_state(runner, sequence_id, "c3")["status"] == "skipped"
    assert runner.state(sequence_id)["status"] == "failed"


def test_the_dataset_is_injected_into_the_fields_each_entry_exposes(rig):
    store, processes, launcher, runner = rig
    dataset     = {"dataset_path": "/data/sets/17sartom", "parameters_path": "/data/sets/17sartom/params/run_a/parameters.npy"}
    sequence_id = _saved(store, [_block("a1", overrides={"paths.dataset_path": "/old", "training.seed": "7"}), _block("b2", script="benchmark")], dataset=dataset)

    runner.start_sequence(sequence_id)

    train, bench = launcher.calls
    assert train["overrides"]["paths.dataset_path"] == "/data/sets/17sartom"
    assert train["overrides"]["paths.parameters_path"] == dataset["parameters_path"]
    assert train["overrides"]["training.seed"] == "7"
    assert bench["overrides"] == {"dataset_base_path": str(Path("/data/sets")), "dataset_filter": "17sartom"}


def test_an_empty_dataset_leaves_the_block_overrides_untouched(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1", overrides={"training.seed": "3"})])

    runner.start_sequence(sequence_id)

    assert launcher.calls[0]["overrides"] == {"training.seed": "3"}


def test_a_block_edited_while_pending_launches_with_the_new_overrides(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1"), _block("b2", predecessors=["a1"])])

    runner.start_sequence(sequence_id)
    store.save(sequence_id, _definition([_block("a1"), _block("b2", predecessors=["a1"], overrides={"training.seed": "42"})]))
    processes.finish("job-0")
    runner.scan()

    assert launcher.calls[1]["overrides"] == {"training.seed": "42"}


def test_stop_stops_running_jobs_and_skips_waiting_blocks(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1"), _block("b2", predecessors=["a1"])])

    runner.start_sequence(sequence_id)
    result = runner.stop_sequence(sequence_id)

    assert result["ok"]
    assert processes.stopped == ["job-0"]
    assert runner.state(sequence_id)["status"] == "stopped"
    assert _block_state(runner, sequence_id, "a1")["status"] == "cancelled"
    assert _block_state(runner, sequence_id, "b2")["status"] == "skipped"


def test_clear_state_refuses_active_then_clears_a_stopped_sequence(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1")])

    runner.start_sequence(sequence_id)
    refused = runner.clear_state(sequence_id)
    runner.stop_sequence(sequence_id)
    cleared = runner.clear_state(sequence_id)
    again   = runner.clear_state(sequence_id)

    assert not refused["ok"]
    assert cleared["ok"] and cleared["cleared"]
    assert again["ok"] and not again["cleared"]
    assert runner.state(sequence_id) is None


def test_the_runner_lock_is_released_while_a_block_launches(tmp_path):
    store     = SequenceStore(SequencePaths(tmp_path), WebLogger(), decision_engine())
    processes = FakeProcesses()
    launcher  = LockProbingLauncher(processes)
    runner    = SequenceRunner(store, processes, launcher, FakeResolver(LEAVES), FakeNotifier(), WebLogger())
    launcher.runner = runner

    runner.start_sequence(_saved(store, [_block("a1")]))

    assert launcher.lock_free == [True]


def test_a_stop_landing_mid_launch_stops_the_job_it_started(tmp_path):
    store     = SequenceStore(SequencePaths(tmp_path), WebLogger(), decision_engine())
    processes = FakeProcesses()
    launcher  = StoppingLauncher(processes)
    runner    = SequenceRunner(store, processes, launcher, FakeResolver(LEAVES), FakeNotifier(), WebLogger())
    launcher.runner      = runner
    sequence_id          = _saved(store, [_block("a1")])
    launcher.sequence_id = sequence_id

    runner.start_sequence(sequence_id)

    assert processes.stopped == ["job-0"]
    assert runner.state(sequence_id)["status"] == "stopped"
    assert _block_state(runner, sequence_id, "a1")["status"] == "cancelled"


def test_a_finished_sequence_reports_done_and_can_restart(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1"), _block("b2", predecessors=["a1"])])

    runner.start_sequence(sequence_id)
    processes.finish("job-0")
    runner.scan()
    processes.finish("job-1")
    runner.scan()

    assert runner.state(sequence_id)["status"] == "done"

    restarted = runner.start_sequence(sequence_id)

    assert restarted["ok"]
    assert _block_state(runner, sequence_id, "b2")["status"] == "pending"


def test_a_skip_toggled_block_is_bypassed_and_its_successors_still_run(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1", skip=True), _block("b2", script="infer_backbone", predecessors=["a1"])])

    runner.start_sequence(sequence_id)

    assert [call["key"] for call in launcher.calls] == ["infer_backbone"]
    assert _block_state(runner, sequence_id, "a1")["status"] == "bypassed"

    processes.finish("job-0")
    runner.scan()

    assert runner.state(sequence_id)["status"] == "done"


def test_a_skipped_block_waits_for_its_predecessors_before_bypassing(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1"), _block("b2", predecessors=["a1"], skip=True), _block("c3", script="infer_backbone", predecessors=["b2"])])

    runner.start_sequence(sequence_id)

    assert [call["key"] for call in launcher.calls] == ["train_backbone"]
    assert _block_state(runner, sequence_id, "b2")["status"] == "pending"
    assert _block_state(runner, sequence_id, "c3")["status"] == "pending"

    processes.finish("job-0")
    runner.scan()

    assert _block_state(runner, sequence_id, "b2")["status"] == "bypassed"
    assert [call["key"] for call in launcher.calls] == ["train_backbone", "infer_backbone"]


def test_consecutive_skipped_blocks_cascade_in_order_within_one_scan(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1"), _block("b2", predecessors=["a1"], skip=True), _block("c3", predecessors=["b2"], skip=True), _block("d4", script="infer_backbone", predecessors=["c3"])])

    runner.start_sequence(sequence_id)
    assert [call["key"] for call in launcher.calls] == ["train_backbone"]

    processes.finish("job-0")
    runner.scan()

    assert _block_state(runner, sequence_id, "b2")["status"] == "bypassed"
    assert _block_state(runner, sequence_id, "c3")["status"] == "bypassed"
    assert [call["key"] for call in launcher.calls] == ["train_backbone", "infer_backbone"]


def test_skip_toggled_mid_run_bypasses_a_waiting_block(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1"), _block("b2", predecessors=["a1"])])

    runner.start_sequence(sequence_id)
    store.save(sequence_id, _definition([_block("a1"), _block("b2", predecessors=["a1"], skip=True)]))
    processes.finish("job-0")
    runner.scan()

    assert len(launcher.calls) == 1
    assert _block_state(runner, sequence_id, "b2")["status"] == "bypassed"
    assert runner.state(sequence_id)["status"] == "done"


def test_skip_overrides_a_checkpoint_hold(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1"), _block("b2", predecessors=["a1"], checkpoint=True), _block("c3", script="infer_backbone", predecessors=["b2"])])

    runner.start_sequence(sequence_id)
    processes.finish("job-0")
    runner.scan()
    assert _block_state(runner, sequence_id, "b2")["status"] == "held"

    store.save(sequence_id, _definition([_block("a1"), _block("b2", predecessors=["a1"], checkpoint=True, skip=True), _block("c3", script="infer_backbone", predecessors=["b2"])]))
    runner.scan()

    assert _block_state(runner, sequence_id, "b2")["status"] == "bypassed"
    assert [call["key"] for call in launcher.calls] == ["train_backbone", "infer_backbone"]


def test_an_isolated_sequence_creates_a_workspace_and_redirects_the_run_roots(rig, tmp_path):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1"), _block("b2", script="infer_backbone", predecessors=["a1"])], isolate=True, name="My Ladder")

    runner.start_sequence(sequence_id)
    processes.finish("job-0")
    runner.scan()

    run_dir = runner.state(sequence_id)["run_dir"]
    assert Path(run_dir).is_dir()
    assert Path(run_dir).parent == tmp_path / "runs" / "sequences"
    assert Path(run_dir).name.startswith("my_ladder_")
    assert launcher.calls[0]["overrides"]["paths.log_base_dir"] == run_dir
    assert launcher.calls[1]["overrides"]["runs_dir"] == run_dir


def test_a_dataset_rooted_runs_dir_is_not_redirected(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1", script="generate_dataset")], isolate=True)

    runner.start_sequence(sequence_id)

    assert "runs_dir" not in launcher.calls[0]["overrides"]




def test_a_training_logdir_is_redirected_into_the_workspace(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1", script="train_jepa")], isolate=True)

    runner.start_sequence(sequence_id)

    assert launcher.calls[0]["overrides"]["logdir"] == runner.state(sequence_id)["run_dir"]


def test_a_logdir_override_outside_the_runs_tree_is_kept(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1", script="train_jepa", overrides={"logdir": "/data/scratch_runs"})], isolate=True)

    runner.start_sequence(sequence_id)

    assert launcher.calls[0]["overrides"]["logdir"] == "/data/scratch_runs"


def test_a_runs_dir_override_outside_the_runs_tree_is_kept(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1", script="infer_backbone", overrides={"runs_dir": "/data/pulled_runs"})], isolate=True)

    runner.start_sequence(sequence_id)

    assert launcher.calls[0]["overrides"]["runs_dir"] == "/data/pulled_runs"


def test_each_start_gets_its_own_workspace(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1")], isolate=True)

    runner.start_sequence(sequence_id)
    first = runner.state(sequence_id)["run_dir"]
    processes.finish("job-0")
    runner.scan()
    runner.start_sequence(sequence_id)
    second = runner.state(sequence_id)["run_dir"]

    assert first != second
    assert Path(first).is_dir()
    assert Path(second).is_dir()


def test_resume_reuses_the_given_workspace_instead_of_creating_one(rig, tmp_path):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1"), _block("b2", script="infer_backbone", predecessors=["a1"])], isolate=True)
    workspace   = tmp_path / "runs" / "sequences" / "seq_2026-08-01_10-00-00"
    workspace.mkdir(parents=True)

    result = runner.start_sequence(sequence_id, str(workspace), ["a1"])

    assert result["ok"], result
    assert runner.state(sequence_id)["run_dir"] == str(workspace)
    assert _block_state(runner, sequence_id, "a1")["status"] == "bypassed"
    assert [call["key"] for call in launcher.calls] == ["infer_backbone"]
    assert launcher.calls[0]["overrides"]["runs_dir"] == str(workspace)
    assert list((tmp_path / "runs" / "sequences").iterdir()) == [workspace]


def test_a_resume_bypass_is_not_persisted_and_a_plain_start_runs_every_block(rig, tmp_path):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1"), _block("b2", script="infer_backbone", predecessors=["a1"])], isolate=True)
    workspace   = tmp_path / "runs" / "sequences" / "old"
    workspace.mkdir(parents=True)

    runner.start_sequence(sequence_id, str(workspace), ["a1"])
    processes.finish("job-0")
    runner.scan()

    assert [block["skip"] for block in store.get(sequence_id)["blocks"]] == [False, False]

    result = runner.start_sequence(sequence_id)

    assert result["ok"], result
    assert runner.state(sequence_id)["run_dir"] != str(workspace)
    assert [call["key"] for call in launcher.calls] == ["infer_backbone", "train_backbone"]


def test_resume_refuses_a_missing_workspace(rig, tmp_path):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1")], isolate=True)

    result = runner.start_sequence(sequence_id, str(tmp_path / "runs" / "sequences" / "ghost"))

    assert not result["ok"]
    assert "does not exist" in result["error"]
    assert launcher.calls == []


def test_resume_refuses_a_workspace_outside_the_sequences_tree(rig, tmp_path):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1")], isolate=True)
    elsewhere   = tmp_path / "elsewhere"
    elsewhere.mkdir()

    result = runner.start_sequence(sequence_id, str(elsewhere))

    assert not result["ok"]
    assert "directly under" in result["error"]


def test_the_live_state_is_mirrored_into_the_workspace(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1"), _block("b2", predecessors=["a1"])], isolate=True)

    runner.start_sequence(sequence_id)
    state_path = Path(runner.state(sequence_id)["run_dir"]) / SequenceRunner.STATE_FILENAME
    first      = json.loads(state_path.read_text())

    processes.finish("job-0")
    runner.scan()
    second = json.loads(state_path.read_text())

    assert first["blocks"]["a1"]["status"] == "running"
    assert second["blocks"]["a1"]["status"] == "done"
    assert second["blocks"]["b2"]["status"] == "running"


def test_the_state_file_write_is_staged_and_serialized_under_the_runner_lock(rig, monkeypatch):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1")], isolate=True)
    lock_free   = []
    original    = Path.write_text

    def probing(self, text, encoding=None):
        if self.name == "sequence_state.tmp":
            acquired = runner.lock.acquire(blocking=False)
            lock_free.append(acquired)
            if acquired:
                runner.lock.release()
        return original(self, text, encoding=encoding)

    monkeypatch.setattr(Path, "write_text", probing)
    runner.start_sequence(sequence_id)

    run_dir = Path(runner.state(sequence_id)["run_dir"])
    assert lock_free == [False]
    assert not (run_dir / "sequence_state.tmp").exists()
    assert json.loads((run_dir / SequenceRunner.STATE_FILENAME).read_text())["blocks"]["a1"]["status"] == "running"


def test_a_failed_staging_write_leaves_the_state_file_intact_and_retries(rig, monkeypatch):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1")], isolate=True)

    runner.start_sequence(sequence_id)
    run_dir = Path(runner.state(sequence_id)["run_dir"])
    before  = (run_dir / SequenceRunner.STATE_FILENAME).read_text()
    processes.finish("job-0")

    original = Path.write_text

    def failing(self, text, encoding=None):
        if self.name == "sequence_state.tmp":
            raise OSError("disk full")
        return original(self, text, encoding=encoding)

    monkeypatch.setattr(Path, "write_text", failing)
    with pytest.raises(OSError):
        runner.scan()

    assert (run_dir / SequenceRunner.STATE_FILENAME).read_text() == before

    monkeypatch.setattr(Path, "write_text", original)
    runner.scan()

    assert json.loads((run_dir / SequenceRunner.STATE_FILENAME).read_text())["blocks"]["a1"]["status"] == "done"


def test_a_stop_is_recorded_in_the_workspace_state_file(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1"), _block("b2", predecessors=["a1"])], isolate=True)

    runner.start_sequence(sequence_id)
    runner.stop_sequence(sequence_id)
    recorded = json.loads((Path(runner.state(sequence_id)["run_dir"]) / SequenceRunner.STATE_FILENAME).read_text())

    assert recorded["status"] == "stopped"
    assert recorded["blocks"]["a1"]["status"] == "cancelled"
    assert recorded["blocks"]["b2"]["status"] == "skipped"


def test_resume_scan_reports_the_blocks_recorded_done(rig, tmp_path):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1"), _block("b2", predecessors=["a1"])], isolate=True)

    runner.start_sequence(sequence_id)
    workspace = runner.state(sequence_id)["run_dir"]
    processes.finish("job-0")
    runner.scan()
    runner.stop_sequence(sequence_id)

    restarted = SequenceRunner(store, FakeProcesses(), FakeLauncher(FakeProcesses()), FakeResolver(LEAVES), FakeNotifier(), WebLogger())
    result    = restarted.resume_scan(sequence_id, workspace)

    assert result["ok"], result
    assert result["recorded"] is True
    assert result["done"] == ["a1"]
    assert result["statuses"]["b2"] == "cancelled"


def test_resume_scan_keeps_proposing_blocks_bypassed_by_an_earlier_resume(rig, tmp_path):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1"), _block("b2", script="infer_backbone", predecessors=["a1"])], isolate=True)
    workspace   = tmp_path / "runs" / "sequences" / "old"
    workspace.mkdir(parents=True)

    runner.start_sequence(sequence_id, str(workspace), ["a1"])
    processes.finish("job-0")
    runner.scan()

    result = runner.resume_scan(sequence_id, str(workspace))

    assert result["ok"], result
    assert result["done"] == ["a1", "b2"]
    assert result["statuses"]["a1"] == "bypassed"


def test_resume_scan_does_not_propose_a_manually_skipped_block(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1", skip=True), _block("b2", script="infer_backbone", predecessors=["a1"])], isolate=True)

    runner.start_sequence(sequence_id)
    workspace = runner.state(sequence_id)["run_dir"]
    processes.finish("job-0")
    runner.scan()

    result = runner.resume_scan(sequence_id, workspace)

    assert result["ok"], result
    assert result["done"] == ["b2"]
    assert result["statuses"]["a1"] == "bypassed"


def test_resume_scan_ignores_done_blocks_no_longer_in_the_sequence(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1")], isolate=True)

    runner.start_sequence(sequence_id)
    workspace = runner.state(sequence_id)["run_dir"]
    processes.finish("job-0")
    runner.scan()
    store.save(sequence_id, _definition([_block("z9")], isolate=True))

    result = runner.resume_scan(sequence_id, workspace)

    assert result["ok"]
    assert result["done"] == []
    assert result["statuses"]["a1"] == "done"


def test_resume_scan_reports_a_workspace_without_a_state_file(rig, tmp_path):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1")], isolate=True)
    workspace   = tmp_path / "runs" / "sequences" / "bare"
    workspace.mkdir(parents=True)

    result = runner.resume_scan(sequence_id, str(workspace))

    assert result["ok"]
    assert result["recorded"] is False
    assert result["done"] == []


def test_resume_scan_refuses_an_invalid_workspace(rig, tmp_path):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1")], isolate=True)
    elsewhere   = tmp_path / "elsewhere"
    elsewhere.mkdir()

    assert not runner.resume_scan(sequence_id, str(elsewhere))["ok"]
    assert not runner.resume_scan(sequence_id, str(tmp_path / "runs" / "sequences" / "ghost"))["ok"]


def test_resume_refuses_an_unisolated_sequence(rig, tmp_path):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1")])
    workspace   = tmp_path / "runs" / "sequences" / "old"
    workspace.mkdir(parents=True)

    result = runner.start_sequence(sequence_id, str(workspace))

    assert not result["ok"]
    assert "does not isolate" in result["error"]


def test_a_sequence_without_isolation_leaves_the_run_roots_alone(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1")])

    runner.start_sequence(sequence_id)

    assert runner.state(sequence_id)["run_dir"] == ""
    assert "paths.log_base_dir" not in launcher.calls[0]["overrides"]










def _workspace_rel(runner: SequenceRunner, sequence_id: str) -> str:
    return Path(runner.state(sequence_id)["run_dir"]).relative_to(runner.store.paths.runs_dir).as_posix()
























def test_start_refuses_an_empty_or_running_sequence(rig):
    store, processes, launcher, runner = rig
    empty_id = store.create("empty")["entry"]["sequence_id"]
    chain_id = _saved(store, [_block("a1")])

    runner.start_sequence(chain_id)

    assert not runner.start_sequence(empty_id)["ok"]
    assert not runner.start_sequence(chain_id)["ok"]


def test_start_refuses_a_mail_step_without_the_smtp_password(rig, monkeypatch):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1", finalize=_finalize("mail_experiment_reports"))], isolate=True)
    monkeypatch.delenv(ReportMailEntryConfig.password_env)

    result = runner.start_sequence(sequence_id)

    assert not result["ok"]
    assert ReportMailEntryConfig.password_env in result["error"]
    assert launcher.calls == []


def test_a_skipped_mail_block_starts_without_the_smtp_password(rig, monkeypatch):
    store, processes, launcher, runner = rig
    blocks      = [_block("a1", finalize=_finalize("mail_experiment_reports"), skip=True), _block("b2", script="infer_backbone", finalize=_finalize("compare_trials"))]
    sequence_id = _saved(store, blocks, isolate=True)
    monkeypatch.delenv(ReportMailEntryConfig.password_env)

    assert runner.start_sequence(sequence_id)["ok"]


def _tune_block(block_id: str, **extra) -> dict:
    return _block(block_id, script="tune", overrides={"run_tag": "tag"}, decisions=[{"path": "model_overrides", "options": [], "waiting": "", "choice": ""}], **extra)


def _write_best_config(runner: SequenceRunner, sequence_id: str, params: dict, model: str = "unet_skip-set_pred") -> None:
    path = Path(runner.state(sequence_id)["run_dir"]) / "tag" / model / "best_config.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"model": model, "trial": 7, "val_loss": 0.1, "params": params}))


def test_a_tuning_block_records_its_best_configuration_as_the_downstream_model_overrides(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_tune_block("t1"), _block("c2", script="cross_validate", predecessors=["t1"])], isolate=True)

    runner.start_sequence(sequence_id)
    _write_best_config(runner, sequence_id, {"encoder_lr": 0.0003, "dropout": 0.25})
    processes.finish("job-0")
    runner.scan()

    assert _block_state(runner, sequence_id, "t1")["status"] == "done"
    assert _block_state(runner, sequence_id, "t1")["verdict"]["values"] == {"model_overrides": "{'encoder_lr': 0.0003, 'dropout': 0.25}"}

    blocks = {block["block_id"]: block for block in store.get(sequence_id)["blocks"]}
    assert blocks["t1"]["decisions"][0]["choice"] == "{'encoder_lr': 0.0003, 'dropout': 0.25}"
    assert blocks["c2"]["overrides"]["model_overrides"] == "{'encoder_lr': 0.0003, 'dropout': 0.25}"
    assert blocks["c2"]["field_notes"]["model_overrides"] == "frozen"

    runner.scan()
    assert launcher.calls[-1]["key"] == "cross_validate"
    assert launcher.calls[-1]["overrides"]["model_overrides"] == "{'encoder_lr': 0.0003, 'dropout': 0.25}"

    decided = [alert for alert in runner.notifier.alerts if alert[0].startswith("sequence block decided")]
    assert len(decided) == 1
    assert "model_overrides = {'encoder_lr': 0.0003, 'dropout': 0.25}" in decided[0][1]


def test_a_tuning_block_holds_its_successor_until_the_verdict_is_recorded(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_tune_block("t1"), _block("c2", script="cross_validate", predecessors=["t1"])], isolate=True)

    runner.start_sequence(sequence_id)
    _write_best_config(runner, sequence_id, {"dropout": 0.1})
    processes.finish("job-0")

    with runner.lock:
        launches = runner._advance(sequence_id, runner.active[sequence_id])

    assert [launch[0] for launch in launches] == ["verdict"]
    assert _block_state(runner, sequence_id, "t1")["status"] == "deciding"
    assert _block_state(runner, sequence_id, "c2")["status"] == "pending"
    assert runner.state(sequence_id)["status"] == "running"


def test_a_tuning_block_without_a_best_configuration_fails_and_skips_its_successor(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_tune_block("t1"), _block("c2", script="cross_validate", predecessors=["t1"])], isolate=True)

    runner.start_sequence(sequence_id)
    processes.finish("job-0")
    runner.scan()
    runner.scan()

    state = _block_state(runner, sequence_id, "t1")
    assert state["status"] == "failed"
    assert "best_config.json" in state["error"]
    assert _block_state(runner, sequence_id, "c2")["status"] == "skipped"
    assert runner.state(sequence_id)["status"] == "failed"
    assert store.get(sequence_id)["blocks"][0]["decisions"][0]["choice"] == ""


def test_a_tuning_block_with_two_best_configurations_fails(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_tune_block("t1")], isolate=True)

    runner.start_sequence(sequence_id)
    _write_best_config(runner, sequence_id, {"dropout": 0.1}, model="unet_skip-set_pred")
    _write_best_config(runner, sequence_id, {"dropout": 0.2}, model="resunet-set_pred")
    processes.finish("job-0")
    runner.scan()

    assert _block_state(runner, sequence_id, "t1")["status"] == "failed"
    assert "found 2" in _block_state(runner, sequence_id, "t1")["error"]


def test_a_tuning_block_records_the_batch_size_on_its_own_axis(rig):
    store, processes, launcher, runner = rig
    block       = _block("t1", script="tune", overrides={"run_tag": "tag"}, decisions=[{"path": "model_overrides", "options": [], "waiting": "", "choice": ""}, {"path": "training.batch_size", "options": [], "waiting": "", "choice": ""}])
    sequence_id = _saved(store, [block, _block("c2", script="cross_validate", predecessors=["t1"])], isolate=True)

    runner.start_sequence(sequence_id)
    _write_best_config(runner, sequence_id, {"dropout": 0.2, "features": [32, 64, 128, 256], "batch_size": 512})
    processes.finish("job-0")
    runner.scan()

    assert _block_state(runner, sequence_id, "t1")["status"] == "done"
    blocks = {block["block_id"]: block for block in store.get(sequence_id)["blocks"]}
    assert blocks["c2"]["overrides"]["model_overrides"]     == "{'dropout': 0.2, 'features': [32, 64, 128, 256]}"
    assert blocks["c2"]["overrides"]["training.batch_size"] == "512"
    assert blocks["c2"]["field_notes"]["training.batch_size"] == "frozen"
    assert blocks["t1"]["decisions"][1]["choice"] == "512"


def test_a_tuning_block_that_sampled_the_batch_without_declaring_its_axis_fails(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_tune_block("t1")], isolate=True)

    runner.start_sequence(sequence_id)
    _write_best_config(runner, sequence_id, {"dropout": 0.2, "batch_size": 512})
    processes.finish("job-0")
    runner.scan()

    assert _block_state(runner, sequence_id, "t1")["status"] == "failed"
    assert "no 'training.batch_size' axis" in _block_state(runner, sequence_id, "t1")["error"]


def test_a_tuning_block_declaring_another_axis_fails_to_record(rig):
    store, processes, launcher, runner = rig
    block       = _block("t1", script="tune", overrides={"run_tag": "tag"}, decisions=[{"path": "training.epochs", "options": ["60"], "waiting": "", "choice": ""}])
    sequence_id = _saved(store, [block], isolate=True)

    runner.start_sequence(sequence_id)
    processes.finish("job-0")
    runner.scan()

    assert _block_state(runner, sequence_id, "t1")["status"] == "failed"
    assert "records only the 'model_overrides' and 'training.batch_size' axes" in _block_state(runner, sequence_id, "t1")["error"]


def test_a_tuning_block_without_decisions_is_plainly_done(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("t1", script="tune", overrides={"run_tag": "tag"})], isolate=True)

    runner.start_sequence(sequence_id)
    processes.finish("job-0")
    runner.scan()

    assert _block_state(runner, sequence_id, "t1")["status"] == "done"
    assert _block_state(runner, sequence_id, "t1")["verdict"] is None


def test_study_progress_reads_the_workspace_snapshots(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_tune_block("t1")], isolate=True)
    snapshot    = {"trials": {"complete": 40, "pruned": 5, "fail": 1, "running": 4}, "finished": 46, "target": 3999, "best": {"trial": 12, "value": 0.39, "params": {}}, "stalled": 28, "plateau_patience": 400, "trials_per_hour": 23.0, "eta_target_h": 172.1}

    runner.start_sequence(sequence_id)
    study = Path(runner.state(sequence_id)["run_dir"]) / "tag" / "unet_skip-set_pred" / "study_progress.json"
    study.parent.mkdir(parents=True)
    study.write_text(json.dumps(snapshot))

    assert runner.study_progress(sequence_id, "t1") == {"ok": True, "studies": {"unet_skip-set_pred": snapshot}}


def test_study_progress_refuses_what_it_cannot_read(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_tune_block("t1"), _block("b2", predecessors=["t1"])], isolate=True)

    assert runner.study_progress(sequence_id, "zz")["error"] == "block not found"
    assert "runs no tuning study" in runner.study_progress(sequence_id, "b2")["error"]
    assert "has not run" in runner.study_progress(sequence_id, "t1")["error"]

    runner.start_sequence(sequence_id)
    assert runner.study_progress(sequence_id, "t1") == {"ok": True, "studies": {}}


def _experiment_tree(run_dir: str, mode: str = "curriculum", trials: tuple = ("rung_a", "rung_b")) -> Path:
    experiment = Path(run_dir) / mode
    for trial in trials:
        (experiment / trial / "seed0").mkdir(parents=True)
    (experiment / "batch_train_logs").mkdir()
    return experiment


def test_a_finalize_block_chains_compare_and_mail_then_ends_done(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1", finalize=_finalize(*FinalizeChain.STEPS))], isolate=True)

    runner.start_sequence(sequence_id)
    experiment = _experiment_tree(runner.state(sequence_id)["run_dir"])
    processes.finish("job-0")
    runner.scan()

    assert _block_state(runner, sequence_id, "a1")["status"] == "finalizing"
    assert launcher.calls[1]["key"] == "compare_seeds"
    assert launcher.calls[1]["overrides"] == {"runs_dir": str(experiment), "group_tags": "rung_a, rung_b"}

    processes.finish("job-1")
    runner.scan()
    assert launcher.calls[2]["key"] == "compare_trials"
    assert launcher.calls[2]["overrides"] == {"runs_dir": str(experiment), "run_tags": "rung_a, rung_b"}

    processes.finish("job-2")
    runner.scan()
    assert launcher.calls[3]["key"] == "mail_experiment_reports"
    assert launcher.calls[3]["overrides"] == {"experiments_dir": str(experiment), "staging_dir": str(Path(runner.state(sequence_id)["run_dir"]) / "_mailed_reports")}

    processes.finish("job-3")
    runner.scan()
    assert _block_state(runner, sequence_id, "a1")["status"] == "done"
    assert runner.state(sequence_id)["status"] == "done"


def test_a_plain_sequence_block_finalizes_on_the_entrys_own_sweep(rig, tmp_path):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1", finalize=_finalize("compare_trials"), overrides={"logdir": str(tmp_path / "own")})])

    runner.start_sequence(sequence_id)
    experiment = _experiment_tree(str(tmp_path / "own"))
    processes.finish("job-0")
    runner.scan()

    assert runner.state(sequence_id)["run_dir"] == ""
    assert _block_state(runner, sequence_id, "a1")["status"] == "finalizing"
    assert launcher.calls[1]["key"] == "compare_trials"
    assert launcher.calls[1]["overrides"] == {"runs_dir": str(experiment), "run_tags": "rung_a, rung_b"}


def test_the_chain_steps_launch_attached_in_a_detached_sequence(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1", finalize=_finalize(*FinalizeChain.STEPS))], isolate=True, detach=True)

    runner.start_sequence(sequence_id)
    _experiment_tree(runner.state(sequence_id)["run_dir"])
    processes.finish("job-0")
    runner.scan()

    assert launcher.calls[0]["detach"] is True
    assert launcher.calls[1]["key"] == "compare_seeds"
    assert launcher.calls[1]["detach"] is False


def test_the_chain_runs_only_the_enabled_switches(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1", finalize=_finalize("compare_trials", "mail_experiment_reports"))], isolate=True)

    runner.start_sequence(sequence_id)
    _experiment_tree(runner.state(sequence_id)["run_dir"])
    processes.finish("job-0")
    runner.scan()

    assert launcher.calls[1]["key"] == "compare_trials"

    processes.finish("job-1")
    runner.scan()
    assert launcher.calls[2]["key"] == "mail_experiment_reports"

    processes.finish("job-2")
    runner.scan()
    assert _block_state(runner, sequence_id, "a1")["status"] == "done"
    assert [call["key"] for call in launcher.calls] == ["train_backbone", "compare_trials", "mail_experiment_reports"]


def test_a_trials_mode_override_names_the_experiment_directory(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1", finalize=_finalize(*FinalizeChain.STEPS), overrides={"trials_mode": "patch"})], isolate=True)

    runner.start_sequence(sequence_id)
    experiment = _experiment_tree(runner.state(sequence_id)["run_dir"], mode="patch")
    processes.finish("job-0")
    runner.scan()

    assert launcher.calls[1]["overrides"]["runs_dir"] == str(experiment)


def test_the_finalize_chain_follows_an_external_logdir(rig, tmp_path):
    store, processes, launcher, runner = rig
    external    = tmp_path / "scratch_runs"
    sequence_id = _saved(store, [_block("a1", finalize=_finalize("compare_trials"), overrides={"logdir": str(external)})], isolate=True)

    runner.start_sequence(sequence_id)
    experiment = _experiment_tree(str(external))
    processes.finish("job-0")
    runner.scan()

    assert launcher.calls[0]["overrides"]["logdir"] == str(external)
    assert _block_state(runner, sequence_id, "a1")["status"] == "finalizing"
    assert launcher.calls[1]["overrides"]["runs_dir"] == str(experiment)


def test_finalizing_does_not_hold_back_the_successors(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1", finalize=_finalize(*FinalizeChain.STEPS)), _block("b2", script="infer_backbone", predecessors=["a1"])], isolate=True)

    runner.start_sequence(sequence_id)
    _experiment_tree(runner.state(sequence_id)["run_dir"])
    processes.finish("job-0")
    runner.scan()

    assert [call["key"] for call in launcher.calls] == ["train_backbone", "compare_seeds", "infer_backbone"]
    assert _block_state(runner, sequence_id, "a1")["status"] == "finalizing"
    assert _block_state(runner, sequence_id, "b2")["status"] == "running"


def test_a_failed_chain_step_marks_finalize_failed_and_fails_the_sequence(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1", finalize=_finalize(*FinalizeChain.STEPS)), _block("b2", script="infer_backbone", predecessors=["a1"])], isolate=True)

    runner.start_sequence(sequence_id)
    _experiment_tree(runner.state(sequence_id)["run_dir"])
    processes.finish("job-0")
    runner.scan()
    processes.finish("job-1", "failed")
    runner.scan()

    state = _block_state(runner, sequence_id, "a1")
    assert state["status"] == "finalize_failed"
    assert "compare_seeds" in state["error"]
    assert _block_state(runner, sequence_id, "b2")["status"] == "running"

    processes.finish("job-2")
    runner.scan()
    assert runner.state(sequence_id)["status"] == "failed"


def test_a_missing_experiment_directory_fails_the_chain(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1", finalize=_finalize(*FinalizeChain.STEPS))], isolate=True)

    runner.start_sequence(sequence_id)
    processes.finish("job-0")
    runner.scan()

    state = _block_state(runner, sequence_id, "a1")
    assert state["status"] == "finalize_failed"
    assert "experiment directory not found" in state["error"]
    assert len(launcher.calls) == 1


def test_finalize_refuses_an_entry_without_trials_mode(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1", script="train_jepa", finalize=_finalize(*FinalizeChain.STEPS))], isolate=True)

    runner.start_sequence(sequence_id)

    state = _block_state(runner, sequence_id, "a1")
    assert state["status"] == "failed"
    assert "trials_mode" in state["error"]
    assert launcher.calls == []


def _cv_fold_tree(run_dir: str, run_tag: str) -> Path:
    folds = Path(run_dir) / run_tag / "folds"
    for fold in ("unet.conv::fold_0", "unet.conv::fold_1"):
        (folds / fold / "seed0").mkdir(parents=True)
    (folds / "logs").mkdir()
    return folds


def test_a_cross_validation_block_finalizes_on_its_fold_sweep(rig):
    store, processes, launcher, runner = rig
    block       = _block("a1", script="cross_validate", finalize=_finalize(*FinalizeChain.STEPS), overrides={"run_tag": "cv1"})
    sequence_id = _saved(store, [block], isolate=True)

    runner.start_sequence(sequence_id)
    folds = _cv_fold_tree(runner.state(sequence_id)["run_dir"], "cv1")
    processes.finish("job-0")
    runner.scan()

    assert _block_state(runner, sequence_id, "a1")["status"] == "finalizing"
    assert launcher.calls[1]["key"] == "compare_seeds"
    assert launcher.calls[1]["overrides"] == {"runs_dir": str(folds), "group_tags": "unet.conv::fold_0, unet.conv::fold_1"}

    processes.finish("job-1")
    runner.scan()
    assert launcher.calls[2]["key"] == "compare_trials"
    assert launcher.calls[2]["overrides"] == {"runs_dir": str(folds), "run_tags": "unet.conv::fold_0, unet.conv::fold_1"}

    processes.finish("job-2")
    runner.scan()
    assert launcher.calls[3]["key"] == "mail_experiment_reports"
    assert launcher.calls[3]["overrides"] == {"experiments_dir": str(folds), "staging_dir": str(Path(runner.state(sequence_id)["run_dir"]) / "_mailed_reports")}

    processes.finish("job-3")
    runner.scan()
    assert _block_state(runner, sequence_id, "a1")["status"] == "done"
    assert runner.state(sequence_id)["status"] == "done"


def test_finalize_refuses_a_cross_validation_block_without_run_tag(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1", script="cross_validate", finalize=_finalize("compare_trials"))], isolate=True)

    runner.start_sequence(sequence_id)

    state = _block_state(runner, sequence_id, "a1")
    assert state["status"] == "failed"
    assert "run_tag" in state["error"]
    assert launcher.calls == []


def test_stop_stops_a_running_chain_job(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1", finalize=_finalize(*FinalizeChain.STEPS))], isolate=True)

    runner.start_sequence(sequence_id)
    _experiment_tree(runner.state(sequence_id)["run_dir"])
    processes.finish("job-0")
    runner.scan()
    runner.stop_sequence(sequence_id)

    assert "job-1" in processes.stopped
    assert _block_state(runner, sequence_id, "a1")["status"] == "cancelled"


def test_resume_scan_proposes_a_finalize_failed_block_for_skipping(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1", finalize=_finalize(*FinalizeChain.STEPS))], isolate=True)

    runner.start_sequence(sequence_id)
    workspace = runner.state(sequence_id)["run_dir"]
    _experiment_tree(workspace)
    processes.finish("job-0")
    runner.scan()
    processes.finish("job-1", "failed")
    runner.scan()

    result = runner.resume_scan(sequence_id, workspace)

    assert result["ok"], result
    assert result["done"] == ["a1"]


def test_a_failed_job_alerts_the_block_and_the_ended_sequence(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1"), _block("b2", predecessors=["a1"])])

    runner.start_sequence(sequence_id)
    processes.finish("job-0", "failed")
    runner.scan()

    titles = [title for title, body, priority in runner.notifier.alerts]

    assert titles == ["sequence block failed: train_backbone", "sequence failed: seq"]
    assert all(priority == "high" for title, body, priority in runner.notifier.alerts)
    assert "job-0 ended as failed" in runner.notifier.alerts[0][1]
    assert "job-0 ended as failed" in runner.notifier.alerts[1][1]


def test_a_finished_sequence_alerts_at_default_priority(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1"), _block("b2", predecessors=["a1"])])

    runner.start_sequence(sequence_id)
    processes.finish("job-0")
    runner.scan()
    processes.finish("job-1")
    runner.scan()

    assert runner.notifier.alerts == [("sequence finished: seq", "all 2 block(s) completed", "default")]




def test_a_stopped_sequence_does_not_alert(rig):
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1"), _block("b2", predecessors=["a1"])])

    runner.start_sequence(sequence_id)
    runner.stop_sequence(sequence_id)
    runner.scan()

    assert runner.notifier.alerts == []
