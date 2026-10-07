"""Tests of the sequence runner: DAG advance, checkpoints, skips, isolation and finalize chains.

Covers block launching over the process manager, failure propagation, held
checkpoints, the isolated workspace redirection, the persisted state file and
resume scans, the finalize chains, and the notifier alerts on failures.
"""

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
    """Process-manager stand-in exposing the job table, lock and stop the runner uses.

    Attributes:
        jobs    : Job records keyed by job id, each carrying only a status.
        lock    : Lock guarding the job table.
        stopped : Job ids the runner asked to stop, in call order.
    """

    def __init__(self) -> None:
        """Starts with an empty job table."""
        self.jobs    = {}
        self.lock    = threading.Lock()
        self.stopped = []

    def finish(self, job_id: str, status: str = "finished") -> None:
        """Moves one job to a terminal status."""
        with self.lock:
            self.jobs[job_id]["status"] = status

    def stop(self, job_id: str) -> dict:
        """Records the stop request and marks the job cancelled."""
        self.stopped.append(job_id)
        self.finish(job_id, "cancelled")
        return {"ok": True}


class FakeLauncher:
    """Launcher stand-in that registers a running job per launch and records the call.

    Attributes:
        processes : Fake process manager the launched jobs are registered in.
        calls     : One record per launch with the key, interpreter, overrides and detach flag.
    """

    def __init__(self, processes: FakeProcesses) -> None:
        """Stores the fake process manager and starts with no calls."""
        self.processes = processes
        self.calls     = []

    def preferred_interpreter(self, key: str) -> str:
        """Returns a fixed interpreter path."""
        return "/usr/bin/python3"

    def execute(self, key: str, interpreter: str, overrides: dict, follow_up, detach: bool, queue: bool) -> dict:
        """Registers a running job for the launch and returns its id."""
        job_id = f"job-{len(self.calls)}"
        self.calls.append({"key": key, "interpreter": interpreter, "overrides": overrides, "detach": detach})
        with self.processes.lock:
            self.processes.jobs[job_id] = {"status": "running"}
        return {"ok": True, "job_id": job_id}








class FakeNotifier:
    """Notifier stand-in recording every alert the runner pushes.

    Attributes:
        alerts : Tuples of (title, body, priority) in call order.
    """

    def __init__(self) -> None:
        """Starts with no recorded alerts."""
        self.alerts = []

    def system_alert(self, title: str, body: str, priority: str = "high") -> None:
        """Records one alert."""
        self.alerts.append((title, body, priority))


class LockProbingLauncher(FakeLauncher):
    """Launcher stand-in recording whether the runner lock is free while a block launches.

    Attributes:
        runner    : Runner whose lock is probed, wired in after construction.
        lock_free : One flag per launch, True when the runner lock was acquirable.
    """

    def __init__(self, processes: FakeProcesses) -> None:
        """Starts with no runner wired and no probes recorded."""
        super().__init__(processes)
        self.runner    = None
        self.lock_free = []

    def execute(self, key: str, interpreter: str, overrides: dict, follow_up, detach: bool, queue: bool) -> dict:
        """Probes the runner lock, then launches as the plain fake does."""
        acquired = self.runner.lock.acquire(blocking=False)
        self.lock_free.append(acquired)
        if acquired:
            self.runner.lock.release()
        return super().execute(key, interpreter, overrides, follow_up, detach, queue)


class StoppingLauncher(FakeLauncher):
    """Launcher stand-in that stops its sequence from inside the launch call.

    Attributes:
        runner      : Runner the stop is requested on, wired in after construction.
        sequence_id : Identifier of the sequence stopped mid-launch.
    """

    def __init__(self, processes: FakeProcesses) -> None:
        """Starts with no runner and no sequence wired."""
        super().__init__(processes)
        self.runner      = None
        self.sequence_id = ""

    def execute(self, key: str, interpreter: str, overrides: dict, follow_up, detach: bool, queue: bool) -> dict:
        """Stops the sequence, then launches as the plain fake does."""
        self.runner.stop_sequence(self.sequence_id)
        return super().execute(key, interpreter, overrides, follow_up, detach, queue)


@pytest.fixture
def rig(tmp_path, monkeypatch):
    """Returns a store, fake backends and runner wired together, with the mail password set."""
    monkeypatch.setenv(ReportMailEntryConfig.password_env, "pw")
    store     = SequenceStore(SequencePaths(tmp_path), WebLogger(), decision_engine())
    processes = FakeProcesses()
    launcher  = FakeLauncher(processes)
    runner    = SequenceRunner(store, processes, launcher, FakeResolver(LEAVES), FakeNotifier(), WebLogger())
    return store, processes, launcher, runner


def _saved(store: SequenceStore, blocks: list, **extra) -> str:
    """Creates a sequence holding the given blocks and returns its id."""
    sequence_id = store.create("seq")["entry"]["sequence_id"]
    result      = store.save(sequence_id, _definition(blocks, **extra))
    assert result["ok"], result
    return sequence_id


def _block_state(runner: SequenceRunner, sequence_id: str, block_id: str) -> dict:
    """Returns one block's live state snapshot."""
    return runner.state(sequence_id)["blocks"][block_id]


def _finalize(*steps: str) -> dict:
    """Returns a finalize-switch dict with the given chain steps on and the rest off."""
    return {step: step in steps for step in FinalizeChain.STEPS}


def test_start_launches_only_the_blocks_without_predecessors(rig):
    """Starting launches every root block and leaves dependent blocks pending."""
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1"), _block("b2", predecessors=["a1"])])

    runner.start_sequence(sequence_id)

    assert [call["key"] for call in launcher.calls] == ["train_backbone"]
    assert _block_state(runner, sequence_id, "a1")["status"] == "running"
    assert _block_state(runner, sequence_id, "b2")["status"] == "pending"


def test_a_block_launches_when_its_predecessor_finishes(rig):
    """A dependent block launches on the scan after its predecessor's job finishes."""
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1"), _block("b2", script="infer_backbone", predecessors=["a1"])])

    runner.start_sequence(sequence_id)
    processes.finish("job-0")
    runner.scan()

    assert [call["key"] for call in launcher.calls] == ["train_backbone", "infer_backbone"]
    assert _block_state(runner, sequence_id, "a1")["status"] == "done"
    assert _block_state(runner, sequence_id, "b2")["status"] == "running"


def test_independent_blocks_launch_in_parallel(rig):
    """Two blocks sharing one finished predecessor launch in the same scan."""
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1"), _block("b2", predecessors=["a1"]), _block("c3", predecessors=["a1"])])

    runner.start_sequence(sequence_id)
    processes.finish("job-0")
    runner.scan()

    assert len(launcher.calls) == 3
    assert _block_state(runner, sequence_id, "b2")["status"] == "running"
    assert _block_state(runner, sequence_id, "c3")["status"] == "running"


def test_a_checkpoint_block_holds_until_released(rig):
    """A ready checkpoint block goes to held, does not launch, and launches on release."""
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


def test_release_refuses_a_block_that_is_not_held(rig):
    """Releasing a pending or unknown block is refused."""
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1"), _block("b2", predecessors=["a1"])])

    runner.start_sequence(sequence_id)

    assert not runner.release(sequence_id, "b2")["ok"]
    assert not runner.release(sequence_id, "ghost")["ok"]


def test_a_failed_block_skips_its_descendants(rig):
    """When a job fails, every downstream block is skipped and the sequence ends failed."""
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
    """Training blocks get both path fields and multi-dataset entries get base plus filter."""
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
    """Without a dataset selection, the stored overrides launch exactly as saved."""
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1", overrides={"training.seed": "3"})])

    runner.start_sequence(sequence_id)

    assert launcher.calls[0]["overrides"] == {"training.seed": "3"}


def test_a_block_edited_while_pending_launches_with_the_new_overrides(rig):
    """Edits saved while a block waits are picked up at its launch."""
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1"), _block("b2", predecessors=["a1"])])

    runner.start_sequence(sequence_id)
    store.save(sequence_id, _definition([_block("a1"), _block("b2", predecessors=["a1"], overrides={"training.seed": "42"})]))
    processes.finish("job-0")
    runner.scan()

    assert launcher.calls[1]["overrides"] == {"training.seed": "42"}


def test_stop_stops_running_jobs_and_skips_waiting_blocks(rig):
    """Stopping cancels the running blocks' jobs and skips everything not yet launched."""
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
    """Clearing refuses while the sequence runs, drops a stopped state, and reports nothing left afterwards."""
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
    """The blocking launch call runs with the runner lock free so the API stays responsive."""
    store     = SequenceStore(SequencePaths(tmp_path), WebLogger(), decision_engine())
    processes = FakeProcesses()
    launcher  = LockProbingLauncher(processes)
    runner    = SequenceRunner(store, processes, launcher, FakeResolver(LEAVES), FakeNotifier(), WebLogger())
    launcher.runner = runner

    runner.start_sequence(_saved(store, [_block("a1")]))

    assert launcher.lock_free == [True]


def test_a_stop_landing_mid_launch_stops_the_job_it_started(tmp_path):
    """A stop during the launch cancels the block and tears down the job the launch produced."""
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
    """The sequence ends done after every block finishes and a restart resets the states."""
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
    """A block marked skip never launches, counts as satisfied downstream, and the sequence ends done."""
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1", skip=True), _block("b2", script="infer_backbone", predecessors=["a1"])])

    runner.start_sequence(sequence_id)

    assert [call["key"] for call in launcher.calls] == ["infer_backbone"]
    assert _block_state(runner, sequence_id, "a1")["status"] == "bypassed"

    processes.finish("job-0")
    runner.scan()

    assert runner.state(sequence_id)["status"] == "done"


def test_a_skipped_block_waits_for_its_predecessors_before_bypassing(rig):
    """A skip mark must not unlock the chain early: the bypass lands only when the skipped block's own predecessors finish."""
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
    """Two adjacent skip marks bypass one after the other in a single scan once the block before them finishes."""
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
    """Flipping skip on a pending block while the sequence runs bypasses it when its turn comes."""
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
    """A held checkpoint block toggled to skip is bypassed on the next scan and the chain continues."""
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
    """Each isolated execution gets a fresh directory; training writes into it and inference reads from it."""
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
    """A runs_dir whose default points at the dataset tree keeps its value under isolation."""
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1", script="generate_dataset")], isolate=True)

    runner.start_sequence(sequence_id)

    assert "runs_dir" not in launcher.calls[0]["overrides"]




def test_a_training_logdir_is_redirected_into_the_workspace(rig):
    """An entry naming its run-output root logdir is isolated like one naming it paths.log_base_dir."""
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1", script="train_jepa")], isolate=True)

    runner.start_sequence(sequence_id)

    assert launcher.calls[0]["overrides"]["logdir"] == runner.state(sequence_id)["run_dir"]


def test_a_logdir_override_outside_the_runs_tree_is_kept(rig):
    """A logdir deliberately pointing away from the runs tree survives isolation."""
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1", script="train_jepa", overrides={"logdir": "/data/scratch_runs"})], isolate=True)

    runner.start_sequence(sequence_id)

    assert launcher.calls[0]["overrides"]["logdir"] == "/data/scratch_runs"


def test_a_runs_dir_override_outside_the_runs_tree_is_kept(rig):
    """An explicit runs_dir override pointing elsewhere survives isolation, default notwithstanding."""
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1", script="infer_backbone", overrides={"runs_dir": "/data/pulled_runs"})], isolate=True)

    runner.start_sequence(sequence_id)

    assert launcher.calls[0]["overrides"]["runs_dir"] == "/data/pulled_runs"


def test_each_start_gets_its_own_workspace(rig):
    """Restarting a finished sequence creates a second directory instead of reusing the first."""
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
    """A resumed start runs inside the fed workspace and bypasses the blocks fed as finished."""
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
    """The bypass list dies with its execution: the stored skip toggles stay off and a fresh start replays the bypassed block."""
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
    """Feeding a directory that does not exist refuses the start instead of creating it."""
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1")], isolate=True)

    result = runner.start_sequence(sequence_id, str(tmp_path / "runs" / "sequences" / "ghost"))

    assert not result["ok"]
    assert "does not exist" in result["error"]
    assert launcher.calls == []


def test_resume_refuses_a_workspace_outside_the_sequences_tree(rig, tmp_path):
    """Feeding a directory outside runs/sequences refuses the start."""
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1")], isolate=True)
    elsewhere   = tmp_path / "elsewhere"
    elsewhere.mkdir()

    result = runner.start_sequence(sequence_id, str(elsewhere))

    assert not result["ok"]
    assert "directly under" in result["error"]


def test_the_live_state_is_mirrored_into_the_workspace(rig):
    """An isolated execution writes its state file on start and refreshes it as blocks finish."""
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
    """The persist pass holds the runner lock across the staging write and replaces the target in one step, leaving no staging file behind."""
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
    """An interrupted staging write never touches the recorded state, and the next pass rewrites it instead of caching the lost text."""
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
    """Stopping refreshes the state file, so a later scan sees the cancelled blocks."""
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1"), _block("b2", predecessors=["a1"])], isolate=True)

    runner.start_sequence(sequence_id)
    runner.stop_sequence(sequence_id)
    recorded = json.loads((Path(runner.state(sequence_id)["run_dir"]) / SequenceRunner.STATE_FILENAME).read_text())

    assert recorded["status"] == "stopped"
    assert recorded["blocks"]["a1"]["status"] == "cancelled"
    assert recorded["blocks"]["b2"]["status"] == "skipped"


def test_resume_scan_reports_the_blocks_recorded_done(rig, tmp_path):
    """A fresh runner, as after a server restart, reads the finished blocks from the state file."""
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
    """A block the previous execution bypassed as resumed-finished stays proposed on the next resume scan."""
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
    """A block bypassed by its skip toggle never ran, so the scan leaves it out of the finished proposal."""
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
    """A block recorded done but removed from the definition is not proposed for skipping."""
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
    """A valid workspace holding no state file answers recorded False instead of guessing."""
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1")], isolate=True)
    workspace   = tmp_path / "runs" / "sequences" / "bare"
    workspace.mkdir(parents=True)

    result = runner.resume_scan(sequence_id, str(workspace))

    assert result["ok"]
    assert result["recorded"] is False
    assert result["done"] == []


def test_resume_scan_refuses_an_invalid_workspace(rig, tmp_path):
    """The scan applies the same workspace validation as the resumed start."""
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1")], isolate=True)
    elsewhere   = tmp_path / "elsewhere"
    elsewhere.mkdir()

    assert not runner.resume_scan(sequence_id, str(elsewhere))["ok"]
    assert not runner.resume_scan(sequence_id, str(tmp_path / "runs" / "sequences" / "ghost"))["ok"]


def test_resume_refuses_an_unisolated_sequence(rig, tmp_path):
    """A sequence with isolation off has no workspace concept, so resuming is refused."""
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1")])
    workspace   = tmp_path / "runs" / "sequences" / "old"
    workspace.mkdir(parents=True)

    result = runner.start_sequence(sequence_id, str(workspace))

    assert not result["ok"]
    assert "does not isolate" in result["error"]


def test_a_sequence_without_isolation_leaves_the_run_roots_alone(rig):
    """With isolation off, no workspace is created and the log directory stays at its default."""
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1")])

    runner.start_sequence(sequence_id)

    assert runner.state(sequence_id)["run_dir"] == ""
    assert "paths.log_base_dir" not in launcher.calls[0]["overrides"]










def _workspace_rel(runner: SequenceRunner, sequence_id: str) -> str:
    """Returns the runs-tree-relative path of a started sequence's workspace."""
    return Path(runner.state(sequence_id)["run_dir"]).relative_to(runner.store.paths.runs_dir).as_posix()
























def test_start_refuses_an_empty_or_running_sequence(rig):
    """Sequences without blocks and sequences already running are refused."""
    store, processes, launcher, runner = rig
    empty_id = store.create("empty")["entry"]["sequence_id"]
    chain_id = _saved(store, [_block("a1")])

    runner.start_sequence(chain_id)

    assert not runner.start_sequence(empty_id)["ok"]
    assert not runner.start_sequence(chain_id)["ok"]


def test_start_refuses_a_mail_step_without_the_smtp_password(rig, monkeypatch):
    """A block mailing reports blocks the start when the server environment lacks the password."""
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1", finalize=_finalize("mail_experiment_reports"))], isolate=True)
    monkeypatch.delenv(ReportMailEntryConfig.password_env)

    result = runner.start_sequence(sequence_id)

    assert not result["ok"]
    assert ReportMailEntryConfig.password_env in result["error"]
    assert launcher.calls == []


def test_a_skipped_mail_block_starts_without_the_smtp_password(rig, monkeypatch):
    """Skip-marked mail blocks and compare-only chains do not demand the password at start."""
    store, processes, launcher, runner = rig
    blocks      = [_block("a1", finalize=_finalize("mail_experiment_reports"), skip=True), _block("b2", script="infer_backbone", finalize=_finalize("compare_trials"))]
    sequence_id = _saved(store, blocks, isolate=True)
    monkeypatch.delenv(ReportMailEntryConfig.password_env)

    assert runner.start_sequence(sequence_id)["ok"]


def _experiment_tree(run_dir: str, mode: str = "curriculum", trials: tuple = ("rung_a", "rung_b")) -> Path:
    """Creates a trial-sweep directory with seed runs inside a workspace and returns it."""
    experiment = Path(run_dir) / mode
    for trial in trials:
        (experiment / trial / "seed0").mkdir(parents=True)
    (experiment / "batch_train_logs").mkdir()
    return experiment


def test_a_finalize_block_chains_compare_and_mail_then_ends_done(rig):
    """The chain runs compare_seeds, compare_trials and the mailer on the block's sweep, then the block is done."""
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
    """Without a workspace, a block with finalize steps on chains on the sweep under its own log directory."""
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
    """The chain steps run attached whatever the sequence's detach flag, so a crashed step ends as failed instead of finished."""
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
    """A block with only the trial comparison and the mailing on skips the seed comparison."""
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
    """The chain follows the block's trials_mode override instead of the entry default."""
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1", finalize=_finalize(*FinalizeChain.STEPS), overrides={"trials_mode": "patch"})], isolate=True)

    runner.start_sequence(sequence_id)
    experiment = _experiment_tree(runner.state(sequence_id)["run_dir"], mode="patch")
    processes.finish("job-0")
    runner.scan()

    assert launcher.calls[1]["overrides"]["runs_dir"] == str(experiment)


def test_the_finalize_chain_follows_an_external_logdir(rig, tmp_path):
    """A block whose logdir deliberately points outside the runs tree finalizes on that directory, not the workspace."""
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
    """Successors launch as soon as the block's own job finishes, beside the running chain."""
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
    """A chain job ending failed marks the block finalize_failed, spares the successors, and fails the sequence."""
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
    """Without the sweep directory on disk, the first chain step refuses to launch and the block ends finalize_failed."""
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
    """A finalize block whose entry exposes no trials_mode fails at launch instead of guessing a directory."""
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1", script="train_jepa", finalize=_finalize(*FinalizeChain.STEPS))], isolate=True)

    runner.start_sequence(sequence_id)

    state = _block_state(runner, sequence_id, "a1")
    assert state["status"] == "failed"
    assert "trials_mode" in state["error"]
    assert launcher.calls == []


def _cv_fold_tree(run_dir: str, run_tag: str) -> Path:
    """Creates a cross-validation fold sweep with seed runs inside a workspace and returns it."""
    folds = Path(run_dir) / run_tag / "folds"
    for fold in ("unet.conv::fold_0", "unet.conv::fold_1"):
        (folds / fold / "seed0").mkdir(parents=True)
    (folds / "logs").mkdir()
    return folds


def test_a_cross_validation_block_finalizes_on_its_fold_sweep(rig):
    """The chain of a cross-validation block compares the fold runs under <workspace>/<run_tag>/folds and mails that folds directory."""
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
    """A cross-validation finalize block whose run tag is left to the timestamp default fails at launch."""
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1", script="cross_validate", finalize=_finalize("compare_trials"))], isolate=True)

    runner.start_sequence(sequence_id)

    state = _block_state(runner, sequence_id, "a1")
    assert state["status"] == "failed"
    assert "run_tag" in state["error"]
    assert launcher.calls == []


def test_stop_stops_a_running_chain_job(rig):
    """Stopping a sequence cancels the chain job of a finalizing block."""
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
    """A block whose chain failed still counts as finished on resume: its own job completed."""
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
    """A job failure pushes one block alert and one sequence-failed alert through the notifier."""
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
    """The sequence-finished notification carries the block count at default priority and nothing else fires."""
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1"), _block("b2", predecessors=["a1"])])

    runner.start_sequence(sequence_id)
    processes.finish("job-0")
    runner.scan()
    processes.finish("job-1")
    runner.scan()

    assert runner.notifier.alerts == [("sequence finished: seq", "all 2 block(s) completed", "default")]




def test_a_stopped_sequence_does_not_alert(rig):
    """A manual stop cancels the blocks without pushing any notification."""
    store, processes, launcher, runner = rig
    sequence_id = _saved(store, [_block("a1"), _block("b2", predecessors=["a1"])])

    runner.start_sequence(sequence_id)
    runner.stop_sequence(sequence_id)
    runner.scan()

    assert runner.notifier.alerts == []
