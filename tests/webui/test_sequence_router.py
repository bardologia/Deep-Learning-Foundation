"""Tests for the sequence API router.

Covers the create/list/detail/save/delete routes over a real store, the
execution routes over a real runner with faked processes, and the guard that
refuses deleting a sequence while it runs.
"""

from __future__ import annotations

import json

import pytest

from routers.sequence_router import SequenceRouter
from sequence_runner         import SequenceRunner
from sequence_store          import SequenceStore
from web_logger              import WebLogger

from tests.webui.conftest             import FakeHandler
from tests.webui.test_sequence_store  import LEAVES, FakeResolver, SequencePaths, _block, _definition, decision_engine
from tests.webui.test_sequence_runner import FakeLauncher, FakeNotifier, FakeProcesses


@pytest.fixture
def rig(tmp_path):
    """Returns a router over a real store and runner with faked backends."""
    store     = SequenceStore(SequencePaths(tmp_path), WebLogger(), decision_engine())
    processes = FakeProcesses()
    runner    = SequenceRunner(store, processes, FakeLauncher(processes), FakeResolver(LEAVES), FakeNotifier(), WebLogger())
    return SequenceRouter(store, runner), store, runner, processes


def _request(router: SequenceRouter, method: str, path: str, body: dict | None = None) -> FakeHandler:
    """Routes one request through the router's table and returns the handler."""
    from routers.dispatch import HttpExchange

    handler  = FakeHandler(method, path, json.dumps(body or {}).encode("utf-8"))
    exchange = HttpExchange.of(handler)
    exchange.read_body()
    assert router.handle(exchange)
    return handler


def test_create_then_list_round_trips(rig):
    """A created sequence appears in the listing with a null live state."""
    router, store, runner, processes = rig

    created = _request(router, "POST", "/api/sequences", {"name": "ladder"})
    listed  = _request(router, "GET", "/api/sequences")

    assert created.status == 200
    assert listed.payload()["sequences"][0]["name"] == "ladder"
    assert listed.payload()["sequences"][0]["state"] is None


def test_detail_answers_404_for_an_unknown_sequence(rig):
    """The detail route answers 404 when the id matches no file."""
    router = rig[0]

    handler = _request(router, "GET", "/api/sequences/ffffffffffff")

    assert handler.status == 404


def test_save_validates_and_persists_the_definition(rig):
    """A valid save lands on disk and an invalid one answers 400."""
    router, store = rig[0], rig[1]
    sequence_id   = store.create("seq")["entry"]["sequence_id"]

    good = _request(router, "POST", f"/api/sequences/{sequence_id}/save", _definition([_block("a1")]))
    bad  = _request(router, "POST", f"/api/sequences/{sequence_id}/save", _definition([_block("a1", predecessors=["a1"])]))

    assert good.status == 200
    assert bad.status == 400
    assert len(store.get(sequence_id)["blocks"]) == 1


def test_start_release_and_stop_drive_the_runner(rig):
    """The execution routes start the sequence, release a held block and stop the run."""
    router, store, runner, processes = rig
    sequence_id = store.create("seq")["entry"]["sequence_id"]
    store.save(sequence_id, _definition([_block("a1"), _block("b2", predecessors=["a1"], checkpoint=True)]))

    started = _request(router, "POST", f"/api/sequences/{sequence_id}/start")
    processes.finish("job-0")
    runner.scan()

    detail = _request(router, "GET", f"/api/sequences/{sequence_id}").payload()
    assert started.status == 200
    assert detail["state"]["blocks"]["b2"]["status"] == "held"

    released = _request(router, "POST", f"/api/sequences/{sequence_id}/release", {"block_id": "b2"})
    stopped  = _request(router, "POST", f"/api/sequences/{sequence_id}/stop")

    assert released.status == 200
    assert stopped.status == 200
    assert runner.state(sequence_id)["status"] == "stopped"


def test_clear_route_refuses_a_running_sequence_then_resets_a_stopped_one(rig):
    """The clear route answers 400 while the sequence runs and drops the state once it is stopped."""
    router, store, runner, processes = rig
    sequence_id = store.create("seq")["entry"]["sequence_id"]
    store.save(sequence_id, _definition([_block("a1")]))

    _request(router, "POST", f"/api/sequences/{sequence_id}/start")
    refused = _request(router, "POST", f"/api/sequences/{sequence_id}/clear")
    _request(router, "POST", f"/api/sequences/{sequence_id}/stop")
    cleared = _request(router, "POST", f"/api/sequences/{sequence_id}/clear")
    detail  = _request(router, "GET", f"/api/sequences/{sequence_id}").payload()

    assert refused.status == 400
    assert cleared.status == 200
    assert cleared.payload()["cleared"] is True
    assert detail["state"] is None


def test_resume_scan_route_reports_the_finished_blocks(rig, tmp_path):
    """The resume-scan route answers with the block ids the workspace records as done."""
    router, store, runner, processes = rig
    sequence_id = store.create("seq")["entry"]["sequence_id"]
    store.save(sequence_id, _definition([_block("a1"), _block("b2", predecessors=["a1"])], isolate=True))

    runner.start_sequence(sequence_id)
    workspace = runner.state(sequence_id)["run_dir"]
    processes.finish("job-0")
    runner.scan()
    runner.stop_sequence(sequence_id)

    handler = _request(router, "POST", f"/api/sequences/{sequence_id}/resume-scan", {"resume_dir": workspace})
    payload = json.loads(handler.wfile.getvalue())

    assert handler.status == 200
    assert payload["recorded"] is True
    assert payload["done"] == ["a1"]


def test_start_accepts_a_resume_workspace_and_bypass_list_in_the_body(rig, tmp_path):
    """A start carrying resume_dir and bypass runs inside that workspace with the listed blocks bypassed."""
    router, store, runner, processes = rig
    sequence_id = store.create("seq")["entry"]["sequence_id"]
    store.save(sequence_id, _definition([_block("a1"), _block("b2", predecessors=["a1"])], isolate=True))
    workspace = tmp_path / "runs" / "sequences" / "old"
    workspace.mkdir(parents=True)

    started = _request(router, "POST", f"/api/sequences/{sequence_id}/start", {"resume_dir": str(workspace), "bypass": ["a1"]})

    assert started.status == 200
    assert runner.state(sequence_id)["run_dir"] == str(workspace)
    assert runner.state(sequence_id)["blocks"]["a1"]["status"] == "bypassed"
    assert runner.state(sequence_id)["blocks"]["b2"]["status"] == "running"


def test_delete_refuses_a_running_sequence_then_deletes_a_stopped_one(rig):
    """Deletion answers 400 while the sequence runs and succeeds after it is stopped."""
    router, store, runner, processes = rig
    sequence_id = store.create("seq")["entry"]["sequence_id"]
    store.save(sequence_id, _definition([_block("a1")]))
    runner.start_sequence(sequence_id)

    refused = _request(router, "POST", f"/api/sequences/{sequence_id}/delete")
    runner.stop_sequence(sequence_id)
    deleted = _request(router, "POST", f"/api/sequences/{sequence_id}/delete")

    assert refused.status == 400
    assert deleted.status == 200
    assert store.get(sequence_id) is None


def test_decide_route_freezes_the_verdict_into_the_later_blocks(rig):
    """The decide route records the winner and answers with the re-stamped definition."""
    router, store, runner, processes = rig
    sequence_id = store.create("seq")["entry"]["sequence_id"]
    decisions   = [{"path": "training.seed", "options": ["1", "7"], "waiting": "1", "choice": ""}]
    store.save(sequence_id, _definition([_block("a1", decisions=decisions), _block("b2", predecessors=["a1"])]))

    handler = _request(router, "POST", f"/api/sequences/{sequence_id}/decide", {"block_id": "a1", "path": "training.seed", "choice": "7"})
    payload = json.loads(handler.wfile.getvalue())

    saved = {block["block_id"]: block for block in payload["entry"]["blocks"]}
    assert handler.status == 200
    assert saved["b2"]["overrides"] == {"training.seed": "7"}
    assert saved["b2"]["field_notes"] == {"training.seed": "frozen"}


def test_study_route_answers_the_tuning_block_snapshots(rig):
    """The study route names the block in its query and answers with the runner's snapshots."""
    router, store, runner, processes = rig
    sequence_id = store.create("seq")["entry"]["sequence_id"]
    store.save(sequence_id, _definition([_block("t1", script="tune", overrides={"run_tag": "tag"})], isolate=True))
    runner.start_sequence(sequence_id)

    handler = _request(router, "GET", f"/api/sequences/{sequence_id}/study?block=t1")

    assert handler.status == 200
    assert json.loads(handler.wfile.getvalue()) == {"ok": True, "studies": {}}

    handler = _request(router, "GET", f"/api/sequences/{sequence_id}/study?block=zz")

    assert handler.status == 400
    assert json.loads(handler.wfile.getvalue())["error"] == "block not found"


def test_decide_route_reports_an_unknown_axis(rig):
    """Deciding an axis the block never declared answers with an error result."""
    router, store, runner, processes = rig
    sequence_id = store.create("seq")["entry"]["sequence_id"]
    store.save(sequence_id, _definition([_block("a1")]))

    handler = _request(router, "POST", f"/api/sequences/{sequence_id}/decide", {"block_id": "a1", "path": "training.seed", "choice": "7"})
    payload = json.loads(handler.wfile.getvalue())

    assert not payload["ok"]
    assert "does not decide" in payload["error"]
