from __future__ import annotations

import json
import queue
import sys
import time
from pathlib import Path

import pytest

from process_manager        import ProcessManager
from request_router         import RequestRouter
from routers.launch_routers import JobRouter
from web_logger             import WebLogger

from tools.orchestration.gpu_queue import GpuProgressFile

from tests.webui.conftest import SLEEP_LONG, FakeHandler, wait_for_status


@pytest.fixture
def manager(make_manager):
    return make_manager({"train_backbone": SLEEP_LONG})


def _progress_path(manager: ProcessManager, job_id: str) -> Path:
    with manager.lock:
        return GpuProgressFile.resolve(Path(manager.jobs[job_id]["overrides"]["gpus_file"]))


def _unit(name: str, log: Path, status: str, gpu: int | None = None, returncode: int | None = None) -> dict:
    return {"name": name, "log": str(log), "status": status, "gpu": gpu, "duration_s": None, "returncode": returncode}


def _write_snapshot(manager: ProcessManager, job_id: str, units: list[dict]) -> Path:
    path = _progress_path(manager, job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"total": len(units), "done": 0, "failed": 0, "queued": 0, "running": [], "units": units}))
    return path


def _launch_with_units(manager: ProcessManager, tmp_path: Path, units: list[dict]) -> str:
    job_id = manager.launch("train_backbone", sys.executable)["job_id"]
    assert wait_for_status(manager, job_id, "running")
    _write_snapshot(manager, job_id, units)
    return job_id


def _request_log(manager: ProcessManager, path: str) -> FakeHandler:
    router  = RequestRouter(WebLogger(), [JobRouter(None, manager, None, None)])
    handler = FakeHandler("GET", path)

    router.route(handler)
    return handler


def _collect(stream, timeout: float = 6.0) -> list[dict]:
    sub      = stream.subscribe()
    events   = []
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        try:
            event = sub.get(timeout=0.2)
        except queue.Empty:
            continue
        events.append(event)
        if event.get("type") == "end":
            break

    stream.unsubscribe(sub)
    return events


def test_unit_entry_resolves_the_unit_and_its_log_path(manager, tmp_path):
    log    = tmp_path / "stage" / "a" / "worker.log"
    job_id = _launch_with_units(manager, tmp_path, [_unit("a", log, "done", gpu=1, returncode=0)])

    entry = manager.unit_entry(job_id, 0)

    assert entry["ok"] is True
    assert entry["unit"]["name"] == "a"
    assert entry["path"] == log

    manager.stop(job_id)


def test_unit_entry_rejects_an_out_of_range_index(manager, tmp_path):
    job_id = _launch_with_units(manager, tmp_path, [_unit("a", tmp_path / "a.log", "queued")])

    assert "out of range" in manager.unit_entry(job_id, 5)["error"]
    assert "out of range" in manager.unit_entry(job_id, -1)["error"]

    manager.stop(job_id)


def test_unit_entry_before_any_snapshot_explains_itself(manager, tmp_path):
    job_id = manager.launch("train_backbone", sys.executable)["job_id"]
    assert wait_for_status(manager, job_id, "running")

    assert "no progress snapshot yet" in manager.unit_entry(job_id, 0)["error"]

    manager.stop(job_id)


def test_job_log_serves_a_unit_log_flattened(manager, tmp_path):
    log = tmp_path / "stage" / "a" / "worker.log"
    log.parent.mkdir(parents=True)
    log.write_text("epoch 1\r\x1b[31mepoch 2\x1b[0m\n")

    job_id = _launch_with_units(manager, tmp_path, [_unit("a", log, "done", gpu=0, returncode=0)])

    data = manager.job_log(job_id, unit=0)

    assert data["ok"] is True
    assert data["available"] is True
    assert data["path"] == str(log)
    assert data["text"] == "epoch 2\n"

    manager.stop(job_id)


def test_job_log_for_a_unit_without_a_file_is_unavailable(manager, tmp_path):
    job_id = _launch_with_units(manager, tmp_path, [_unit("a", tmp_path / "missing.log", "queued")])

    data = manager.job_log(job_id, unit=0)

    assert data["ok"] is True
    assert data["available"] is False
    assert "has not started" in data["reason"]

    manager.stop(job_id)


def test_log_route_without_a_unit_answers_the_job_log(manager, tmp_path):
    job_id = _launch_with_units(manager, tmp_path, [_unit("a", tmp_path / "a.log", "queued")])

    handler = _request_log(manager, f"/api/jobs/{job_id}/log")
    data    = handler.payload()

    assert handler.status == 200
    assert data["ok"] is True
    assert data["available"] is False
    assert "writes to a pipe" in data["reason"]

    manager.stop(job_id)


def test_log_route_with_a_unit_answers_that_unit_log(manager, tmp_path):
    log = tmp_path / "stage" / "a" / "worker.log"
    log.parent.mkdir(parents=True)
    log.write_text("epoch 1\r\x1b[31mepoch 2\x1b[0m\n")

    job_id = _launch_with_units(manager, tmp_path, [_unit("a", log, "done", gpu=0, returncode=0)])

    handler = _request_log(manager, f"/api/jobs/{job_id}/log?unit=0")
    data    = handler.payload()

    assert handler.status == 200
    assert data["available"] is True
    assert data["path"] == str(log)
    assert data["text"] == "epoch 2\n"

    manager.stop(job_id)


def test_log_route_refuses_an_unparseable_unit_index(manager, tmp_path):
    job_id = _launch_with_units(manager, tmp_path, [_unit("a", tmp_path / "a.log", "queued")])

    handler = _request_log(manager, f"/api/jobs/{job_id}/log?unit=abc")

    assert handler.status == 400
    assert handler.payload() == {"ok": False, "error": "invalid unit index 'abc'"}

    manager.stop(job_id)


def test_unit_stream_of_a_finished_unit_replays_the_log_and_ends(manager, tmp_path):
    log = tmp_path / "stage" / "a" / "worker.log"
    log.parent.mkdir(parents=True)
    log.write_text("all done here\n")

    job_id = _launch_with_units(manager, tmp_path, [_unit("a", log, "done", gpu=0, returncode=0)])

    opened = manager.unit_stream(job_id, 0)
    assert opened["ok"] is True

    events = _collect(opened["stream"])
    kinds  = [event["type"] for event in events]

    assert kinds[0] == "unit"
    assert events[0]["name"] == "a"
    assert "all done here" in "".join(event["data"] for event in events if event["type"] == "chunk")
    assert events[-2]["type"] == "status" and events[-2]["status"] == "done" and events[-2]["code"] == 0
    assert kinds[-1] == "end"

    manager.stop(job_id)


def test_unit_stream_of_an_unstarted_unit_ends_immediately(manager, tmp_path):
    job_id = _launch_with_units(manager, tmp_path, [_unit("a", tmp_path / "missing.log", "queued")])

    opened = manager.unit_stream(job_id, 0)
    events = _collect(opened["stream"])

    assert [event["type"] for event in events] == ["unit", "status", "end"]
    assert events[1]["missing"] is True

    manager.stop(job_id)


def test_unit_stream_follows_a_running_unit_until_stopped(manager, tmp_path):
    log = tmp_path / "stage" / "a" / "worker.log"
    log.parent.mkdir(parents=True)
    log.write_text("first line\n")

    job_id = _launch_with_units(manager, tmp_path, [_unit("a", log, "running", gpu=0)])

    opened = manager.unit_stream(job_id, 0)
    sub    = opened["stream"].subscribe()

    def next_event(timeout: float = 6.0) -> dict:
        return sub.get(timeout=timeout)

    assert next_event()["type"] == "unit"
    assert "first line" in next_event()["data"]

    with open(log, "a") as handle:
        handle.write("second line\n")

    assert "second line" in next_event()["data"]

    opened["stop"].set()
    tail = [next_event(), next_event()]

    assert tail[0]["type"] == "status"
    assert tail[1]["type"] == "end"

    opened["stream"].unsubscribe(sub)
    manager.stop(job_id)


def test_unit_stream_ends_when_the_parent_job_stops(manager, tmp_path):
    log = tmp_path / "stage" / "a" / "worker.log"
    log.parent.mkdir(parents=True)
    log.write_text("running output\n")

    job_id = _launch_with_units(manager, tmp_path, [_unit("a", log, "running", gpu=0)])

    opened = manager.unit_stream(job_id, 0)
    manager.stop(job_id)

    events = _collect(opened["stream"])
    assert events[-1]["type"] == "end"
