from __future__ import annotations

import sys

import pytest

from process_manager import JobStream

from tests.webui.conftest import ARGS_DUMP, SLEEP_LONG, job_record, wait_for_status

SLEEP_OK   = "import time\ntime.sleep(0.6)\n"
SLEEP_FAIL = "import sys, time\ntime.sleep(0.6)\nsys.exit(3)\n"
WRITER     = "import pathlib, sys\npathlib.Path('order.txt').open('a').write(pathlib.Path(__file__).stem + '\\n')\n"


@pytest.fixture
def manager(make_manager):
    return make_manager({
        "sleep_ok"   : SLEEP_OK,
        "sleep_fail" : SLEEP_FAIL,
        "sleep_long" : SLEEP_LONG,
        "writer_a"   : WRITER,
        "writer_b"   : WRITER,
        "args_dump"  : ARGS_DUMP,
    })


def test_enqueue_idle_starts_immediately(manager):
    result = manager.enqueue("sleep_ok", sys.executable)

    assert result["ok"]
    assert result["queued"] is False
    assert wait_for_status(manager, result["job_id"], "finished")


def test_enqueue_waits_for_running_job(manager):
    running = manager.launch("sleep_ok", sys.executable)
    queued  = manager.enqueue("sleep_ok", sys.executable)

    assert queued["ok"]
    assert queued["queued"] is True
    assert job_record(manager, queued["job_id"])["status"] == "queued"

    assert wait_for_status(manager, queued["job_id"], "finished")
    assert job_record(manager, running["job_id"])["status"] == "finished"


def test_queued_job_starts_after_predecessor_fails(manager):
    failing = manager.launch("sleep_fail", sys.executable)
    queued  = manager.enqueue("sleep_ok", sys.executable)

    assert wait_for_status(manager, queued["job_id"], "finished")
    assert job_record(manager, failing["job_id"])["status"] == "failed"


def test_queued_jobs_run_in_order(manager, tmp_path):
    manager.launch("sleep_ok", sys.executable)
    first  = manager.enqueue("writer_a", sys.executable)
    second = manager.enqueue("writer_b", sys.executable)

    assert wait_for_status(manager, first["job_id"], "finished")
    assert wait_for_status(manager, second["job_id"], "finished")
    assert (tmp_path / "order.txt").read_text().splitlines() == ["writer_a", "writer_b"]


def test_queued_launch_keeps_overrides(manager, tmp_path):
    manager.launch("sleep_ok", sys.executable)
    queued = manager.enqueue("args_dump", sys.executable, {"training.seed": "7"})

    assert wait_for_status(manager, queued["job_id"], "finished")
    assert (tmp_path / "argv.txt").read_text() == "--training.seed 7"


def test_cancelled_queued_job_is_skipped(manager, tmp_path):
    manager.launch("sleep_ok", sys.executable)
    first  = manager.enqueue("writer_a", sys.executable)
    second = manager.enqueue("writer_b", sys.executable)

    assert manager.stop(first["job_id"])["ok"]
    assert job_record(manager, first["job_id"])["status"] == "cancelled"

    assert wait_for_status(manager, second["job_id"], "finished")
    assert job_record(manager, first["job_id"])["status"] == "cancelled"
    assert (tmp_path / "order.txt").read_text().splitlines() == ["writer_b"]


def test_stop_all_purges_queue(manager):
    running = manager.launch("sleep_long", sys.executable)
    queued  = manager.enqueue("sleep_ok", sys.executable)

    assert manager.stop_all() == 1
    assert job_record(manager, queued["job_id"])["status"] == "cancelled"
    assert job_record(manager, queued["job_id"])["pid"] is None

    assert wait_for_status(manager, running["job_id"], "failed")
    assert job_record(manager, queued["job_id"])["status"] == "cancelled"


def test_queue_respects_follow_up_chain(manager, tmp_path):
    manager.launch("sleep_ok", sys.executable)
    parent = manager.enqueue("writer_a", sys.executable, follow_up="args_dump")
    last   = manager.enqueue("writer_b", sys.executable)

    assert wait_for_status(manager, last["job_id"], "finished")

    follow_id = job_record(manager, parent["job_id"])["follow_up"]
    assert follow_id is not None
    assert job_record(manager, follow_id)["status"] == "finished"

    order = (tmp_path / "order.txt").read_text().splitlines()
    assert order == ["writer_a", "writer_b"]
    assert (tmp_path / "argv.txt").exists()


def test_enqueue_rejects_missing_script(manager):
    result = manager.enqueue("missing", sys.executable)
    assert result == {"ok": False, "error": "script not found"}


def test_notifications_fire_on_start_and_finish_for_direct_and_queued(manager):
    events = []
    manager.notifier.job_started  = lambda record: events.append(("started", record["script"]))
    manager.notifier.job_finished = lambda record: events.append(("finished", record["script"]))

    manager.launch("sleep_ok", sys.executable)
    queued = manager.enqueue("writer_a", sys.executable)

    assert wait_for_status(manager, queued["job_id"], "finished")
    assert events == [("started", "sleep_ok"), ("finished", "sleep_ok"), ("started", "writer_a"), ("finished", "writer_a")]


def test_the_console_history_drops_the_oldest_ended_jobs(manager):
    manager.HISTORY_LIMIT = 2

    with manager.lock:
        manager.jobs["live"]    = {"job_id": "live", "status": "running", "started": "2026-07-14T00:00:00"}
        manager.streams["live"] = JobStream()
        for index in range(4):
            job_id                  = f"job{index}"
            manager.jobs[job_id]    = {"job_id": job_id, "status": "finished", "started": f"2026-07-14T00:0{index + 1}:00"}
            manager.streams[job_id] = JobStream()

    manager._prune_history()

    with manager.lock:
        assert sorted(manager.jobs)    == ["job2", "job3", "live"]
        assert sorted(manager.streams) == ["job2", "job3", "live"]
