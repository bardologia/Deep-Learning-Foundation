from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from process_manager import ProcessManager

from tools.orchestration.gpu_queue import GpuProgressFile

from tests.webui.conftest import SLEEP_LONG, wait_for_status, wait_until_finished


@pytest.fixture
def manager(make_manager):
    return make_manager({name: SLEEP_LONG for name in ("train_backbone", "tune_dataloader")})


def _progress_path(manager: ProcessManager, job_id: str) -> Path:
    with manager.lock:
        return GpuProgressFile.resolve(Path(manager.jobs[job_id]["overrides"]["gpus_file"]))


def _snapshot(done: int = 12, failed: int = 1, total: int = 30) -> dict:
    return {
        "total"        : total,
        "done"         : done,
        "failed"       : failed,
        "queued"       : total - done - failed - 2,
        "running"      : [{"name": "aug-on/seed3", "gpu": 0, "elapsed_s": 310.0}, {"name": "aug-off/seed1", "gpu": 1, "elapsed_s": 95.0}],
        "workers"      : 2,
        "failed_units" : ["aug-off/seed0"] if failed else [],
        "average_s"    : 600.0,
        "elapsed_s"    : 4200.0,
        "eta_s"        : 5400.0,
        "total_s"      : 9600.0,
        "started_at"   : "2026-07-17T10:00:00",
        "finish_at"    : "2026-07-17T14:30:00",
        "updated_at"   : "2026-07-17T13:00:00",
    }


def test_progress_reports_unsupported_for_non_pool_scripts(manager):
    result = manager.launch("tune_dataloader", sys.executable)

    assert manager.progress(result["job_id"]) == {"ok": True, "supported": False, "live": False}

    manager.stop(result["job_id"])


def test_progress_before_the_file_exists_is_not_live(manager):
    result = manager.launch("train_backbone", sys.executable)
    job_id = result["job_id"]

    assert wait_for_status(manager, job_id, "running")

    info = manager.progress(job_id)

    assert info["ok"] is True
    assert info["supported"] is True
    assert info["live"] is False
    assert "progress" not in info
    assert info["path"].endswith(f"{job_id}_progress.json")

    manager.stop(job_id)


def test_progress_reads_the_live_snapshot(manager):
    result = manager.launch("train_backbone", sys.executable)
    job_id = result["job_id"]

    assert wait_for_status(manager, job_id, "running")

    path = _progress_path(manager, job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_snapshot()))

    info = manager.progress(job_id)

    assert info["ok"] is True
    assert info["live"] is True
    assert info["progress"] == _snapshot()

    manager.stop(job_id)


def test_list_jobs_embeds_the_progress_of_running_fan_outs(manager):
    result = manager.launch("train_backbone", sys.executable)
    job_id = result["job_id"]

    assert wait_for_status(manager, job_id, "running")

    path = _progress_path(manager, job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_snapshot()))

    record = next(r for r in manager.list_jobs() if r["job_id"] == job_id)

    assert record["progress"] == _snapshot()

    manager.stop(job_id)


def test_list_jobs_leaves_other_jobs_without_progress(manager):
    result = manager.launch("tune_dataloader", sys.executable)
    job_id = result["job_id"]

    assert wait_for_status(manager, job_id, "running")

    record = next(r for r in manager.list_jobs() if r["job_id"] == job_id)

    assert record["progress"] is None

    manager.stop(job_id)


def test_progress_of_a_finished_job_keeps_the_snapshot_but_is_not_live(manager):
    result = manager.launch("train_backbone", sys.executable)
    job_id = result["job_id"]

    assert wait_for_status(manager, job_id, "running")

    path = _progress_path(manager, job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_snapshot(done=30, failed=0)))

    manager.stop(job_id)
    assert wait_until_finished(manager, job_id)

    info = manager.progress(job_id)

    assert info["live"] is False
    assert info["progress"]["done"] == 30

    record = next(r for r in manager.list_jobs() if r["job_id"] == job_id)
    assert record["progress"] is None


def test_progress_rejects_an_unreadable_file(manager):
    result = manager.launch("train_backbone", sys.executable)
    job_id = result["job_id"]

    assert wait_for_status(manager, job_id, "running")

    path = _progress_path(manager, job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{oops")

    info = manager.progress(job_id)

    assert info["ok"] is False
    assert "unreadable progress file" in info["error"]

    manager.stop(job_id)


def test_progress_reports_an_unknown_job(manager):
    assert manager.progress("nope") == {"ok": False, "error": "unknown job"}
