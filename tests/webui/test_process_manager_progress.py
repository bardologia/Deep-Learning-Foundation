"""Tests for the fan-out progress snapshot exposed by the process manager.

Covers the progress file derived from a job's GPU pool file, liveness while
the job runs, embedding of the snapshot in list_jobs, and the unknown-job and
unreadable-file error paths.
"""
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
    """Returns a process manager whose backbone and dataloader scripts are long sleeps."""
    return make_manager({name: SLEEP_LONG for name in ("train_backbone", "tune_dataloader")})


def _progress_path(manager: ProcessManager, job_id: str) -> Path:
    """Returns the progress file path derived from the job's GPU pool file."""
    with manager.lock:
        return GpuProgressFile.resolve(Path(manager.jobs[job_id]["overrides"]["gpus_file"]))


def _snapshot(done: int = 12, failed: int = 1, total: int = 30) -> dict:
    """Builds a fan-out progress snapshot with the given done, failed and total unit counts."""
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
    """A script without a GPU pool reports progress as unsupported."""
    result = manager.launch("tune_dataloader", sys.executable)

    assert manager.progress(result["job_id"]) == {"ok": True, "supported": False, "live": False}

    manager.stop(result["job_id"])


def test_progress_before_the_file_exists_is_not_live(manager):
    """Before the file is written, progress is supported but not live and names the expected path."""
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
    """A written progress file is read back verbatim and reported as live."""
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
    """list_jobs embeds the snapshot of a running fan-out job."""
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
    """list_jobs leaves non-fan-out jobs with a null progress field."""
    result = manager.launch("tune_dataloader", sys.executable)
    job_id = result["job_id"]

    assert wait_for_status(manager, job_id, "running")

    record = next(r for r in manager.list_jobs() if r["job_id"] == job_id)

    assert record["progress"] is None

    manager.stop(job_id)


def test_progress_of_a_finished_job_keeps_the_snapshot_but_is_not_live(manager):
    """A finished job still serves its last snapshot from progress but not from list_jobs."""
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
    """A malformed progress file is reported as an unreadable progress file."""
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
    """progress reports an unknown job id."""
    assert manager.progress("nope") == {"ok": False, "error": "unknown job"}


def test_detached_verdict_settles_from_the_completion_snapshot(manager):
    """The detached verdict follows the final progress snapshot: settled failures error, full completion is ok, a short snapshot died, no snapshot stays unknown."""
    assert manager._detached_verdict(None) == ("finished", "unknown")
    assert manager._detached_verdict({"total": 0, "done": 0, "failed": 0}) == ("finished", "unknown")
    assert manager._detached_verdict({"total": 4, "done": 3, "failed": 1}) == ("failed", "error")
    assert manager._detached_verdict({"total": 4, "done": 4, "failed": 0}) == ("finished", "ok")
    assert manager._detached_verdict({"total": 4, "done": 2, "failed": 0}) == ("failed", "died")
    assert manager._detached_verdict({"total": 4, "done": 2, "failed": 1}) == ("failed", "died")


def test_job_fate_reads_detached_verdicts(manager):
    """A detached fan-out that settled with failed units reads as finished, one that died mid-run as crashed, and a completed one as finished."""
    base = {"status": "failed", "pid": 1, "exit_code": None}

    manager.jobs["error"] = {**base, "job_id": "error", "verdict": "error"}
    manager.jobs["died"]  = {**base, "job_id": "died",  "verdict": "died"}
    manager.jobs["ok"]    = {"job_id": "ok", "status": "finished", "pid": 1, "exit_code": None, "verdict": "ok"}

    assert manager.job_fate("error", 1) == "finished"
    assert manager.job_fate("died", 1)  == "crashed"
    assert manager.job_fate("ok", 1)    == "finished"
