"""Tests for the live GPU pool file the process manager injects into fan-out runs.

Covers which launch scripts receive a per-job pool file, honouring an explicit
gpus_file override, and the validation, parking, preemption and error paths of
set_gpus and gpu_pool.
"""
from __future__ import annotations

import json
import sys
from dataclasses import fields
from pathlib     import Path

import pytest

from process_manager import ProcessManager

from configuration.benchmark.general        import BenchmarkConfig
from configuration.cross_validation.general import CrossValidationConfig
from configuration.inference                import BackboneInferenceEntryConfig, DualInferenceEntryConfig, ImageAeInferenceEntryConfig, ProfileAeInferenceEntryConfig
from configuration.training                 import BackboneEntryConfig, DualEntryConfig, ImageAeEntryConfig, JepaEntryConfig, ProfileAeEntryConfig
from configuration.tuning.general           import TuningEntryConfig
from tools.orchestration.gpu_queue          import GpuPoolFile

from tests.webui.conftest import SLEEP_LONG, wait_for_status, wait_until_finished

_SCHEDULING_PAGES = [
    ("train_backbone",            BackboneEntryConfig),
    ("train_dual",                DualEntryConfig),
    ("train_jepa",                JepaEntryConfig),
    ("train_profile_autoencoder", ProfileAeEntryConfig),
    ("train_image_autoencoder",   ImageAeEntryConfig),
    ("benchmark",                 BenchmarkConfig),
    ("cross_validate",            CrossValidationConfig),
    ("tune",                      TuningEntryConfig),
    ("infer_backbone",            BackboneInferenceEntryConfig),
    ("infer_profile_autoencoder", ProfileAeInferenceEntryConfig),
    ("infer_image_autoencoder",   ImageAeInferenceEntryConfig),
    ("infer_dual",                DualInferenceEntryConfig),
]


@pytest.fixture
def manager(make_manager):
    """Returns a process manager whose fan-out and tuning scripts are long sleeps."""
    return make_manager({name: SLEEP_LONG for name in ("train_backbone", "train_dual", "train_jepa", "tune_dataloader")})


def _pool_path(manager: ProcessManager, job_id: str) -> Path:
    """Returns the GPU pool file path injected into the job's overrides."""
    with manager.lock:
        return Path(manager.jobs[job_id]["overrides"]["gpus_file"])


def test_pool_scripts_match_the_configs_exposing_a_pool_file():
    """POOL_SCRIPTS matches exactly the launch pages whose config exposes a gpus_file field."""
    capable = {key for key, config in _SCHEDULING_PAGES if "gpus_file" in {field.name for field in fields(config())}}

    assert set(ProcessManager.POOL_SCRIPTS) == capable


def test_launch_injects_a_per_job_pool_file_for_fan_out_scripts(manager):
    """A fan-out script gets a per-job pool file under gpu_pools_dir passed on the command line."""
    result = manager.launch("train_backbone", sys.executable)
    job_id = result["job_id"]

    with manager.lock:
        record = dict(manager.jobs[job_id])

    assert record["overrides"]["gpus_file"] == str(manager.paths.gpu_pools_dir / f"{job_id}.json")
    assert "--gpus_file" in record["command"]

    manager.stop(job_id)


def test_launch_leaves_other_scripts_without_a_pool_file(manager):
    """A non-fan-out script gets no pool file and reports the pool as unsupported."""
    result = manager.launch("tune_dataloader", sys.executable)

    with manager.lock:
        record = dict(manager.jobs[result["job_id"]])

    assert "gpus_file" not in record["overrides"]
    assert manager.gpu_pool(result["job_id"]) == {"ok": True, "supported": False, "live": False}

    manager.stop(result["job_id"])


def test_launch_keeps_an_explicit_pool_file_override(manager, tmp_path):
    """An explicit gpus_file override is kept instead of the per-job default."""
    chosen = tmp_path / "mine.json"
    result = manager.launch("train_backbone", sys.executable, {"gpus_file": str(chosen)})

    with manager.lock:
        record = dict(manager.jobs[result["job_id"]])

    assert record["overrides"]["gpus_file"] == str(chosen)
    assert record["request"]["overrides"]["gpus_file"] == str(chosen)

    manager.stop(result["job_id"])


def test_the_stored_request_omits_the_per_job_pool_file(manager):
    """The request kept for a retry carries the overrides as sent, without the injected pool file."""
    result = manager.launch("train_backbone", sys.executable, {"epochs": "3"}, "tune_dataloader", False)

    with manager.lock:
        record = dict(manager.jobs[result["job_id"]])

    assert record["request"] == {"overrides": {"epochs": "3"}, "follow_up": "tune_dataloader", "finalize": {}}
    assert "gpus_file" in record["overrides"]

    manager.stop(result["job_id"])


def test_a_recovered_pool_file_under_gpu_pools_dir_is_dropped_from_the_request(manager, tmp_path):
    """An adopted command line loses the pool file the console injected, but keeps one chosen elsewhere."""
    injected = str(manager.paths.gpu_pools_dir / "abc.json")
    chosen   = str(tmp_path / "mine.json")

    assert manager._without_pool_file({"epochs": "3", "gpus_file": injected}) == {"epochs": "3"}
    assert manager._without_pool_file({"epochs": "3", "gpus_file": chosen}) == {"epochs": "3", "gpus_file": chosen}
    assert manager._without_pool_file({"epochs": "3"}) == {"epochs": "3"}


def test_attach_finalize_stores_the_switches_on_the_request(manager):
    """The finalize switches a launch was sent with land on its stored request."""
    result = manager.launch("train_backbone", sys.executable)
    manager.attach_finalize(result["job_id"], {"compare_trials": True})

    with manager.lock:
        assert manager.jobs[result["job_id"]]["request"]["finalize"] == {"compare_trials": True}

    manager.stop(result["job_id"])


def test_set_gpus_writes_the_pool_file_of_a_running_fan_out(manager):
    """set_gpus rewrites the pool file of a running fan-out job and gpu_pool reads it back."""
    result = manager.launch("train_backbone", sys.executable)
    job_id = result["job_id"]

    assert wait_for_status(manager, job_id, "running")

    pool = _pool_path(manager, job_id)
    pool.parent.mkdir(parents=True, exist_ok=True)
    pool.write_text(json.dumps({"gpus": [0]}))

    applied = manager.set_gpus(job_id, [0, 2, 3])

    assert applied == {"ok": True, "gpus": [0, 2, 3], "parked": False, "preempted": []}
    assert json.loads(pool.read_text()) == {"gpus": [0, 2, 3]}
    assert manager.gpu_pool(job_id)["gpus"] == [0, 2, 3]

    manager.stop(job_id)


def test_set_gpus_reports_a_pool_file_write_failure(manager, monkeypatch):
    """set_gpus returns the OSError of a failed pool file write as an error and leaves the pool unchanged."""
    result = manager.launch("train_backbone", sys.executable)
    job_id = result["job_id"]

    assert wait_for_status(manager, job_id, "running")

    pool = _pool_path(manager, job_id)
    pool.parent.mkdir(parents=True, exist_ok=True)
    pool.write_text(json.dumps({"gpus": [0]}))

    def refuse(self, gpus, preempt):
        """Fails the write as a full disk would."""
        raise OSError("disk full")

    monkeypatch.setattr(GpuPoolFile, "write", refuse)

    applied = manager.set_gpus(job_id, [0, 1])

    assert applied["ok"] is False
    assert "failed to write pool file: disk full" in applied["error"]
    assert json.loads(pool.read_text()) == {"gpus": [0]}

    manager.stop(job_id)


def test_set_gpus_refuses_a_job_that_seeded_no_pool(manager):
    """set_gpus fails when the running job has not written a pool file yet."""
    result = manager.launch("train_backbone", sys.executable)
    job_id = result["job_id"]

    assert wait_for_status(manager, job_id, "running")

    applied = manager.set_gpus(job_id, [0, 1])

    assert applied["ok"] is False
    assert "no live GPU pool" in applied["error"]
    assert manager.gpu_pool(job_id)["live"] is False

    manager.stop(job_id)


def test_set_gpus_parks_only_when_parking_is_confirmed(manager):
    """An empty selection needs park=True, and a later non-empty selection resumes the job."""
    result = manager.launch("train_backbone", sys.executable)
    job_id = result["job_id"]

    assert wait_for_status(manager, job_id, "running")

    pool = _pool_path(manager, job_id)
    pool.parent.mkdir(parents=True, exist_ok=True)
    pool.write_text(json.dumps({"gpus": [0, 1]}))

    refused = manager.set_gpus(job_id, [])

    assert refused["ok"] is False
    assert "confirm parking" in refused["error"]
    assert json.loads(pool.read_text()) == {"gpus": [0, 1]}

    parked = manager.set_gpus(job_id, [], park=True)

    assert parked == {"ok": True, "gpus": [], "parked": True, "preempted": []}
    assert json.loads(pool.read_text()) == {"gpus": []}
    assert manager.gpu_pool(job_id)["gpus"] == []

    resumed = manager.set_gpus(job_id, [2])

    assert resumed == {"ok": True, "gpus": [2], "parked": False, "preempted": []}
    assert json.loads(pool.read_text()) == {"gpus": [2]}

    manager.stop(job_id)


def test_set_gpus_writes_the_preempt_list_beside_the_pool(manager):
    """A preempt list is written next to the new pool so the scheduler frees those devices at once."""
    result = manager.launch("train_backbone", sys.executable)
    job_id = result["job_id"]

    assert wait_for_status(manager, job_id, "running")

    pool = _pool_path(manager, job_id)
    pool.parent.mkdir(parents=True, exist_ok=True)
    pool.write_text(json.dumps({"gpus": [0, 1, 2]}))

    applied = manager.set_gpus(job_id, [0], preempt=[1, 2])

    assert applied == {"ok": True, "gpus": [0], "parked": False, "preempted": [1, 2]}
    assert json.loads(pool.read_text()) == {"gpus": [0], "preempt": [1, 2]}
    assert manager.gpu_pool(job_id)["gpus"] == [0]

    manager.stop(job_id)


def test_set_gpus_rejects_a_device_kept_and_preempted_at_once(manager):
    """A device listed both in the pool and in preempt is refused and the file stays untouched."""
    result = manager.launch("train_backbone", sys.executable)
    job_id = result["job_id"]

    assert wait_for_status(manager, job_id, "running")

    pool = _pool_path(manager, job_id)
    pool.parent.mkdir(parents=True, exist_ok=True)
    pool.write_text(json.dumps({"gpus": [0, 1]}))

    applied = manager.set_gpus(job_id, [0, 1], preempt=[1])

    assert applied["ok"] is False
    assert "listed in both" in applied["error"]
    assert json.loads(pool.read_text()) == {"gpus": [0, 1]}

    manager.stop(job_id)


def test_set_gpus_parks_now_when_every_device_is_preempted(manager):
    """An empty pool with every device preempted parks the job and stops its runs at once."""
    result = manager.launch("train_backbone", sys.executable)
    job_id = result["job_id"]

    assert wait_for_status(manager, job_id, "running")

    pool = _pool_path(manager, job_id)
    pool.parent.mkdir(parents=True, exist_ok=True)
    pool.write_text(json.dumps({"gpus": [0, 1]}))

    parked = manager.set_gpus(job_id, [], park=True, preempt=[0, 1])

    assert parked == {"ok": True, "gpus": [], "parked": True, "preempted": [0, 1]}
    assert json.loads(pool.read_text()) == {"gpus": [], "preempt": [0, 1]}

    manager.stop(job_id)


@pytest.mark.parametrize("gpus, reason", [
    ([0, 0],       "repeat"),
    ([-1],         "non-negative"),
    (["0"],        "non-negative"),
    ("0,1",        "list"),
])
def test_set_gpus_rejects_an_invalid_selection(manager, gpus, reason):
    """An invalid GPU selection is rejected with a reason and leaves the pool file untouched."""
    result = manager.launch("train_backbone", sys.executable)
    job_id = result["job_id"]

    assert wait_for_status(manager, job_id, "running")

    pool = _pool_path(manager, job_id)
    pool.parent.mkdir(parents=True, exist_ok=True)
    pool.write_text(json.dumps({"gpus": [0]}))

    applied = manager.set_gpus(job_id, gpus)

    assert applied["ok"] is False
    assert reason in applied["error"]
    assert json.loads(pool.read_text()) == {"gpus": [0]}

    manager.stop(job_id)


def test_set_gpus_refuses_a_finished_job(manager):
    """set_gpus refuses a job that is no longer running."""
    result = manager.launch("train_backbone", sys.executable)
    job_id = result["job_id"]

    assert wait_for_status(manager, job_id, "running")
    manager.stop(job_id)

    assert wait_until_finished(manager, job_id)

    applied = manager.set_gpus(job_id, [0, 1])

    assert applied["ok"] is False
    assert applied["error"] == "job is not running"


def test_set_gpus_reports_an_unknown_job(manager):
    """set_gpus and gpu_pool both report an unknown job id."""
    assert manager.set_gpus("nope", [0]) == {"ok": False, "error": "unknown job"}
    assert manager.gpu_pool("nope")      == {"ok": False, "error": "unknown job"}
