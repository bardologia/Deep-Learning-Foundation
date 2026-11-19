
from __future__ import annotations

import socket

import torch

from tools.runtime.environment_capture import EnvironmentCapture


def test_snapshot_carries_the_version_and_host_keys():
    snapshot = EnvironmentCapture.snapshot()

    assert snapshot["hostname"] == socket.gethostname()
    assert snapshot["torch"]    == torch.__version__
    for key in ("platform", "python", "numpy", "scipy", "torch_cuda_build", "cudnn", "driver_version", "gpus", "conda_env", "cuda_visible_devices"):
        assert key in snapshot


def test_snapshot_gpu_entries_match_the_visible_devices():
    snapshot = EnvironmentCapture.snapshot()

    if torch.cuda.is_available():
        assert len(snapshot["gpus"]) == torch.cuda.device_count()
        assert all(entry["name"] and entry["total_memory_bytes"] > 0 for entry in snapshot["gpus"])
        assert snapshot["driver_version"]
    else:
        assert snapshot["gpus"]           == []
        assert snapshot["driver_version"] is None


def test_snapshot_is_json_serialisable():
    import json

    snapshot = EnvironmentCapture.snapshot()

    assert json.loads(json.dumps(snapshot)) == snapshot
