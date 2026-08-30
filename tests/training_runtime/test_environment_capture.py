"""Tests for the environment snapshot recorded alongside run outputs."""

from __future__ import annotations

import socket

import torch

from tools.runtime.environment_capture import EnvironmentCapture


def test_snapshot_carries_the_version_and_host_keys():
    """The snapshot names the host and every library version a rerun would need to match."""
    snapshot = EnvironmentCapture.snapshot()

    assert snapshot["hostname"] == socket.gethostname()
    assert snapshot["torch"]    == torch.__version__
    for key in ("platform", "python", "numpy", "scipy", "torch_cuda_build", "cudnn", "driver_version", "gpus", "conda_env", "cuda_visible_devices"):
        assert key in snapshot


def test_snapshot_gpu_entries_match_the_visible_devices():
    """The GPU list holds one named entry per visible CUDA device and is empty without CUDA."""
    snapshot = EnvironmentCapture.snapshot()

    if torch.cuda.is_available():
        assert len(snapshot["gpus"]) == torch.cuda.device_count()
        assert all(entry["name"] and entry["total_memory_bytes"] > 0 for entry in snapshot["gpus"])
        assert snapshot["driver_version"]
    else:
        assert snapshot["gpus"]           == []
        assert snapshot["driver_version"] is None


def test_snapshot_is_json_serialisable():
    """Every snapshot value survives a JSON round trip unchanged."""
    import json

    snapshot = EnvironmentCapture.snapshot()

    assert json.loads(json.dumps(snapshot)) == snapshot
