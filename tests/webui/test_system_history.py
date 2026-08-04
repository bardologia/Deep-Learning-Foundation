from __future__ import annotations

import pytest

from system_monitor import SystemHistory

FAKE_DEVICES = [
    {"index": 0, "util": 50,   "mem_used": 8000, "mem_total": 16000, "name": "GPU A", "temp": 40, "power": 30, "power_limit": 90, "uuid": "GPU-a"},
    {"index": 1, "util": None, "mem_used": 0,    "mem_total": 0,     "name": "GPU B", "temp": 40, "power": 30, "power_limit": 90, "uuid": "GPU-b"},
]


@pytest.fixture
def history(monitor, monkeypatch):
    monkeypatch.setattr(monitor, "_gpu_devices", lambda: FAKE_DEVICES)
    return SystemHistory(monitor)


def test_sample_records_all_series(history):
    history.sample()
    history.sample()

    state = history.state()
    assert len(state["cpu"]) == 2
    assert len(state["ram"]) == 2
    assert set(state["gpus"]) == {"0", "1"}
    assert state["gpus"]["0"]["util"] == [50, 50]
    assert state["gpus"]["0"]["mem"]  == [50.0, 50.0]
    assert state["gpus"]["1"]["util"] == [0.0, 0.0]
    assert state["gpus"]["1"]["mem"]  == [0.0, 0.0]


def test_samples_are_bounded_percentages(history):
    history.sample()
    history.sample()

    state = history.state()
    for series in (state["cpu"], state["ram"]):
        assert all(isinstance(value, float) and 0.0 <= value <= 100.0 for value in series)


def test_ring_buffer_caps_history(monitor, monkeypatch):
    monkeypatch.setattr(SystemHistory, "MAX_SAMPLES", 5)
    capped = SystemHistory(monitor)

    for _ in range(8):
        capped.sample()

    state = capped.state()
    assert len(state["cpu"]) == 5
    assert len(state["gpus"]["0"]["util"]) == 5
    assert state["max_samples"] == 5


def test_tracks_follow_the_device_index_not_the_listing_position(history, monitor, monkeypatch):
    monkeypatch.setattr(monitor, "_gpu_devices", lambda: [FAKE_DEVICES[1]])
    history.sample()

    assert set(history.state()["gpus"]) == {"1"}


def test_a_dropped_device_listing_keeps_every_series_aligned(history, monitor, monkeypatch):
    history.sample()
    monkeypatch.setattr(monitor, "_gpu_devices", lambda: [])
    history.sample()

    state = history.state()
    assert len(state["cpu"])                == 2
    assert len(state["gpus"]["0"]["util"])  == 2
    assert len(state["gpus"]["1"]["mem"])   == 2


def test_a_device_without_a_memory_reading_samples_as_zero(history, monitor, monkeypatch):
    monkeypatch.setattr(monitor, "_gpu_devices", lambda: [{**FAKE_DEVICES[0], "mem_used": None}])
    history.sample()

    assert history.state()["gpus"]["0"]["mem"] == [0.0]


def test_history_state_is_independent_of_monitor_snapshots(history, monitor):
    history.sample()
    monitor.snapshot()
    history.sample()

    assert len(history.state()["cpu"]) == 2


def test_snapshot_embeds_history(monitor):
    snapshot = monitor.snapshot()

    assert "history" in snapshot
    assert set(snapshot["history"]) == {"period_s", "max_samples", "cpu", "ram", "gpus"}
