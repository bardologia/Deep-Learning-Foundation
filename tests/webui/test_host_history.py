
from __future__ import annotations

import json
import time

import pytest

from host_history import HostHistoryLog
from web_logger   import WebLogger

from tests.webui.conftest import StubPaths


class StubMonitor:

    def __init__(self) -> None:
        self.devices = [{"index": 0, "name": "GPU A", "util": 60.0, "mem_used": 8000, "mem_total": 16000, "temp": 55, "power": 180.0, "power_limit": 250.0, "uuid": "GPU-a"}]

    def _gpu_devices(self) -> list[dict] | None:
        return self.devices


@pytest.fixture
def log(tmp_path, monkeypatch):
    monkeypatch.setattr(HostHistoryLog, "sample_loop", lambda self: None)
    return HostHistoryLog(StubMonitor(), StubPaths(tmp_path), WebLogger())


def test_a_sample_lands_in_the_hour_ring_with_gpu_telemetry(log):
    log.sample()
    state = log.state("hour")

    assert len(state["host"]) == 1
    assert set(state["host"][0]) == {"t", "cpu", "ram", "rx", "tx"}
    assert state["gpus"][0]["index"] == 0
    assert state["gpus"][0]["buckets"][0]["temp"]  == 55
    assert state["gpus"][0]["buckets"][0]["power"] == 180.0
    assert state["gpus"][0]["buckets"][0]["mem"]   == 50.0


def test_the_hour_ring_is_bounded(log):
    for _ in range(HostHistoryLog.RECENT_SAMPLES + 10):
        log.sample()

    assert len(log.state("hour")["host"]) == HostHistoryLog.RECENT_SAMPLES


def test_bucketed_spans_serve_means(log):
    log.monitor.devices[0]["util"] = 40.0
    log.sample()
    log.monitor.devices[0]["util"] = 80.0
    log.sample()

    state  = log.state("day")
    bucket = state["gpus"][0]["buckets"][-1]

    assert state["bucket_s"] == 300
    assert bucket["util"] == 60.0
    assert len(state["host"]) >= 1


def test_an_unknown_gpu_probe_keeps_the_host_series(log):
    log.monitor.devices = None
    log.sample()

    state = log.state("hour")
    assert len(state["host"]) == 1
    assert state["gpus"] == []


def test_net_rates_are_differences_not_totals(log):
    rx0, tx0 = log._net_rates(100.0)
    assert (rx0, tx0) == (0.0, 0.0)

    log.prev_net = (0, 0)
    log.prev_t   = 100.0
    rx1, tx1     = log._net_rates(101.0)

    assert rx1 >= 0.0
    assert tx1 >= 0.0


def test_closed_buckets_persist_and_reload(log, tmp_path, monkeypatch):
    old = time.time() - 2 * HostHistoryLog.BUCKET_S
    monkeypatch.setattr(time, "time", lambda: old)
    log.sample()
    monkeypatch.undo()
    log.sample()

    assert log.path.is_file()

    monkeypatch.setattr(HostHistoryLog, "sample_loop", lambda self: None)
    reloaded = HostHistoryLog(StubMonitor(), StubPaths(tmp_path), WebLogger())

    assert len(reloaded.buckets) >= 1
    assert reloaded.names["0"] == "GPU A"


def test_expired_buckets_are_dropped_on_load(log, tmp_path, monkeypatch):
    stale = {"t": int(time.time() - HostHistoryLog.RETENTION_S - 3600), "host": {"n": 1, "cpu": 1.0, "ram": 1.0, "rx": 0.0, "tx": 0.0}, "gpus": {}}
    log.path.parent.mkdir(parents=True, exist_ok=True)
    log.path.write_text(json.dumps(stale) + "\n", encoding="utf-8")

    monkeypatch.setattr(HostHistoryLog, "sample_loop", lambda self: None)
    reloaded = HostHistoryLog(StubMonitor(), StubPaths(tmp_path), WebLogger())

    assert reloaded.buckets == {}


def test_an_unknown_span_raises(log):
    with pytest.raises(KeyError):
        log.state("year")
