
from __future__ import annotations

from unittest.mock import Mock
import json
import time

import pytest

from gpu_activity import GpuActivityLog
from web_logger   import WebLogger

from tests.webui.conftest import StubPaths


def occupancy(util_a: float, owner_a: str | None, util_b: float = 0.0) -> list[dict]:
    procs = [{"pid": 10, "mem": 512.0, "owner": owner_a, "cmd": "train"}] if owner_a is not None else []
    return [
        {"index": 0, "name": "GPU A", "util": util_a, "mem_used": 8000, "mem_total": 16000, "uuid": "GPU-a", "procs": procs},
        {"index": 1, "name": "GPU B", "util": util_b, "mem_used": 0,    "mem_total": 16000, "uuid": "GPU-b", "procs": []},
    ]


class StubMonitor:

    def __init__(self, user: str = "me") -> None:
        self.user    = user
        self.reading = []

    def gpu_occupancy(self) -> list[dict] | None:
        return self.reading


@pytest.fixture
def log(tmp_path, monkeypatch):
    monkeypatch.setattr(GpuActivityLog, "sample_loop", lambda self: None)
    return GpuActivityLog(StubMonitor(), StubPaths(tmp_path), WebLogger())


def test_a_busy_sample_is_attributed_to_this_user(log):
    log.monitor.reading = occupancy(80.0, "me")
    log.sample()

    gpus = {gpu["index"]: gpu for gpu in log.state()["gpus"]}
    bucket = gpus[0]["buckets"][-1]

    assert bucket["busy"]    == 1.0
    assert bucket["mine"]    == 1.0
    assert bucket["others"]  == 0.0
    assert bucket["util"]    == 80.0
    assert bucket["holders"] == ["me"]


def test_work_by_another_user_is_kept_apart(log):
    log.monitor.reading = occupancy(60.0, "neighbour")
    log.sample()

    gpus = {gpu["index"]: gpu for gpu in log.state()["gpus"]}
    bucket = gpus[0]["buckets"][-1]

    assert bucket["mine"]    == 0.0
    assert bucket["others"]  == 1.0
    assert bucket["holders"] == ["neighbour"]


def test_an_idle_device_records_an_idle_bucket(log):
    log.monitor.reading = occupancy(80.0, "me", util_b=1.0)
    log.sample()

    gpus = {gpu["index"]: gpu for gpu in log.state()["gpus"]}
    bucket = gpus[1]["buckets"][-1]

    assert bucket["busy"]    == 0.0
    assert bucket["holders"] == []


def test_samples_within_one_bucket_average(log):
    log.monitor.reading = occupancy(80.0, "me")
    log.sample()
    log.monitor.reading = occupancy(0.0, None)
    log.sample()

    gpus = {gpu["index"]: gpu for gpu in log.state()["gpus"]}
    assert len(gpus[0]["buckets"]) == 1
    assert gpus[0]["buckets"][0]["busy"] == 0.5
    assert gpus[0]["buckets"][0]["util"] == 40.0


def test_buckets_older_than_the_retention_are_dropped(log):
    stale = int((log.state()["now"] - 1.5 * GpuActivityLog.RETENTION_S) // GpuActivityLog.BUCKET_S) * GpuActivityLog.BUCKET_S
    log.buckets[stale] = {"gpus": {"0": {"name": "GPU A", "n": 1, "busy": 1, "mine": 1, "others": 0, "util": 90.0, "holders": ["me"]}}}

    log.monitor.reading = occupancy(80.0, "me")
    log.sample()

    assert stale not in log.buckets


def test_week_span_merges_raw_buckets_into_hour_slots(log):
    now  = int(time.time())
    slot = int(now // 3600) * 3600
    log.buckets[slot]       = {"gpus": {"0": {"name": "GPU A", "n": 1, "busy": 1, "mine": 1, "others": 0, "util": 80.0, "holders": ["me"]}}}
    log.buckets[slot + 300] = {"gpus": {"0": {"name": "GPU A", "n": 1, "busy": 0, "mine": 0, "others": 0, "util": 0.0,  "holders": []}}}
    log.names["0"] = "GPU A"

    state  = log.state("week")
    bucket = state["gpus"][0]["buckets"][-1]

    assert state["bucket_s"] == 3600
    assert state["window_s"] == 7 * 86400
    assert bucket["t"]       == slot
    assert bucket["busy"]    == 0.5
    assert bucket["util"]    == 40.0
    assert bucket["holders"] == ["me"]


def test_day_span_hides_what_the_month_span_keeps(log):
    old = int((time.time() - 3 * 86400) // GpuActivityLog.BUCKET_S) * GpuActivityLog.BUCKET_S
    log.buckets[old] = {"gpus": {"0": {"name": "GPU A", "n": 1, "busy": 1, "mine": 1, "others": 0, "util": 90.0, "holders": ["me"]}}}
    log.names["0"] = "GPU A"

    assert log.state("day")["gpus"] == [] or all(b["t"] != old for b in log.state("day")["gpus"][0]["buckets"])
    assert any(b["t"] == int(old // (4 * 3600)) * (4 * 3600) for b in log.state("month")["gpus"][0]["buckets"])


def test_an_unknown_span_breaks_loudly(log):
    with pytest.raises(KeyError):
        log.state("year")


def test_closed_buckets_survive_a_restart(log, tmp_path, monkeypatch):
    log.monitor.reading = occupancy(80.0, "me")
    log.sample()
    log.open_at -= GpuActivityLog.BUCKET_S
    log.sample()

    reloaded = GpuActivityLog(StubMonitor(), StubPaths(tmp_path), WebLogger())
    records  = [json.loads(line) for line in log.path.read_text(encoding="utf-8").splitlines()]

    assert records
    assert {gpu["index"] for gpu in reloaded.state()["gpus"]} == {0, 1}


def test_an_interrupted_rewrite_leaves_the_previous_file_intact(log, monkeypatch):
    log.monitor.reading = occupancy(80.0, "me")
    log.sample()
    log._persist()
    before = log.path.read_text(encoding="utf-8")

    def broken(path, text, encoding=None):
        path.write_bytes(text[: len(text) // 2].encode())
        Mock(side_effect=OSError("disk full"))()

    log.monitor.reading = occupancy(20.0, "me")
    log.sample()

    with monkeypatch.context() as patched:
        patched.setattr(type(log.path), "write_text", broken)
        with pytest.raises(OSError):
            log._persist()

    assert log.path.read_text(encoding="utf-8") == before

    reloaded = GpuActivityLog(StubMonitor(), StubPaths(log.path.parent.parent), WebLogger())
    assert {gpu["index"] for gpu in reloaded.state()["gpus"]} == {0, 1}


def test_an_unreadable_probe_records_nothing(log):
    log.monitor.reading = None
    log.sample()

    assert log.state()["gpus"] == []
