from __future__ import annotations

import threading
import time
from datetime    import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from notifier   import ExperimentProgressWatcher, JobNotifier
from web_logger import WebLogger

from tests.webui.conftest import StubPaths


class CaptureHandler(BaseHTTPRequestHandler):

    requests = []

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        CaptureHandler.requests.append({
            "path"     : self.path,
            "title"    : self.headers.get("Title"),
            "priority" : self.headers.get("Priority"),
            "body"     : self.rfile.read(length).decode("utf-8"),
        })
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, fmt: str, *args) -> None:
        return


@pytest.fixture
def notifier(tmp_path):
    return JobNotifier(StubPaths(tmp_path), WebLogger())


@pytest.fixture
def sent(notifier, monkeypatch):
    calls = []
    monkeypatch.setattr(notifier, "_send", lambda title, body, priority: calls.append((title, body, priority)) or None)
    return calls


@pytest.fixture
def capture_server():
    CaptureHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), CaptureHandler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _drain(calls: list, count: int, timeout: float = 5.0) -> list:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(calls) >= count:
            return calls
        time.sleep(0.02)
    return calls


def _record(status: str = "finished", exit_code: int | None = 0, ago_s: float = 3600.0, stopped: bool = False, description: str = "") -> dict:
    started = (datetime.now() - timedelta(seconds=ago_s)).isoformat(timespec="seconds")
    record  = {"job_id": "abc123def456", "script": "train_backbone", "status": status, "exit_code": exit_code, "started": started, "pid": 4242, "description": description}
    if stopped:
        record["stopped"] = True
    return record


def _arm(notifier) -> None:
    result = notifier.configure({"enabled": True, "topic": "unit-test-topic", "server": "http://127.0.0.1:9"})
    assert result["ok"]


def test_defaults_are_disabled(notifier):
    state = notifier.state()
    assert state["enabled"] is False
    assert state["topic"]   == ""
    assert state["server"]  == "https://ntfy.sh"


def test_configure_persists_across_instances(notifier, tmp_path):
    _arm(notifier)

    reloaded = JobNotifier(StubPaths(tmp_path), WebLogger())
    state    = reloaded.state()
    assert state["enabled"] is True
    assert state["topic"]   == "unit-test-topic"


def test_configure_rejects_invalid_topic(notifier):
    assert not notifier.configure({"enabled": True, "topic": "has spaces"})["ok"]
    assert not notifier.configure({"enabled": True, "topic": "a/b"})["ok"]
    assert not notifier.configure({"enabled": True, "topic": "x" * 65})["ok"]


def test_configure_requires_topic_to_enable(notifier):
    assert not notifier.configure({"enabled": True, "topic": ""})["ok"]
    assert notifier.configure({"enabled": False, "topic": ""})["ok"]


def test_configure_rejects_invalid_server(notifier):
    assert not notifier.configure({"enabled": True, "topic": "ok-topic", "server": "ntfy.sh"})["ok"]


def test_disabled_notifier_sends_nothing(notifier, sent):
    notifier.job_started(_record(status="running"))
    notifier.job_finished(_record())
    time.sleep(0.2)
    assert sent == []


def test_start_notifies(notifier, sent):
    _arm(notifier)
    notifier.job_started(_record(status="running"))

    assert len(_drain(sent, 1)) == 1
    title, body, priority = sent[0]
    assert title    == "train_backbone started"
    assert priority == "default"
    assert "abc123def456" in body
    assert "4242" in body


def test_start_body_carries_description(notifier, sent):
    _arm(notifier)
    notifier.job_started(_record(status="running", description="presence trials experiment · resunet-conv"))

    assert len(_drain(sent, 1)) == 1
    body = sent[0][1]
    assert body.startswith("presence trials experiment · resunet-conv\n")
    assert "abc123def456" in body


def test_finish_body_carries_description(notifier, sent):
    _arm(notifier)
    notifier.job_finished(_record(ago_s=600.0, description="single training · resunet-conv"))

    assert len(_drain(sent, 1)) == 1
    body = sent[0][1]
    assert body.startswith("single training · resunet-conv\n")
    assert "10m 00s" in body


def test_record_without_description_keeps_plain_body(notifier, sent):
    _arm(notifier)
    notifier.job_finished({key: value for key, value in _record(ago_s=600.0).items() if key != "description"})

    assert len(_drain(sent, 1)) == 1
    assert sent[0][1].startswith("runtime 10m 00s")


def test_quick_success_notifies(notifier, sent):
    _arm(notifier)
    notifier.job_finished(_record(ago_s=5.0))

    assert len(_drain(sent, 1)) == 1
    assert sent[0][0] == "train_backbone finished"


def test_long_success_notifies(notifier, sent):
    _arm(notifier)
    notifier.job_finished(_record(ago_s=7500.0))

    assert len(_drain(sent, 1)) == 1
    title, body, priority = sent[0]
    assert title    == "train_backbone finished"
    assert priority == "default"
    assert "2h 05m" in body
    assert "abc123def456" in body


def test_failure_notifies(notifier, sent):
    _arm(notifier)
    notifier.job_finished(_record(status="failed", exit_code=1, ago_s=5.0))

    assert len(_drain(sent, 1)) == 1
    title, body, priority = sent[0]
    assert title    == "train_backbone FAILED (exit 1)"
    assert priority == "high"


def test_stopped_job_notifies_as_stopped(notifier, sent):
    _arm(notifier)
    notifier.job_finished(_record(status="failed", exit_code=-15, stopped=True))

    assert len(_drain(sent, 1)) == 1
    title, body, priority = sent[0]
    assert title    == "train_backbone stopped (exit -15)"
    assert priority == "default"


def test_queued_cancellation_stays_silent(notifier, sent):
    _arm(notifier)
    notifier.job_finished(_record(status="cancelled", exit_code=None))
    time.sleep(0.2)
    assert sent == []


def test_unknown_exit_is_labelled(notifier, sent):
    _arm(notifier)
    notifier.job_finished(_record(exit_code=None, ago_s=600.0))

    assert len(_drain(sent, 1)) == 1
    assert sent[0][0] == "train_backbone finished (exit status unknown)"


def test_delivery_over_http(notifier, capture_server):
    result = notifier.configure({"enabled": True, "topic": "unit-test-topic", "server": capture_server})
    assert result["ok"]

    notifier.job_started(_record(status="running"))
    notifier.job_finished(_record(ago_s=600.0))

    requests = _drain(CaptureHandler.requests, 2)
    assert len(requests) == 2
    titles = {request["title"] for request in requests}
    assert titles == {"train_backbone started", "train_backbone finished"}
    assert all(request["path"] == "/unit-test-topic" for request in requests)
    finished = next(request for request in requests if request["title"] == "train_backbone finished")
    assert "10m 00s" in finished["body"]


def test_test_message_requires_topic(notifier):
    assert not notifier.test()["ok"]


def test_test_message_is_delivered(notifier, capture_server):
    result = notifier.configure({"enabled": False, "topic": "unit-test-topic", "server": capture_server})
    assert result["ok"]

    assert notifier.test()["ok"]
    assert len(CaptureHandler.requests) == 1
    assert CaptureHandler.requests[0]["title"] == "DLR-TomoSAR test notification"


def _progress(total: int = 30, done: int = 15, failed: int = 0, eta_s: float | None = 9000.0, failed_units: list | None = None) -> dict:
    return {
        "total"        : total,
        "done"         : done,
        "failed"       : failed,
        "queued"       : max(0, total - done - failed - 2),
        "running"      : [{"name": "aug-on/seed3", "gpu": 0, "elapsed_s": 310.0}],
        "workers"      : 2,
        "failed_units" : failed_units if failed_units is not None else [],
        "average_s"    : 600.0 if eta_s is not None else None,
        "elapsed_s"    : 4200.0,
        "eta_s"        : eta_s,
        "total_s"      : 4200.0 + eta_s if eta_s is not None else None,
        "started_at"   : "2026-07-17T10:00:00",
        "finish_at"    : "2026-07-17T19:13:00" if eta_s is not None else None,
        "updated_at"   : "2026-07-17T14:00:00",
    }


def test_experiment_progress_notifies_with_eta(notifier, sent):
    _arm(notifier)
    notifier.experiment_progress(_record(status="running"), _progress(total=30, done=15))

    assert len(_drain(sent, 1)) == 1
    title, body, priority = sent[0]
    assert title    == "train_backbone 15/30 units (50%)"
    assert priority == "default"
    assert "ETA 2h 30m" in body
    assert "finish ≈ 19:13" in body
    assert "avg 10m 00s/unit" in body


def test_experiment_progress_without_eta_says_so(notifier, sent):
    _arm(notifier)
    notifier.experiment_progress(_record(status="running"), _progress(done=0, eta_s=None))

    assert len(_drain(sent, 1)) == 1
    title, body, _priority = sent[0]
    assert title == "train_backbone 0/30 units (0%)"
    assert "no ETA yet" in body


def test_experiment_progress_counts_failed_units(notifier, sent):
    _arm(notifier)
    notifier.experiment_progress(_record(status="running"), _progress(done=14, failed=1, failed_units=["aug-on/seed3"]))

    assert len(_drain(sent, 1)) == 1
    title, body, _priority = sent[0]
    assert title == "train_backbone 15/30 units (50%)"
    assert "1 FAILED" in body


def test_experiment_unit_failure_notifies_high_priority(notifier, sent):
    _arm(notifier)
    notifier.experiment_unit_failed(_record(status="running"), _progress(done=10, failed=1, failed_units=["aug-on/seed3"]), ["aug-on/seed3"])

    assert len(_drain(sent, 1)) == 1
    title, body, priority = sent[0]
    assert title    == "train_backbone unit FAILED: aug-on/seed3"
    assert priority == "high"
    assert "1 of 30 units failed" in body


def test_several_unit_failures_notify_as_one_push(notifier, sent):
    _arm(notifier)
    notifier.experiment_unit_failed(_record(status="running"), _progress(done=10, failed=2, failed_units=["a", "b"]), ["a", "b"])

    assert len(_drain(sent, 1)) == 1
    title, body, _priority = sent[0]
    assert title == "train_backbone: 2 units FAILED"
    assert "a, b" in body


def test_finish_body_reports_the_unit_tally(notifier, sent):
    _arm(notifier)
    record             = _record(ago_s=600.0)
    record["progress"] = _progress(total=30, done=28, failed=2, eta_s=0.0, failed_units=["a", "b"])
    notifier.job_finished(record)

    assert len(_drain(sent, 1)) == 1
    body = sent[0][1]
    assert "28/30 units done, 2 FAILED" in body


class StubProcesses:

    def __init__(self) -> None:
        self.records = []

    def list_jobs(self) -> list[dict]:
        return [dict(record) for record in self.records]


def _running_job(progress: dict | None, job_id: str = "abc123def456") -> dict:
    return {**_record(status="running"), "job_id": job_id, "progress": progress}


@pytest.fixture
def watched(notifier):
    processes = StubProcesses()
    return ExperimentProgressWatcher(processes, notifier, WebLogger()), processes


def test_watcher_pushes_once_when_the_first_eta_appears(notifier, sent, watched):
    _arm(notifier)
    watcher, processes = watched

    processes.records = [_running_job(_progress(done=0, eta_s=None))]
    watcher.scan()
    time.sleep(0.1)
    assert sent == []

    processes.records = [_running_job(_progress(done=1))]
    watcher.scan()
    assert len(_drain(sent, 1)) == 1
    assert sent[0][0] == "train_backbone 1/30 units (3%)"

    watcher.scan()
    time.sleep(0.1)
    assert len(sent) == 1


def test_watcher_pushes_a_single_milestone_when_several_are_crossed(notifier, sent, watched):
    _arm(notifier)
    watcher, processes = watched

    processes.records = [_running_job(_progress(done=1))]
    watcher.scan()

    processes.records = [_running_job(_progress(done=16))]
    watcher.scan()

    assert len(_drain(sent, 1)) == 1
    assert sent[0][0] == "train_backbone 16/30 units (53%)"


def test_watcher_baselines_a_job_already_mid_run(notifier, sent, watched):
    _arm(notifier)
    watcher, processes = watched

    processes.records = [_running_job(_progress(done=20))]
    watcher.scan()
    processes.records = [_running_job(_progress(done=21))]
    watcher.scan()
    time.sleep(0.1)
    assert sent == []

    processes.records = [_running_job(_progress(done=23))]
    watcher.scan()
    assert len(_drain(sent, 1)) == 1
    assert "(77%)" in sent[0][0]


def test_watcher_pushes_new_unit_failures_once(notifier, sent, watched):
    _arm(notifier)
    watcher, processes = watched

    processes.records = [_running_job(_progress(done=5))]
    watcher.scan()

    processes.records = [_running_job(_progress(done=5, failed=2, failed_units=["a", "b"]))]
    watcher.scan()
    assert len(_drain(sent, 1)) == 1
    assert sent[0][0] == "train_backbone: 2 units FAILED"

    watcher.scan()
    time.sleep(0.1)
    assert len(sent) == 1


def test_watcher_forgets_jobs_that_stop_running(notifier, watched):
    watcher, processes = watched

    processes.records = [_running_job(_progress(done=5))]
    watcher.scan()
    assert watcher.tracked

    processes.records = []
    watcher.scan()
    assert watcher.tracked == {}


def test_watcher_ignores_jobs_without_progress(notifier, watched):
    watcher, processes = watched

    processes.records = [_running_job(None)]
    watcher.scan()

    assert watcher.tracked == {}
