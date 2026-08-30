"""Tests for the GPU watchdog that detects foreign processes joining our devices.

Covers which co-tenancy patterns count as an intrusion, how a job fate observed
during an intrusion window escalates the incident to critical, and how an intruder
that disappears within the grace period is recorded as bounced.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from gpu_watchdog import GpuWatchdog
from web_logger   import WebLogger

from tests.webui.conftest import StubNotifier, StubPaths


class StubSystem:
    """Stand-in system monitor serving a fixed GPU occupancy listing.

    Attributes:
        user      : Name of the local user.
        occupancy : Device records returned by gpu_occupancy, or None for unknown.
        owners    : Process owner name keyed by pid.
    """

    def __init__(self) -> None:
        """Starts with no devices and no known process owners."""
        self.user      = "me"
        self.occupancy = []
        self.owners    = {}

    def gpu_occupancy(self) -> list[dict] | None:
        """Returns a deep copy of the configured device records, or None when occupancy is unknown."""
        if self.occupancy is None:
            return None

        return [{**device, "procs": [dict(proc) for proc in device["procs"]]} for device in self.occupancy]

    def pid_owner(self, pid: int) -> str | None:
        """Returns the configured owner name for a pid, or None when unknown."""
        return self.owners.get(pid)


class StubProcesses:
    """Stand-in process manager mapping pids to jobs and jobs to exit fates.

    Attributes:
        jobs  : Job id keyed by pid.
        fates : Recorded exit fate keyed by job id.
    """

    def __init__(self) -> None:
        """Starts with no tracked pids and no recorded fates."""
        self.jobs  = {}
        self.fates = {}

    def job_for_pid(self, pid: int) -> str | None:
        """Returns the job id owning a pid, or None when untracked."""
        return self.jobs.get(pid)

    def job_fate(self, job_id: str, pid: int) -> str:
        """Returns the recorded exit fate for a job, defaulting to unknown."""
        return self.fates.get(job_id, "unknown")


def _device(procs: list[dict]) -> dict:
    """Returns a single-GPU occupancy record carrying the given process list."""
    return {"index": 0, "name": "A100", "util": 50.0, "mem_used": 1000.0, "mem_total": 40000.0, "uuid": "GPU-0", "procs": procs}


def _proc(pid: int, owner: str) -> dict:
    """Returns a GPU process record for a pid and owner."""
    return {"pid": pid, "owner": owner, "mem": 1000.0, "cmd": f"python train_{pid}.py"}


def _watchdog(tmp_path: Path) -> tuple[GpuWatchdog, StubSystem, StubProcesses]:
    """Returns a watchdog wired to fresh system and process stubs."""
    system    = StubSystem()
    processes = StubProcesses()
    watchdog  = GpuWatchdog(system, StubPaths(tmp_path), WebLogger(), processes, StubNotifier())
    return watchdog, system, processes


def _invaded(watchdog: GpuWatchdog, system: StubSystem) -> None:
    """Drives the watchdog through one solo evaluation then a second where a foreign process joins the same GPU."""
    system.occupancy = [_device([_proc(100, "me")])]
    system.owners    = {100: "me", 200: "them"}
    watchdog._evaluate()

    system.occupancy = [_device([_proc(100, "me"), _proc(200, "them")])]
    watchdog._evaluate()


def test_shared_gpu_at_startup_is_not_an_intrusion(tmp_path):
    """A GPU already shared on the first evaluation raises no intrusion."""
    watchdog, system, _ = _watchdog(tmp_path)

    system.occupancy = [_device([_proc(100, "me"), _proc(200, "them")])]
    system.owners    = {100: "me", 200: "them"}
    watchdog._evaluate()

    assert watchdog.state()["active"] == []
    assert watchdog.state()["count"]  == 0


def test_joining_an_occupied_gpu_is_not_an_intrusion(tmp_path):
    """Our own process joining a GPU another user already held raises no intrusion."""
    watchdog, system, _ = _watchdog(tmp_path)

    system.occupancy = [_device([_proc(200, "them")])]
    system.owners    = {200: "them"}
    watchdog._evaluate()

    system.occupancy = [_device([_proc(100, "me"), _proc(200, "them")])]
    system.owners    = {100: "me", 200: "them"}
    watchdog._evaluate()

    assert watchdog.state()["active"] == []
    assert watchdog.state()["count"]  == 0


def test_late_arrival_on_my_gpu_is_an_intrusion(tmp_path):
    """A foreign process appearing on a GPU we already held is reported as an active intrusion naming our pids."""
    watchdog, system, _ = _watchdog(tmp_path)

    _invaded(watchdog, system)

    state = watchdog.state()
    assert state["count"] == 1
    assert [record["user"] for record in state["active"]] == ["them"]
    assert state["active"][0]["mine_pids"] == [100]


def test_an_unknown_occupancy_leaves_the_intrusion_state_alone(tmp_path):
    """An unreadable occupancy probe preserves the previous intrusion records and device flags."""
    watchdog, system, _ = _watchdog(tmp_path)

    _invaded(watchdog, system)
    system.occupancy = None
    watchdog._evaluate()

    state = watchdog.state()
    assert state["count"] == 1
    assert [record["user"] for record in state["active"]] == ["them"]
    assert [device["index"] for device in state["gpus"]]  == [0]
    assert state["gpus"][0]["others"] is True


def test_clean_finish_during_intrusion_is_not_critical(tmp_path):
    """A job that finishes cleanly while an intruder is present ends the incident without marking it critical."""
    watchdog, system, processes = _watchdog(tmp_path)
    processes.jobs  = {100: "j1"}
    processes.fates = {"j1": "finished"}

    _invaded(watchdog, system)

    system.occupancy = [_device([_proc(200, "them")])]
    system.owners    = {200: "them"}
    watchdog._evaluate()

    state = watchdog.state()
    assert state["critical"] == 0
    assert state["active"]   == []
    assert watchdog.incidents[("GPU-0", 200)]["status"] == "ended"


def test_crash_during_intrusion_window_is_critical(tmp_path):
    """A job that crashes during the intrusion window marks the incident critical and lists the dead pid."""
    watchdog, system, processes = _watchdog(tmp_path)
    processes.jobs  = {100: "j1"}
    processes.fates = {"j1": "crashed"}

    _invaded(watchdog, system)

    system.occupancy = [_device([_proc(200, "them")])]
    system.owners    = {200: "them"}
    watchdog._evaluate()

    state = watchdog.state()
    assert state["critical"] == 1
    assert state["active"][0]["status"]    == "critical"
    assert state["active"][0]["dead_pids"] == [100]


def test_user_stopped_job_is_not_critical(tmp_path):
    """A job stopped by the user ends the incident without escalating it."""
    watchdog, system, processes = _watchdog(tmp_path)
    processes.jobs  = {100: "j1"}
    processes.fates = {"j1": "stopped"}

    _invaded(watchdog, system)

    system.occupancy = [_device([_proc(200, "them")])]
    system.owners    = {200: "them"}
    watchdog._evaluate()

    state = watchdog.state()
    assert state["critical"] == 0
    assert watchdog.incidents[("GPU-0", 200)]["status"] == "ended"


def test_untracked_pid_exit_is_not_critical(tmp_path):
    """The disappearance of a pid belonging to no tracked job ends the incident without escalating it."""
    watchdog, system, _ = _watchdog(tmp_path)

    _invaded(watchdog, system)

    system.occupancy = [_device([_proc(200, "them")])]
    system.owners    = {200: "them"}
    watchdog._evaluate()

    state = watchdog.state()
    assert state["critical"] == 0
    assert watchdog.incidents[("GPU-0", 200)]["status"] == "ended"


def test_a_vanished_intruder_is_recorded_as_bounced(tmp_path):
    """An intruder gone longer than the grace period is drained once as a bounce and marked survived."""
    watchdog, system, _ = _watchdog(tmp_path)

    _invaded(watchdog, system)

    system.occupancy = [_device([_proc(100, "me")])]
    system.owners    = {100: "me"}
    watchdog._evaluate()
    assert watchdog.take_bounced() == []

    watchdog.incidents[("GPU-0", 200)]["last_seen"] -= GpuWatchdog.GRACE_S + 1
    watchdog._evaluate()

    bounced = watchdog.take_bounced()
    assert [record["user"] for record in bounced]           == ["them"]
    assert bounced[0]["gpu_index"]                          == 0
    assert watchdog.take_bounced()                          == []
    assert watchdog.incidents[("GPU-0", 200)]["status"]     == "survived"


def test_a_persisting_intruder_is_not_bounced(tmp_path):
    """An intruder still holding the GPU stays active and produces no bounce record."""
    watchdog, system, _ = _watchdog(tmp_path)

    _invaded(watchdog, system)
    watchdog._evaluate()

    assert watchdog.take_bounced() == []
    assert watchdog.incidents[("GPU-0", 200)]["status"] == "active"


def test_pending_fate_defers_until_exit_resolves(tmp_path):
    """An unresolved job fate keeps the incident active until the fate settles on a crash, which then marks it critical."""
    watchdog, system, processes = _watchdog(tmp_path)
    processes.jobs  = {100: "j1"}
    processes.fates = {"j1": "pending"}

    _invaded(watchdog, system)

    system.occupancy = [_device([_proc(200, "them")])]
    system.owners    = {200: "them"}
    watchdog._evaluate()
    assert watchdog.incidents[("GPU-0", 200)]["status"] == "active"

    processes.fates = {"j1": "crashed"}
    watchdog._evaluate()

    state = watchdog.state()
    assert state["critical"] == 1
    assert state["active"][0]["dead_pids"] == [100]


def test_an_intrusion_is_pushed_through_the_notifier(tmp_path):
    """A detected intrusion pushes one high-priority notification naming the intruder."""
    watchdog, system, _ = _watchdog(tmp_path)
    _invaded(watchdog, system)
    alerts = watchdog.notifier.alerts

    assert len(alerts) == 1
    assert alerts[0][0] == "GPU intrusion"
    assert alerts[0][2] == "high"
    assert "them" in alerts[0][1]


def test_a_critical_escalation_is_pushed_as_urgent(tmp_path):
    """An incident escalated to critical pushes an urgent notification."""
    watchdog, system, processes = _watchdog(tmp_path)
    processes.jobs  = {100: "job-1"}
    processes.fates = {"job-1": "crashed"}

    _invaded(watchdog, system)
    system.occupancy = [_device([_proc(200, "them")])]
    system.owners    = {200: "them"}
    watchdog._evaluate()

    urgent = [alert for alert in watchdog.notifier.alerts if alert[2] == "urgent"]

    assert len(urgent) == 1
    assert urgent[0][0] == "GPU intrusion killed your job"


def test_state_hands_out_incident_record_copies(tmp_path):
    """A record returned by state() is a copy the watch thread's later escalation cannot mutate."""
    watchdog, system, processes = _watchdog(tmp_path)
    processes.jobs  = {100: "j1"}
    processes.fates = {"j1": "crashed"}

    _invaded(watchdog, system)
    snapshot = watchdog.state()["active"][0]

    system.occupancy = [_device([_proc(200, "them")])]
    system.owners    = {200: "them"}
    watchdog._evaluate()

    assert snapshot["status"] == "active"
    assert "dead_pids" not in snapshot
    assert watchdog.state()["active"][0]["status"]    == "critical"
    assert watchdog.state()["active"][0]["dead_pids"] == [100]


def test_history_returns_journal_entries_newest_first(tmp_path):
    """The intrusion and critical journal entries come back newest first with the full count."""
    watchdog, system, processes = _watchdog(tmp_path)
    watchdog.paths.gpu_guard_dir.mkdir(parents=True)
    processes.jobs  = {100: "j1"}
    processes.fates = {"j1": "crashed"}

    _invaded(watchdog, system)
    system.occupancy = [_device([_proc(200, "them")])]
    system.owners    = {200: "them"}
    watchdog._evaluate()

    history = watchdog.history()

    assert history["total"] == 2
    assert [entry["kind"] for entry in history["records"]] == ["critical", "intrusion"]
    assert history["records"][0]["dead_pids"]              == [100]
    assert history["records"][1]["intruder"]["pid"]        == 200


def test_history_is_served_from_memory_without_rereading_the_journal(tmp_path):
    """A history poll answers from the retained tail instead of re-parsing the journal file."""
    watchdog, system, _ = _watchdog(tmp_path)
    watchdog.paths.gpu_guard_dir.mkdir(parents=True)
    _invaded(watchdog, system)

    watchdog.log_path.write_text("")
    history = watchdog.history()

    assert history["total"] == 1
    assert history["records"][0]["kind"] == "intrusion"
    assert history["records"][0]["intruder"]["pid"] == 200


def test_start_seeds_the_journal_tail_from_the_existing_file(tmp_path):
    """Seeding parses only the retained tail of an existing journal while counting every entry."""
    watchdog, _, _ = _watchdog(tmp_path)
    watchdog.paths.gpu_guard_dir.mkdir(parents=True)

    entries = [{"kind": "intrusion", "detected_at": index} for index in range(GpuWatchdog.JOURNAL_KEEP + 5)]
    watchdog.log_path.write_text("".join(json.dumps(entry) + "\n" for entry in entries))

    watchdog._seed_journal()
    history = watchdog.history(limit=2)

    assert history["total"] == GpuWatchdog.JOURNAL_KEEP + 5
    assert [record["detected_at"] for record in history["records"]] == [GpuWatchdog.JOURNAL_KEEP + 4, GpuWatchdog.JOURNAL_KEEP + 3]
    assert len(watchdog.journal) == GpuWatchdog.JOURNAL_KEEP
    assert watchdog.journal[0]["detected_at"] == 5


def test_journal_append_and_history_read_serialise_on_the_watchdog_lock(tmp_path):
    """Neither a history read nor a journal append proceeds while the watchdog lock is held."""
    watchdog, system, _ = _watchdog(tmp_path)
    watchdog.paths.gpu_guard_dir.mkdir(parents=True)
    _invaded(watchdog, system)

    record  = watchdog.incidents[("GPU-0", 200)]["record"]
    results = []
    reader  = threading.Thread(target=lambda: results.append(watchdog.history()))
    writer  = threading.Thread(target=lambda: watchdog._persist(record, [], "intrusion"))

    with watchdog.lock:
        reader.start()
        writer.start()
        reader.join(timeout=0.3)
        writer.join(timeout=0.3)
        assert results == []
        assert watchdog.log_path.read_text().count("\n") == 1

    reader.join(timeout=5.0)
    writer.join(timeout=5.0)

    assert results[0]["total"] >= 1
    assert watchdog.history()["total"] == 2
