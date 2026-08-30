"""Detection and journalling of other users landing on GPUs you already hold.

A background thread polls GPU occupancy, opens an incident whenever a foreign
process joins a device after one of your own processes, escalates it to
critical when your process dies inside the critical window, and records every
incident to a JSONL journal. Incidents whose intruder gives up first are
reported to the GPU schedule as bounced launch attempts.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from collections import deque
from datetime    import datetime

from notifier        import JobNotifier
from process_manager import ProcessManager
from project_paths   import ProjectPaths
from system_monitor  import SystemMonitor
from web_logger      import WebLogger


class GpuWatchdog:
    """Watches for foreign processes intruding on GPUs held by this user.

    Attributes:
        system        : System monitor supplying GPU occupancy and process ownership.
        paths         : Project paths supplying the guard journal directory.
        logger        : Web UI logger raising the intrusion and critical alerts.
        processes     : Process manager mapping own pids to jobs and their fates.
        notifier      : Notifier the intrusion and critical alerts are pushed through.
        host          : Hostname stamped into every journal entry.
        armed         : Whether the polling thread has been started.
        incidents     : Open incidents keyed by (GPU uuid, intruder pid).
        residents     : First and last sighting time of every observed GPU process.
        gpus          : Latest per-device occupancy summary.
        events        : Recent intrusion and critical records, newest last.
        bounced       : Intrusions that ended with the intruder leaving first.
        count         : Total intrusions raised.
        critical      : Total critical escalations raised.
        journal       : In-memory tail of the journal entries, oldest first.
        journal_total : Total number of entries the journal file holds.
    """

    INTERVAL          = 4.0
    GRACE_S           = 30.0
    CRITICAL_WINDOW_S = 300.0
    INCIDENT_HOLD_S   = 900.0
    EVENT_LIMIT       = 50
    JOURNAL_KEEP      = 500
    LOG_NAME          = "intrusions.jsonl"

    def __init__(self, system: SystemMonitor, paths: ProjectPaths, logger: WebLogger, processes: ProcessManager, notifier: JobNotifier) -> None:
        """Prepares the watchdog state and resolves the journal path.

        Args:
            system    : System monitor supplying GPU occupancy and process owners.
            paths     : Project paths supplying the GPU guard directory.
            logger    : Web UI logger.
            processes : Process manager used to attribute own pids to jobs.
            notifier  : Notifier the intrusion and critical alerts are pushed through.
        """
        self.system        = system
        self.paths         = paths
        self.logger        = logger
        self.processes     = processes
        self.notifier      = notifier
        self.host          = socket.gethostname()
        self.lock          = threading.Lock()
        self.armed         = False
        self.incidents     = {}
        self.residents     = {}
        self.gpus          = []
        self.events        = deque(maxlen=self.EVENT_LIMIT)
        self.bounced       = deque(maxlen=self.EVENT_LIMIT)
        self.count         = 0
        self.critical      = 0
        self.journal       = deque(maxlen=self.JOURNAL_KEEP)
        self.journal_total = 0
        self.log_path      = self.paths.gpu_guard_dir / self.LOG_NAME

    def start(self) -> None:
        """Creates the journal directory, seeds the in-memory journal tail and arms the polling thread."""
        self.paths.gpu_guard_dir.mkdir(parents=True, exist_ok=True)
        self._seed_journal()
        worker = threading.Thread(target=self._watch, daemon=True)
        worker.start()
        self.armed = True
        self.logger.muted(f"gpu watchdog armed (interval {self.INTERVAL:.0f}s, journal {self.log_path})")

    def _seed_journal(self) -> None:
        """Reads the journal file once, keeping its entry count and the parsed tail in memory."""
        try:
            with open(self.log_path, encoding="utf-8") as handle:
                lines = [line.strip() for line in handle if line.strip()]
        except FileNotFoundError:
            return

        with self.lock:
            self.journal_total = len(lines)
            self.journal.extend(json.loads(line) for line in lines[-self.JOURNAL_KEEP:])

    def state(self) -> dict:
        """Returns the live panel state: armed flag, copies of the open incident records, GPU summary and counters."""
        with self.lock:
            active = [dict(incident["record"]) for incident in self.incidents.values() if incident["status"] in ("active", "critical")]
            return {
                "armed"    : self.armed,
                "active"   : active,
                "gpus"     : self.gpus,
                "events"   : list(self.events),
                "count"    : self.count,
                "critical" : self.critical,
                "log_path" : str(self.log_path),
            }

    def take_bounced(self) -> list[dict]:
        """Returns and clears the intrusion records whose intruder left first."""
        with self.lock:
            bounced = list(self.bounced)
            self.bounced.clear()

        return bounced

    def history(self, limit: int = 100) -> dict:
        """Returns the retained journal tail from memory, never re-reading the journal file.

        Args:
            limit : Maximum number of records to return, newest first, capped by the retained tail.

        Returns:
            Dict with the trimmed ``records``, the ``total`` entry count of the
            journal file and its ``log_path``; empty when nothing was journalled.
        """
        with self.lock:
            records = [dict(record) for record in list(self.journal)[-limit:]] if limit > 0 else []
            total   = self.journal_total

        records.reverse()
        return {"records": records, "total": total, "log_path": str(self.log_path)}

    def _watch(self) -> None:
        """Polls occupancy on the fixed interval until the process exits."""
        while True:
            time.sleep(self.INTERVAL)
            try:
                self._evaluate()
            except Exception as exc:
                self.logger.error(f"gpu watchdog error: {exc}")

    def _evaluate(self) -> None:
        """Runs one detection pass over the current occupancy snapshot."""
        occupancy = self.system.gpu_occupancy()
        if occupancy is None:
            return

        server_gpus = self._server_gpus(occupancy)
        intruders   = self._detect(occupancy)

        self._register(intruders, server_gpus)
        self._escalate(intruders, server_gpus)
        self._forget(time.monotonic())

        with self.lock:
            self.gpus = server_gpus

    def _server_gpus(self, occupancy: list[dict]) -> list[dict]:
        """Summarises each device with its holders and whether this user or others own it.

        Args:
            occupancy : Per-device occupancy records from the system monitor.

        Returns:
            One summary dict per device with utilisation, memory in MiB,
            holders, and the mine / others / free flags.
        """
        summary = []
        for device in occupancy:
            owners = {proc["owner"] for proc in device["procs"] if proc["owner"]}
            summary.append({
                "index"     : device["index"],
                "name"      : device["name"],
                "util"      : device["util"],
                "mem_used"  : device["mem_used"],
                "mem_total" : device["mem_total"],
                "holders"   : [{"user": proc["owner"], "pid": proc["pid"], "mem_mib": proc["mem"], "cmd": proc["cmd"]} for proc in device["procs"]],
                "mine"      : self.system.user in owners,
                "others"    : bool(owners - {self.system.user}),
                "free"      : len(device["procs"]) == 0,
            })
        return summary

    def _detect(self, occupancy: list[dict]) -> dict:
        """Finds foreign processes that joined a device after this user's own.

        Args:
            occupancy : Per-device occupancy records from the system monitor.

        Returns:
            Intrusion records keyed by (GPU uuid, intruder pid).
        """
        now = time.monotonic()
        self._remember(occupancy, now)

        intruders = {}
        for device in occupancy:
            mine = [proc for proc in device["procs"] if proc["owner"] == self.system.user]
            if not mine:
                continue

            mine_first = min(self.residents[(device["uuid"], proc["pid"])]["first"] for proc in mine)
            for proc in device["procs"]:
                owner = proc["owner"]
                if owner is None or owner == self.system.user:
                    continue
                if self.residents[(device["uuid"], proc["pid"])]["first"] <= mine_first:
                    continue
                intruders[(device["uuid"], proc["pid"])] = self._intrusion(device, proc, mine)
        return intruders

    def _remember(self, occupancy: list[dict], now: float) -> None:
        """Updates first and last sighting times per GPU process and drops stale entries.

        Args:
            occupancy : Per-device occupancy records from the system monitor.
            now       : Monotonic clock reading of this pass, in seconds.
        """
        for device in occupancy:
            for proc in device["procs"]:
                key      = (device["uuid"], proc["pid"])
                resident = self.residents.get(key)
                if resident is None:
                    self.residents[key] = {"first": now, "last": now}
                else:
                    resident["last"] = now

        stale = [key for key, resident in self.residents.items() if now - resident["last"] > self.GRACE_S]
        for key in stale:
            del self.residents[key]

    def _intrusion(self, device: dict, proc: dict, mine: list[dict]) -> dict:
        """Builds the record describing one foreign process sharing a device.

        Args:
            device : Occupancy record of the shared GPU.
            proc   : Occupancy record of the foreign process.
            mine   : Occupancy records of this user's processes on the device.

        Returns:
            Intrusion record with GPU identity, both parties' pids and memory
            in MiB, and an ``active`` status stamped with the current time.
        """
        return {
            "gpu_index" : device["index"],
            "gpu_uuid"  : device["uuid"],
            "gpu_name"  : device["name"],
            "user"      : proc["owner"],
            "pid"       : proc["pid"],
            "mem_mib"   : proc["mem"],
            "cmd"       : proc["cmd"],
            "mine_pids" : [m["pid"] for m in mine],
            "mine_mib"  : round(sum(m["mem"] for m in mine), 1),
            "util"      : device["util"],
            "mem_used"  : device["mem_used"],
            "mem_total" : device["mem_total"],
            "status"    : "active",
            "since"     : datetime.now().isoformat(timespec="seconds"),
        }

    def _register(self, intruders: dict, server_gpus: list[dict]) -> None:
        """Opens incidents for newly seen intruders and refreshes the ones already open.

        Args:
            intruders   : Intrusion records keyed by (GPU uuid, intruder pid).
            server_gpus : Device summaries stored alongside each journal entry.
        """
        now   = time.monotonic()
        fresh = []

        with self.lock:
            for key, record in intruders.items():
                incident = self.incidents.get(key)
                if incident is None:
                    jobs                = {pid: self.processes.job_for_pid(pid) for pid in record["mine_pids"]}
                    self.incidents[key] = {"record": record, "mine_pids": list(record["mine_pids"]), "jobs": jobs, "status": "active", "started": now, "last_seen": now}
                    fresh.append(record)
                else:
                    incident["last_seen"] = now

        for record in fresh:
            self._persist(record, server_gpus, "intrusion")
            self._raise(record)

    def _escalate(self, intruders: dict, server_gpus: list[dict]) -> None:
        """Closes each open incident as critical, ended, or survived.

        Args:
            intruders   : Intrusion records seen in this pass.
            server_gpus : Device summaries stored alongside each journal entry.
        """
        now = time.monotonic()

        with self.lock:
            snapshot = [(key, incident) for key, incident in self.incidents.items() if incident["status"] == "active"]

        for key, incident in snapshot:
            gone  = [pid for pid in incident["mine_pids"] if self.system.pid_owner(pid) != self.system.user]
            fates = {pid: self._fate(incident, pid) for pid in gone}
            gone  = [pid for pid in gone if fates[pid] != "pending"]

            crashed = [pid for pid in gone if fates[pid] == "crashed"]
            if crashed and now - incident["started"] <= self.CRITICAL_WINDOW_S:
                self._flag_critical(incident, crashed, server_gpus)
            elif gone:
                self._retire(incident, gone, fates)
            elif key not in intruders and now - incident["last_seen"] > self.GRACE_S:
                with self.lock:
                    incident["status"]           = "survived"
                    incident["record"]["status"] = "survived"
                    incident["closed"]           = now
                    self.bounced.append({**incident["record"], "bounced_at": datetime.now().isoformat(timespec="seconds")})

    def _fate(self, incident: dict, pid: int) -> str:
        """Returns how one of this user's pids ended: its job fate, or "unknown" when unattributed."""
        job_id = incident["jobs"].get(pid)
        if job_id is None:
            return "unknown"
        return self.processes.job_fate(job_id, pid)

    def _retire(self, incident: dict, gone: list[int], fates: dict) -> None:
        """Drops departed own pids from an incident and ends it once none remain.

        Args:
            incident : Open incident being updated.
            gone     : Own pids no longer owned by this user.
            fates    : Fate string per departed pid.
        """
        with self.lock:
            incident["mine_pids"] = [pid for pid in incident["mine_pids"] if pid not in gone]
            if not incident["mine_pids"]:
                incident["status"]           = "ended"
                incident["record"]["status"] = "ended"
                incident["closed"]           = time.monotonic()

        detail = ", ".join(f"{pid} {fates[pid]}" for pid in gone)
        if incident["status"] == "ended":
            self.logger.muted(f"gpu incident on gpu {incident['record']['gpu_index']} ended, own pid(s) exited ({detail})")

    def _flag_critical(self, incident: dict, dead: list[int], server_gpus: list[dict]) -> None:
        """Marks an incident critical because own processes crashed inside the window.

        Args:
            incident    : Open incident being escalated.
            dead        : Own pids that crashed.
            server_gpus : Device summaries stored alongside the journal entry.
        """
        with self.lock:
            record                = incident["record"]
            incident["status"]    = "critical"
            incident["closed"]    = time.monotonic()
            record["status"]      = "critical"
            record["dead_pids"]   = dead
            record["critical_at"] = datetime.now().isoformat(timespec="seconds")
            self.critical        += 1
            self.events.append(dict(record))

        self._persist(record, server_gpus, "critical")
        message = f"CRITICAL: your pid(s) {dead} on gpu {record['gpu_index']} [{record['gpu_name']}] DIED within {self.CRITICAL_WINDOW_S:.0f}s of {record['user']} (pid {record['pid']}) invading it"
        self.logger.error(message)
        self.notifier.system_alert("GPU intrusion killed your job", message, "urgent")

    def _forget(self, now: float) -> None:
        """Removes closed incidents from the live panel once the hold time has passed."""
        with self.lock:
            done = {key: incident["record"] for key, incident in self.incidents.items() if incident["status"] != "active" and now - incident["closed"] >= self.INCIDENT_HOLD_S}
            for key in done:
                del self.incidents[key]

        for record in done.values():
            self.logger.muted(f"gpu incident on gpu {record['gpu_index']} ({record['user']} pid {record['pid']}, {record['status']}) left the live panel after {self.INCIDENT_HOLD_S / 60:.0f} min; the journal keeps it")

    def _persist(self, record: dict, server_gpus: list[dict], kind: str) -> None:
        """Appends one incident entry to the JSONL journal and the in-memory tail in one lock hold.

        Args:
            record      : Intrusion record to serialise.
            server_gpus : Device summaries captured with the entry.
            kind        : Entry kind, either ``intrusion`` or ``critical``.
        """
        entry = {
            "kind"        : kind,
            "detected_at" : record["since"],
            "host"        : self.host,
            "user_me"     : self.system.user,
            "gpu"         : {"index": record["gpu_index"], "uuid": record["gpu_uuid"], "name": record["gpu_name"], "util": record["util"], "mem_used": record["mem_used"], "mem_total": record["mem_total"]},
            "intruder"    : {"user": record["user"], "pid": record["pid"], "mem_mib": record["mem_mib"], "cmd": record["cmd"]},
            "mine"        : {"pids": record["mine_pids"], "mem_mib": record["mine_mib"]},
            "all_gpus"    : server_gpus,
        }

        if kind == "critical":
            entry["dead_pids"]   = record.get("dead_pids", [])
            entry["critical_at"] = record.get("critical_at")

        try:
            with self.lock, open(self.log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
                self.journal.append(entry)
                self.journal_total += 1
        except OSError as exc:
            self.logger.error(f"gpu watchdog could not write journal: {exc}")

    def _raise(self, record: dict) -> None:
        """Counts an intrusion, adds it to the event ring, logs the alert and pushes it.

        Args:
            record : Intrusion record to announce.
        """
        with self.lock:
            self.count += 1
            self.events.append(dict(record))

        message = f"GPU INTRUSION: {record['user']} (pid {record['pid']}, {round(record['mem_mib'])} MiB) joined gpu {record['gpu_index']} [{record['gpu_name']}] while you hold pids {record['mine_pids']}"
        self.logger.error(message)
        self.notifier.system_alert("GPU intrusion", message, "high")
