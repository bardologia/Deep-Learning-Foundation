from __future__ import annotations

import os
import threading
import time
from collections import deque
from datetime    import datetime

from proc_stats      import CpuCounters, MemInfo
from process_manager import ProcessManager
from web_logger      import WebLogger


class ResourceWatchdog:

    INTERVAL      = 5.0
    CPU_ALERT     = 90.0
    LOAD_RATIO    = 1.0
    RAM_WARN      = 80.0
    RAM_KILL      = 90.0
    CLEAR_MARGIN  = 5.0
    SUSTAIN       = {"cpu": 3, "load": 3, "ram": 2, "ram_kill": 2}
    KILL_COOLDOWN = 60.0

    def __init__(self, processes: ProcessManager, logger: WebLogger) -> None:
        self.processes = processes
        self.logger    = logger
        self.lock      = threading.Lock()
        self.armed     = False
        self.active    = {}
        self.events    = deque(maxlen=20)
        self.streaks   = {key: 0 for key in self.SUSTAIN}
        self.prev_cpu  = None
        self.last_kill = 0.0

    def start(self) -> None:
        worker = threading.Thread(target=self._watch, daemon=True)
        worker.start()
        self.armed = True
        self.logger.muted(f"resource watchdog armed (cpu {self.CPU_ALERT:.0f}%, ram warn {self.RAM_WARN:.0f}%, ram kill {self.RAM_KILL:.0f}%)")

    def state(self) -> dict:
        limits = {"cpu_alert": self.CPU_ALERT, "load_ratio": self.LOAD_RATIO, "ram_warn": self.RAM_WARN, "ram_kill": self.RAM_KILL, "interval": self.INTERVAL, "cooldown": self.KILL_COOLDOWN}

        with self.lock:
            return {"armed": self.armed, "limits": limits, "active": list(self.active.values()), "events": list(self.events)}

    def _watch(self) -> None:
        while True:
            time.sleep(self.INTERVAL)
            try:
                self._evaluate()
            except Exception as exc:
                self.logger.error(f"watchdog error: {exc}")

    def _evaluate(self) -> None:
        cpu   = self._cpu_percent()
        ram   = self._ram_percent()
        cores = os.cpu_count() or 1
        load1 = os.getloadavg()[0]
        ratio = load1 / cores

        if cpu is not None:
            self._track("cpu", "warn", cpu >= self.CPU_ALERT, cpu < self.CPU_ALERT - self.CLEAR_MARGIN,
                        f"CPU at {cpu:.0f}%, above the {self.CPU_ALERT:.0f}% threshold")

        self._track("load", "warn", ratio >= self.LOAD_RATIO, ratio < self.LOAD_RATIO - 0.1,
                    f"load {load1:.1f} saturates all {cores} cores, no spare CPU capacity")

        if ram is not None:
            self._track("ram", "danger" if ram >= self.RAM_KILL else "warn", ram >= self.RAM_WARN, ram < self.RAM_WARN - self.CLEAR_MARGIN,
                        f"RAM at {ram:.0f}%, jobs are terminated at {self.RAM_KILL:.0f}%")
            self._check_kill(ram)

    def _track(self, kind: str, level: str, breached: bool, cleared: bool, message: str) -> None:
        if breached:
            self.streaks[kind] += 1
            if self.streaks[kind] >= self.SUSTAIN[kind]:
                self._raise(kind, level, message)
        else:
            self.streaks[kind] = 0
            if cleared:
                self._drop(kind)

    def _check_kill(self, ram: float) -> None:
        if ram < self.RAM_KILL:
            self.streaks["ram_kill"] = 0
            return

        self.streaks["ram_kill"] += 1
        now = time.monotonic()
        if self.streaks["ram_kill"] < self.SUSTAIN["ram_kill"] or now - self.last_kill < self.KILL_COOLDOWN:
            return

        self.last_kill = now
        killed         = self.processes.stop_all()
        detail         = f"terminated {killed} running job(s)" if killed else "no console jobs were running to terminate"
        message        = f"RAM at {ram:.0f}% crossed the {self.RAM_KILL:.0f}% limit, {detail}"

        with self.lock:
            self.events.append({"kind": "ram_kill", "level": "danger", "message": message, "time": datetime.now().isoformat(timespec="seconds")})
        self.logger.error(message)

    def _raise(self, kind: str, level: str, message: str) -> None:
        with self.lock:
            known = self.active.get(kind)
            if known is None:
                self.active[kind] = {"kind": kind, "level": level, "message": message, "since": datetime.now().isoformat(timespec="seconds")}
                self.logger.warning(f"alert raised: {message}")
            else:
                known["level"]   = level
                known["message"] = message

    def _drop(self, kind: str) -> None:
        with self.lock:
            known = self.active.pop(kind, None)
        if known is not None:
            self.logger.muted(f"alert cleared: {kind}")

    def _cpu_percent(self) -> float | None:
        current       = CpuCounters.read().get("cpu")
        prev          = self.prev_cpu
        self.prev_cpu = current

        if current is None or prev is None or current[1] <= prev[1]:
            return None
        return 100.0 * (current[0] - prev[0]) / (current[1] - prev[1])

    def _ram_percent(self) -> float | None:
        return MemInfo.used_percent()
