
from __future__ import annotations

import threading
import time

from web_logger import WebLogger


class JobEnergyMeter:

    INTERVAL_S = 15.0

    def __init__(self, system, processes, logger: WebLogger) -> None:
        self.system    = system
        self.processes = processes
        self.logger    = logger
        self.prev_t    = None

    def start(self) -> None:
        worker = threading.Thread(target=self._watch, daemon=True)
        worker.start()
        self.logger.muted(f"job energy meter armed (interval {self.INTERVAL_S:.0f}s, device power split by GPU memory share)")

    def _job_watts(self, occupancy: list[dict]) -> dict:
        watts = {}
        for device in occupancy:
            power = device["power"]
            procs = device["procs"]
            if power is None or not procs:
                continue

            total_mem = sum(proc["mem"] for proc in procs)
            for proc in procs:
                if proc["owner"] != self.system.user:
                    continue

                job_id = self.processes.job_for_pid(proc["pid"])
                if job_id is None:
                    continue

                share          = (proc["mem"] / total_mem) if total_mem > 0 else 1.0 / len(procs)
                watts[job_id]  = watts.get(job_id, 0.0) + power * share
        return watts

    def sample(self) -> None:
        now       = time.monotonic()
        occupancy = self.system.gpu_occupancy()

        if occupancy is None:
            self.prev_t = None
            return

        prev        = self.prev_t
        self.prev_t = now
        if prev is None or now <= prev:
            return

        dt = now - prev
        for job_id, job_watts in self._job_watts(occupancy).items():
            self.processes.add_energy(job_id, job_watts * dt / 3600.0)

    def _watch(self) -> None:
        while True:
            time.sleep(self.INTERVAL_S)
            try:
                self.sample()
            except Exception as error:
                self.logger.error(f"job energy meter error: {error}")
