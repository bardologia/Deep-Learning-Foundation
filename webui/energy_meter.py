"""Per-job GPU energy accounting integrated from the live power draw.

Every INTERVAL_S seconds the meter reads the GPU occupancy, splits each
device's power draw across the compute processes on it proportional to their
GPU memory, attributes the shares of this user's processes to their console
jobs, and integrates the result into each job's energy counter in watt-hours.
"""

from __future__ import annotations

import threading
import time

from web_logger import WebLogger


class JobEnergyMeter:
    """Integrates GPU power draw into per-job energy counters.

    Attributes:
        system    : System monitor supplying the GPU occupancy with power draw.
        processes : Process manager mapping pids to jobs and holding the counters.
        logger    : Console logger for sampling failures.
        prev_t    : Monotonic time of the previous integrated sample, or None.
    """

    INTERVAL_S = 15.0

    def __init__(self, system, processes, logger: WebLogger) -> None:
        """Stores the collaborators and starts with no previous sample.

        Args:
            system    : System monitor supplying GPU occupancy and the console user.
            processes : Process manager the integrated energy is booked into.
            logger    : Console logger for sampling failures.
        """
        self.system    = system
        self.processes = processes
        self.logger    = logger
        self.prev_t    = None

    def start(self) -> None:
        """Starts the background integration thread."""
        worker = threading.Thread(target=self._watch, daemon=True)
        worker.start()
        self.logger.muted(f"job energy meter armed (interval {self.INTERVAL_S:.0f}s, device power split by GPU memory share)")

    def _job_watts(self, occupancy: list[dict]) -> dict:
        """Attributes each device's power draw to this user's console jobs.

        A device's power is split across the compute processes on it in
        proportion to their GPU memory (equally when no memory is reported),
        and each share owned by this user is booked to the job owning the pid.

        Args:
            occupancy : Per-device occupancy records from the system monitor.

        Returns:
            Watts currently attributed per job id.
        """
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
        """Integrates one power reading into the counters of the jobs holding GPUs."""
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
        """Samples on the fixed interval forever, logging any integration failure."""
        while True:
            time.sleep(self.INTERVAL_S)
            try:
                self.sample()
            except Exception as error:
                self.logger.error(f"job energy meter error: {error}")
