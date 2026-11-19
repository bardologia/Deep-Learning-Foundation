
from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime    import datetime, timedelta
from pathlib     import Path

from tools.data.io           import FileIO
from tools.monitoring.logger import Logger
from tools.runtime.cuda_mask import InheritedCudaMask


@dataclass
class GpuJob:

    name     : str
    command  : list[str]
    log_path : Path


@dataclass
class GpuJobResult:

    name       : str
    gpu        : int
    status     : str
    returncode : int
    duration_s : float
    log_file   : str


@dataclass
class PoolEdit:

    gpus    : list[int]
    preempt : list[int]


class GpuPoolFile:

    FILE_NAME        = "gpu_pool.json"
    PROCESS_START_NS = time.time_ns()

    def __init__(self, path: Path, logger: Logger) -> None:
        self.path   = Path(path)
        self.logger = logger
        self.stamp  : int | None = None

    @classmethod
    def resolve(cls, configured: str, run_dir: Path) -> Path:
        return Path(configured) if configured else Path(run_dir) / cls.FILE_NAME

    def _stamp(self) -> int | None:
        return self.path.stat().st_mtime_ns if self.path.exists() else None

    def write(self, gpus: list[int], preempt: list[int] = ()) -> None:
        payload = {"gpus": list(gpus)}
        if preempt:
            payload["preempt"] = list(preempt)

        FileIO.save_json(payload, self.path, indent=2, atomic=True)
        self.stamp = self._stamp()

    def live(self) -> list[int] | None:
        stamp = self._stamp()

        if stamp is None or stamp <= self.PROCESS_START_NS:
            return None

        try:
            return self.parse(FileIO.load_json(self.path)).gpus
        except (ValueError, TypeError, json.JSONDecodeError, OSError) as error:
            self.logger.error(f"Ignored the malformed GPU pool in {self.path}: {error}. This pass falls back to the launch selection; fix the file to steer the experiment.")
            return None

    def seed(self, gpus: list[int]) -> list[int]:
        live = self.live()
        pool = gpus if live is None else live

        self.write(pool)
        self.logger.info(f"GPU pool is live-editable at {self.path} — write {{\"gpus\": [...]}} to resize this experiment while it runs, add \"preempt\": [...] to free devices at once (their runs stop and restart from the head of the queue)")

        if live is not None and sorted(live) != sorted(gpus):
            self.logger.info(f"GPU pool {pool} carried over from the edit made while this experiment was already running; the launch selection {list(gpus)} no longer applies")

        return list(pool)

    @staticmethod
    def _devices(payload: dict, key: str) -> list[int]:
        requested = payload[key]

        if not isinstance(requested, list):
            raise ValueError(f"'{key}' must be a list of device ids, got {requested!r}")

        invalid = [gpu for gpu in requested if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0]
        if invalid:
            raise ValueError(f"'{key}' must hold non-negative integers, got {invalid}")

        duplicates = sorted({gpu for gpu in requested if requested.count(gpu) > 1})
        if duplicates:
            raise ValueError(f"'{key}' must not repeat a device, duplicated: {duplicates}")

        return list(requested)

    @classmethod
    def validate(cls, payload) -> list[int]:
        if not isinstance(payload, dict) or "gpus" not in payload:
            raise ValueError(f"expected an object holding a 'gpus' key, got {payload!r}")

        return cls._devices(payload, "gpus")

    @classmethod
    def parse(cls, payload) -> PoolEdit:
        gpus    = cls.validate(payload)
        preempt = cls._devices(payload, "preempt") if "preempt" in payload else []
        overlap = sorted(set(gpus) & set(preempt))

        if overlap:
            raise ValueError(f"a device cannot stay in 'gpus' and be freed through 'preempt' at the same time, listed in both: {overlap}")

        return PoolEdit(gpus=gpus, preempt=preempt)

    def requested(self) -> PoolEdit | None:
        stamp = self._stamp()

        if stamp is None or stamp == self.stamp:
            return None

        self.stamp = stamp

        try:
            return self.parse(FileIO.load_json(self.path))
        except (ValueError, TypeError, json.JSONDecodeError, OSError) as error:
            self.logger.error(f"Rejected the GPU pool edit in {self.path}: {error}. The pool is unchanged; fix the file to resize the experiment.")
            return None


class GpuProgressFile:

    SUFFIX           = "_progress.json"
    WRITE_INTERVAL_S = 15.0
    TERMINAL         = ("done", "failed", "stale")

    def __init__(self, path: Path, jobs: list[GpuJob], logger: Logger) -> None:
        self.path          = Path(path)
        self.total         = len(jobs)
        self.logger        = logger
        self.started_mono  = time.monotonic()
        self.started_at    = datetime.now()
        self.durations     : list[float] = []
        self.done          = 0
        self.failed        = 0
        self.failed_units  : list[str] = []
        self.preempted     = 0
        self.last_write    = float("-inf")
        self.written_units = None

        self._reject_duplicate_logs(jobs)

        self.units       = {str(job.log_path): self._fresh_unit(job) for job in jobs}
        self.prior_units = self._load_prior(set(self.units))

    @staticmethod
    def _reject_duplicate_logs(jobs: list[GpuJob]) -> None:
        counts     = Counter(str(job.log_path) for job in jobs)
        duplicates = sorted(log for log, count in counts.items() if count > 1)

        if duplicates:
            raise ValueError(f"Queued units share log path(s): {', '.join(duplicates)}. Two units writing the same log file would overwrite each other's output and collapse into a single progress entry — rename the runs so their log paths differ")

    @classmethod
    def resolve(cls, pool_file: Path) -> Path:
        pool = Path(pool_file)
        return pool.with_name(pool.stem + cls.SUFFIX)

    @staticmethod
    def _fresh_unit(job: GpuJob) -> dict:
        return {"name": job.name, "log": str(job.log_path), "status": "queued", "gpu": None, "duration_s": None, "returncode": None, "preemptions": 0}

    def _load_prior(self, current: set[str]) -> list[dict]:
        if not self.path.exists():
            return []

        payload = FileIO.load_json(self.path)
        if "units" not in payload:
            raise ValueError(f"progress file {self.path} predates per-unit tracking — delete it and relaunch")

        prior = [dict(unit) for unit in payload["units"] if unit["log"] not in current]
        for unit in prior:
            if unit["status"] not in self.TERMINAL:
                unit["status"] = "stale"

        return prior

    @property
    def completed(self) -> int:
        return self.done + self.failed

    def preempt(self, job: GpuJob) -> None:
        unit                 = self.units[str(job.log_path)]
        unit["status"]       = "queued"
        unit["gpu"]          = None
        unit["preemptions"] += 1

        self.preempted += 1

    def record(self, result: GpuJobResult) -> None:
        unit               = self.units[result.log_file]
        unit["status"]     = "done" if result.status == "DONE" else "failed"
        unit["gpu"]        = result.gpu
        unit["duration_s"] = round(result.duration_s, 1)
        unit["returncode"] = result.returncode

        if result.status == "DONE":
            self.done += 1
        else:
            self.failed += 1
            self.failed_units.append(result.name)

        self.durations.append(result.duration_s)

    def _average_s(self) -> float | None:
        return sum(self.durations) / len(self.durations) if self.durations else None

    def _eta_s(self, running: list[dict], queued: int, workers: int) -> float | None:
        average = self._average_s()
        if average is None:
            return None

        remaining = [max(average - unit["elapsed_s"], 0.0) for unit in running]
        pooled    = (queued * average + sum(remaining)) / max(workers, 1)

        return max(pooled, max(remaining, default=0.0))

    def _unit_states(self, running: list[dict]) -> list[dict]:
        live = {unit["log"]: unit for unit in running}

        states = []
        for log, unit in self.units.items():
            state = dict(unit)
            if state["status"] == "queued" and log in live:
                state["status"] = "running"
                state["gpu"]    = live[log]["gpu"]
            states.append(state)

        return self.prior_units + states

    def snapshot(self, running: list[dict], queued: int, workers: int) -> dict:
        elapsed = time.monotonic() - self.started_mono
        eta     = self._eta_s(running, queued, workers)

        return {
            "total"        : self.total,
            "done"         : self.done,
            "failed"       : self.failed,
            "queued"       : queued,
            "running"      : running,
            "workers"      : workers,
            "units"        : self._unit_states(running),
            "failed_units" : list(self.failed_units),
            "preempted"    : self.preempted,
            "average_s"    : self._average_s(),
            "elapsed_s"    : elapsed,
            "eta_s"        : eta,
            "total_s"      : elapsed + eta if eta is not None else None,
            "started_at"   : self.started_at.isoformat(timespec="seconds"),
            "finish_at"    : (datetime.now() + timedelta(seconds=eta)).isoformat(timespec="seconds") if eta is not None else None,
            "updated_at"   : datetime.now().isoformat(timespec="seconds"),
        }

    def _label(self, seconds: float) -> str:
        minutes, _     = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)

        if hours:
            return f"{hours}h {minutes:02d}m"
        if minutes:
            return f"{minutes} min"
        return "<1 min"

    def log_progress(self, snapshot: dict) -> None:
        share  = f"[{self.completed}/{self.total}]"
        failed = f" · {self.failed} FAILED" if self.failed else ""

        if snapshot["eta_s"] is None:
            self.logger.info(f"{share} units finished{failed}")
            return

        average = self._label(snapshot["average_s"])
        eta     = self._label(snapshot["eta_s"])
        finish  = snapshot["finish_at"][11:16]

        self.logger.info(f"{share} avg {average}/unit · ETA {eta} · finish ≈ {finish}{failed}")

    def write(self, snapshot: dict, force: bool = False) -> None:
        signature = tuple((unit["log"], unit["status"]) for unit in snapshot["units"])
        now       = time.monotonic()

        if not force and signature == self.written_units and now - self.last_write < self.WRITE_INTERVAL_S:
            return

        self.last_write    = now
        self.written_units = signature
        FileIO.save_json(snapshot, self.path, indent=2, atomic=True)


class GpuQueue:

    EMPTY_POOL_MESSAGE = "at least one GPU id is required; an empty gpus list would poll forever without launching any job"

    def __init__(self, gpus: list[int], logger: Logger, poll_interval_s: float = 5.0, handle_signals: bool = True, terminate_deadline_s: float = 30.0, pool_file: Path | None = None, resume_hint: str = "") -> None:
        if not gpus:
            raise ValueError(f"GpuQueue requires {self.EMPTY_POOL_MESSAGE}.")

        self.gpus                 = list(gpus)
        self.logger               = logger
        self.poll_interval_s      = poll_interval_s
        self.handle_signals       = handle_signals
        self.terminate_deadline_s = terminate_deadline_s
        self.pool                 = GpuPoolFile(pool_file, logger) if pool_file is not None else None
        self.resume_hint          = resume_hint
        self.progress             : GpuProgressFile | None = None
        self.running              : list[dict] = []
        self.retiring             : set[int] = set()
        self.results              : list[GpuJobResult] = []

    def _install_signal_handlers(self) -> dict:
        previous = {
            signal.SIGTERM : signal.getsignal(signal.SIGTERM),
            signal.SIGINT  : signal.getsignal(signal.SIGINT),
        }

        signal.signal(signal.SIGTERM, self._terminate_running)
        signal.signal(signal.SIGINT,  self._terminate_running)

        return previous

    def _stop_workers(self, records: list[dict]) -> None:
        for record in records:
            if record["process"].poll() is None:
                record["process"].terminate()

        deadline = time.time() + self.terminate_deadline_s
        for record in records:
            try:
                record["process"].wait(timeout=max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                record["process"].kill()
                record["process"].wait()
                self.logger.warning(f"[GPU {record['gpu']}] {record['job'].name} did not exit within {self.terminate_deadline_s:.0f}s of SIGTERM; killed")

            record["log_fh"].close()

    def _shutdown_running(self) -> list[str]:
        stopped = [f"{record['job'].name} (GPU {record['gpu']})" for record in self.running]

        self._stop_workers(self.running)
        self.running = []

        return stopped

    def _terminate_running(self, signum, frame) -> None:
        stopped = self._shutdown_running()

        self.logger.warning(f"Received signal {signum} — terminated {len(stopped)} workers: {', '.join(stopped) or 'none'}{self._resume_suffix()}")

        sys.exit(128 + signum)

    def _resume_suffix(self) -> str:
        return f"; {self.resume_hint}" if self.resume_hint else ""

    def _reap(self, running: list[dict], gpu_pool: list[int], results: list[GpuJobResult]) -> None:
        finished = [record for record in running if record["process"].poll() is not None]

        for record in finished:
            running.remove(record)
            record["log_fh"].close()

            job        = record["job"]
            returncode = record["process"].returncode
            status     = "DONE" if returncode == 0 else "FAILED"
            duration_s = time.monotonic() - record["started"]

            if returncode == 0:
                self.logger.info(f"[GPU {record['gpu']}] finished  {job.name}  ({duration_s / 60:.1f} min)")
            else:
                self.logger.error(f"[GPU {record['gpu']}] failed    {job.name}  (exit {returncode}, see {job.log_path})")

            results.append(GpuJobResult(
                name       = job.name,
                gpu        = record["gpu"],
                status     = status,
                returncode = returncode,
                duration_s = duration_s,
                log_file   = str(job.log_path),
            ))

            if record["gpu"] in self.retiring:
                self.retiring.discard(record["gpu"])
                self.logger.info(f"[GPU {record['gpu']}] drained — released from the pool")
                continue

            gpu_pool.append(record["gpu"])

    def _reconcile(self, gpu_pool: list[int], queue: list[GpuJob]) -> None:
        if self.pool is None:
            return

        requested = self.pool.requested()
        if requested is None:
            return

        self._preempt(requested.preempt, gpu_pool, queue)

        busy    = [record["gpu"] for record in self.running]
        active  = set(gpu_pool) | set(busy)
        added   = sorted(set(requested.gpus) - active)
        removed = sorted(active - set(requested.gpus))

        gpu_pool.extend(added)
        gpu_pool.sort()

        for gpu in removed:
            if gpu in gpu_pool:
                gpu_pool.remove(gpu)

        self.retiring |= set(removed) & set(busy)
        self.retiring -= set(requested.gpus)

        self._log_resize(added, removed, gpu_pool, queue)

        if requested.preempt:
            self.pool.write(requested.gpus)

    def _preempt(self, devices: list[int], gpu_pool: list[int], queue: list[GpuJob]) -> None:
        if not devices:
            return

        stopped = [record for record in self.running if record["gpu"] in devices and record["process"].poll() is None]
        self._stop_workers(stopped)

        for record in stopped:
            self.running.remove(record)

            minutes = (time.monotonic() - record["started"]) / 60
            self.logger.warning(f"[GPU {record['gpu']}] preempted {record['job'].name} after {minutes:.1f} min — device freed, the unit restarts from the head of the queue")

            if self.progress is not None:
                self.progress.preempt(record["job"])

        queue[:0] = [record["job"] for record in stopped]

        idle = [gpu for gpu in devices if gpu in gpu_pool]
        for gpu in idle:
            gpu_pool.remove(gpu)

        self.retiring -= set(devices)

        if idle:
            self.logger.info(f"GPU pool released {idle} — idle, no longer used")

    def _log_resize(self, added: list[int], removed: list[int], gpu_pool: list[int], queue: list[GpuJob]) -> None:
        if not added and not removed:
            return

        if added:
            self.logger.info(f"GPU pool grew by {added} — {len(queue)} jobs still queued")

        draining = sorted(self.retiring)
        released = [gpu for gpu in removed if gpu not in self.retiring]

        if released:
            self.logger.info(f"GPU pool released {released} — idle, no longer used")

        if draining:
            self.logger.info(f"GPU pool retiring {draining} — released once the job in flight finishes")

        if not gpu_pool and not self.running:
            self.logger.warning(f"GPU pool is empty — {len(queue)} jobs parked until a device is added back to {self.pool.path}")

    def _launch_banner(self, job: GpuJob, gpu_id: int) -> str:
        preemptions = 0 if self.progress is None else self.progress.units[str(job.log_path)]["preemptions"]
        restart     = f" (restart after {preemptions} preemption(s))" if preemptions else ""

        return f"===== {datetime.now():%Y-%m-%d %H:%M:%S} launch of {job.name} on GPU {gpu_id}{restart} ====="

    def _launch(self, job: GpuJob, gpu_id: int) -> dict:
        job.log_path.parent.mkdir(parents=True, exist_ok=True)

        command = job.command + ["--gpu", str(gpu_id)]
        log_fh  = open(job.log_path, "a", encoding="utf-8")

        log_fh.write(self._launch_banner(job, gpu_id) + "\n")
        log_fh.flush()

        try:
            process = subprocess.Popen(command, stdout=log_fh, stderr=log_fh, env=InheritedCudaMask.restored())
        except BaseException:
            log_fh.close()
            raise

        self.logger.info(f"[GPU {gpu_id}] started   {job.name}")

        return {"job": job, "gpu": gpu_id, "process": process, "log_fh": log_fh, "started": time.monotonic()}

    def _publish_progress(self, queue: list[GpuJob], gpu_pool: list[int], results: list[GpuJobResult]) -> None:
        if self.progress is None:
            return

        announce = len(results) > self.progress.completed
        for result in results[self.progress.completed:]:
            self.progress.record(result)

        running  = [{"name": record["job"].name, "gpu": record["gpu"], "log": str(record["job"].log_path), "elapsed_s": round(time.monotonic() - record["started"], 1)} for record in self.running]
        workers  = max(1, len(gpu_pool) + len(self.running) - len(self.retiring))
        snapshot = self.progress.snapshot(running, len(queue), workers)

        if announce:
            self.progress.log_progress(snapshot)

        self.progress.write(snapshot, force=announce)

    def _restore_signal_handlers(self, previous: dict) -> None:
        for signum, handler in previous.items():
            signal.signal(signum, handler)

    def run(self, jobs: list[GpuJob]) -> list[GpuJobResult]:
        queue = list(jobs)

        self.results  = []
        self.running  = []
        self.retiring = set()

        if self.pool is not None and jobs:
            self.gpus     = self.pool.seed(self.gpus)
            self.progress = GpuProgressFile(GpuProgressFile.resolve(self.pool.path), jobs, self.logger)

            if not self.gpus:
                self.logger.warning(f"GPU pool is empty — {len(jobs)} jobs parked until a device is added back to {self.pool.path}")

        gpu_pool = list(self.gpus)

        previous_handlers = self._install_signal_handlers() if self.handle_signals else {}

        try:
            while queue or self.running:
                self._reap(self.running, gpu_pool, self.results)
                self._reconcile(gpu_pool, queue)

                while queue and gpu_pool:
                    job    = queue.pop(0)
                    gpu_id = gpu_pool.pop(0)
                    self.running.append(self._launch(job, gpu_id))

                self._publish_progress(queue, gpu_pool, self.results)

                if queue or self.running:
                    time.sleep(self.poll_interval_s)

            self.gpus = sorted(gpu_pool)
        finally:
            if self.running:
                stopped = self._shutdown_running()
                self.logger.error(f"Scheduler aborted with {len(stopped)} workers still in flight — terminated {', '.join(stopped)} so no unsupervised job keeps holding a GPU{self._resume_suffix()}")

            self._restore_signal_handlers(previous_handlers)

        return self.results
