"""Time-of-day GPU pool scheduling for running web UI jobs.

A weekly weekend window and a nightly window split the clock into weekday,
night and weekend phases, each with its own GPU pool. A background thread moves
live jobs onto the pool for the current phase, reclaims GPUs that come free
(greedy mode), and can donate a GPU to another user whose launch bounced off a
full machine (charity mode).
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta

from tools.orchestration.gpu_queue import GpuPoolFile

from gpu_watchdog    import GpuWatchdog
from process_manager import ProcessManager
from project_paths   import ProjectPaths
from system_monitor  import SystemMonitor
from web_logger      import WebLogger


class Window:
    """Base class for a recurring time window on a cyclic minute clock.

    Subclasses define BOUNDS (payload key to inclusive maximum), SAME_MOMENT
    (the error message for a zero-length window) and the three minute
    conversions used to test containment.
    """

    DAY_MINUTES = 24 * 60
    BOUNDS      = {}
    SAME_MOMENT = ""

    @classmethod
    def validate(cls, payload: dict) -> "Window":
        """Builds a window from a payload, rejecting out-of-range or degenerate values.

        Args:
            payload : Mapping holding one integer per key in ``BOUNDS``.

        Returns:
            The validated window instance.

        Raises:
            ValueError : If a bound is missing, is not a whole number in range, or the start and end fall on the same minute.
        """
        values = {}

        for key, maximum in cls.BOUNDS.items():
            value = payload[key]
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
                raise ValueError(f"'{key}' must be a whole number in [0, {maximum}], got {value!r}")
            values[key] = value

        window = cls(**values)
        if window.start_minute() == window.end_minute():
            raise ValueError(cls.SAME_MOMENT)

        return window

    def contains(self, moment: datetime) -> bool:
        """Returns whether the moment falls inside the window, wrapping past midnight."""
        now   = self.minute_of(moment)
        start = self.start_minute()
        end   = self.end_minute()

        if start < end:
            return start <= now < end

        return now >= start or now < end


class WeekWindow(Window):
    """Weekly window spanning from a weekday and hour to another weekday and hour.

    Attributes:
        start_day  : Weekday index of the window start, Monday is 0.
        start_hour : Hour of day the window starts.
        end_day    : Weekday index of the window end, Monday is 0.
        end_hour   : Hour of day the window ends.
    """

    DAYS        = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
    BOUNDS      = {"start_day": 6, "start_hour": 23, "end_day": 6, "end_hour": 23}
    SAME_MOMENT = "the weekend window starts and ends at the same moment, so it would never switch"

    def __init__(self, start_day: int, start_hour: int, end_day: int, end_hour: int) -> None:
        """Stores the weekday and hour bounds of the weekend window.

        Args:
            start_day  : Weekday index of the start, Monday is 0.
            start_hour : Hour of day the window opens.
            end_day    : Weekday index of the end, Monday is 0.
            end_hour   : Hour of day the window closes.
        """
        self.start_day  = start_day
        self.start_hour = start_hour
        self.end_day    = end_day
        self.end_hour   = end_hour

    def minute_of(self, moment: datetime) -> int:
        """Returns the minute of the week for the moment."""
        return moment.weekday() * self.DAY_MINUTES + moment.hour * 60 + moment.minute

    def start_minute(self) -> int:
        """Returns the window start as a minute of the week."""
        return self.start_day * self.DAY_MINUTES + self.start_hour * 60

    def end_minute(self) -> int:
        """Returns the window end as a minute of the week."""
        return self.end_day * self.DAY_MINUTES + self.end_hour * 60

    def label(self) -> str:
        """Returns a human-readable description of the weekly window."""
        return f"{self.DAYS[self.start_day]} {self.start_hour:02d}:00 to {self.DAYS[self.end_day]} {self.end_hour:02d}:00"

    def as_dict(self) -> dict:
        """Returns the window bounds as the schedule payload keys."""
        return {"start_day": self.start_day, "start_hour": self.start_hour, "end_day": self.end_day, "end_hour": self.end_hour}


class NightWindow(Window):
    """Daily window spanning from one hour of the evening to one hour of the morning.

    Attributes:
        night_start_hour : Hour of day the night window opens.
        night_end_hour   : Hour of day the night window closes.
    """

    BOUNDS      = {"night_start_hour": 23, "night_end_hour": 23}
    SAME_MOMENT = "the night window starts and ends at the same hour, so it would never switch"

    def __init__(self, night_start_hour: int, night_end_hour: int) -> None:
        """Stores the hour bounds of the nightly window.

        Args:
            night_start_hour : Hour of day the window opens.
            night_end_hour   : Hour of day the window closes.
        """
        self.night_start_hour = night_start_hour
        self.night_end_hour   = night_end_hour

    def minute_of(self, moment: datetime) -> int:
        """Returns the minute of the day for the moment."""
        return moment.hour * 60 + moment.minute

    def start_minute(self) -> int:
        """Returns the window start as a minute of the day."""
        return self.night_start_hour * 60

    def end_minute(self) -> int:
        """Returns the window end as a minute of the day."""
        return self.night_end_hour * 60

    def label(self) -> str:
        """Returns a human-readable description of the nightly window."""
        return f"{self.night_start_hour:02d}:00 to {self.night_end_hour:02d}:00 every day"

    def as_dict(self) -> dict:
        """Returns the window bounds as the schedule payload keys."""
        return {"night_start_hour": self.night_start_hour, "night_end_hour": self.night_end_hour}


class GpuAvailability:
    """Reads live GPU occupancy and reports which devices may be used."""

    def __init__(self, system: SystemMonitor) -> None:
        """Binds the availability view to a system monitor.

        Args:
            system : Monitor supplying GPU occupancy and the owning user name.
        """
        self.system = system

    def busy(self) -> set[int] | None:
        """Returns the indices of GPUs held by another user, or None if the probe failed."""
        occupancy = self.system.gpu_occupancy()
        if occupancy is None:
            return None

        held = set()

        for device in occupancy:
            owners = {proc["owner"] for proc in device["procs"] if proc["owner"]}
            if owners - {self.system.user}:
                held.add(device["index"])

        return held

    def idle(self) -> set[int] | None:
        """Returns the indices of GPUs running no process, or None if the probe failed."""
        occupancy = self.system.gpu_occupancy()
        if occupancy is None:
            return None

        return {device["index"] for device in occupancy if not device["procs"]}

    def grant(self, target: list[int]) -> list[int] | None:
        """Returns the target GPUs that nobody else holds.

        Args:
            target : Candidate GPU indices.

        Returns:
            The subset of ``target`` free of other users' processes, or None
            when GPU occupancy could not be probed.
        """
        busy = self.busy()
        if busy is None:
            return None

        return [gpu for gpu in target if gpu not in busy]


class GpuSchedule:
    """Moves live jobs between phase GPU pools and shares GPUs under contention.

    Attributes:
        paths        : Project paths supplying the log directory holding the schedule file.
        logger       : Web UI logger for schedule announcements.
        processes    : Process manager owning the live jobs and their GPU pools.
        guard        : Watchdog reporting other users' bounced launch attempts.
        availability : Live GPU occupancy view.
        settings     : Validated schedule settings, seeded from DEFAULTS.
        applied      : Per job, the phase last applied and the GPUs withheld from it.
        pending      : Per job, the phase whose move is still waiting.
        gifted       : GPU indices donated under charity mode.
    """

    INTERVAL       = 30.0
    SWEEP_INTERVAL = 600.0
    CHARITY_NAP    = timedelta(hours=4)
    BOUNCE_TTL     = timedelta(minutes=10)
    FILE_NAME      = "gpu_schedule.json"
    GPU_KEYS       = ("weekday_gpus", "night_gpus", "weekend_gpus")
    FLAG_KEYS      = ("enabled", "greedy", "charity")

    DEFAULTS = {
        "enabled"          : False,
        "greedy"           : True,
        "charity"          : False,
        "weekday_gpus"     : [2, 3],
        "night_gpus"       : [0, 1, 2, 3],
        "weekend_gpus"     : [0, 1, 2, 3],
        "start_day"        : 4,
        "start_hour"       : 18,
        "end_day"          : 0,
        "end_hour"         : 8,
        "night_start_hour" : 20,
        "night_end_hour"   : 8,
    }

    def __init__(self, paths: ProjectPaths, logger: WebLogger, processes: ProcessManager, system: SystemMonitor, guard: GpuWatchdog) -> None:
        """Loads the persisted schedule and prepares the scheduling state.

        Args:
            paths     : Project paths supplying the log directory.
            logger    : Web UI logger.
            processes : Process manager owning live jobs and their GPU pools.
            system    : System monitor used to probe GPU occupancy.
            guard     : Watchdog reporting bounced launch attempts by other users.
        """
        self.paths        = paths
        self.logger       = logger
        self.processes    = processes
        self.guard        = guard
        self.availability = GpuAvailability(system)
        self.lock         = threading.Lock()
        self.path         = self.paths.logs_dir / self.FILE_NAME
        self.applied      = {}
        self.pending      = {}
        self.settings     = self.seed(paths.profile)
        self.last_tick    = None
        self.last_sweep   = None
        self.swept_at     = None
        self.nap_until    = None
        self.gifted       = set()
        self.last_charity = None

        self._load()

    @classmethod
    def seed(cls, profile) -> dict:
        """Returns the built-in defaults with every GPU pool set to the machine profile's pool when it names one.

        Args:
            profile : Machine profile of this installation.

        Returns:
            Settings mapping used until a schedule file exists.
        """
        settings = dict(cls.DEFAULTS)
        pool     = profile.gpu_pool

        if pool:
            for key in cls.GPU_KEYS:
                settings[key] = list(pool)

        return settings

    def _load(self) -> None:
        """Reads the schedule file, keeping the seeded defaults when it is missing or invalid."""
        if not self.path.is_file():
            return

        try:
            self.settings = self.validate(json.loads(self.path.read_text(encoding="utf-8")))
        except (ValueError, TypeError, KeyError, json.JSONDecodeError, OSError) as error:
            self.logger.error(f"ignoring unreadable GPU schedule {self.path}: {error}; the built-in defaults apply until it is fixed")

    @classmethod
    def validate(cls, payload) -> dict:
        """Returns the normalised schedule settings for a payload.

        Args:
            payload : Object holding the flags, window bounds and per-phase GPU lists.

        Returns:
            Settings dict with the flags, both windows expanded and each phase
            GPU list normalised.

        Raises:
            ValueError : If the payload is not an object, a key is missing, a flag is not boolean, a window is degenerate, or a GPU pool is empty.
        """
        if not isinstance(payload, dict):
            raise ValueError(f"expected an object holding the schedule, got {payload!r}")

        missing = [key for key in cls.DEFAULTS if key not in payload]
        if missing:
            raise ValueError(f"the schedule must define {missing}")

        for key in cls.FLAG_KEYS:
            if not isinstance(payload[key], bool):
                raise ValueError(f"'{key}' must be true or false, got {payload[key]!r}")

        settings = {**{key: payload[key] for key in cls.FLAG_KEYS}, **WeekWindow.validate(payload).as_dict(), **NightWindow.validate(payload).as_dict()}

        for key in cls.GPU_KEYS:
            gpus = GpuPoolFile.validate({"gpus": payload[key]})
            if not gpus:
                raise ValueError(f"'{key}' must hold at least one GPU; park an experiment from its console tile instead")
            settings[key] = gpus

        return settings

    def week_window(self) -> WeekWindow:
        """Returns the configured weekend window."""
        with self.lock:
            return WeekWindow(self.settings["start_day"], self.settings["start_hour"], self.settings["end_day"], self.settings["end_hour"])

    def night_window(self) -> NightWindow:
        """Returns the configured nightly window."""
        with self.lock:
            return NightWindow(self.settings["night_start_hour"], self.settings["night_end_hour"])

    def phase(self, moment: datetime) -> str:
        """Returns the active phase at the moment: weekend, night or weekday."""
        if self.week_window().contains(moment):
            return "weekend"

        if self.night_window().contains(moment):
            return "night"

        return "weekday"

    def gpus_for(self, phase: str) -> list[int]:
        """Returns the configured GPU pool for a phase."""
        with self.lock:
            return list(self.settings[f"{phase}_gpus"])

    def _live_jobs(self) -> list[str]:
        """Returns the ids of running jobs whose GPU pool file is live."""
        live = []

        for record in self.processes.list_jobs():
            job_id = record["job_id"]
            if record["status"] == "running" and self.processes.gpu_pool(job_id).get("live"):
                live.append(job_id)

        return live

    def _gifts(self) -> set[int]:
        """Returns the GPU indices currently donated to other users."""
        with self.lock:
            return set(self.gifted)

    def _enter(self, job_id: str, phase: str, enabled: bool) -> bool:
        """Records the job in the phase and returns whether a phase crossing is due.

        Args:
            job_id  : Identifier of the live job.
            phase   : Phase active at this tick.
            enabled : Whether the schedule is enabled.

        Returns:
            True when the schedule is enabled and the job was last placed in a
            different phase, so its pool must be crossed over.
        """
        with self.lock:
            record = self.applied.get(job_id)
            if record is None:
                self.applied[job_id] = {"phase": phase, "withheld": []}
                return False

            if not enabled:
                record["phase"] = phase
                return False

            return record["phase"] != phase

    def _cross(self, job_id: str, phase: str) -> list[str]:
        """Moves a job onto the phase pool, skipping GPUs that are busy or gifted.

        Args:
            job_id : Identifier of the live job.
            phase  : Phase whose pool the job should run on.

        Returns:
            A single-element list holding the job id when the pool was written,
            empty when the move stays pending.
        """
        target = self.gpus_for(phase)
        gifts  = self._gifts()
        free   = self.availability.grant(target)

        if free is None:
            with self.lock:
                unknown              = self.pending.get(job_id) == phase
                self.pending[job_id] = phase

            if not unknown:
                self.logger.warning(f"gpu schedule left job {job_id} where it is: nvidia-smi is not answering, so GPU occupancy is unknown and the move to the {phase} pool {target} waits for a working probe")
            return []

        granted  = [gpu for gpu in free if gpu not in gifts]
        withheld = [gpu for gpu in target if gpu not in granted]

        with self.lock:
            self.applied[job_id]["withheld"] = withheld
            announced                        = self.pending.get(job_id) == phase
            self.pending[job_id]             = phase

        if not granted:
            if not announced:
                self.logger.warning(f"gpu schedule left job {job_id} where it is: every GPU in the {phase} pool {target} is busy with someone else's work, so the move stays pending until one frees")
            return []

        result = self.processes.set_gpus(job_id, granted)
        if not result.get("ok"):
            if not announced:
                self.logger.warning(f"gpu schedule could not move job {job_id} onto the {phase} pool {granted}: {result.get('error', 'the pool write failed')}, so the move stays pending")
            return []

        with self.lock:
            self.applied[job_id]["phase"] = phase
            self.pending.pop(job_id, None)

        if withheld:
            self.logger.ok(f"gpu schedule moved job {job_id} onto the {phase} pool {granted}; {withheld} held by someone else")
            return [job_id]

        self.logger.ok(f"gpu schedule moved job {job_id} onto the {phase} pool {granted}")
        return [job_id]

    def tick(self, moment: datetime) -> list[str]:
        """Crosses every live job onto the pool of the current phase.

        Args:
            moment : Wall-clock instant the tick represents.

        Returns:
            The ids of jobs whose GPU pool was rewritten.
        """
        with self.lock:
            enabled        = self.settings["enabled"]
            self.last_tick = moment.isoformat(timespec="seconds")

        phase   = self.phase(moment)
        applied = []
        live    = self._live_jobs()

        for job_id in live:
            if self._enter(job_id, phase, enabled):
                applied += self._cross(job_id, phase)

        with self.lock:
            self.applied = {job_id: record for job_id, record in self.applied.items() if job_id in live}
            self.pending = {job_id: pending for job_id, pending in self.pending.items() if job_id in live}

        return applied

    def _wake(self, moment: datetime) -> None:
        """Ends the charity nap and releases the gifted GPUs once the nap expires."""
        with self.lock:
            if self.nap_until is None or moment < self.nap_until:
                return
            gifts          = sorted(self.gifted)
            self.nap_until = None
            self.gifted.clear()

        self.logger.ok(f"charity nap over: greedy is awake again and will pick {gifts} back up once they free")

    def _pools(self) -> dict[str, list[int]]:
        """Returns the current GPU pool of each live job that holds at least one GPU."""
        pools = {}

        for job_id in self._live_jobs():
            gpus = self.processes.gpu_pool(job_id).get("gpus", [])
            if gpus:
                pools[job_id] = gpus

        return pools

    def _all_taken(self, pools: dict[str, list[int]]) -> bool:
        """Returns whether every process-free GPU is already claimed by one of the pools."""
        idle = self.availability.idle()
        if idle is None:
            return False

        pooled = {gpu for gpus in pools.values() for gpu in gpus}
        return all(index in pooled for index in idle)

    def _releasable(self, pools: dict[str, list[int]]) -> dict[int, list[str]]:
        """Returns each GPU whose every holder would keep at least one other GPU, mapped to its holders."""
        holders = {}

        for job_id, gpus in pools.items():
            for gpu in gpus:
                holders.setdefault(gpu, []).append(job_id)

        return {gpu: jobs for gpu, jobs in holders.items() if all(len(pools[job_id]) > 1 for job_id in jobs)}

    def _pick(self, releasable: dict[int, list[str]], pools: dict[str, list[int]], wanted: int) -> int | None:
        """Chooses the GPU to donate, preferring the one the bounced attempt wanted.

        Args:
            releasable : GPU index to the job ids holding it.
            pools      : Per job GPU pools.
            wanted     : GPU index the bounced attempt targeted.

        Returns:
            The GPU index to hand over, or None when nothing can be released.
        """
        if wanted in releasable:
            return wanted

        if not releasable:
            return None

        return max(releasable, key=lambda gpu: (min(len(pools[job_id]) for job_id in releasable[gpu]), gpu))

    def _donate(self, gpu: int, jobs: list[str], attempt: dict, moment: datetime) -> list[str]:
        """Drops a GPU from every holding job and puts greedy reclaim to sleep.

        Args:
            gpu     : GPU index to hand over.
            jobs    : Job ids currently holding that GPU.
            attempt : Bounced launch record of the other user.
            moment  : Wall-clock instant of the donation.

        Returns:
            The ids of the jobs that gave the GPU up, empty if none could.
        """
        donors = []

        for job_id in jobs:
            pool   = self.processes.gpu_pool(job_id).get("gpus", [])
            result = self.processes.set_gpus(job_id, [g for g in pool if g != gpu])
            if not result.get("ok"):
                continue

            donors.append(job_id)
            with self.lock:
                record = self.applied.get(job_id)
                if record is not None and gpu not in record["withheld"]:
                    record["withheld"].append(gpu)

        if not donors:
            return []

        until = moment + self.CHARITY_NAP

        with self.lock:
            self.nap_until    = until
            self.gifted.add(gpu)
            self.last_charity = {"gpu": gpu, "users": [attempt["user"]], "jobs": donors, "at": moment.isoformat(timespec="seconds"), "until": until.isoformat(timespec="seconds")}

        self.logger.ok(f"charity opened gpu {gpu} for {attempt['user']}, whose attempt on gpu {attempt['gpu_index']} bounced off a full machine: job(s) {donors} leave it once the unit in flight finishes, greedy sleeps until {until.strftime('%A %H:%M')}")
        return donors

    def _fresh_bounces(self, moment: datetime) -> list[dict]:
        """Returns the bounced launch attempts recorded within the bounce time-to-live."""
        cutoff = moment - self.BOUNCE_TTL

        return [attempt for attempt in self.guard.take_bounced() if datetime.fromisoformat(attempt["bounced_at"]) >= cutoff]

    def charity(self, moment: datetime) -> list[str]:
        """Hands one GPU to another user whose launch bounced off a full machine.

        Args:
            moment : Wall-clock instant of the check.

        Returns:
            The ids of the jobs that gave a GPU up, empty when charity is off,
            napping, or nothing needs donating.
        """
        self._wake(moment)

        with self.lock:
            active  = self.settings["enabled"] and self.settings["greedy"] and self.settings["charity"]
            napping = self.nap_until is not None

        if not active or napping:
            return []

        bounced = self._fresh_bounces(moment)
        if not bounced:
            return []

        pools = self._pools()
        if not self._all_taken(pools):
            return []

        attempt    = bounced[-1]
        releasable = self._releasable(pools)
        gpu        = self._pick(releasable, pools, attempt["gpu_index"])
        if gpu is None:
            return []

        return self._donate(gpu, releasable[gpu], attempt, moment)

    def _claim(self, job_id: str, current: list[int], freed: list[int]) -> list[int]:
        """Grows a job pool with GPUs that came free and returns the resulting pool."""
        merged = sorted(set(current) | set(freed))
        if merged == sorted(current):
            return merged

        result = self.processes.set_gpus(job_id, merged)
        if not result.get("ok"):
            return sorted(current)

        self.logger.ok(f"gpu schedule grew job {job_id} onto {merged}: {freed} came free")
        return merged

    def sweep(self, moment: datetime) -> list[str]:
        """Reclaims previously withheld GPUs for jobs once their owners release them.

        Args:
            moment : Wall-clock instant of the sweep.

        Returns:
            The ids of jobs whose pool grew.
        """
        with self.lock:
            greedy          = self.settings["enabled"] and self.settings["greedy"]
            napping         = self.nap_until is not None and moment < self.nap_until
            self.last_sweep = moment.isoformat(timespec="seconds")

        if not greedy or napping:
            return []

        claimed = []

        for job_id in self._live_jobs():
            with self.lock:
                record   = self.applied.get(job_id)
                withheld = list(record["withheld"]) if record is not None else []

            if not withheld:
                continue

            current = self.processes.gpu_pool(job_id).get("gpus", [])
            if not current:
                continue

            freed = self.availability.grant(withheld)
            if not freed:
                continue

            pool = self._claim(job_id, current, freed)
            if pool != sorted(current):
                claimed.append(job_id)

            with self.lock:
                record["withheld"] = [gpu for gpu in record["withheld"] if gpu not in pool]

        return claimed

    def _due(self, now: float) -> bool:
        """Returns whether a sweep is due, stamping the sweep clock when it is."""
        if self.swept_at is None or now - self.swept_at >= self.SWEEP_INTERVAL:
            self.swept_at = now
            return True

        return False

    def _watch(self) -> None:
        """Runs the tick, charity and periodic sweep loop until the process exits."""
        while True:
            try:
                moment = datetime.now()
                self.tick(moment)
                self.charity(moment)

                if self._due(time.monotonic()):
                    self.sweep(moment)
            except Exception as error:
                self.logger.error(f"gpu schedule tick failed: {error}")

            time.sleep(self.INTERVAL)

    def start(self) -> None:
        """Starts the scheduling loop on a daemon thread."""
        worker = threading.Thread(target=self._watch, daemon=True)
        worker.start()
        self.logger.muted(f"gpu schedule armed from {self.path} (enabled={self.settings['enabled']}, greedy={self.settings['greedy']}, charity={self.settings['charity']}, every {self.INTERVAL:.0f}s)")

    def state(self) -> dict:
        """Returns the schedule settings plus the current phase, pools and charity state."""
        with self.lock:
            settings = dict(self.settings)
            last     = self.last_tick
            swept    = self.last_sweep
            nap      = self.nap_until.isoformat(timespec="seconds") if self.nap_until else None
            gifts    = sorted(self.gifted)
            given    = dict(self.last_charity) if self.last_charity else None
            waiting  = sorted({gpu for record in self.applied.values() for gpu in record["withheld"]})

        phase = self.phase(datetime.now())

        return {
            **settings,
            "phase"        : phase,
            "gpus_now"     : list(settings[f"{phase}_gpus"]),
            "waiting"      : waiting,
            "napping"      : nap is not None,
            "nap_until"    : nap,
            "gifted"       : gifts,
            "last_charity" : given,
            "window"       : self.week_window().label(),
            "night_window" : self.night_window().label(),
            "last_tick"    : last,
            "last_sweep"   : swept,
            "path"         : str(self.path),
        }

    def update(self, payload) -> dict:
        """Validates, applies and persists a new schedule.

        Args:
            payload : Object holding the flags, window bounds and per-phase GPU lists.

        Returns:
            ``{"ok": False, "error": ...}`` when the payload is rejected,
            otherwise ``{"ok": True}`` merged with the new state.
        """
        try:
            settings = self.validate(payload)
        except (ValueError, TypeError) as error:
            return {"ok": False, "error": str(error)}

        with self.lock:
            self.settings = settings
            if not (settings["enabled"] and settings["greedy"] and settings["charity"]):
                self.nap_until = None
                self.gifted.clear()

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")

        self.logger.ok(f"gpu schedule saved: {'on' if settings['enabled'] else 'off'}, greedy {'on' if settings['greedy'] else 'off'}, charity {'on' if settings['charity'] else 'off'}, weekday {settings['weekday_gpus']}, night {settings['night_gpus']}, weekend {settings['weekend_gpus']}")

        return {"ok": True, **self.state()}
