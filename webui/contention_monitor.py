
from __future__ import annotations

import os
import threading
import time
from collections import deque
from datetime    import datetime

from notifier   import JobNotifier
from politeness import JobPoliteness, JobThrottle
from proc_stats import MemInfo, ProcStats, ProcSweep, UserSlices
from web_logger import WebLogger


class ContentionMonitor:

    INTERVAL            = 5.0
    SECTOR              = 512
    SHARE_FLOOR         = 0.1
    IO_SHARE            = 0.5
    CPU_STALL           = 20.0
    MEM_STALL           = 10.0
    IO_STALL            = 20.0
    CPU_FULL            = 5.0
    MEM_FULL            = 3.0
    IO_FULL             = 5.0
    SWAP_OUT_MBS        = 0.5
    CLEAR_MARGIN        = 0.8
    SUSTAIN             = 2
    NUKE_SUSTAIN        = 6
    NUKE_COOLDOWN       = 120.0
    LEAK_WINDOW_SAMPLES = 48
    LEAK_RECENT_SAMPLES = 12
    LEAK_MEM_FLOOR_GB   = 1.5
    LEAK_GROWTH_MBPM    = 120.0
    LEAK_RECENT_MB      = 60.0

    def __init__(self, paths, logger: WebLogger, notifier: JobNotifier, nuke, throttle: JobThrottle, politeness: JobPoliteness) -> None:
        self.paths      = paths
        self.logger     = logger
        self.notifier   = notifier
        self.nuke       = nuke
        self.throttle   = throttle
        self.politeness = politeness
        self.lock       = threading.Lock()
        self.uid      = os.getuid()
        self.page     = os.sysconf("SC_PAGE_SIZE")
        self.cores    = os.cpu_count() or 1
        self.device   = self._resolve_device()
        self.sources  = {"psi": UserSlices.host_available(), "cgroup": UserSlices.available(), "slice": UserSlices.self_slice_uid() == self.uid, "device": self.device}

        self.signals   = {}
        self.active    = {}
        self.events    = deque(maxlen=20)
        self.streaks   = {}
        self.announced = set()
        self.throttled = set()

        self.auto_nuke   = False
        self.nuke_streak = 0
        self.last_nuke   = 0.0

        self.proc_hist = {}

        self.prev_swap   = None
        self.prev_disk   = None
        self.prev_io     = {}
        self.prev_slices = {}
        self.prev_t      = 0.0

    def start(self) -> None:
        if not self.sources["psi"]:
            self.logger.error(f"contention monitor disabled: {UserSlices.PSI_ROOT} is missing, the kernel has no pressure-stall accounting")
            return
        if not self.sources["cgroup"]:
            self.logger.error(f"contention monitor disabled: {UserSlices.ROOT} is missing, the host has no per-user cgroup slices")
            return
        if not self.sources["slice"]:
            self.logger.warning(f"contention monitor: this console runs outside user-{self.uid}.slice, so the jobs it launches are not charged to your slice")

        worker = threading.Thread(target=self._watch, name="ContentionMonitor", daemon=True)
        worker.start()
        self.logger.muted(f"contention monitor armed (device {self.device or 'n/a'}, share floor {self.SHARE_FLOOR:.0%})")

    def arm(self, value: bool) -> dict:
        with self.lock:
            self.auto_nuke = bool(value)
            if not self.auto_nuke:
                self.nuke_streak = 0
        self.logger.warning(f"auto-nuke {'ARMED' if self.auto_nuke else 'disarmed'} (fires after {self.NUKE_SUSTAIN * self.INTERVAL:.0f}s of severe self-caused stall)")
        return self.state()

    def mine_memory_share(self) -> float | None:
        with self.lock:
            signals = self.signals

        if not signals:
            return None
        return signals["mem"]["mine_share"]

    def state(self) -> dict:
        with self.lock:
            return {
                "auto_nuke"  : self.auto_nuke,
                "politeness" : self.politeness.state(),
                "throttled"  : sorted(self.throttled),
                "sources"    : dict(self.sources),
                "signals"    : dict(self.signals),
                "active"     : list(self.active.values()),
                "events"     : list(self.events),
            }

    def _watch(self) -> None:
        while True:
            time.sleep(self.INTERVAL)
            try:
                signals = self._sample()
                with self.lock:
                    self.signals = signals
                self._evaluate(signals)
            except Exception as exc:
                self.logger.error(f"contention monitor error: {exc}")

    def _sample(self) -> dict:
        now = time.monotonic()
        dt  = max(now - self.prev_t, 1e-6) if self.prev_t else self.INTERVAL

        psi    = UserSlices.host_pressure()
        swap   = self._read_swap(dt)
        mem    = self._read_memory()
        disk   = self._read_disk(dt)
        procs  = self._scan(now, dt)
        users  = self._users(UserSlices.rows(), dt, mem["used_gb"], procs["user_n"])
        others = self._others(users)
        me     = next((user for user in users if user["is_mine"]), {"cpu_pct": 0.0, "cpu_share": 0.0, "mem_gb": 0.0, "mem_share": 0.0, "stall": {"cpu": 0.0, "mem": 0.0, "io": 0.0}})

        self.prev_t = now

        io_share = (procs["io_mbs"] / disk["total_mbs"]) if disk["total_mbs"] > 0.05 else 0.0
        rivals   = [user for user in users if not user["is_mine"]]
        dominant = {
            "cpu" : me["cpu_share"] >= self.SHARE_FLOOR and me["cpu_share"] >= max((user["cpu_share"] for user in rivals), default=0.0),
            "mem" : me["mem_share"] >= self.SHARE_FLOOR and me["mem_share"] >= max((user["mem_share"] for user in rivals), default=0.0),
            "io"  : io_share >= self.IO_SHARE,
        }

        return {
            "psi"    : psi,
            "swap"   : swap,
            "mem"    : {"used_pct": mem["used_pct"], "mine_gb": me["mem_gb"], "mine_share": me["mem_share"], "dominant": dominant["mem"]},
            "io"     : {**disk, "mine_mbs": procs["io_mbs"], "mine_share": min(io_share, 1.0), "dominant": dominant["io"]},
            "cpu"    : {"load_ratio": os.getloadavg()[0] / self.cores, "mine_pct": me["cpu_pct"], "mine_share": me["cpu_share"], "dominant": dominant["cpu"]},
            "mine"   : {"nproc": procs["nproc"], "mem_gb": me["mem_gb"], "stall": me["stall"]},
            "leak"   : procs["leak"],
            "top"    : self._top_consumer(users, procs["biggest"]),
            "users"  : users,
            "others" : others,
        }

    def _read_swap(self, dt: float) -> dict:
        pin, pout = 0, 0
        try:
            for line in open("/proc/vmstat").read().splitlines():
                if line.startswith("pswpin "):
                    pin = int(line.split()[1])
                elif line.startswith("pswpout "):
                    pout = int(line.split()[1])
        except (OSError, ValueError):
            return {"in_mbs": 0.0, "out_mbs": 0.0}

        prev           = self.prev_swap
        self.prev_swap = (pin, pout)
        if prev is None:
            return {"in_mbs": 0.0, "out_mbs": 0.0}

        in_mbs  = max(0, pin - prev[0])  * self.page / dt / (1024.0 ** 2)
        out_mbs = max(0, pout - prev[1]) * self.page / dt / (1024.0 ** 2)
        return {"in_mbs": in_mbs, "out_mbs": out_mbs}

    def _read_memory(self) -> dict:
        info = MemInfo.fields()
        if not info:
            return {"used_pct": 0.0, "used_gb": 0.0}

        total     = info.get("MemTotal", 0)
        available = info.get("MemAvailable", 0)
        used      = max(0, total - available)
        return {"used_pct": (100.0 * used / total) if total else 0.0, "used_gb": used / (1024.0 ** 3)}

    def _read_disk(self, dt: float) -> dict:
        empty = {"device": self.device, "util": 0.0, "await_ms": 0.0, "read_mbs": 0.0, "write_mbs": 0.0, "total_mbs": 0.0}
        if self.device is None:
            return empty

        row = self._diskstat_row()
        if row is None:
            return empty

        reads, sec_read, ms_read, writes, sec_write, ms_write, io_ticks = row
        prev           = self.prev_disk
        self.prev_disk = row
        if prev is None:
            return empty

        d_reads  = max(0, reads     - prev[0])
        d_secr   = max(0, sec_read  - prev[1])
        d_msr    = max(0, ms_read   - prev[2])
        d_writes = max(0, writes    - prev[3])
        d_secw   = max(0, sec_write - prev[4])
        d_msw    = max(0, ms_write  - prev[5])
        d_ticks  = max(0, io_ticks  - prev[6])

        ops       = d_reads + d_writes
        read_mbs  = d_secr * self.SECTOR / dt / (1024.0 ** 2)
        write_mbs = d_secw * self.SECTOR / dt / (1024.0 ** 2)
        return {
            "device"    : self.device,
            "util"      : min(100.0, d_ticks / (dt * 1000.0) * 100.0),
            "await_ms"  : ((d_msr + d_msw) / ops) if ops else 0.0,
            "read_mbs"  : read_mbs,
            "write_mbs" : write_mbs,
            "total_mbs" : read_mbs + write_mbs,
        }

    def _diskstat_row(self):
        try:
            lines = open("/proc/diskstats").read().splitlines()
        except OSError:
            return None

        for line in lines:
            parts = line.split()
            if len(parts) < 14 or parts[2] != self.device:
                continue
            nums = [int(v) for v in parts[3:]]
            return (nums[0], nums[2], nums[3], nums[4], nums[6], nums[7], nums[9])
        return None

    def _scan(self, now: float, dt: float) -> dict:
        nproc    = 0
        per_mem  = {}
        per_comm = {}
        counters = {}
        user_n   = {}
        biggest  = (0, None, "", self.uid)

        for row in ProcSweep.rows():
            pid      = row["pid"]
            uid      = row["uid"]
            stat     = row["stat"]
            proc_mem = row["attributed"] if row["attributed"] is not None else stat["rss"]

            user_n[uid] = user_n.get(uid, 0) + 1
            if proc_mem > biggest[0]:
                biggest = (proc_mem, pid, stat["comm"], uid)

            if uid == self.uid:
                nproc        += 1
                counters[pid] = self._pid_io(pid)
                per_mem [pid] = proc_mem
                per_comm[pid] = stat["comm"]

        prev         = self.prev_io
        self.prev_io = counters

        io_bytes = sum(max(0, counter - prev[pid]) for pid, counter in counters.items() if pid in prev)
        io_mbs   = io_bytes / dt / (1024.0 ** 2)

        return {"nproc": nproc, "io_mbs": io_mbs, "leak": self._growth_candidate(now, per_mem, per_comm), "user_n": user_n, "biggest": biggest}

    def _users(self, slices: list, dt: float, used_gb: float, user_n: dict) -> list[dict]:
        prev             = self.prev_slices
        self.prev_slices = {row["uid"]: row["cpu_usec"] for row in slices}

        deltas = {row["uid"]: max(0, row["cpu_usec"] - prev[row["uid"]]) for row in slices if row["uid"] in prev}
        total  = sum(deltas.values())
        rows   = []

        for row in slices:
            uid   = row["uid"]
            usec  = deltas.get(uid, 0)
            stall = {key: row["stall"][key]["some"] for key, _ in UserSlices.RESOURCES}
            full  = {key: row["stall"][key]["full"] for key, _ in UserSlices.RESOURCES}
            rows.append({
                "user"      : ProcStats.username(uid),
                "uid"       : uid,
                "is_mine"   : uid == self.uid,
                "nproc"     : user_n.get(uid, 0),
                "cpu_pct"   : 100.0 * usec / (dt * 1e6),
                "cpu_share" : (usec / total) if total > 0 else 0.0,
                "mem_gb"    : row["mem"] / (1024.0 ** 3),
                "mem_share" : min(1.0, row["mem"] / (1024.0 ** 3) / used_gb) if used_gb > 0 else 0.0,
                "stall"     : stall,
                "full"      : full,
            })

        rows.sort(key=lambda r: (-max(r["stall"].values()), -r["cpu_pct"]))
        return rows

    def _others(self, users: list[dict]) -> dict:
        limits = {"cpu": self.CPU_STALL, "mem": self.MEM_STALL, "io": self.IO_STALL}
        out    = {}

        for key in limits:
            rivals = [user for user in users if not user["is_mine"]]
            worst  = max(rivals, key=lambda user: user["stall"][key], default=None)
            out[key] = {
                "stall" : worst["stall"][key] if worst else 0.0,
                "full"  : worst["full"][key]  if worst else 0.0,
                "user"  : worst["user"]       if worst else None,
                "n"     : sum(1 for user in rivals if user["stall"][key] >= limits[key]),
            }

        return out

    def _top_consumer(self, users: list[dict], biggest: tuple) -> dict | None:
        if not users:
            return None

        lead                                = max(users, key=lambda user: user["mem_gb"])
        top_mem, top_pid, top_comm, top_uid = biggest

        return {
            "user"        : lead["user"],
            "mem_gb"      : lead["mem_gb"],
            "nproc"       : lead["nproc"],
            "is_mine"     : lead["is_mine"],
            "proc_pid"    : top_pid,
            "proc_user"   : ProcStats.username(top_uid),
            "proc_comm"   : top_comm,
            "proc_mem_gb" : top_mem / (1024.0 ** 3),
        }

    def _growth_candidate(self, now: float, per_mem: dict, per_comm: dict) -> dict | None:
        for pid in [p for p in self.proc_hist if p not in per_mem]:
            del self.proc_hist[pid]
        for pid, proc_mem in per_mem.items():
            self.proc_hist.setdefault(pid, deque(maxlen=self.LEAK_WINDOW_SAMPLES)).append((now, proc_mem))

        best = None
        for pid, hist in self.proc_hist.items():
            if len(hist) < self.LEAK_RECENT_SAMPLES:
                continue

            t_first, r_first = hist[0]
            t_last,  r_last  = hist[-1]
            span             = t_last - t_first
            mem_gb           = r_last / (1024.0 ** 3)
            if span < 1.0 or mem_gb < self.LEAK_MEM_FLOOR_GB:
                continue

            rate_mbpm  = (r_last - r_first) / span * 60.0 / (1024.0 ** 2)
            recent_ref = hist[max(0, len(hist) - 1 - self.LEAK_RECENT_SAMPLES)][1]
            recent_mb  = (r_last - recent_ref) / (1024.0 ** 2)
            if rate_mbpm < self.LEAK_GROWTH_MBPM or recent_mb < self.LEAK_RECENT_MB:
                continue

            if best is None or rate_mbpm > best["rate_mbpm"]:
                best = {"pid": pid, "comm": per_comm.get(pid, "?"), "mem_gb": mem_gb, "rate_mbpm": rate_mbpm, "growth_gb": (r_last - r_first) / (1024.0 ** 3), "window_min": span / 60.0}

        if best is None:
            return None

        best["message"] = f"process {best['comm']} (pid {best['pid']}) is accumulating RAM: {best['mem_gb']:.1f} GB now, +{best['growth_gb']:.1f} GB in {best['window_min']:.0f} min ({best['rate_mbpm']:.0f} MB/min, still climbing)"
        return best

    def _pid_io(self, pid: int) -> int:
        try:
            total = 0
            for line in open(f"/proc/{pid}/io").read().splitlines():
                if line.startswith("read_bytes:") or line.startswith("write_bytes:"):
                    total += int(line.split()[1])
            return total
        except (OSError, ValueError, IndexError):
            return 0

    def _evaluate(self, signals: dict) -> None:
        swap   = signals["swap"]
        mem    = signals["mem"]
        disk   = signals["io"]
        cpu    = signals["cpu"]
        others = signals["others"]
        top    = signals.get("top")

        o_mem      = others["mem"]
        mem_breach = o_mem["stall"] >= self.MEM_STALL or swap["out_mbs"] >= self.SWAP_OUT_MBS
        mem_clear  = o_mem["stall"] < self.MEM_STALL * self.CLEAR_MARGIN and swap["out_mbs"] < self.SWAP_OUT_MBS
        mem_level  = "danger" if (o_mem["full"] >= self.MEM_FULL or swap["out_mbs"] >= self.SWAP_OUT_MBS) else "warn"
        swap_note  = f", machine swapping out {swap['out_mbs']:.1f} MB/s" if swap["out_mbs"] >= self.SWAP_OUT_MBS else ""
        culprit    = f"; dominant consumer is {top['user']} ({top['mem_gb']:.0f} GB across {top['nproc']} procs, biggest pid {top['proc_pid']} {top['proc_comm']} {top['proc_mem_gb']:.0f} GB)" if (top and not mem["dominant"] and not top["is_mine"]) else ""
        self._track("mem", mem_level, mem_breach, mem_clear, mem["dominant"],
                    f"memory: {self._who(o_mem)} stalled on memory {o_mem['stall']:.0f}% of the time{swap_note}; "
                    f"you hold {mem['mine_share']:.0%} of used RAM ({mem['mine_gb']:.1f} GB){culprit}")

        o_io      = others["io"]
        io_breach = o_io["stall"] >= self.IO_STALL
        io_clear  = o_io["stall"] < self.IO_STALL * self.CLEAR_MARGIN
        io_level  = "danger" if o_io["full"] >= self.IO_FULL else "warn"
        self._track("io", io_level, io_breach, io_clear, disk["dominant"],
                    f"disk: {self._who(o_io)} waited on I/O {o_io['stall']:.0f}% of the time; "
                    f"your processes drive {disk['mine_share']:.0%} of {disk['device'] or 'the repo disk'} ({disk['mine_mbs']:.0f} MB/s, {disk['await_ms']:.0f} ms per request)")

        o_cpu      = others["cpu"]
        cpu_breach = o_cpu["stall"] >= self.CPU_STALL
        cpu_clear  = o_cpu["stall"] < self.CPU_STALL * self.CLEAR_MARGIN
        cpu_level  = "danger" if o_cpu["full"] >= self.CPU_FULL else "warn"
        self._track("cpu", cpu_level, cpu_breach, cpu_clear, cpu["dominant"],
                    f"CPU: {self._who(o_cpu)} waited for a core {o_cpu['stall']:.0f}% of the time; "
                    f"you hold {cpu['mine_share']:.0%} of the users' CPU ({cpu['mine_pct'] / 100.0:.0f} cores busy)")

        leak = signals.get("leak")
        self._track("leak", "warn", leak is not None, leak is None, True,
                    leak["message"] if leak else "")

        self._check_nuke()

    def _who(self, other: dict) -> str:
        name  = other["user"] or "no neighbour"
        extra = other["n"] - 1
        return f"{name} and {extra} more" if extra > 0 else name

    def _check_nuke(self) -> None:
        with self.lock:
            if not self.auto_nuke:
                self.nuke_streak = 0
                return

            severe = [a for a in self.active.values() if a["mine"] and a["level"] == "danger" and a["kind"] != "leak"]
            if severe:
                self.nuke_streak += 1
            else:
                self.nuke_streak = 0
            streak  = self.nuke_streak
            reasons = [a["message"] for a in severe]

        if streak < self.NUKE_SUSTAIN:
            return

        now = time.monotonic()
        if now - self.last_nuke < self.NUKE_COOLDOWN:
            return

        self.last_nuke   = now
        self.nuke_streak = 0
        detail  = reasons[0] if reasons else "severe self-caused stall"
        result  = self.nuke.nuke()
        message = f"AUTO-NUKE fired after {self.NUKE_SUSTAIN * self.INTERVAL:.0f}s slowing other users ({detail}); terminated {result.get('signalled', 0)}, force-killed {result.get('killed', 0)}"

        with self.lock:
            self.events.append({"kind": "auto_nuke", "level": "danger", "message": message, "time": datetime.now().isoformat(timespec="seconds")})
        self.logger.error(message)
        self.notifier.system_alert("auto-nuke fired", message, "urgent")

    def _track(self, kind: str, level: str, breached: bool, cleared: bool, mine: bool, message: str) -> None:
        if breached:
            self.streaks[kind] = self.streaks.get(kind, 0) + 1
            if self.streaks[kind] >= self.SUSTAIN:
                self._raise(kind, level, mine, message)
        else:
            self.streaks[kind] = 0
            if cleared:
                self._drop(kind)

    def _raise(self, kind: str, level: str, mine: bool, message: str) -> None:
        with self.lock:
            known = self.active.get(kind)
            if known is None:
                self.active[kind] = {"kind": kind, "level": level, "mine": mine, "message": message, "since": datetime.now().isoformat(timespec="seconds")}
                announce          = mine
            else:
                announce         = mine and (not known["mine"] or (level == "danger" and known["level"] != "danger"))
                known["level"]   = level
                known["mine"]    = mine
                known["message"] = message

            if announce:
                self.events.append({"kind": kind, "level": level, "message": message, "time": datetime.now().isoformat(timespec="seconds")})
                self.announced.add(kind)

        if announce:
            self.logger.warning(f"neighbour impact: {message}")
            self.notifier.system_alert(f"neighbour impact: {kind}", message, "high" if level == "danger" else "default")
            self._throttle_once(kind)

    def _throttle_once(self, kind: str) -> None:
        if kind == "leak" or not self.politeness.auto_throttle():
            return

        with self.lock:
            if kind in self.throttled:
                return
            self.throttled.add(kind)

        result  = self.throttle.apply()
        outcome = f"lowered {result['processes']} processes to nice {JobThrottle.NICE} and idle I/O" if result["ok"] else f"throttle FAILED on {result['processes']} processes: {result['error']}"
        message = f"AUTO-THROTTLE after {kind} alarm: {outcome}"

        with self.lock:
            self.events.append({"kind": "throttle", "level": "warn" if result["ok"] else "danger", "message": message, "time": datetime.now().isoformat(timespec="seconds")})
        self.notifier.system_alert("auto-throttle", message, "default" if result["ok"] else "high")

    def _drop(self, kind: str) -> None:
        with self.lock:
            known     = self.active.pop(kind, None)
            announced = kind in self.announced
            self.announced.discard(kind)
            self.throttled.discard(kind)

        if known is not None and announced:
            self.logger.muted(f"stall cleared: {kind}")

    def _resolve_device(self):
        target = self.paths.repo_root.resolve()
        best   = ("", None)
        try:
            for line in open("/proc/mounts").read().splitlines():
                parts = line.split()
                if len(parts) < 2:
                    continue
                source, mount = parts[0], parts[1]
                if target.is_relative_to(mount) and len(mount) >= len(best[0]) and source.startswith("/dev/"):
                    best = (mount, os.path.realpath(source).rsplit("/", 1)[-1])
        except OSError:
            return None
        return best[1]
