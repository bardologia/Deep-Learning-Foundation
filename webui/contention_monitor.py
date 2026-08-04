from __future__ import annotations

import os
import threading
import time
from collections import deque
from datetime    import datetime

from proc_stats import MemInfo, ProcStats
from web_logger import WebLogger


class ContentionMonitor:

    INTERVAL            = 5.0
    SECTOR              = 512
    SHARE               = 0.5
    MEM_PSI_SOME        = 10.0
    MEM_PSI_FULL        = 3.0
    IO_PSI_SOME         = 25.0
    IO_PSI_FULL         = 3.0
    IO_UTIL             = 80.0
    CPU_PSI_SOME        = 40.0
    LOAD_RATIO          = 1.0
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

    def __init__(self, paths, logger: WebLogger, nuke) -> None:
        self.paths  = paths
        self.logger = logger
        self.nuke   = nuke
        self.lock   = threading.Lock()
        self.uid    = os.getuid()
        self.clk    = os.sysconf("SC_CLK_TCK")
        self.page   = os.sysconf("SC_PAGE_SIZE")
        self.cores  = os.cpu_count() or 1
        self.device = self._resolve_device()

        self.signals   = {}
        self.active    = {}
        self.events    = deque(maxlen=20)
        self.streaks   = {}
        self.announced = set()

        self.auto_nuke   = False
        self.nuke_streak = 0
        self.last_nuke   = 0.0

        self.proc_hist = {}

        self.prev_swap = None
        self.prev_disk = None
        self.prev_mine = {}
        self.prev_t    = 0.0

    def start(self) -> None:
        worker = threading.Thread(target=self._watch, name="ContentionMonitor", daemon=True)
        worker.start()
        self.logger.muted(f"contention monitor armed (device {self.device or 'n/a'}, share {self.SHARE:.0%})")

    def arm(self, value: bool) -> dict:
        with self.lock:
            self.auto_nuke = bool(value)
            if not self.auto_nuke:
                self.nuke_streak = 0
        self.logger.warning(f"auto-nuke {'ARMED' if self.auto_nuke else 'disarmed'} (fires after {self.NUKE_SUSTAIN * self.INTERVAL:.0f}s of severe self-caused contention)")
        return self.state()

    def state(self) -> dict:
        with self.lock:
            return {
                "auto_nuke" : self.auto_nuke,
                "signals"   : dict(self.signals),
                "active"    : list(self.active.values()),
                "events"    : list(self.events),
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

        psi  = self._read_psi()
        swap = self._read_swap(dt)
        mem  = self._read_memory()
        disk = self._read_disk(dt)
        mine = self._scan(now, dt)

        self.prev_t = now

        mem_used_gb  = mem["used_gb"]
        mem_share    = (mine["mem_gb"] / mem_used_gb) if mem_used_gb > 0 else 0.0
        cpu_capacity = self.cores * 100.0
        cpu_share    = (mine["cpu_pct"] / cpu_capacity) if cpu_capacity > 0 else 0.0
        io_share     = (mine["io_mbs"] / disk["total_mbs"]) if disk["total_mbs"] > 0.05 else 0.0

        return {
            "psi"  : psi,
            "swap" : swap,
            "mem"  : {"used_pct": mem["used_pct"], "mine_gb": mine["mem_gb"], "mine_share": min(mem_share, 1.0)},
            "io"   : {**disk, "mine_mbs": mine["io_mbs"], "mine_share": min(io_share, 1.0)},
            "cpu"  : {"load_ratio": os.getloadavg()[0] / self.cores, "mine_pct": mine["cpu_pct"], "mine_share": min(cpu_share, 1.0)},
            "mine" : {"nproc": mine["nproc"], "mem_gb": mine["mem_gb"]},
            "leak" : mine["leak"],
            "top"  : mine["top"],
        }

    def _read_psi(self) -> dict:
        out = {}
        for resource in ("cpu", "memory", "io"):
            some, full = 0.0, 0.0
            try:
                for line in open(f"/proc/pressure/{resource}").read().splitlines():
                    fields = dict(token.split("=") for token in line.split()[1:] if "=" in token)
                    if line.startswith("some"):
                        some = float(fields.get("avg10", 0.0))
                    elif line.startswith("full"):
                        full = float(fields.get("avg10", 0.0))
            except (OSError, ValueError):
                pass
            out["mem" if resource == "memory" else resource] = {"some": some, "full": full}
        return out

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
        mine_mem = 0
        nproc    = 0
        per_mem  = {}
        per_comm = {}
        counters = {}
        user_mem = {}
        user_n   = {}
        top      = (0, None, "", self.uid)

        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            try:
                uid = os.stat(f"/proc/{pid}").st_uid
            except OSError:
                continue

            stat = ProcStats.stat(pid)
            if stat is None:
                continue

            attributed = ProcStats.attributed(pid)
            proc_mem   = attributed if attributed is not None else stat["rss"]

            user_mem[uid] = user_mem.get(uid, 0) + proc_mem
            user_n  [uid] = user_n.get(uid, 0) + 1
            if proc_mem > top[0]:
                top = (proc_mem, pid, stat["comm"], uid)

            if uid == self.uid:
                mine_mem     += proc_mem
                nproc        += 1
                counters[pid] = (stat["jiffies"], self._pid_io(pid))
                per_mem [pid] = proc_mem
                per_comm[pid] = stat["comm"]

        prev           = self.prev_mine
        self.prev_mine = counters

        jiffies  = sum(max(0, counter[0] - prev[pid][0]) for pid, counter in counters.items() if pid in prev)
        io_bytes = sum(max(0, counter[1] - prev[pid][1]) for pid, counter in counters.items() if pid in prev)
        cpu_pct  = 100.0 * jiffies / self.clk / dt
        io_mbs   = io_bytes / dt / (1024.0 ** 2)

        leak = self._growth_candidate(now, per_mem, per_comm)

        return {"nproc": nproc, "mem_gb": mine_mem / (1024.0 ** 3), "cpu_pct": cpu_pct, "io_mbs": io_mbs, "leak": leak, "top": self._top_consumer(user_mem, user_n, top)}

    def _top_consumer(self, user_mem: dict, user_n: dict, top: tuple) -> dict | None:
        if not user_mem:
            return None

        lead_uid             = max(user_mem, key=user_mem.get)
        top_mem, top_pid, top_comm, top_uid = top

        return {
            "user"        : ProcStats.username(lead_uid),
            "mem_gb"      : user_mem[lead_uid] / (1024.0 ** 3),
            "nproc"       : user_n[lead_uid],
            "is_mine"     : lead_uid == self.uid,
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
        psi  = signals["psi"]
        swap = signals["swap"]
        mem  = signals["mem"]
        disk = signals["io"]
        cpu  = signals["cpu"]

        top        = signals.get("top")
        mem_breach = psi["mem"]["some"] >= self.MEM_PSI_SOME or swap["out_mbs"] >= self.SWAP_OUT_MBS
        mem_clear  = psi["mem"]["some"] < self.MEM_PSI_SOME * self.CLEAR_MARGIN and swap["out_mbs"] < self.SWAP_OUT_MBS
        mem_mine   = mem["mine_share"] >= self.SHARE
        mem_level  = "danger" if (psi["mem"]["full"] >= self.MEM_PSI_FULL or swap["out_mbs"] >= self.SWAP_OUT_MBS) else "warn"
        swap_note  = f", swapping out {swap['out_mbs']:.1f} MB/s" if swap["out_mbs"] >= self.SWAP_OUT_MBS else ""
        culprit    = f"; dominant consumer is {top['user']} ({top['mem_gb']:.0f} GB across {top['nproc']} procs, biggest pid {top['proc_pid']} {top['proc_comm']} {top['proc_mem_gb']:.0f} GB)" if (top and not mem_mine and not top["is_mine"]) else ""
        self._track("mem", mem_level, mem_breach, mem_clear, mem_mine,
                    f"memory contention: tasks stalled {psi['mem']['some']:.0f}% of the time{swap_note}; "
                    f"you hold {mem['mine_share']:.0%} of used RAM ({mem['mine_gb']:.1f} GB){culprit}")

        io_breach = psi["io"]["some"] >= self.IO_PSI_SOME or disk["util"] >= self.IO_UTIL
        io_clear  = psi["io"]["some"] < self.IO_PSI_SOME * self.CLEAR_MARGIN and disk["util"] < self.IO_UTIL * self.CLEAR_MARGIN
        io_mine   = disk["mine_share"] >= self.SHARE
        io_level  = "danger" if psi["io"]["full"] >= self.IO_PSI_FULL else "warn"
        self._track("io", io_level, io_breach, io_clear, io_mine,
                    f"disk contention on {disk['device']}: {disk['util']:.0f}% busy, tasks stalled {psi['io']['some']:.0f}%; "
                    f"you drive {disk['mine_share']:.0%} of the I/O ({disk['mine_mbs']:.0f} MB/s)")

        cpu_breach = psi["cpu"]["some"] >= self.CPU_PSI_SOME or cpu["load_ratio"] >= self.LOAD_RATIO
        cpu_clear  = psi["cpu"]["some"] < self.CPU_PSI_SOME * self.CLEAR_MARGIN and cpu["load_ratio"] < self.LOAD_RATIO * self.CLEAR_MARGIN
        cpu_mine   = cpu["mine_share"] >= self.SHARE
        self._track("cpu", "warn", cpu_breach, cpu_clear, cpu_mine,
                    f"CPU saturated: load/core {cpu['load_ratio']:.2f}, tasks stalled {psi['cpu']['some']:.0f}%; "
                    f"you use {cpu['mine_share']:.0%} of all cores ({cpu['mine_pct']:.0f}% busy)")

        leak = signals.get("leak")
        self._track("leak", "warn", leak is not None, leak is None, True,
                    leak["message"] if leak else "")

        self._check_nuke()

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
        detail  = reasons[0] if reasons else "severe self-caused contention"
        result  = self.nuke.nuke()
        message = f"AUTO-NUKE fired after {self.NUKE_SUSTAIN * self.INTERVAL:.0f}s slowing other users ({detail}); terminated {result.get('signalled', 0)}, force-killed {result.get('killed', 0)}"

        with self.lock:
            self.events.append({"kind": "auto_nuke", "level": "danger", "message": message, "time": datetime.now().isoformat(timespec="seconds")})
        self.logger.error(message)

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

    def _drop(self, kind: str) -> None:
        with self.lock:
            known     = self.active.pop(kind, None)
            announced = kind in self.announced
            self.announced.discard(kind)

        if known is not None and announced:
            self.logger.muted(f"contention cleared: {kind}")

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
