from __future__ import annotations

import io
import os

import pytest

import proc_stats
from proc_stats import CpuCounters, MemInfo, ProcStats, PssCache


MEMINFO = """MemTotal:       32000000 kB
MemFree:         1000000 kB
MemAvailable:    8000000 kB
Buffers:          200000 kB
"""

PROC_STAT = """cpu  100 20 30 400 50 0 10 5
cpu0 50 10 15 200 25 0 5 2
cpu1 50 10 15 200 25 0 5 3
intr 12345 1 2 3
ctxt 999
"""

PID_STAT = "4242 (python (worker) 3.11) S 4000 4242 4242 0 -1 4194304 100 0 0 0 700 300 0 0 20 0 9 0 55555 123456789 512 " + " ".join(["0"] * 30)


class FakeProc:

    def __init__(self, files: dict) -> None:
        self.files = files
        self.reads = []

    def __call__(self, path, *args, **kwargs):
        self.reads.append(str(path))
        content = self.files.get(str(path))
        if content is None:
            raise OSError(f"no such file: {path}")
        return io.StringIO(content)


@pytest.fixture
def proc(monkeypatch):
    fake = FakeProc({
        "/proc/meminfo"           : MEMINFO,
        "/proc/stat"              : PROC_STAT,
        "/proc/4242/stat"         : PID_STAT,
        "/proc/4242/statm"        : "2000 1500 400 10 0 300 0\n",
        "/proc/4242/smaps_rollup" : "Rss:  4096 kB\nPss:  1024 kB\nShared_Clean: 12 kB\n",
    })
    monkeypatch.setattr(proc_stats, "open", fake, raising=False)
    return fake


def test_meminfo_reports_bytes_not_kilobytes(proc):
    fields = MemInfo.fields()

    assert fields["MemTotal"]     == 32000000 * 1024
    assert fields["MemAvailable"] == 8000000 * 1024


def test_used_percent_counts_unavailable_memory(proc):
    assert MemInfo.used_percent() == pytest.approx(75.0)


def test_used_percent_is_none_when_meminfo_is_unreadable(monkeypatch):
    monkeypatch.setattr(proc_stats, "open", FakeProc({}), raising=False)

    assert MemInfo.fields()       == {}
    assert MemInfo.used_percent() is None


def test_cpu_counters_exclude_idle_and_iowait(proc):
    counters = CpuCounters.read()

    assert sorted(counters) == ["cpu", "cpu0", "cpu1"]
    assert counters["cpu"]  == (100 + 20 + 30 + 0 + 10 + 5, 615)
    assert counters["cpu0"] == (50 + 10 + 15 + 5 + 2, 307)


def test_cpu_counters_are_empty_when_proc_stat_is_unreadable(monkeypatch):
    monkeypatch.setattr(proc_stats, "open", FakeProc({}), raising=False)

    assert CpuCounters.read() == {}


def test_stat_survives_a_command_name_full_of_parentheses(proc):
    stat = ProcStats.stat(4242)

    assert stat["comm"]    == "python (worker) 3.11"
    assert stat["state"]   == "S"
    assert stat["ppid"]    == 4000
    assert stat["jiffies"] == 700 + 300
    assert stat["threads"] == 9
    assert stat["started"] == "55555"
    assert stat["rss"]     == 512 * ProcStats.PAGE


def test_stat_and_ppid_degrade_on_a_dead_pid(proc):
    assert ProcStats.stat(9999) is None
    assert ProcStats.ppid(9999) == 0
    assert ProcStats.ppid(4242) == 4000


def test_private_memory_subtracts_the_shared_pages(proc):
    assert ProcStats.private(4242) == (1500 - 400) * ProcStats.PAGE
    assert ProcStats.private(9999) is None


def test_private_memory_never_goes_negative(monkeypatch):
    monkeypatch.setattr(proc_stats, "open", FakeProc({"/proc/1/statm": "10 5 900 0 0 0 0\n"}), raising=False)

    assert ProcStats.private(1) == 0


def test_attributed_prefers_pss_over_private(proc, monkeypatch):
    monkeypatch.setattr(ProcStats, "CACHE", PssCache())

    assert ProcStats.attributed(4242) == 1024 * 1024


def test_attributed_falls_back_to_private_without_a_rollup(monkeypatch):
    fake = FakeProc({"/proc/4242/statm": "2000 1500 400 10 0 300 0\n"})
    monkeypatch.setattr(proc_stats, "open", fake, raising=False)
    monkeypatch.setattr(ProcStats, "CACHE", PssCache())

    assert ProcStats.attributed(4242) == (1500 - 400) * ProcStats.PAGE


def test_the_pss_cache_reads_once_inside_its_ttl(proc):
    cache = PssCache()

    assert cache.value(4242) == 1024 * 1024
    assert cache.value(4242) == 1024 * 1024
    assert proc.reads.count("/proc/4242/smaps_rollup") == 1


def test_an_expired_pss_entry_is_read_again(proc):
    cache = PssCache()
    cache.value(4242)

    cache.entries[4242] = (cache.entries[4242][0] - PssCache.TTL_S - 1.0, 1)
    assert cache.value(4242) == 1024 * 1024
    assert proc.reads.count("/proc/4242/smaps_rollup") == 2


def test_pruning_drops_entries_older_than_the_window(proc):
    cache = PssCache()
    cache.value(4242)

    cache.entries[7] = (cache.entries[4242][0] - PssCache.PRUNE_S - 1.0, 99)
    cache.pruned_t   = 0.0
    cache.value(4243)

    assert 7 not in cache.entries
    assert 4242 in cache.entries


def test_pruning_only_runs_on_a_cache_miss(proc):
    cache = PssCache()
    cache.value(4242)

    cache.entries[7] = (cache.entries[4242][0] - PssCache.PRUNE_S - 1.0, 99)
    cache.pruned_t   = 0.0
    cache.value(4242)

    assert 7 in cache.entries


def test_a_missing_rollup_is_cached_as_absent(monkeypatch):
    fake = FakeProc({})
    monkeypatch.setattr(proc_stats, "open", fake, raising=False)
    cache = PssCache()

    assert cache.value(4242) is None
    assert cache.value(4242) is None
    assert fake.reads.count("/proc/4242/smaps_rollup") == 1


def test_the_running_process_reports_plausible_live_numbers():
    stat = ProcStats.stat(os.getpid())

    assert stat["comm"]
    assert stat["threads"] >= 1
    assert stat["rss"]      > 0
    assert stat["ppid"]    == os.getppid()
    assert ProcStats.attributed(os.getpid()) > 0
    assert ProcStats.username(os.getuid()) != ""
