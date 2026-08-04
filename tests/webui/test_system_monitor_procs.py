from __future__ import annotations

import os
import threading


def test_rows_report_thread_counts(monitor):
    rows = monitor._procs({})

    assert rows
    assert all(isinstance(row["threads"], int) and row["threads"] >= 1 for row in rows)


def test_own_process_thread_count_tracks_spawned_threads(monitor):
    release = threading.Event()
    workers = [threading.Thread(target=release.wait, daemon=True) for _ in range(5)]
    for worker in workers:
        worker.start()

    try:
        monitor.PROC_LIMIT = 100000
        row = next(r for r in monitor._procs({}) if r["pid"] == os.getpid())
        assert row["threads"] >= 6
    finally:
        release.set()
        for worker in workers:
            worker.join()
