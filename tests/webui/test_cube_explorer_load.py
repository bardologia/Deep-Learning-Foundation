"""Tests for the cube load path of the cube explorer.

Covers the chunked npy reader (whole files, several files at once, a row band
of a cube, refused Fortran order and truncated files), the byte counter that
drives the progress bar and names the files in flight, the stage timings
reported with a ready cube, the reference cube read as the band the cube
covers, the cube kept in memory for a later switch, and the silence report of
a load that stops moving.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from cube_explorer import LoadProgress, LoadTimer, NpyReader

from tests.webui.conftest import N_COLS, N_ROWS, N_Z, load_cube, make_cube_run, make_preproc, make_stamp, open_explorer


def _reader(chunk: int = 7) -> tuple[NpyReader, list]:
    """Returns a reader with a tiny chunk size whose progress records every advance, and the record list."""
    events = []
    reader = NpyReader(LoadProgress(lambda fraction, reading: events.append((fraction, reading))))
    reader.CHUNK_BYTES = chunk
    return reader, events


def test_read_returns_the_saved_array_and_books_every_byte(tmp_path):
    """A whole-file read returns the array np.save wrote and books exactly its data bytes on the counter."""
    array = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    path  = tmp_path / "cube.npy"
    np.save(path, array)

    reader, events = _reader()
    reader.progress.plan(array.nbytes)
    read = reader.read(path)

    assert read.dtype == array.dtype and read.flags["C_CONTIGUOUS"]
    assert np.array_equal(read, array)
    assert reader.progress.done == array.nbytes
    assert events[-1][0] == 1.0
    assert events[-1][1] == []


def test_read_many_returns_every_array_in_order(tmp_path):
    """Several files read together come back in request order, complex dtypes included."""
    arrays = [np.random.default_rng(seed).random((5, 3)).astype(np.float32) for seed in range(3)]
    arrays.append((np.ones((2, 2)) + 1j * np.arange(4).reshape(2, 2)).astype(np.complex64))
    paths  = []
    for index, array in enumerate(arrays):
        paths.append(tmp_path / f"a{index}.npy")
        np.save(paths[-1], array)

    reader, _ = _reader()
    for read, array in zip(reader.read_many(paths), arrays):
        assert np.array_equal(read, array)


def test_read_band_reads_only_the_rows_asked_for(tmp_path):
    """A band read returns rows lo:hi of every plane and books only the band's bytes."""
    cube = np.arange(4 * 6 * 5, dtype=np.float32).reshape(4, 6, 5)
    path = tmp_path / "cube.npy"
    np.save(path, cube)

    reader, _ = _reader()
    band      = reader.read_band(path, 2, 5)

    assert np.array_equal(band, cube[:, 2:5, :])
    assert reader.progress.done == cube[:, 2:5, :].nbytes

    with pytest.raises(ValueError, match="fall outside"):
        reader.read_band(path, 4, 7)


def test_header_reads_shape_and_dtype_without_the_data(tmp_path):
    """The header names the shape and dtype and books no bytes."""
    path = tmp_path / "cube.npy"
    np.save(path, np.zeros((2, 3, 4), dtype=np.float64))

    reader, _    = _reader()
    shape, dtype = reader.header(path)

    assert shape == (2, 3, 4) and dtype == np.float64
    assert reader.progress.done == 0


def test_fortran_ordered_and_truncated_files_are_refused(tmp_path):
    """A Fortran-ordered array and a file cut short both raise instead of returning garbage."""
    fortran = tmp_path / "fortran.npy"
    np.save(fortran, np.asfortranarray(np.zeros((3, 4), dtype=np.float32)))

    truncated = tmp_path / "truncated.npy"
    np.save(truncated, np.zeros((64,), dtype=np.float32))
    truncated.write_bytes(truncated.read_bytes()[:-16])

    reader, _ = _reader()
    with pytest.raises(ValueError, match="Fortran"):
        reader.read(fortran)
    with pytest.raises(ValueError, match="ends after"):
        reader.read(truncated)


def test_progress_names_the_files_in_flight_with_the_most_outstanding_first():
    """The in-flight snapshot lists unfinished files only, the one with the most bytes left first."""
    events   = []
    progress = LoadProgress(lambda fraction, reading: events.append((fraction, reading)))
    progress.plan(300)
    progress.begin("/data/runs/group/run_a/cubes/small.npy", 100)
    progress.begin("/data/runs/group/run_a/cubes/large.npy", 200)

    progress.advance("/data/runs/group/run_a/cubes/small.npy", 50)
    fraction, reading = events[-1]
    assert fraction == pytest.approx(50 / 300)
    assert [row["file"] for row in reading] == ["runs/group/run_a/cubes/large.npy", "runs/group/run_a/cubes/small.npy"]

    progress.advance("/data/runs/group/run_a/cubes/small.npy", 50)
    assert [row["file"] for row in events[-1][1]] == ["runs/group/run_a/cubes/large.npy"]


def test_timer_accumulates_repeated_stages():
    """A stage named twice adds both laps, and the summary lists stages in completion order."""
    timer = LoadTimer()
    timer.lap("pred")
    timer.lap("gt")
    timer.lap("pred")

    assert list(timer.stages) == ["pred", "gt"]
    assert timer.summary().startswith("pred ")
    assert timer.total() >= 0.0


def test_a_ready_cube_reports_its_load_timings(tmp_path):
    """The cube metadata carries the total load time and one entry per stage that ran."""
    make_cube_run(tmp_path, with_full=True, with_params=("pred", "gt"), with_metrics=True)
    explorer, ids = open_explorer(tmp_path, expected=1)

    status  = load_cube(explorer, ids[0])
    timings = status["cube"]["timings"]

    assert timings["total"] > 0.0
    assert {"pred", "gt", "full", "primary", "params", "metric maps", "aux"} <= set(timings["stages"])
    assert status["progress"] == 1.0


def test_the_reference_cube_is_read_as_the_band_the_cube_covers(tmp_path):
    """A stamp covering rows 2:6 of the dataset reads exactly that band of the full reference cube."""
    preproc = make_preproc(tmp_path, with_full=True)
    stamp   = make_stamp(tmp_path / "group" / "run_a", preproc, np.random.default_rng(1), shape=(N_Z, 4, N_COLS))
    metrics = stamp / "metrics.json"
    metrics.write_text(metrics.read_text().replace(f"[0, {N_ROWS}, 0, {N_COLS}]", f"[2, 6, 0, {N_COLS}]"))

    explorer, ids = open_explorer(tmp_path, expected=1)
    load_cube(explorer, ids[0])

    full     = np.abs(np.load(preproc / "data" / "cube_full.npy"))[:, 2:6, :]
    profiles = explorer.profiles(ids[0], row=1, col=4)["sources"]["full"]["values"]
    assert np.allclose(profiles, full[:, 1, 4])


def test_a_cube_loaded_before_is_served_from_memory(tmp_path):
    """Switching back to a cube loaded earlier answers at once, reports a memory stage and drops the attached comparison."""
    preproc = make_preproc(tmp_path)
    stamp_a = make_stamp(tmp_path / "run_a", preproc, np.random.default_rng(1))
    stamp_b = make_stamp(tmp_path / "run_b", preproc, np.random.default_rng(2))

    explorer, _ = open_explorer(tmp_path, expected=2)
    load_cube(explorer, str(stamp_a))
    assert explorer.attach_second(str(stamp_a), str(stamp_b))["ok"]

    load_cube(explorer, str(stamp_b))
    assert explorer.start_load(str(stamp_a))["ok"]

    status = explorer.load_status()
    assert status["state"] == "ready" and status["id"] == str(stamp_a)
    assert status["cube"]["timings"]["stages"] == {"memory": 0.0}
    assert status["cube"]["attached"] is None
    assert status["cube"]["sources"] == ["pred", "gt"]


def test_a_silent_load_reports_how_long_and_where_it_waits(tmp_path, monkeypatch):
    """While a load sits in one stage, the status names the seconds since it last moved and the frames of the load thread."""
    make_cube_run(tmp_path)
    explorer, ids = open_explorer(tmp_path, expected=1)

    def stalled_primary(reader, n_rows, n_cols):
        """Holds the primary stage long enough to be observed."""
        time.sleep(0.6)
        return np.zeros((n_rows, n_cols), dtype=np.float32)

    monkeypatch.setattr(explorer, "_primary_db", stalled_primary)
    assert explorer.start_load(ids[0])["ok"]

    deadline = time.time() + 10.0
    status   = explorer.load_status()
    while status["state"] == "loading" and status.get("stage") != "primary" and time.time() < deadline:
        time.sleep(0.02)
        status = explorer.load_status()

    time.sleep(0.3)
    status = explorer.load_status()
    assert status["state"] == "loading", status
    assert status["stalled_for"] >= 0.2
    assert any("stalled_primary" in frame for frame in status["where"])

    while explorer.load_status()["state"] == "loading" and time.time() < deadline:
        time.sleep(0.05)
    assert explorer.load_status()["state"] == "ready"
