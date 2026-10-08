"""Tests for the Gaussian parameter layers of the cube explorer.

Covers the parameter metadata and per-field colour ranges, intensity metadata,
the binary point blobs for curve peaks and parameter scatterers with amplitude
thresholds and subsampling caps, parameter and error map rendering with slot
clipping, per-pixel slot lookups, the aux grid served for the
three-dimensional view, NaN tolerance throughout, and the refusal of stamps
whose metrics predate the recorded activity threshold.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from cube_explorer import CubeExplorer

from tests.webui.conftest import ACTIVITY_THRESHOLD, N_ROWS, N_Z, N_COLS, N_SLOTS, load_cube, loaded_cube, make_cube_run, open_explorer

PARAM_SOURCES = ("pred", "gt")


def _loaded_explorer(base: Path, with_params: tuple = PARAM_SOURCES, with_full: bool = False) -> tuple[CubeExplorer, str]:
    """Returns an explorer with a parameter-carrying cube loaded, plus its cube id.

    Args:
        base        : Parent directory for the generated run.
        with_params : Parameter sources written next to the cube.
        with_full   : Whether the dataset also holds the full cube, served as the full curve source.

    Returns:
        The explorer and the id of its loaded cube.
    """
    return loaded_cube(base, with_params=with_params, with_full=with_full)


def test_meta_reports_param_block(tmp_path):
    """Checks the parameter metadata reports sources, slot count, the active-amplitude threshold and per-field ranges."""
    explorer, _ = _loaded_explorer(tmp_path)

    meta = explorer.load_status()["cube"]["params"]

    assert meta["sources"] == ["pred", "gt"]
    assert meta["n_slots"] == N_SLOTS
    assert meta["error"] is True
    assert meta["threshold"] == ACTIVITY_THRESHOLD
    assert meta["ranges"]["mu"] == [-10.0, 30.0]
    assert meta["ranges"]["count"] == [0.0, float(N_SLOTS)]
    assert meta["ranges"]["error_count"] == [-float(N_SLOTS), float(N_SLOTS)]
    assert set(meta["ranges"]) >= {"amp", "sigma", "error_amp", "error_mu", "error_sigma"}


def test_meta_without_params_is_none(tmp_path):
    """Checks a run without parameter blocks exposes no parameter metadata."""
    explorer, _ = _loaded_explorer(tmp_path, with_params=())

    assert explorer.load_status()["cube"]["params"] is None


def test_meta_intensity_ranges_per_source(tmp_path):
    """Checks every curve source reports finite intensity bounds."""
    explorer, _ = _loaded_explorer(tmp_path, with_full=True)

    intensity = explorer.load_status()["cube"]["intensity"]

    assert set(intensity) == {"pred", "gt", "full"}
    assert all(np.isfinite(v) for bounds in intensity.values() for v in bounds)


def test_curve_points_threshold_and_values(tmp_path):
    """Checks thresholded curve points carry the expected row, column, z value and amplitude."""
    explorer, cube_id = _loaded_explorer(tmp_path, with_full=True)

    raw    = np.frombuffer(explorer.points_bin(cube_id, "full", amp_min=0.9, max_points=100), dtype=np.float32)
    header = raw[:4]
    rows   = raw[4:].reshape(-1, 4)

    z_values = np.arange(N_Z, dtype=np.float64)

    assert header[0] == 2 and header[1] == 2
    assert rows[0].tolist() == [3.0, 4.0, float(z_values[2]), 2.0]
    assert rows[1].tolist() == [5.0, 1.0, float(z_values[4]), 1.0]


def test_curve_points_subsample_cap(tmp_path):
    """Checks the point cap limits the returned rows while the header still reports the full count."""
    explorer, cube_id = _loaded_explorer(tmp_path, with_full=True)

    raw    = np.frombuffer(explorer.points_bin(cube_id, "full", amp_min=-1.0, max_points=50), dtype=np.float32)
    header = raw[:4]
    rows   = raw[4:].reshape(-1, 4)

    assert header[0] == 50
    assert header[1] == N_Z * N_ROWS * N_COLS
    assert rows.shape == (50, 4)


def test_curve_points_absent_source_is_none(tmp_path):
    """Checks requesting points from an unloaded curve source yields no blob."""
    explorer, cube_id = _loaded_explorer(tmp_path)

    assert explorer.points_bin(cube_id, "full", amp_min=0.0, max_points=100) is None


def test_param_map_png_for_all_fields(tmp_path):
    """Checks every parameter field renders for the predicted, ground-truth and error sources."""
    explorer, cube_id = _loaded_explorer(tmp_path)

    for source in ("pred", "gt", "error"):
        for field in ("amp", "mu", "sigma", "count"):
            png = explorer.param_map_png(cube_id, source, field, slot=0)
            assert png and png[:4] == b"\x89PNG", (source, field)


def test_param_map_rejects_unknown_source_and_field(tmp_path):
    """Checks unknown sources, unknown fields and unloaded cubes yield no map."""
    explorer, cube_id = _loaded_explorer(tmp_path)

    assert explorer.param_map_png(cube_id, "banana", "amp", 0) is None
    assert explorer.param_map_png(cube_id, "pred", "banana", 0) is None
    assert explorer.param_map_png("wrong_id", "pred", "amp", 0) is None


def test_param_map_error_requires_both_sources(tmp_path):
    """Checks the error and ground-truth maps are unavailable when only predictions were saved."""
    explorer, cube_id = _loaded_explorer(tmp_path, with_params=("pred",))

    assert explorer.param_map_png(cube_id, "pred", "amp", 0) is not None
    assert explorer.param_map_png(cube_id, "error", "amp", 0) is None
    assert explorer.param_map_png(cube_id, "gt", "amp", 0) is None


def test_param_cbar_png(tmp_path):
    """Checks parameter colour bars render for known sources and are refused for unknown ones."""
    explorer, cube_id = _loaded_explorer(tmp_path)

    assert explorer.param_cbar_png(cube_id, "pred", "amp")[:4] == b"\x89PNG"
    assert explorer.param_cbar_png(cube_id, "error", "count")[:4] == b"\x89PNG"
    assert explorer.param_cbar_png(cube_id, "banana", "amp") is None


def test_params_at_returns_slot_values(tmp_path):
    """Checks a pixel lookup returns each slot's amplitude, mean and width with the correct active flag."""
    explorer, cube_id = _loaded_explorer(tmp_path)

    stamp  = Path(cube_id)
    pred   = np.load(stamp / "cubes" / "params_pred.npy")
    result = explorer.params_at(cube_id, row=2, col=3)

    assert result["ok"]
    assert result["n_slots"] == N_SLOTS

    slot0 = result["sources"]["pred"][0]
    assert slot0["amp"]   == float(pred[0, 2, 3])
    assert slot0["mu"]    == float(pred[1, 2, 3])
    assert slot0["sigma"] == float(pred[2, 2, 3])
    assert slot0["active"] is True

    slot1 = result["sources"]["pred"][1]
    assert slot1["amp"]    == 0.0
    assert slot1["active"] is False


def test_params_at_clips_indices_and_requires_params(tmp_path):
    """Checks out-of-range indices clamp to the scene and a cube without parameters fails the lookup."""
    explorer, cube_id = _loaded_explorer(tmp_path)

    clipped = explorer.params_at(cube_id, row=999, col=-4)
    assert clipped["ok"] and clipped["row"] == N_ROWS - 1 and clipped["col"] == 0

    bare_explorer, bare_id = _loaded_explorer(tmp_path / "bare", with_params=())
    assert not bare_explorer.params_at(bare_id, row=0, col=0)["ok"]


def test_param_map_slot_clipped(tmp_path):
    """Checks slot indices beyond either end of the range clamp instead of failing."""
    explorer, cube_id = _loaded_explorer(tmp_path)

    assert explorer.param_map_png(cube_id, "pred", "amp", slot=99) is not None
    assert explorer.param_map_png(cube_id, "pred", "amp", slot=-3) is not None


def test_points_bin_thresholds_and_subsamples(tmp_path):
    """Checks the parameter point blob honours the amplitude threshold and the cap, and rejects unknown sources and cubes."""
    explorer, cube_id = _loaded_explorer(tmp_path)

    blob = explorer.points_bin(cube_id, "pred", amp_min=1e-3, max_points=100000)
    raw  = np.frombuffer(blob, dtype=np.float32)

    n_sent, total = int(raw[0]), int(raw[1])
    rows          = raw[4:].reshape(n_sent, 4)

    stamp = Path(cube_id)
    pred  = np.load(stamp / "cubes" / "params_pred.npy")
    amps  = pred[0::3]
    mus   = pred[1::3]
    mask  = np.isfinite(amps) & (amps >= 1e-3) & np.isfinite(mus)

    assert total == int(mask.sum())
    assert n_sent == total
    assert rows[:, 3].min() >= 1e-3
    assert rows[:, 0].max() < N_ROWS and rows[:, 1].max() < N_COLS

    capped = np.frombuffer(explorer.points_bin(cube_id, "pred", amp_min=1e-3, max_points=5), dtype=np.float32)
    assert int(capped[0]) == 5 and int(capped[1]) == total

    assert explorer.points_bin(cube_id, "banana", 1e-3, 100) is None
    assert explorer.points_bin("wrong", "pred", 1e-3, 100) is None


def test_dem_grid_absent_without_artifact(tmp_path):
    """Checks a run without an aux artifact reports no aux layer and serves no grid."""
    explorer, cube_id = _loaded_explorer(tmp_path)

    assert explorer.load_status()["cube"]["aux"] is False
    assert explorer.aux_grid_bin(cube_id) is None


def test_dem_grid_with_artifact(tmp_path):
    """Checks the aux grid is served relative to the scene median, preserving NaN gaps."""
    make_cube_run(tmp_path, with_params=PARAM_SOURCES)

    preproc = tmp_path / "preproc"
    layout  = json.loads((preproc / "data" / "dataset.json").read_text())

    aux       = np.linspace(600.0, 700.0, N_ROWS * N_COLS).reshape(N_ROWS, N_COLS).astype(np.float32)
    aux[2, 3] = np.nan
    np.save(preproc / "data" / "aux.npy", aux)
    layout["artifacts"]["aux_full"] = "aux.npy"
    (preproc / "data" / "dataset.json").write_text(json.dumps(layout))

    explorer, cube_ids = open_explorer(tmp_path, expected=1)
    cube_id            = cube_ids[0]

    assert load_cube(explorer, cube_id)["cube"]["aux"] is True

    raw    = np.frombuffer(explorer.aux_grid_bin(cube_id), dtype=np.float32)
    header = raw[:4]
    grid   = raw[4:].reshape(N_ROWS, N_COLS)

    median = float(np.median(aux[np.isfinite(aux)]))

    assert header[0] == N_ROWS and header[1] == N_COLS
    assert abs(float(header[2]) - median) < 1e-3
    assert np.isnan(grid[2, 3])
    assert np.allclose(grid[0, 0], aux[0, 0] - median, atol=1e-3)


def test_params_with_nan_survive_load_and_lookup(tmp_path):
    """Checks NaN parameter entries leave the ranges finite, surface in lookups and still render."""
    stamp = make_cube_run(tmp_path, with_params=PARAM_SOURCES)
    pred  = np.load(stamp / "cubes" / "params_pred.npy")
    pred[1, 0, 0] = np.nan
    pred[4]       = np.nan
    np.save(stamp / "cubes" / "params_pred.npy", pred)

    explorer, cube_ids = open_explorer(tmp_path, expected=1)
    cube_id            = cube_ids[0]

    meta = load_cube(explorer, cube_id)["cube"]["params"]
    assert all(np.isfinite(v) for bounds in meta["ranges"].values() for v in bounds)

    result = explorer.params_at(cube_id, row=0, col=0)
    assert result["ok"]
    assert np.isnan(result["sources"]["pred"][0]["mu"])
    assert result["sources"]["pred"][1]["active"] is False

    assert explorer.param_map_png(cube_id, "pred", "mu", 0) is not None
    assert explorer.param_map_png(cube_id, "error", "mu", 1) is not None


def test_params_without_a_recorded_threshold_fail_the_load(tmp_path):
    """Checks a parameter-carrying stamp whose metrics predate the recorded threshold is refused with a request to re-run inference."""
    stamp   = make_cube_run(tmp_path, with_params=PARAM_SOURCES)
    payload = json.loads((stamp / "metrics.json").read_text(encoding="utf-8"))
    del payload["activity_threshold"]
    (stamp / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")

    explorer, cube_ids = open_explorer(tmp_path, expected=1)
    assert explorer.start_load(cube_ids[0])["ok"]

    deadline = time.time() + 30.0
    while explorer.load_status()["state"] == "loading" and time.time() < deadline:
        time.sleep(0.05)

    status = explorer.load_status()
    assert status["state"] == "error"
    assert "re-run inference" in status["error"]
