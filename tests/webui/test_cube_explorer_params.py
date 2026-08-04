from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cube_explorer import CubeExplorer

from tools.loss.param_loss import ParamMatcher

from tests.webui.conftest import N_AZ, N_ELEV, N_RG, N_SLOTS, load_cube, loaded_cube, make_cube_run, open_explorer

PARAM_SOURCES = ("pred", "gt")


def _loaded_explorer(base: Path, with_params: tuple = PARAM_SOURCES, with_spacing: bool = False, with_reduced: bool = False) -> tuple[CubeExplorer, str]:
    return loaded_cube(base, with_params=with_params, with_spacing=with_spacing, with_reduced=with_reduced)


def test_meta_reports_param_block(tmp_path):
    explorer, _ = _loaded_explorer(tmp_path)

    meta = explorer.load_status()["cube"]["params"]

    assert meta["sources"] == ["pred", "gt"]
    assert meta["n_slots"] == N_SLOTS
    assert meta["error"] is True
    assert meta["threshold"] == ParamMatcher.ACTIVE_AMP_THR
    assert meta["ranges"]["mu"] == [-10.0, 30.0]
    assert meta["ranges"]["count"] == [0.0, float(N_SLOTS)]
    assert meta["ranges"]["error_count"] == [-float(N_SLOTS), float(N_SLOTS)]
    assert set(meta["ranges"]) >= {"amp", "sigma", "error_amp", "error_mu", "error_sigma"}


def test_meta_without_params_is_none(tmp_path):
    explorer, _ = _loaded_explorer(tmp_path, with_params=())

    assert explorer.load_status()["cube"]["params"] is None


def test_meta_spacing_absent_without_track_parameters(tmp_path):
    explorer, _ = _loaded_explorer(tmp_path)

    assert explorer.load_status()["cube"]["spacing"] is None


def test_meta_spacing_from_reference_track(tmp_path):
    explorer, _ = _loaded_explorer(tmp_path, with_spacing=True)

    assert explorer.load_status()["cube"]["spacing"] == {"az": 0.4, "rg": 0.6}


def test_meta_intensity_ranges_per_source(tmp_path):
    explorer, _ = _loaded_explorer(tmp_path, with_reduced=True)

    intensity = explorer.load_status()["cube"]["intensity"]

    assert set(intensity) == {"pred", "gt", "reduced"}
    assert all(np.isfinite(v) for bounds in intensity.values() for v in bounds)


def test_curve_points_threshold_and_values(tmp_path):
    explorer, cube_id = _loaded_explorer(tmp_path, with_reduced=True)

    raw    = np.frombuffer(explorer.points_bin(cube_id, "reduced", amp_min=0.9, max_points=100), dtype=np.float32)
    header = raw[:4]
    rows   = raw[4:].reshape(-1, 4)

    heights = np.linspace(-10.0, 30.0, N_ELEV)

    assert header[0] == 2 and header[1] == 2
    assert rows[0].tolist() == [3.0, 4.0, float(heights[2]), 2.0]
    assert rows[1].tolist() == [5.0, 1.0, float(heights[4]), 1.0]


def test_curve_points_subsample_cap(tmp_path):
    explorer, cube_id = _loaded_explorer(tmp_path, with_reduced=True)

    raw    = np.frombuffer(explorer.points_bin(cube_id, "reduced", amp_min=-1.0, max_points=50), dtype=np.float32)
    header = raw[:4]
    rows   = raw[4:].reshape(-1, 4)

    assert header[0] == 50
    assert header[1] == N_ELEV * N_AZ * N_RG
    assert rows.shape == (50, 4)


def test_curve_points_absent_source_is_none(tmp_path):
    explorer, cube_id = _loaded_explorer(tmp_path)

    assert explorer.points_bin(cube_id, "full", amp_min=0.0, max_points=100) is None


def test_param_map_png_for_all_fields(tmp_path):
    explorer, cube_id = _loaded_explorer(tmp_path)

    for source in ("pred", "gt", "error"):
        for field in ("amp", "mu", "sigma", "count"):
            png = explorer.param_map_png(cube_id, source, field, slot=0)
            assert png and png[:4] == b"\x89PNG", (source, field)


def test_param_map_rejects_unknown_source_and_field(tmp_path):
    explorer, cube_id = _loaded_explorer(tmp_path)

    assert explorer.param_map_png(cube_id, "banana", "amp", 0) is None
    assert explorer.param_map_png(cube_id, "pred", "banana", 0) is None
    assert explorer.param_map_png("wrong_id", "pred", "amp", 0) is None


def test_param_map_error_requires_both_sources(tmp_path):
    explorer, cube_id = _loaded_explorer(tmp_path, with_params=("pred",))

    assert explorer.param_map_png(cube_id, "pred", "amp", 0) is not None
    assert explorer.param_map_png(cube_id, "error", "amp", 0) is None
    assert explorer.param_map_png(cube_id, "gt", "amp", 0) is None


def test_param_cbar_png(tmp_path):
    explorer, cube_id = _loaded_explorer(tmp_path)

    assert explorer.param_cbar_png(cube_id, "pred", "amp")[:4] == b"\x89PNG"
    assert explorer.param_cbar_png(cube_id, "error", "count")[:4] == b"\x89PNG"
    assert explorer.param_cbar_png(cube_id, "banana", "amp") is None


def test_params_at_returns_slot_values(tmp_path):
    explorer, cube_id = _loaded_explorer(tmp_path)

    stamp  = Path(cube_id)
    pred   = np.load(stamp / "cubes" / "params_pred.npy")
    result = explorer.params_at(cube_id, az=2, rg=3)

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
    explorer, cube_id = _loaded_explorer(tmp_path)

    clipped = explorer.params_at(cube_id, az=999, rg=-4)
    assert clipped["ok"] and clipped["az"] == N_AZ - 1 and clipped["rg"] == 0

    bare_explorer, bare_id = _loaded_explorer(tmp_path / "bare", with_params=())
    assert not bare_explorer.params_at(bare_id, az=0, rg=0)["ok"]


def test_param_map_slot_clipped(tmp_path):
    explorer, cube_id = _loaded_explorer(tmp_path)

    assert explorer.param_map_png(cube_id, "pred", "amp", slot=99) is not None
    assert explorer.param_map_png(cube_id, "pred", "amp", slot=-3) is not None


def test_points_bin_thresholds_and_subsamples(tmp_path):
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
    assert rows[:, 0].max() < N_AZ and rows[:, 1].max() < N_RG

    capped = np.frombuffer(explorer.points_bin(cube_id, "pred", amp_min=1e-3, max_points=5), dtype=np.float32)
    assert int(capped[0]) == 5 and int(capped[1]) == total

    assert explorer.points_bin(cube_id, "banana", 1e-3, 100) is None
    assert explorer.points_bin("wrong", "pred", 1e-3, 100) is None


def test_dem_grid_absent_without_artifact(tmp_path):
    explorer, cube_id = _loaded_explorer(tmp_path)

    assert explorer.load_status()["cube"]["dem"] is False
    assert explorer.dem_grid_bin(cube_id) is None


def test_dem_grid_with_artifact(tmp_path):
    make_cube_run(tmp_path, with_params=PARAM_SOURCES)

    preproc = tmp_path / "preproc"
    layout  = json.loads((preproc / "data" / "dataset.json").read_text())

    dem       = np.linspace(600.0, 700.0, N_AZ * N_RG).reshape(N_AZ, N_RG).astype(np.float32)
    dem[2, 3] = np.nan
    np.save(preproc / "data" / "dem.npy", dem)
    layout["artifacts"]["dem_full"] = "dem.npy"
    (preproc / "data" / "dataset.json").write_text(json.dumps(layout))

    explorer, cube_ids = open_explorer(tmp_path, expected=1)
    cube_id            = cube_ids[0]

    assert load_cube(explorer, cube_id)["cube"]["dem"] is True

    raw    = np.frombuffer(explorer.dem_grid_bin(cube_id), dtype=np.float32)
    header = raw[:4]
    grid   = raw[4:].reshape(N_AZ, N_RG)

    median = float(np.median(dem[np.isfinite(dem)]))

    assert header[0] == N_AZ and header[1] == N_RG
    assert abs(float(header[2]) - median) < 1e-3
    assert np.isnan(grid[2, 3])
    assert np.allclose(grid[0, 0], dem[0, 0] - median, atol=1e-3)


def test_params_with_nan_survive_load_and_lookup(tmp_path):
    stamp = make_cube_run(tmp_path, with_params=PARAM_SOURCES)
    pred  = np.load(stamp / "cubes" / "params_pred.npy")
    pred[1, 0, 0] = np.nan
    pred[4]       = np.nan
    np.save(stamp / "cubes" / "params_pred.npy", pred)

    explorer, cube_ids = open_explorer(tmp_path, expected=1)
    cube_id            = cube_ids[0]

    meta = load_cube(explorer, cube_id)["cube"]["params"]
    assert all(np.isfinite(v) for bounds in meta["ranges"].values() for v in bounds)

    result = explorer.params_at(cube_id, az=0, rg=0)
    assert result["ok"]
    assert np.isnan(result["sources"]["pred"][0]["mu"])
    assert result["sources"]["pred"][1]["active"] is False

    assert explorer.param_map_png(cube_id, "pred", "mu", 0) is not None
    assert explorer.param_map_png(cube_id, "error", "mu", 1) is not None
