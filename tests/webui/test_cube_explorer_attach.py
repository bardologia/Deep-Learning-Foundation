"""Tests for run-vs-run comparison in the cube explorer: the attached pred B and A minus B sources, their delta metric layers, transects and saved figures."""

from __future__ import annotations

import io
import json
from pathlib import Path
from urllib.parse import urlencode

import matplotlib.pyplot as plt
import numpy as np
import pytest

from cube_explorer        import CubeExplorer
from routers.cube_routers import CubeRouter
from routers.dispatch     import HttpExchange

from tests.webui.conftest import N_ROWS, N_Z, N_COLS, FakeHandler, load_cube, make_preproc, make_stamp, open_explorer


def _make_run(base: Path, preproc: Path, name: str, seed: int, x_min: float = -10.0) -> Path:
    return make_stamp(base / name, preproc, np.random.default_rng(seed), x_min=x_min)


def _loaded(base: Path) -> tuple[CubeExplorer, str, str]:
    preproc = make_preproc(base)
    stamp_a = _make_run(base, preproc, "run_a", seed=1)
    stamp_b = _make_run(base, preproc, "run_b", seed=2)

    explorer, _ = open_explorer(base, expected=2)
    cube_a      = str(stamp_a)

    load_cube(explorer, cube_a)

    return explorer, cube_a, str(stamp_b)


def test_attach_adds_predb_and_diff(tmp_path):
    explorer, cube_a, cube_b = _loaded(tmp_path)

    result = explorer.attach_second(cube_a, cube_b)
    assert result["ok"], result

    meta = result["cube"]
    assert meta["sources"] == ["pred", "predb", "diff", "gt"]
    assert meta["attached"]["id"] == cube_b
    assert meta["attached"]["run"] == "run_b"

    pred_a = np.load(Path(cube_a) / "cubes" / "pred_curves.npy")
    pred_b = np.load(Path(cube_b) / "cubes" / "pred_curves.npy")

    profiles = explorer.profiles(cube_a, row=2, col=3)
    assert set(profiles["sources"]) == {"pred", "predb", "diff", "gt"}

    expected_diff = (pred_a - pred_b)[:, 2, 3]
    order         = np.argsort(np.linspace(-10.0, 30.0, N_Z))
    assert np.allclose(profiles["sources"]["diff"]["values"], expected_diff[order])

    assert explorer.slice_png(cube_a, "predb", "col", row=0, col=0)[:4] == b"\x89PNG"
    assert explorer.slice_png(cube_a, "diff",  "col", row=0, col=0)[:4] == b"\x89PNG"
    assert explorer.plane_png(cube_a, "diff", frac=0.5, space="normalized")[:4] == b"\x89PNG"


def test_diff_range_is_symmetric(tmp_path):
    explorer, cube_a, cube_b = _loaded(tmp_path)
    explorer.attach_second(cube_a, cube_b)

    entry = explorer._entry(cube_a, "diff")
    assert entry["diverging"] is True
    assert entry["vmin"] == -entry["vmax"]
    assert entry["vmax"] > 0


def test_detach_removes_comparison(tmp_path):
    explorer, cube_a, cube_b = _loaded(tmp_path)
    explorer.attach_second(cube_a, cube_b)

    result = explorer.detach_second(cube_a)
    assert result["ok"]
    assert result["cube"]["sources"] == ["pred", "gt"]
    assert result["cube"]["attached"] is None
    assert explorer.slice_png(cube_a, "predb", "col", row=0, col=0) is None


def test_attach_rejects_self_and_unknown(tmp_path):
    explorer, cube_a, _ = _loaded(tmp_path)

    assert not explorer.attach_second(cube_a, cube_a)["ok"]
    assert not explorer.attach_second(cube_a, str(tmp_path / "nowhere"))["ok"]
    assert not explorer.attach_second(str(tmp_path / "other"), cube_a)["ok"]


def test_attach_rejects_mismatched_axis(tmp_path):
    preproc = make_preproc(tmp_path)
    stamp_a = _make_run(tmp_path, preproc, "run_a", seed=1)
    stamp_c = _make_run(tmp_path, preproc, "run_c", seed=3, x_min=-99.0)

    explorer, _ = open_explorer(tmp_path, expected=2)
    load_cube(explorer, str(stamp_a))

    result = explorer.attach_second(str(stamp_a), str(stamp_c))
    assert not result["ok"] and "z axis" in result["error"]


def test_ssim_covers_predb(tmp_path):
    explorer, cube_a, cube_b = _loaded(tmp_path)
    explorer.attach_second(cube_a, cube_b)

    result = explorer.slice_ssim(cube_a, row=2, col=2)
    assert result["ok"]
    assert "predb" in result["col"]
    assert "diff" not in result["col"]


def test_transect_png_samples_line(tmp_path):
    """Checks straight and bent transects render in both colour spaces and degenerate lines, unknown sources or cubes return None."""
    explorer, cube_a, _ = _loaded(tmp_path)

    png = explorer.transect_png(cube_a, "pred", [(0, 0), (N_ROWS - 1, N_COLS - 1)])
    assert png and png[:4] == b"\x89PNG"

    normalized = explorer.transect_png(cube_a, "pred", [(2, 1), (2, 4)], space="normalized")
    assert normalized and normalized[:4] == b"\x89PNG"

    bent = explorer.transect_png(cube_a, "pred", [(0, 0), (N_ROWS - 1, 0), (N_ROWS - 1, N_COLS - 1)])
    assert bent and bent[:4] == b"\x89PNG"

    assert explorer.transect_png(cube_a, "pred", [(1, 1)]) is None
    assert explorer.transect_png(cube_a, "pred", [(1, 1), (1, 1)]) is None
    assert explorer.transect_png(cube_a, "pred", [(N_ROWS + 5, 1), (N_ROWS - 1, 1)]) is None
    assert explorer.transect_png(cube_a, "banana", [(0, 0), (1, 1)]) is None
    assert explorer.transect_png("wrong", "pred", [(0, 0), (1, 1)]) is None


def test_bent_transect_shares_its_vertices_between_segments(tmp_path):
    """Checks a bent polyline samples every pixel of each segment once, the shared corner included, so its PNG is as wide as the path."""
    explorer, cube_a, _ = _loaded(tmp_path)

    row_idx, col_idx = explorer._path_indices([(0, 0), (0, 3), (2, 3)])
    assert row_idx.tolist() == [0, 0, 0, 0, 1, 2]
    assert col_idx.tolist() == [0, 1, 2, 3, 3, 3]

    png = plt.imread(io.BytesIO(explorer.transect_png(cube_a, "pred", [(0, 0), (0, 3), (2, 3)])))
    assert png.shape[:2] == (N_Z, 6)


def test_clip_points_folds_repeats_after_clipping():
    """Checks vertices are clipped into the footprint before consecutive repeats are dropped, and fewer than two distinct ones give None."""
    assert CubeExplorer._clip_points([(-3, 2), (0, 2), (9, 9), (7, 5)], 8, 6) == [(0, 2), (7, 5)]
    assert CubeExplorer._clip_points([(1, 1), (2, 2), (1, 1)], 8, 6) == [(1, 1), (2, 2), (1, 1)]
    assert CubeExplorer._clip_points([(1, 1), (1, 1)], 8, 6) is None
    assert CubeExplorer._clip_points([], 8, 6) is None


def test_transect_cut_geometry(tmp_path):
    """Checks a diagonal cut spans the longer axis, keeps the ascending z axis and the source's colour limits."""
    explorer, cube_a, _ = _loaded(tmp_path)

    entry            = explorer._entry(cube_a, "pred")
    row_idx, col_idx = explorer._path_indices([(0, 0), (N_ROWS - 1, N_COLS - 1)])
    data, z_values, vmin, vmax = explorer._transect_cut(entry, row_idx, col_idx, "physical")

    assert data.shape == (N_Z, max(N_ROWS, N_COLS))
    assert z_values[0] == -10.0 and z_values[-1] == 30.0
    assert vmin == entry["vmin"] and vmax == entry["vmax"]

    cube = entry["cube"]
    assert np.allclose(data[:, 0], cube[:, 0, 0])


def test_save_transect_writes_figures(tmp_path):
    """Checks saving a transect writes one non-empty figure per source under a coordinate-named directory."""
    explorer, cube_a, _ = _loaded(tmp_path)

    result = explorer.save_transect(cube_a, [(0, 0), (4, 5)])
    assert result["ok"], result

    out_dir = Path(result["dir"])
    assert result["rel"] == "figures/cube_transects/row0000_col0000_to_row0004_col0005"

    expected = {f"transect_{source}_physical.png" for source in ("pred", "gt")}
    assert set(result["files"]) == expected
    assert all((out_dir / name).stat().st_size > 0 for name in expected)


def test_save_transect_names_a_bent_line_by_its_interior_vertices(tmp_path):
    """Checks a polyline's figure folder carries a _via suffix counting its interior vertices."""
    explorer, cube_a, _ = _loaded(tmp_path)

    result = explorer.save_transect(cube_a, [(0, 0), (4, 0), (4, 5)])

    assert result["ok"], result
    assert result["rel"] == "figures/cube_transects/row0000_col0000_to_row0004_col0005_via1"


def test_save_transect_rejects_bad_space_degenerate_lines_and_unloaded(tmp_path):
    """Checks an unknown colour space, a line without two distinct points and an unloaded cube are all refused."""
    explorer, cube_a, _ = _loaded(tmp_path)

    assert not explorer.save_transect(cube_a, [(0, 0), (1, 1)], space="banana")["ok"]
    assert not explorer.save_transect(cube_a, [(1, 1)])["ok"]
    assert not explorer.save_transect(cube_a, [(1, 1), (1, 1)])["ok"]
    assert not explorer.save_transect(str(tmp_path / "nowhere"), [(0, 0), (1, 1)])["ok"]
    assert not (Path(cube_a) / "figures" / "cube_transects").exists()


def test_save_slices_includes_comparison_sources(tmp_path):
    explorer, cube_a, cube_b = _loaded(tmp_path)
    explorer.attach_second(cube_a, cube_b)

    result = explorer.save_slices(cube_a, row=1, col=1)
    assert result["ok"]

    expected = {f"{axis}_{source}_physical.png" for source in ("pred", "predb", "diff", "gt") for axis in ("col", "row")}
    assert set(result["files"]) == expected


def test_cmap_selection_and_diverging_override(tmp_path):
    explorer, cube_a, cube_b = _loaded(tmp_path)
    explorer.attach_second(cube_a, cube_b)

    pred = explorer._entry(cube_a, "pred")
    diff = explorer._entry(cube_a, "diff")

    assert explorer._entry_cmap(pred, "viridis") == "viridis"
    assert explorer._entry_cmap(pred, "banana") == "jet"
    assert explorer._entry_cmap(diff, "viridis") == "coolwarm"

    assert explorer.slice_png(cube_a, "pred", "col", row=0, col=0, cmap="viridis")[:4] == b"\x89PNG"
    assert explorer.plane_png(cube_a, "pred", frac=0.5, cmap="gray")[:4] == b"\x89PNG"
    assert explorer.transect_png(cube_a, "pred", [(0, 0), (2, 2)], cmap="inferno")[:4] == b"\x89PNG"


def _delta_runs(base: Path) -> tuple[CubeExplorer, Path, Path]:
    """Loads run A beside run B, both carrying a pixel MSE and a failure-mode map, with a map only run A carries."""
    preproc = make_preproc(base)
    stamp_a = _make_run(base, preproc, "run_a", seed=1)
    stamp_b = _make_run(base, preproc, "run_b", seed=2)

    rng = np.random.default_rng(3)
    for stamp in (stamp_a, stamp_b):
        np.save(stamp / "cubes" / "pixel_mse.npy", rng.random((N_ROWS, N_COLS)).astype(np.float32))
        np.save(stamp / "cubes" / "failure_mode.npy", np.zeros((N_ROWS, N_COLS), dtype=np.float32))
    np.save(stamp_a / "cubes" / "only_a.npy", rng.random((N_ROWS, N_COLS)).astype(np.float32))

    explorer, _ = open_explorer(base, expected=2)
    load_cube(explorer, str(stamp_a))

    return explorer, stamp_a, stamp_b


def test_attach_adds_delta_metric_layers(tmp_path):
    """Shared continuous metric maps gain a signed delta layer that detach removes again."""
    explorer, stamp_a, stamp_b = _delta_runs(tmp_path)

    attached = explorer.attach_second(str(stamp_a), str(stamp_b))
    assert attached["ok"]

    layers = {layer["key"]: layer for layer in attached["cube"]["metric_maps"]}
    assert "delta_pixel_mse" in layers
    assert "delta_failure_mode" not in layers
    assert "delta_only_a" not in layers

    delta = layers["delta_pixel_mse"]
    assert delta["label"] == "Δ MSE"
    assert delta["diverging"] is True
    assert delta["vmin"] == -delta["vmax"] and delta["vmax"] > 0
    assert layers["pixel_mse"]["diverging"] is False

    a     = np.load(stamp_a / "cubes" / "pixel_mse.npy")
    b     = np.load(stamp_b / "cubes" / "pixel_mse.npy")
    value = explorer.metric_value_at(str(stamp_a), "delta_pixel_mse", row=2, col=3)
    assert abs(value["value"] - float(a[2, 3] - b[2, 3])) < 1e-6

    png = explorer.metric_overlay_png(str(stamp_a), "delta_pixel_mse", delta["vmin"], delta["vmax"], float("-inf"), float("inf"), 0.8)
    assert png and png[:4] == b"\x89PNG"

    detached = explorer.detach_second(str(stamp_a))
    assert all(not layer["key"].startswith("delta_") for layer in detached["cube"]["metric_maps"])
    assert explorer.metric_value_at(str(stamp_a), "delta_pixel_mse", row=2, col=3)["ok"] is False


def test_delta_overlay_paints_through_the_diverging_palette(tmp_path):
    """At full opacity a delta layer shows coolwarm colours over its symmetric range instead of viridis."""
    explorer, stamp_a, stamp_b = _delta_runs(tmp_path)
    delta = next(layer for layer in explorer.attach_second(str(stamp_a), str(stamp_b))["cube"]["metric_maps"] if layer["key"] == "delta_pixel_mse")

    png  = explorer.metric_overlay_png(str(stamp_a), "delta_pixel_mse", delta["vmin"], delta["vmax"], float("-inf"), float("inf"), 1.0)
    rgb  = plt.imread(io.BytesIO(png))[..., :3]
    data = np.load(stamp_a / "cubes" / "pixel_mse.npy") - np.load(stamp_b / "cubes" / "pixel_mse.npy")
    norm = np.clip((data - delta["vmin"]) / (delta["vmax"] - delta["vmin"]), 0.0, 1.0)

    assert np.allclose(rgb, plt.get_cmap("coolwarm")(norm)[..., :3], atol=1.0 / 255.0 + 1e-6)


def test_attaching_another_run_replaces_the_delta_layers(tmp_path):
    """Attaching a third run swaps the delta layers for A minus that run instead of stacking them."""
    explorer, stamp_a, stamp_b = _delta_runs(tmp_path)
    stamp_c                    = _make_run(tmp_path, tmp_path / "preproc", "run_c", seed=4)
    np.save(stamp_c / "cubes" / "pixel_mse.npy", np.full((N_ROWS, N_COLS), 0.25, dtype=np.float32))

    explorer.attach_second(str(stamp_a), str(stamp_b))
    attached = explorer.attach_second(str(stamp_a), str(stamp_c))

    keys = [layer["key"] for layer in attached["cube"]["metric_maps"] if layer["key"].startswith("delta_")]
    a    = np.load(stamp_a / "cubes" / "pixel_mse.npy")
    assert keys == ["delta_pixel_mse"]
    assert abs(explorer.metric_value_at(str(stamp_a), "delta_pixel_mse", row=1, col=1)["value"] - float(a[1, 1] - 0.25)) < 1e-6


def test_region_rows_label_delta_layers(tmp_path):
    """Region statistics name delta layers with the same delta label the layer list carries."""
    explorer, stamp_a, stamp_b = _delta_runs(tmp_path)
    explorer.attach_second(str(stamp_a), str(stamp_b))

    rows = {row["key"]: row for row in explorer.region_metrics(str(stamp_a), 0, 1, 0, 1)["rows"]}
    assert rows["delta_pixel_mse"]["label"] == "Δ MSE"


def test_reloading_a_cube_from_memory_drops_its_delta_layers(tmp_path):
    """Switching back to a cube kept in memory detaches its comparison, delta layers included."""
    explorer, stamp_a, stamp_b = _delta_runs(tmp_path)
    explorer.attach_second(str(stamp_a), str(stamp_b))

    load_cube(explorer, str(stamp_b))
    status = load_cube(explorer, str(stamp_a))

    assert status["cube"]["attached"] is None
    assert all(not layer["key"].startswith("delta_") for layer in status["cube"]["metric_maps"])
    assert "delta_pixel_mse" not in explorer.loaded["metric_maps"]


def test_attach_reports_a_comparison_stamp_without_metrics(tmp_path):
    """A comparison stamp missing its metrics.json is refused with the reason instead of raising."""
    explorer, cube_a, cube_b = _loaded(tmp_path)
    (Path(cube_b) / "metrics.json").unlink()

    result = explorer.attach_second(cube_a, cube_b)

    assert result["ok"] is False and "metrics.json missing" in result["error"]
    assert explorer.load_status()["cube"]["sources"] == ["pred", "gt"]


def test_attach_reports_an_unreadable_comparison_cube(tmp_path):
    """A comparison prediction cube that is not a valid npy file is refused with the reason instead of raising."""
    explorer, cube_a, cube_b = _loaded(tmp_path)
    (Path(cube_b) / "cubes" / "pred_curves.npy").write_bytes(b"not an npy file")

    result = explorer.attach_second(cube_a, cube_b)

    assert result["ok"] is False and result["error"]


def test_attach_route_answers_a_broken_comparison_with_an_error_status(tmp_path):
    """The /api/cubes/attach route turns a broken comparison stamp into a 400 answer naming the reason."""
    explorer, cube_a, cube_b = _loaded(tmp_path)
    (Path(cube_b) / "metrics.json").unlink()

    handler  = FakeHandler("POST", "/api/cubes/attach", json.dumps({"id": cube_a, "other": cube_b}).encode("utf-8"))
    exchange = HttpExchange.of(handler)
    exchange.read_body()

    assert CubeRouter(explorer).handle(exchange)
    assert handler.status == 400 and "metrics.json missing" in handler.payload()["error"]


def test_transect_metrics_samples_polyline(tmp_path):
    """Checks metric maps sample along the polyline with NaN turned into None and a degenerate line refused."""
    preproc = make_preproc(tmp_path)
    stamp   = _make_run(tmp_path, preproc, "run_a", seed=1)

    mse       = np.arange(N_ROWS * N_COLS, dtype=np.float32).reshape(N_ROWS, N_COLS)
    mse[0, 0] = np.nan
    np.save(stamp / "cubes" / "pixel_mse.npy", mse)

    explorer, _ = open_explorer(tmp_path, expected=1)
    load_cube(explorer, str(stamp))

    result = explorer.transect_metrics(str(stamp), [(0, 0), (0, 3), (2, 3)])
    assert result["ok"]
    assert result["n_samples"] == 6
    assert result["row"] == [0, 0, 0, 0, 1, 2]
    assert result["col"] == [0, 1, 2, 3, 3, 3]

    row = next(row for row in result["rows"] if row["key"] == "pixel_mse")
    assert row["label"] == "MSE"
    assert row["values"][0] is None
    assert row["values"][1] == float(mse[0, 1])

    assert explorer.transect_metrics(str(stamp), [(0, 0)])["ok"] is False
    assert explorer.transect_metrics(str(stamp), [(2, 2), (2, 2)])["ok"] is False
    assert explorer.transect_metrics(str(tmp_path / "nowhere"), [(0, 0), (1, 1)])["ok"] is False


def test_transect_metrics_carry_the_delta_layers_of_an_attached_run(tmp_path):
    """Checks an attached comparison's delta layers are sampled along the line under their delta label."""
    explorer, stamp_a, stamp_b = _delta_runs(tmp_path)
    explorer.attach_second(str(stamp_a), str(stamp_b))

    rows  = {row["key"]: row for row in explorer.transect_metrics(str(stamp_a), [(1, 0), (1, 2)])["rows"]}
    delta = np.load(stamp_a / "cubes" / "pixel_mse.npy") - np.load(stamp_b / "cubes" / "pixel_mse.npy")

    assert rows["delta_pixel_mse"]["label"] == "Δ MSE"
    assert np.allclose(rows["delta_pixel_mse"]["values"], delta[1, 0:3])


def _route(explorer: CubeExplorer, method: str, target: str, body: dict | None = None) -> FakeHandler:
    """Runs one request through the cube router and returns the handler holding the answer."""
    handler  = FakeHandler(method, target, json.dumps(body).encode("utf-8") if body is not None else b"")
    exchange = HttpExchange.of(handler)
    exchange.read_body()

    assert CubeRouter(explorer).handle(exchange)
    return handler


def test_transect_routes_take_the_polyline_in_pts(tmp_path):
    """Checks the transect, metric strip and save routes read the polyline from pts and refuse a degenerate one."""
    stamp = _make_run(tmp_path, make_preproc(tmp_path), "run_a", seed=1)
    np.save(stamp / "cubes" / "pixel_mse.npy", np.ones((N_ROWS, N_COLS), dtype=np.float32))

    explorer, _ = open_explorer(tmp_path, expected=1)
    cube_a      = str(stamp)
    load_cube(explorer, cube_a)

    bent  = urlencode({"id": cube_a, "source": "pred", "pts": "0,0;0,3;2,3"})
    png   = _route(explorer, "GET", f"/api/cubes/transect?{bent}")
    strip = _route(explorer, "GET", f"/api/cubes/transect_metrics?{bent}")

    assert png.status == 200 and png.wfile.getvalue()[:4] == b"\x89PNG"
    assert strip.status == 200 and strip.payload()["n_samples"] == 6

    assert _route(explorer, "GET", f"/api/cubes/transect?{urlencode({'id': cube_a, 'source': 'pred', 'pts': '1,1'})}").status == 404
    assert _route(explorer, "GET", f"/api/cubes/transect_metrics?{urlencode({'id': cube_a, 'pts': '1,1;1,1'})}").status == 400

    saved = _route(explorer, "POST", "/api/cubes/save_transect", {"id": cube_a, "pts": [[0, 0], [4, 0], [4, 5]]})
    assert saved.status == 200 and saved.payload()["rel"].endswith("_via1")


def test_transect_points_parser_refuses_malformed_vertices():
    """Checks the pts parser reads ordered pairs, treats an empty value as no vertex and raises on a malformed vertex."""
    assert CubeRouter._parse_points("0,1;2,3") == [(0, 1), (2, 3)]
    assert CubeRouter._parse_points("") == []

    for malformed in ("0,1;2", "0,1;a,3", "0,1,2", "0,1;"):
        with pytest.raises(ValueError):
            CubeRouter._parse_points(malformed)
