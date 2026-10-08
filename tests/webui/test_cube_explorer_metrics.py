"""Tests for the cube explorer's metric layers: overlays, class legends, histograms, pixel readouts, colour bars and selective scoring."""

from __future__ import annotations

import io
from pathlib      import Path
from urllib.parse import quote

import matplotlib.pyplot as plt
import numpy as np

from cube_explorer        import CubeExplorer
from routers.cube_routers import CubeRouter
from routers.dispatch     import HttpExchange

from tests.webui.conftest import N_COLS, N_ROWS, FakeHandler, load_cube, loaded_cube, make_cube_run, open_explorer


def _loaded_explorer(base: Path) -> tuple[CubeExplorer, str]:
    return loaded_cube(base, with_metrics=True)


def _class_explorer(base: Path) -> tuple[CubeExplorer, str]:
    """Loads a cube whose stamp carries failure-mode, label-suspect and extract-fit maps beside the continuous layers."""
    stamp = make_cube_run(base, with_metrics=True)
    np.save(stamp / "cubes" / "failure_mode.npy", np.tile(np.arange(N_COLS) % 6, (N_ROWS, 1)).astype(np.float32))
    np.save(stamp / "cubes" / "label_suspect.npy", (np.arange(N_ROWS * N_COLS).reshape(N_ROWS, N_COLS) % 2).astype(np.float32))
    np.save(stamp / "cubes" / "extract_r2.npy", np.full((N_ROWS, N_COLS), 0.5, dtype=np.float32))

    explorer, cube_ids = open_explorer(base, expected=1)
    load_cube(explorer, cube_ids[0])

    return explorer, cube_ids[0]


def test_meta_lists_metric_layers(tmp_path):
    explorer, _ = _loaded_explorer(tmp_path)

    layers = explorer.load_status()["cube"]["metric_maps"]
    keys   = {layer["key"] for layer in layers}

    assert keys == {"pixel_r2", "label_r2"}
    assert all(np.isfinite(layer["vmin"]) and np.isfinite(layer["vmax"]) and layer["vmax"] > layer["vmin"] for layer in layers)
    assert next(layer for layer in layers if layer["key"] == "pixel_r2")["label"] == "R2"


def test_metric_overlay_png(tmp_path):
    explorer, cube_id = _loaded_explorer(tmp_path)

    png = explorer.metric_overlay_png(cube_id, "pixel_r2", vmin=0.0, vmax=1.0, keep_min=float("-inf"), keep_max=float("inf"), alpha=0.75)
    assert png and png[:4] == b"\x89PNG"

    thresholded = explorer.metric_overlay_png(cube_id, "pixel_r2", vmin=0.0, vmax=1.0, keep_min=0.5, keep_max=float("inf"), alpha=1.0)
    assert thresholded and thresholded[:4] == b"\x89PNG"

    degenerate = explorer.metric_overlay_png(cube_id, "pixel_r2", vmin=2.0, vmax=2.0, keep_min=float("-inf"), keep_max=float("inf"), alpha=0.5)
    assert degenerate and degenerate[:4] == b"\x89PNG"


def test_metric_overlay_greys_the_primary_with_the_cached_limits(tmp_path):
    """The overlay's grey base uses the primary colour limits cached at load, not percentiles recomputed per request."""
    explorer, cube_id = _loaded_explorer(tmp_path)

    primary = explorer.loaded["primary"]
    lo, hi  = float(np.median(primary)), float(np.median(primary)) + 1.0
    explorer.loaded["primary_clim"] = (lo, hi)

    png      = explorer.metric_overlay_png(cube_id, "pixel_r2", vmin=0.0, vmax=1.0, keep_min=float("-inf"), keep_max=float("inf"), alpha=0.0)
    grey     = plt.imread(io.BytesIO(png))[..., 0]
    expected = np.clip((primary - lo) / (hi - lo), 0.0, 1.0)

    assert grey.shape == primary.shape
    assert np.allclose(grey, expected, atol=1.0 / 255.0 + 1e-6)


def test_metric_overlay_rejects_unknown_key(tmp_path):
    explorer, cube_id = _loaded_explorer(tmp_path)

    assert explorer.metric_overlay_png(cube_id, "banana", 0.0, 1.0, float("-inf"), float("inf"), 0.75) is None
    assert explorer.metric_overlay_png(cube_id, "misshaped", 0.0, 1.0, float("-inf"), float("inf"), 0.75) is None
    assert explorer.metric_overlay_png("wrong", "pixel_r2", 0.0, 1.0, float("-inf"), float("inf"), 0.75) is None


def test_metric_value_at(tmp_path):
    explorer, cube_id = _loaded_explorer(tmp_path)

    stamp = Path(cube_id)
    r2    = np.load(stamp / "cubes" / "pixel_r2.npy")

    result = explorer.metric_value_at(cube_id, "pixel_r2", row=2, col=3)
    assert result["ok"] and result["value"] == float(r2[2, 3])

    nan_result = explorer.metric_value_at(cube_id, "pixel_r2", row=0, col=0)
    assert nan_result["ok"] and nan_result["value"] is None

    clipped = explorer.metric_value_at(cube_id, "pixel_r2", row=99, col=-1)
    assert clipped["ok"] and clipped["row"] == N_ROWS - 1 and clipped["col"] == 0

    assert not explorer.metric_value_at(cube_id, "banana", 0, 0)["ok"]


def test_cbar_png_whitelist(tmp_path):
    explorer, _ = _loaded_explorer(tmp_path)

    assert explorer.cbar_png("viridis")[:4] == b"\x89PNG"
    assert explorer.cbar_png("banana") is None


def test_metric_histogram(tmp_path):
    """Continuous layers bin over the robust range with the tails counted apart, and class layers count per class."""
    explorer, cube_id = _class_explorer(tmp_path)

    hist = explorer.metric_histogram(cube_id, "pixel_r2")
    assert hist["ok"] and hist["kind"] == "continuous"
    assert len(hist["edges"]) == len(hist["counts"]) + 1 == CubeExplorer.METRIC_HIST_BINS + 1
    assert sum(hist["counts"]) + hist["n_low"] + hist["n_high"] == N_ROWS * N_COLS - 1
    assert hist["n_low"] >= 1 and hist["n_high"] >= 1

    modes = explorer.metric_histogram(cube_id, "failure_mode")
    assert modes["ok"] and modes["kind"] == "classes"
    assert modes["names"][0] == "ok" and len(modes["counts"]) == 6
    assert sum(modes["counts"]) == N_ROWS * N_COLS

    assert explorer.metric_histogram(cube_id, "banana")["ok"] is False
    assert explorer.metric_histogram("wrong", "pixel_r2")["ok"] is False


def test_metric_hist_route_answers_through_the_cube_router(tmp_path):
    """The /api/cubes/metric_hist route answers a known layer with its counts and an unknown one with an error status."""
    explorer, cube_id = _class_explorer(tmp_path)
    router            = CubeRouter(explorer)

    known   = FakeHandler("GET", f"/api/cubes/metric_hist?id={quote(cube_id)}&key=failure_mode")
    unknown = FakeHandler("GET", f"/api/cubes/metric_hist?id={quote(cube_id)}&key=banana")

    assert router.handle(HttpExchange.of(known)) and router.handle(HttpExchange.of(unknown))
    assert known.status == 200 and known.payload()["counts"] == [N_ROWS] * 6
    assert unknown.status == 400


def test_metric_histogram_rejects_a_layer_without_finite_values(tmp_path):
    """A layer holding no finite value reports that instead of binning an empty array."""
    stamp = make_cube_run(tmp_path, with_metrics=True)
    np.save(stamp / "cubes" / "embedding_error.npy", np.full((N_ROWS, N_COLS), np.nan, dtype=np.float32))

    explorer, cube_ids = open_explorer(tmp_path, expected=1)
    load_cube(explorer, cube_ids[0])

    result = explorer.metric_histogram(cube_ids[0], "embedding_error")
    assert result["ok"] is False and "no finite value" in result["error"]


def test_selective_metrics_keep_low_confidence_share(tmp_path):
    explorer, cube_id = _loaded_explorer(tmp_path)

    full = explorer.selective_metrics(cube_id, "pixel_r2", coverage=1.0)
    half = explorer.selective_metrics(cube_id, "pixel_r2", coverage=0.5)

    assert full["ok"] and half["ok"]
    assert full["n_kept"]  == full["n_total"]
    assert half["n_kept"]  <  full["n_kept"]
    assert half["coverage"] <= 0.6

    r2_full = next(row for row in full["rows"] if row["key"] == "pixel_r2")
    r2_half = next(row for row in half["rows"] if row["key"] == "pixel_r2")

    assert full["direction"] == "high"
    assert r2_full["kept"] == r2_full["full"]
    assert r2_half["kept"] >= r2_half["full"]


def test_selective_metrics_reject_unknown_layer(tmp_path):
    explorer, cube_id = _loaded_explorer(tmp_path)

    assert explorer.selective_metrics(cube_id, "banana", coverage=0.5)["ok"] is False
    assert explorer.selective_metrics("wrong", "pixel_r2", coverage=0.5)["ok"] is False


def test_class_layers_carry_named_legend(tmp_path):
    """Class layers publish a named, coloured legend over their class index range while continuous layers carry none."""
    explorer, _ = _class_explorer(tmp_path)

    layers = {layer["key"]: layer for layer in explorer.load_status()["cube"]["metric_maps"]}

    assert layers["pixel_r2"]["classes"] is None
    assert [entry["name"] for entry in layers["label_suspect"]["classes"]] == ["clean", "suspect"]
    assert [entry["name"] for entry in layers["failure_mode"]["classes"]] == ["ok", "missed", "hallucinated", "position", "width", "amplitude"]
    assert layers["failure_mode"]["vmin"] == 0.0 and layers["failure_mode"]["vmax"] == 5.0
    assert all(entry["color"].startswith("#") for entry in layers["failure_mode"]["classes"])
    assert layers["extract_r2"]["label"] == "extract R2"


def test_class_layer_value_names_and_overlay(tmp_path):
    """Class layers answer point queries with the class name and still render overlays."""
    explorer, cube_id = _class_explorer(tmp_path)

    assert explorer.metric_value_at(cube_id, "failure_mode", row=0, col=2)["name"] == "hallucinated"
    assert explorer.metric_value_at(cube_id, "label_suspect", row=0, col=1)["name"] == "suspect"
    assert "name" not in explorer.metric_value_at(cube_id, "pixel_r2", row=2, col=3)

    png = explorer.metric_overlay_png(cube_id, "failure_mode", vmin=0.0, vmax=5.0, keep_min=float("-inf"), keep_max=float("inf"), alpha=0.9)
    assert png and png[:4] == b"\x89PNG"


def test_class_overlay_paints_each_class_in_its_legend_colour(tmp_path):
    """At full opacity every pixel of a class layer shows its class colour instead of a viridis shade."""
    explorer, cube_id = _class_explorer(tmp_path)

    png    = explorer.metric_overlay_png(cube_id, "failure_mode", vmin=0.0, vmax=5.0, keep_min=float("-inf"), keep_max=float("inf"), alpha=1.0)
    rgb    = plt.imread(io.BytesIO(png))[..., :3]
    legend = next(layer for layer in explorer.load_status()["cube"]["metric_maps"] if layer["key"] == "failure_mode")["classes"]

    for col in range(N_COLS):
        color    = legend[col % 6]["color"]
        expected = [int(color[j:j + 2], 16) / 255.0 for j in (1, 3, 5)]
        assert np.allclose(rgb[:, col], expected, atol=1.0 / 255.0 + 1e-6), col
