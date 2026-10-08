"""Tests for the cube explorer's metric layers: overlays, pixel readouts, colour bars and selective scoring."""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cube_explorer import CubeExplorer

from tests.webui.conftest import N_ROWS, loaded_cube


def _loaded_explorer(base: Path) -> tuple[CubeExplorer, str]:
    return loaded_cube(base, with_metrics=True)


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
