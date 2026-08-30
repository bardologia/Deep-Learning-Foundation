"""Tests for the Fit Lab dataset browser, tomogram loading and Gaussian fitting.

Covers dataset discovery, asynchronous cube loading, PNG preview maps, fit request
validation and the JAX-backed multi-Gaussian fit across its parameter-freezing modes.
"""

from __future__ import annotations

import importlib.util
import json
import struct
import time
from pathlib import Path

import numpy as np
import pytest

from fit_lab    import FitLab
from web_logger import WebLogger


_HAS_JAX = importlib.util.find_spec("jax") is not None

H, N_ROWS, N_COLS    = 24, 12, 10
Z_RANGE    = (-10.0, 30.0)
TRUE_MU    = 5.0
TRUE_SIGMA = 3.0

BASE_CONFIG = {
    "k_max"              : 2,
    "lambda_k"           : 0.01,
    "mode"               : "sigma",
    "threshold_factor"   : 0.25,
    "truncation_index"   : 170,
    "prominence_frac"    : 0.05,
    "sigma_init_divisor" : 4.0,
    "activity_threshold" : 1e-3,
    "adam_steps"         : 200,
    "adam_lr"            : 0.2,
}


def _make_dataset(root: Path) -> Path:
    """Writes a synthetic dataset whose tomogram holds one Gaussian per pixel.

    Args:
        root : Directory the dataset folder is created inside.

    Returns:
        Path of the dataset directory holding data/ and meta/ subfolders.
    """
    dataset = root / "ds"
    (dataset / "data").mkdir(parents=True)
    (dataset / "meta").mkdir()

    z_axis  = np.linspace(Z_RANGE[0], Z_RANGE[1], H, dtype=np.float32)
    profile = np.exp(-((z_axis - TRUE_MU) ** 2) / (2.0 * TRUE_SIGMA ** 2)).astype(np.float32)
    cube    = np.tile(profile[:, None, None], (1, N_ROWS, N_COLS)) * np.linspace(0.5, 2.0, N_COLS, dtype=np.float32)[None, None, :]

    np.save(dataset / "data" / "cube_full.npy", cube.astype(np.complex64))
    np.save(dataset / "data" / "primary.npy", np.ones((N_ROWS, N_COLS), dtype=np.complex64))

    state = {"cube_config": {"z_range": list(Z_RANGE)}}
    (dataset / "meta" / "config_state.json").write_text(json.dumps(state), encoding="utf-8")

    return dataset


def _load(lab: FitLab, dataset: Path) -> None:
    """Starts a dataset load and blocks until the lab reports the ready state."""
    assert lab.start_load(str(dataset))["ok"]
    deadline = time.time() + 30
    while lab.load_status()["state"] == "loading" and time.time() < deadline:
        time.sleep(0.05)
    status = lab.load_status()
    assert status["state"] == "ready", status["error"]


def _fit(lab: FitLab, body: dict) -> dict:
    """Starts a fit, waits for completion and returns the fit result payload."""
    assert lab.start_fit(body)["ok"]
    deadline = time.time() + 120
    while lab.fit_status()["state"] == "fitting" and time.time() < deadline:
        time.sleep(0.05)
    status = lab.fit_status()
    assert status["state"] == "done", status["error"]
    return lab.fit_result_payload()


def test_datasets_listing(tmp_path):
    """Dataset listing finds children of a parent path, accepts a dataset path directly and rejects blank or missing paths."""
    dataset = _make_dataset(tmp_path)
    lab     = FitLab(WebLogger())

    listing = lab.datasets(str(tmp_path))
    assert listing["ok"]
    assert [d["path"] for d in listing["datasets"]] == [str(dataset)]

    direct = lab.datasets(str(dataset))
    assert [d["name"] for d in direct["datasets"]] == ["ds"]

    assert not lab.datasets("")["ok"]
    assert not lab.datasets(str(tmp_path / "missing"))["ok"]


def test_load_meta_and_maps(tmp_path):
    """A loaded dataset reports cube dimensions and height range, and renders SLC and peak PNG maps sized (az, rg)."""
    dataset = _make_dataset(tmp_path)
    lab     = FitLab(WebLogger())

    _load(lab, dataset)

    meta = lab.load_status()["meta"]
    assert meta["h"] == H and meta["row"] == N_ROWS and meta["col"] == N_COLS
    assert meta["z_range"] == list(Z_RANGE)

    for src in ("primary", "peak"):
        png = lab.map_png(src)
        assert png.startswith(b"\x89PNG")
        assert struct.unpack(">II", png[16:24]) == (N_ROWS, N_COLS)
    assert lab.map_png("nope") is None


def test_load_error_on_missing_cube(tmp_path):
    """Loading a directory without cube_full.npy ends in the error state naming the missing file."""
    lab = FitLab(WebLogger())
    assert lab.start_load(str(tmp_path))["ok"]

    deadline = time.time() + 30
    while lab.load_status()["state"] == "loading" and time.time() < deadline:
        time.sleep(0.05)

    status = lab.load_status()
    assert status["state"] == "error"
    assert "cube_full.npy" in status["error"]


def test_fit_request_validation(tmp_path):
    """Fit requests are rejected when no dataset is loaded, pixels are empty, out of bounds or too many, or the config misses keys or names an unknown mode or oversized k_max."""
    dataset = _make_dataset(tmp_path)
    lab     = FitLab(WebLogger())

    assert "no dataset" in lab.start_fit({"pixels": [{"row": 0, "col": 0}], "config": BASE_CONFIG})["error"]

    _load(lab, dataset)

    assert "non-empty" in lab.start_fit({"pixels": [], "config": BASE_CONFIG})["error"]
    assert "outside"   in lab.start_fit({"pixels": [{"row": N_ROWS, "col": 0}], "config": BASE_CONFIG})["error"]
    assert "missing"   in lab.start_fit({"pixels": [{"row": 0, "col": 0}], "config": {"k_max": 2}})["error"]
    assert "mode"      in lab.start_fit({"pixels": [{"row": 0, "col": 0}], "config": {**BASE_CONFIG, "mode": "nope"}})["error"]
    assert "mode"      in lab.start_fit({"pixels": [{"row": 0, "col": 0}], "config": {**BASE_CONFIG, "mode": ""}})["error"]
    assert "mode"      in lab.start_fit({"pixels": [{"row": 0, "col": 0}], "config": {**BASE_CONFIG, "mode": "sigma_sigma"}})["error"]
    assert "k_max"     in lab.start_fit({"pixels": [{"row": 0, "col": 0}], "config": {**BASE_CONFIG, "k_max": 99}})["error"]

    too_many = [{"row": 0, "col": i % N_COLS} for i in range(FitLab.MAX_PIXELS + 1)] + [{"row": 1, "col": i % N_COLS} for i in range(N_COLS)]
    assert "at most" in lab.start_fit({"pixels": too_many, "config": BASE_CONFIG})["error"]


@pytest.mark.skipif(not _HAS_JAX, reason="jax not installed in this environment")
def test_fit_recovers_synthetic_gaussian(tmp_path):
    """The fit returns one row per k with full-length curves, a penalised score of mse plus lambda_k times k, and recovers the injected Gaussian mean."""
    dataset = _make_dataset(tmp_path)
    lab     = FitLab(WebLogger())

    _load(lab, dataset)
    result = _fit(lab, {"pixels": [{"row": 3, "col": 7}, {"row": 0, "col": 0}], "config": BASE_CONFIG})

    assert result["ok"]
    assert len(result["z"]) == H
    assert len(result["pixels"]) == 2

    pixel = result["pixels"][0]
    assert pixel["active"]
    assert len(pixel["per_k"]) == BASE_CONFIG["k_max"]
    assert 1 <= pixel["best_k"] <= BASE_CONFIG["k_max"]

    for row in pixel["per_k"]:
        assert len(row["params"]) == row["k"]
        assert len(row["total"]) == H
        assert len(row["components"]) == row["k"]
        assert all(len(comp) == H for comp in row["components"])
        assert row["penalised"] == pytest.approx(row["mse"] + BASE_CONFIG["lambda_k"] * row["k"])

    best = pixel["per_k"][pixel["best_k"] - 1]
    assert best["mse"] < 0.01
    assert abs(best["params"][0]["mu"] - TRUE_MU) < 2.5


@pytest.mark.skipif(not _HAS_JAX, reason="jax not installed in this environment")
def test_fit_modes_share_kernel(tmp_path):
    """Freeing amplitude and mean alongside sigma never yields a worse MSE than the sigma-only mode."""
    dataset = _make_dataset(tmp_path)
    lab     = FitLab(WebLogger())

    _load(lab, dataset)
    sigma_only = _fit(lab, {"pixels": [{"row": 3, "col": 7}], "config": BASE_CONFIG})
    all_free   = _fit(lab, {"pixels": [{"row": 3, "col": 7}], "config": {**BASE_CONFIG, "mode": "sigma_amp_mu"}})

    assert sigma_only["config"]["mode"] == "sigma"
    assert all_free["config"]["mode"]   == "sigma_amp_mu"

    mse_sigma = sigma_only["pixels"][0]["per_k"][0]["mse"]
    mse_free  = all_free["pixels"][0]["per_k"][0]["mse"]
    assert mse_free <= mse_sigma + 1e-6


@pytest.mark.skipif(not _HAS_JAX, reason="jax not installed in this environment")
def test_fit_composition_freezes_sigma(tmp_path):
    """The amp_mu mode leaves every fitted sigma at the initialiser value derived from the height spacing."""
    dataset = _make_dataset(tmp_path)
    lab     = FitLab(WebLogger())

    _load(lab, dataset)
    result = _fit(lab, {"pixels": [{"row": 3, "col": 7}], "config": {**BASE_CONFIG, "mode": "amp_mu"}})

    assert result["config"]["mode"] == "amp_mu"

    h_span     = Z_RANGE[1] - Z_RANGE[0]
    dh         = h_span / (H - 1)
    sigma_base = max(2.0 * dh, h_span / (8.0 * BASE_CONFIG["k_max"]))
    sigma_init = float(np.clip(sigma_base / BASE_CONFIG["sigma_init_divisor"], dh, h_span / 2.0))

    for row in result["pixels"][0]["per_k"]:
        for params in row["params"]:
            assert params["sigma"] == pytest.approx(sigma_init, abs=1e-5)
