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

H, AZ, RG    = 24, 12, 10
HEIGHT_RANGE = (-10.0, 30.0)
TRUE_MU      = 5.0
TRUE_SIGMA   = 3.0

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
    dataset = root / "ds"
    (dataset / "data").mkdir(parents=True)
    (dataset / "meta").mkdir()

    heights = np.linspace(HEIGHT_RANGE[0], HEIGHT_RANGE[1], H, dtype=np.float32)
    profile = np.exp(-((heights - TRUE_MU) ** 2) / (2.0 * TRUE_SIGMA ** 2)).astype(np.float32)
    cube    = np.tile(profile[:, None, None], (1, AZ, RG)) * np.linspace(0.5, 2.0, RG, dtype=np.float32)[None, None, :]

    np.save(dataset / "data" / "tomogram_full.npy", cube.astype(np.complex64))
    np.save(dataset / "data" / "primary.npy", np.ones((AZ, RG), dtype=np.complex64))

    state = {"tomogram_config": {"height_range": list(HEIGHT_RANGE)}}
    (dataset / "meta" / "config_state.json").write_text(json.dumps(state), encoding="utf-8")

    return dataset


def _load(lab: FitLab, dataset: Path) -> None:
    assert lab.start_load(str(dataset))["ok"]
    deadline = time.time() + 30
    while lab.load_status()["state"] == "loading" and time.time() < deadline:
        time.sleep(0.05)
    status = lab.load_status()
    assert status["state"] == "ready", status["error"]


def _fit(lab: FitLab, body: dict) -> dict:
    assert lab.start_fit(body)["ok"]
    deadline = time.time() + 120
    while lab.fit_status()["state"] == "fitting" and time.time() < deadline:
        time.sleep(0.05)
    status = lab.fit_status()
    assert status["state"] == "done", status["error"]
    return lab.fit_result_payload()


def test_datasets_listing(tmp_path):
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
    dataset = _make_dataset(tmp_path)
    lab     = FitLab(WebLogger())

    _load(lab, dataset)

    meta = lab.load_status()["meta"]
    assert meta["h"] == H and meta["az"] == AZ and meta["rg"] == RG
    assert meta["height_range"] == list(HEIGHT_RANGE)

    for src in ("slc", "peak"):
        png = lab.map_png(src)
        assert png.startswith(b"\x89PNG")
        assert struct.unpack(">II", png[16:24]) == (AZ, RG)
    assert lab.map_png("nope") is None


def test_load_error_on_missing_tomogram(tmp_path):
    lab = FitLab(WebLogger())
    assert lab.start_load(str(tmp_path))["ok"]

    deadline = time.time() + 30
    while lab.load_status()["state"] == "loading" and time.time() < deadline:
        time.sleep(0.05)

    status = lab.load_status()
    assert status["state"] == "error"
    assert "tomogram_full.npy" in status["error"]


def test_fit_request_validation(tmp_path):
    dataset = _make_dataset(tmp_path)
    lab     = FitLab(WebLogger())

    assert "no dataset" in lab.start_fit({"pixels": [{"az": 0, "rg": 0}], "config": BASE_CONFIG})["error"]

    _load(lab, dataset)

    assert "non-empty" in lab.start_fit({"pixels": [], "config": BASE_CONFIG})["error"]
    assert "outside"   in lab.start_fit({"pixels": [{"az": AZ, "rg": 0}], "config": BASE_CONFIG})["error"]
    assert "missing"   in lab.start_fit({"pixels": [{"az": 0, "rg": 0}], "config": {"k_max": 2}})["error"]
    assert "mode"      in lab.start_fit({"pixels": [{"az": 0, "rg": 0}], "config": {**BASE_CONFIG, "mode": "nope"}})["error"]
    assert "mode"      in lab.start_fit({"pixels": [{"az": 0, "rg": 0}], "config": {**BASE_CONFIG, "mode": ""}})["error"]
    assert "mode"      in lab.start_fit({"pixels": [{"az": 0, "rg": 0}], "config": {**BASE_CONFIG, "mode": "sigma_sigma"}})["error"]
    assert "k_max"     in lab.start_fit({"pixels": [{"az": 0, "rg": 0}], "config": {**BASE_CONFIG, "k_max": 99}})["error"]

    too_many = [{"az": 0, "rg": i % RG} for i in range(FitLab.MAX_PIXELS + 1)] + [{"az": 1, "rg": i % RG} for i in range(RG)]
    assert "at most" in lab.start_fit({"pixels": too_many, "config": BASE_CONFIG})["error"]


@pytest.mark.skipif(not _HAS_JAX, reason="jax not installed in this environment")
def test_fit_recovers_synthetic_gaussian(tmp_path):
    dataset = _make_dataset(tmp_path)
    lab     = FitLab(WebLogger())

    _load(lab, dataset)
    result = _fit(lab, {"pixels": [{"az": 3, "rg": 7}, {"az": 0, "rg": 0}], "config": BASE_CONFIG})

    assert result["ok"]
    assert len(result["height"]) == H
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
    dataset = _make_dataset(tmp_path)
    lab     = FitLab(WebLogger())

    _load(lab, dataset)
    sigma_only = _fit(lab, {"pixels": [{"az": 3, "rg": 7}], "config": BASE_CONFIG})
    all_free   = _fit(lab, {"pixels": [{"az": 3, "rg": 7}], "config": {**BASE_CONFIG, "mode": "sigma_amp_mu"}})

    assert sigma_only["config"]["mode"] == "sigma"
    assert all_free["config"]["mode"]   == "sigma_amp_mu"

    mse_sigma = sigma_only["pixels"][0]["per_k"][0]["mse"]
    mse_free  = all_free["pixels"][0]["per_k"][0]["mse"]
    assert mse_free <= mse_sigma + 1e-6


@pytest.mark.skipif(not _HAS_JAX, reason="jax not installed in this environment")
def test_fit_composition_freezes_sigma(tmp_path):
    dataset = _make_dataset(tmp_path)
    lab     = FitLab(WebLogger())

    _load(lab, dataset)
    result = _fit(lab, {"pixels": [{"az": 3, "rg": 7}], "config": {**BASE_CONFIG, "mode": "amp_mu"}})

    assert result["config"]["mode"] == "amp_mu"

    h_span     = HEIGHT_RANGE[1] - HEIGHT_RANGE[0]
    dh         = h_span / (H - 1)
    sigma_base = max(2.0 * dh, h_span / (8.0 * BASE_CONFIG["k_max"]))
    sigma_init = float(np.clip(sigma_base / BASE_CONFIG["sigma_init_divisor"], dh, h_span / 2.0))

    for row in result["pixels"][0]["per_k"]:
        for params in row["params"]:
            assert params["sigma"] == pytest.approx(sigma_init, abs=1e-5)
