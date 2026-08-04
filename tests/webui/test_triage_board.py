from __future__ import annotations

import json
from pathlib import Path
from types   import SimpleNamespace

import numpy as np

from triage_board import TriageBoard, TriageStore
from web_logger   import WebLogger


N_AZ, N_RG = 48, 32


def _board(tmp_path: Path) -> TriageBoard:
    paths = SimpleNamespace(logs_dir=tmp_path / "logs")
    return TriageBoard(paths, WebLogger())


def _stamp(tmp_path: Path, with_modes: bool = True, with_aux: bool = True) -> Path:
    stamp = tmp_path / "run" / "inference" / "stamp"
    cubes = stamp / "cubes"
    cubes.mkdir(parents=True)

    error            = np.full((N_AZ, N_RG), 0.1, dtype=np.float32)
    error[0:16, 0:16] = 5.0

    np.save(cubes / "pixel_mse.npy", error)

    if with_modes:
        modes              = np.zeros((N_AZ, N_RG), dtype=np.int8)
        modes[0:16, 0:16]  = 1
        np.save(cubes / "failure_mode.npy", modes)

    if with_aux:
        np.save(cubes / "label_r2.npy", np.full((N_AZ, N_RG), 0.8, dtype=np.float32))

    return stamp


def test_cases_rank_the_worst_block_first(tmp_path):
    stamp = _stamp(tmp_path)
    out   = _board(tmp_path).cases(str(stamp), top_n=10)

    assert out["ok"] is True
    assert out["has_modes"] is True
    assert out["aux"] == ["label_r2"]

    worst = out["cases"][0]
    assert worst["az0"] == 0 and worst["rg0"] == 0
    assert worst["mse_mean"] > out["cases"][-1]["mse_mean"]
    assert worst["mode"] == "missed"
    assert worst["fail_frac"] == 1.0
    assert worst["label_r2"] == np.float32(0.8)
    assert 0 <= worst["worst_az"] < 16 and 0 <= worst["worst_rg"] < 16


def test_cases_without_optional_layers_still_work(tmp_path):
    stamp = _stamp(tmp_path, with_modes=False, with_aux=False)
    out   = _board(tmp_path).cases(str(stamp), top_n=5)

    assert out["ok"] is True
    assert out["has_modes"] is False
    assert out["aux"] == []
    assert "mode" not in out["cases"][0]


def test_cases_require_the_error_cube(tmp_path):
    stamp = tmp_path / "run" / "inference" / "empty"
    (stamp / "cubes").mkdir(parents=True)

    out = _board(tmp_path).cases(str(stamp))

    assert out["ok"] is False
    assert "pixel_mse" in out["error"]


def test_annotations_persist_and_clear(tmp_path):
    stamp = _stamp(tmp_path)
    board = _board(tmp_path)

    saved = board.annotate({"id": str(stamp), "case": "0_0", "verdict": "label problem", "note": "fit is garbage here"})
    assert saved["ok"] is True

    reloaded = board.cases(str(stamp), top_n=10)
    worst    = reloaded["cases"][0]
    assert worst["annotation"]["verdict"] == "label problem"
    assert worst["annotation"]["note"]    == "fit is garbage here"

    cleared = board.annotate({"id": str(stamp), "case": "0_0", "verdict": "", "note": ""})
    assert cleared["ok"] is True
    assert board.cases(str(stamp), top_n=10)["cases"][0]["annotation"] is None


def test_unknown_verdict_is_rejected(tmp_path):
    store = TriageStore(tmp_path / "logs" / "triage")

    out = store.annotate("cube", "0_0", "who knows", "")

    assert out["ok"] is False


def _curves(stamp: Path, n_elev: int = 6) -> None:
    cubes = stamp / "cubes"
    rng   = np.random.default_rng(0)

    np.save(cubes / "pred_curves.npy", rng.random((n_elev, N_AZ, N_RG)).astype(np.float32))
    np.save(cubes / "gt_curves.npy",   rng.random((n_elev, N_AZ, N_RG)).astype(np.float32))

    (stamp / "metrics.json").write_text(json.dumps({"x_axis_min": -10.0, "x_axis_max": 30.0}), encoding="utf-8")


def test_thumb_renders_a_png_crop(tmp_path):
    stamp = _stamp(tmp_path)
    png   = _board(tmp_path).thumb(str(stamp), az0=0, rg0=0)

    assert png is not None
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_thumb_refuses_bad_blocks(tmp_path):
    stamp = _stamp(tmp_path)
    board = _board(tmp_path)

    assert board.thumb(str(stamp), az0=N_AZ + 16, rg0=0) is None
    assert board.thumb(str(tmp_path / "nowhere"), az0=0, rg0=0) is None


def test_profile_returns_gt_and_prediction(tmp_path):
    stamp = _stamp(tmp_path)
    _curves(stamp)

    out = _board(tmp_path).profile(str(stamp), az=3, rg=4)

    assert out["ok"] is True
    assert len(out["x_axis"]) == 6
    assert len(out["gt"])     == 6
    assert len(out["pred"])   == 6
    assert out["x_axis"][0] == -10.0
    assert out["x_axis"][-1] == 30.0


def test_profile_outside_region_fails(tmp_path):
    stamp = _stamp(tmp_path)
    _curves(stamp)

    out = _board(tmp_path).profile(str(stamp), az=N_AZ + 5, rg=0)

    assert out["ok"] is False
    assert "outside" in out["error"]


def test_profile_without_curves_fails_cleanly(tmp_path):
    stamp = _stamp(tmp_path)

    out = _board(tmp_path).profile(str(stamp), az=0, rg=0)

    assert out["ok"] is False
