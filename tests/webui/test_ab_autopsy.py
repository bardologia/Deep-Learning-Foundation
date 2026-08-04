from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ab_autopsy import AbAutopsy
from web_logger import WebLogger


N_ELEV, N_AZ, N_RG = 6, 32, 32


def _stamp(base: Path, name: str, mse: np.ndarray, rmse: float, x_axis: tuple = (-10.0, 30.0), n_elev: int = N_ELEV) -> Path:
    stamp = base / name / "inference" / "stamp"
    cubes = stamp / "cubes"
    cubes.mkdir(parents=True)

    (stamp / "metrics.json").write_text(json.dumps({
        "curve_rmse_gt" : rmse,
        "overall_r2_gt" : 1.0 - rmse,
        "x_axis_min"    : x_axis[0],
        "x_axis_max"    : x_axis[1],
        "n_pixels"      : N_AZ * N_RG,
    }), encoding="utf-8")

    rng = np.random.default_rng(hash(name) % 2**32)
    np.save(cubes / "pixel_mse.npy", mse.astype(np.float32))
    np.save(cubes / "pred_curves.npy", rng.random((n_elev, N_AZ, N_RG)).astype(np.float32))
    np.save(cubes / "gt_curves.npy",   rng.random((n_elev, N_AZ, N_RG)).astype(np.float32))

    return stamp


def _pair(tmp_path: Path) -> tuple[Path, Path]:
    mse_a              = np.full((N_AZ, N_RG), 0.1)
    mse_b              = np.full((N_AZ, N_RG), 0.1)
    mse_a[0:16, 0:16]  = 3.0
    mse_b[16:32, 0:16] = 3.0

    return _stamp(tmp_path, "run_a", mse_a, rmse=0.2), _stamp(tmp_path, "run_b", mse_b, rmse=0.4)


def test_compare_ranks_metric_gaps_and_finds_hotspots(tmp_path):
    stamp_a, stamp_b = _pair(tmp_path)

    out = AbAutopsy(WebLogger()).compare(str(stamp_a), str(stamp_b))

    assert out["ok"] is True
    assert out["run_a"] == "run_a" and out["run_b"] == "run_b"

    rmse_row = next(row for row in out["metrics"] if row["key"] == "curve_rmse_gt")
    assert rmse_row["winner"] == "A"
    assert "n_pixels" not in {row["key"] for row in out["metrics"]}

    winners = {spot["winner"] for spot in out["hotspots"]}
    assert winners == {"A", "B"}

    b_wins = [spot for spot in out["hotspots"] if spot["winner"] == "B"]
    assert all(spot["az0"] < 16 and spot["rg0"] < 16 for spot in b_wins)


def test_compare_rejects_region_mismatch(tmp_path):
    stamp_a, _ = _pair(tmp_path)
    stamp_c    = _stamp(tmp_path, "run_c", np.full((8, 8), 0.1), rmse=0.3)

    out = AbAutopsy(WebLogger()).compare(str(stamp_a), str(stamp_c))

    assert out["ok"] is False
    assert "different regions" in out["error"]


def test_compare_missing_metrics_fails_cleanly(tmp_path):
    out = AbAutopsy(WebLogger()).compare(str(tmp_path / "nope"), str(tmp_path / "nope2"))

    assert out["ok"] is False


def test_profile_returns_both_predictions(tmp_path):
    stamp_a, stamp_b = _pair(tmp_path)

    out = AbAutopsy(WebLogger()).profile(str(stamp_a), str(stamp_b), az=4, rg=5)

    assert out["ok"] is True
    assert len(out["a"]) == N_ELEV
    assert len(out["b"]) == N_ELEV
    assert len(out["gt"]) == N_ELEV
    assert out["x_axis"][0] == -10.0


def test_profile_outside_region_fails(tmp_path):
    stamp_a, stamp_b = _pair(tmp_path)

    assert AbAutopsy(WebLogger()).profile(str(stamp_a), str(stamp_b), az=999, rg=0)["ok"] is False


def test_profile_rejects_elevation_axis_mismatch(tmp_path):
    stamp_a, _ = _pair(tmp_path)
    stamp_d    = _stamp(tmp_path, "run_d", np.full((N_AZ, N_RG), 0.1), rmse=0.3, x_axis=(-5.0, 45.0))

    out = AbAutopsy(WebLogger()).profile(str(stamp_a), str(stamp_d), az=4, rg=5)

    assert out["ok"] is False
    assert "different elevation axes" in out["error"]


def test_profile_rejects_elevation_length_mismatch(tmp_path):
    stamp_a, _ = _pair(tmp_path)
    stamp_e    = _stamp(tmp_path, "run_e", np.full((N_AZ, N_RG), 0.1), rmse=0.3, n_elev=N_ELEV + 3)

    out = AbAutopsy(WebLogger()).profile(str(stamp_a), str(stamp_e), az=4, rg=5)

    assert out["ok"] is False
    assert "different cubes" in out["error"]


def test_runs_lists_only_comparable_stamps(tmp_path):
    complete = tmp_path / "backbone" / "run_a" / "inference" / "s1"
    (complete / "cubes").mkdir(parents=True)
    (complete / "cubes" / "pred_curves.npy").write_bytes(b"x")
    (complete / "cubes" / "pixel_mse.npy").write_bytes(b"x")
    (complete / "metrics.json").write_text("{}")

    bare = tmp_path / "backbone" / "run_b" / "inference" / "s2"
    (bare / "cubes").mkdir(parents=True)
    (bare / "cubes" / "pred_curves.npy").write_bytes(b"x")

    out = AbAutopsy(WebLogger()).runs(str(tmp_path))

    assert out["ok"] is True
    assert [run["id"] for run in out["runs"]] == [str(complete)]
    assert out["runs"][0]["stamp"] == "s1"


def test_runs_reports_root_errors(tmp_path):
    out = AbAutopsy(WebLogger()).runs(str(tmp_path / "nowhere"))

    assert out["ok"] is False
    assert out["runs"] == []
