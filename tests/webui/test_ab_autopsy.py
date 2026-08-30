"""Tests for the A/B autopsy tab that compares two inference stamps.

Covers metric ranking and error hotspot detection between two runs, per-pixel
profile retrieval, the guards that reject mismatched regions or elevation axes,
the containment check keeping stamps inside catalogued run roots, and the
listing that admits only stamps carrying the cubes a comparison needs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ab_autopsy import AbAutopsy
from web_logger import WebLogger


N_Z, N_ROWS, N_COLS = 6, 32, 32


def _stamp(base: Path, name: str, mse: np.ndarray, rmse: float, z_axis: tuple = (-10.0, 30.0), n_z: int = N_Z) -> Path:
    """Writes an inference stamp with a given error map and metrics.

    Args:
        base   : Parent directory receiving ``<name>/inference/stamp``.
        name   : Run name, also seeding the random curve cubes.
        mse    : Per-pixel squared error map of shape (row, col).
        rmse   : Curve RMSE recorded in metrics.json.
        z_axis : Z axis bounds in metres as (min, max).
        n_z : Number of z samples in the curve cubes.

    Returns:
        Path of the created stamp directory.
    """
    stamp = base / name / "inference" / "stamp"
    cubes = stamp / "cubes"
    cubes.mkdir(parents=True)

    (stamp / "metrics.json").write_text(json.dumps({
        "curve_rmse_gt" : rmse,
        "overall_r2_gt" : 1.0 - rmse,
        "x_axis_min"    : z_axis[0],
        "x_axis_max"    : z_axis[1],
        "n_pixels"      : N_ROWS * N_COLS,
    }), encoding="utf-8")

    rng = np.random.default_rng(hash(name) % 2**32)
    np.save(cubes / "pixel_mse.npy", mse.astype(np.float32))
    np.save(cubes / "pred_curves.npy", rng.random((n_z, N_ROWS, N_COLS)).astype(np.float32))
    np.save(cubes / "gt_curves.npy",   rng.random((n_z, N_ROWS, N_COLS)).astype(np.float32))

    return stamp


def _autopsy(tmp_path: Path) -> AbAutopsy:
    """Returns an autopsy instance with ``tmp_path`` catalogued as a runs root."""
    autopsy = AbAutopsy(WebLogger())
    autopsy.runs(str(tmp_path))
    return autopsy


def _pair(tmp_path: Path) -> tuple[Path, Path]:
    """Writes two stamps whose high-error quadrants sit in complementary row halves.

    Args:
        tmp_path : Parent directory receiving both runs.

    Returns:
        Paths of the run_a and run_b stamps, with run_a the better overall RMSE.
    """
    mse_a              = np.full((N_ROWS, N_COLS), 0.1)
    mse_b              = np.full((N_ROWS, N_COLS), 0.1)
    mse_a[0:16, 0:16]  = 3.0
    mse_b[16:32, 0:16] = 3.0

    return _stamp(tmp_path, "run_a", mse_a, rmse=0.2), _stamp(tmp_path, "run_b", mse_b, rmse=0.4)


def test_compare_ranks_metric_gaps_and_finds_hotspots(tmp_path):
    """Checks compare names the better run per metric, drops n_pixels and locates each run's hotspots."""
    stamp_a, stamp_b = _pair(tmp_path)

    out = _autopsy(tmp_path).compare(str(stamp_a), str(stamp_b))

    assert out["ok"] is True
    assert out["run_a"] == "run_a" and out["run_b"] == "run_b"

    rmse_row = next(row for row in out["metrics"] if row["key"] == "curve_rmse_gt")
    assert rmse_row["winner"] == "A"
    assert "n_pixels" not in {row["key"] for row in out["metrics"]}

    winners = {spot["winner"] for spot in out["hotspots"]}
    assert winners == {"A", "B"}

    b_wins = [spot for spot in out["hotspots"] if spot["winner"] == "B"]
    assert all(spot["row0"] < 16 and spot["col0"] < 16 for spot in b_wins)


def test_compare_rejects_region_mismatch(tmp_path):
    """Checks comparing stamps of different scene sizes reports a region mismatch."""
    stamp_a, _ = _pair(tmp_path)
    stamp_c    = _stamp(tmp_path, "run_c", np.full((8, 8), 0.1), rmse=0.3)

    out = _autopsy(tmp_path).compare(str(stamp_a), str(stamp_c))

    assert out["ok"] is False
    assert "different regions" in out["error"]


def test_compare_missing_metrics_fails_cleanly(tmp_path):
    """Checks comparing nonexistent stamps returns a failure rather than raising."""
    out = _autopsy(tmp_path).compare(str(tmp_path / "nope"), str(tmp_path / "nope2"))

    assert out["ok"] is False


def test_profile_returns_both_predictions(tmp_path):
    """Checks a pixel profile returns both predictions, the ground truth and the height axis."""
    stamp_a, stamp_b = _pair(tmp_path)

    out = _autopsy(tmp_path).profile(str(stamp_a), str(stamp_b), row=4, col=5)

    assert out["ok"] is True
    assert len(out["a"]) == N_Z
    assert len(out["b"]) == N_Z
    assert len(out["gt"]) == N_Z
    assert out["z_axis"][0] == -10.0


def test_profile_outside_region_fails(tmp_path):
    """Checks an azimuth index beyond the scene is rejected."""
    stamp_a, stamp_b = _pair(tmp_path)

    assert _autopsy(tmp_path).profile(str(stamp_a), str(stamp_b), row=999, col=0)["ok"] is False


def test_profile_rejects_z_axis_mismatch(tmp_path):
    """Checks stamps with different z axis bounds cannot be profiled together."""
    stamp_a, _ = _pair(tmp_path)
    stamp_d    = _stamp(tmp_path, "run_d", np.full((N_ROWS, N_COLS), 0.1), rmse=0.3, z_axis=(-5.0, 45.0))

    out = _autopsy(tmp_path).profile(str(stamp_a), str(stamp_d), row=4, col=5)

    assert out["ok"] is False
    assert "different z axes" in out["error"]


def test_profile_rejects_z_length_mismatch(tmp_path):
    """Checks stamps with different numbers of z samples cannot be profiled together."""
    stamp_a, _ = _pair(tmp_path)
    stamp_e    = _stamp(tmp_path, "run_e", np.full((N_ROWS, N_COLS), 0.1), rmse=0.3, n_z=N_Z + 3)

    out = _autopsy(tmp_path).profile(str(stamp_a), str(stamp_e), row=4, col=5)

    assert out["ok"] is False
    assert "different cubes" in out["error"]


def test_compare_refuses_a_stamp_outside_every_opened_root(tmp_path):
    """Checks compare rejects a stamp outside the catalogued roots while still allowing catalogued pairs."""
    stamp_a, stamp_b = _pair(tmp_path / "opened")
    outside          = _stamp(tmp_path / "elsewhere", "run_x", np.full((N_AZ, N_RG), 0.1), rmse=0.3)

    autopsy = AbAutopsy(WebLogger())
    autopsy.runs(str(tmp_path / "opened"))

    out = autopsy.compare(str(stamp_a), str(outside))

    assert out["ok"] is False
    assert "outside every catalogued runs root" in out["error"]
    assert autopsy.compare(str(stamp_a), str(stamp_b))["ok"] is True


def test_profile_refuses_a_stamp_outside_every_opened_root(tmp_path):
    """Checks profile rejects a stamp outside the catalogued roots."""
    stamp_a, _ = _pair(tmp_path / "opened")
    outside    = _stamp(tmp_path / "elsewhere", "run_x", np.full((N_AZ, N_RG), 0.1), rmse=0.3)

    autopsy = AbAutopsy(WebLogger())
    autopsy.runs(str(tmp_path / "opened"))

    out = autopsy.profile(str(outside), str(stamp_a), az=4, rg=5)

    assert out["ok"] is False
    assert "outside every catalogued runs root" in out["error"]


def test_runs_lists_only_comparable_stamps(tmp_path):
    """Checks the listing keeps stamps with metrics and both cubes and drops incomplete ones."""
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
    """Checks scanning a missing root returns a failure with an empty run list."""
    out = AbAutopsy(WebLogger()).runs(str(tmp_path / "nowhere"))

    assert out["ok"] is False
    assert out["runs"] == []
