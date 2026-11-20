from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ab_autopsy import AbAutopsy
from web_logger import WebLogger


N_CHANNELS, N_ROWS, N_COLS = 2, 32, 32


def _stamp(base: Path, name: str, error: np.ndarray, rmse: float) -> Path:
    run_dir = base / name
    stamp   = run_dir / "inference" / "stamp"
    maps    = stamp / "maps"
    maps.mkdir(parents=True)
    (run_dir / "meta").mkdir()

    (run_dir / "meta" / "run_summary.json").write_text(json.dumps({"target_channels": ["first", "second"]}), encoding="utf-8")
    (stamp / "metrics.json").write_text(json.dumps({"rmse_mean": rmse, "r2_mean": 1.0 - rmse, "n_pixels": int(error.size)}), encoding="utf-8")

    targets = np.random.default_rng(0).normal(size=(N_CHANNELS, *error.shape))
    np.save(maps / "targets.npy", targets.astype(np.float32))
    np.save(maps / "predictions.npy", (targets + np.sqrt(error)[None]).astype(np.float32))

    return stamp


def _autopsy(tmp_path: Path) -> AbAutopsy:
    autopsy = AbAutopsy(WebLogger())
    autopsy.runs(str(tmp_path))
    return autopsy


def _pair(tmp_path: Path) -> tuple[Path, Path]:
    error_a              = np.full((N_ROWS, N_COLS), 0.1)
    error_b              = np.full((N_ROWS, N_COLS), 0.1)
    error_a[0:16, 0:16]  = 3.0
    error_b[16:32, 0:16] = 3.0

    return _stamp(tmp_path, "run_a", error_a, rmse=0.2), _stamp(tmp_path, "run_b", error_b, rmse=0.4)


def test_compare_ranks_metric_gaps_and_finds_hotspots(tmp_path):
    stamp_a, stamp_b = _pair(tmp_path)

    out = _autopsy(tmp_path).compare(str(stamp_a), str(stamp_b))

    assert out["ok"] is True
    assert out["run_a"] == "run_a" and out["run_b"] == "run_b"

    rmse_row = next(row for row in out["metrics"] if row["key"] == "rmse_mean")
    assert rmse_row["winner"] == "A"
    assert "n_pixels" not in {row["key"] for row in out["metrics"]}

    winners = {spot["winner"] for spot in out["hotspots"]}
    assert winners == {"A", "B"}

    b_wins = [spot for spot in out["hotspots"] if spot["winner"] == "B"]
    assert all(spot["row0"] < 16 and spot["col0"] < 16 for spot in b_wins)


def test_compare_rejects_region_mismatch(tmp_path):
    stamp_a, _ = _pair(tmp_path)
    stamp_c    = _stamp(tmp_path, "run_c", np.full((8, 8), 0.1), rmse=0.3)

    out = _autopsy(tmp_path).compare(str(stamp_a), str(stamp_c))

    assert out["ok"] is False
    assert "different regions" in out["error"]


def test_compare_missing_maps_fails_cleanly(tmp_path):
    out = _autopsy(tmp_path).compare(str(tmp_path / "nope"), str(tmp_path / "nope2"))

    assert out["ok"] is False


def test_pixel_returns_every_target_channel_of_both_predictions(tmp_path):
    stamp_a, stamp_b = _pair(tmp_path)

    out = _autopsy(tmp_path).pixel(str(stamp_a), str(stamp_b), row=4, col=5)

    assert out["ok"] is True
    assert out["channels"] == ["first", "second"]
    assert len(out["a"]) == len(out["b"]) == len(out["target"]) == N_CHANNELS
    assert abs(out["a"][0] - out["target"][0] - np.sqrt(3.0)) < 1e-5


def test_pixel_outside_region_fails(tmp_path):
    stamp_a, stamp_b = _pair(tmp_path)

    assert _autopsy(tmp_path).pixel(str(stamp_a), str(stamp_b), row=999, col=0)["ok"] is False


def test_pixel_rejects_a_map_shape_mismatch(tmp_path):
    stamp_a, _ = _pair(tmp_path)
    stamp_e    = _stamp(tmp_path, "run_e", np.full((8, 8), 0.1), rmse=0.3)

    out = _autopsy(tmp_path).pixel(str(stamp_a), str(stamp_e), row=4, col=5)

    assert out["ok"] is False
    assert "different maps" in out["error"]


def test_compare_refuses_a_stamp_outside_every_opened_root(tmp_path):
    stamp_a, stamp_b = _pair(tmp_path / "opened")
    outside          = _stamp(tmp_path / "elsewhere", "run_x", np.full((N_ROWS, N_COLS), 0.1), rmse=0.3)

    autopsy = AbAutopsy(WebLogger())
    autopsy.runs(str(tmp_path / "opened"))

    out = autopsy.compare(str(stamp_a), str(outside))

    assert out["ok"] is False
    assert "outside every catalogued runs root" in out["error"]
    assert autopsy.compare(str(stamp_a), str(stamp_b))["ok"] is True


def test_pixel_refuses_a_stamp_outside_every_opened_root(tmp_path):
    stamp_a, _ = _pair(tmp_path / "opened")
    outside    = _stamp(tmp_path / "elsewhere", "run_x", np.full((N_ROWS, N_COLS), 0.1), rmse=0.3)

    autopsy = AbAutopsy(WebLogger())
    autopsy.runs(str(tmp_path / "opened"))

    out = autopsy.pixel(str(outside), str(stamp_a), row=4, col=5)

    assert out["ok"] is False
    assert "outside every catalogued runs root" in out["error"]


def test_runs_lists_only_comparable_stamps(tmp_path):
    complete = tmp_path / "backbone" / "run_a" / "inference" / "s1"
    (complete / "maps").mkdir(parents=True)
    (complete / "maps" / "predictions.npy").write_bytes(b"x")
    (complete / "maps" / "targets.npy").write_bytes(b"x")
    (complete / "metrics.json").write_text("{}")

    bare = tmp_path / "backbone" / "run_b" / "inference" / "s2"
    (bare / "maps").mkdir(parents=True)
    (bare / "maps" / "predictions.npy").write_bytes(b"x")

    out = AbAutopsy(WebLogger()).runs(str(tmp_path))

    assert out["ok"] is True
    assert [run["id"] for run in out["runs"]] == [str(complete)]
    assert out["runs"][0]["stamp"] == "s1"


def test_runs_reports_root_errors(tmp_path):
    out = AbAutopsy(WebLogger()).runs(str(tmp_path / "nowhere"))

    assert out["ok"] is False
    assert out["runs"] == []
