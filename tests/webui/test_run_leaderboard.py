from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from run_leaderboard import RunAxes, RunLeaderboard
from web_logger      import WebLogger


STANDARD_NAME = "resunet-conv-sorted_gt-K_5-hvn-none-param_l1_1_20260617_210314"

METRICS = {
    "curve_mse_gt"                : 0.5,
    "overall_r2_gt"               : 0.8,
    "pixel_r2_gt_median"          : 0.9,
    "fraction_pred_beats_reduced" : 0.7,
    "data_consistency_status"     : "evaluated",
    "n_pixels"                    : 150000,
}


def _make_run(base: Path, name: str, stamp: str = "20260701_000000", metrics: dict | None = None, trainer: dict | None = None, summary: dict | None = None) -> Path:
    run       = base / name
    stamp_dir = run / "inference" / stamp
    stamp_dir.mkdir(parents=True)

    (stamp_dir / "metrics.json").write_text(json.dumps(metrics if metrics is not None else METRICS))

    if trainer is not None:
        (run / "docs").mkdir(exist_ok=True)
        (run / "docs" / "trainer_config.json").write_text(json.dumps(trainer))

    if summary is not None:
        (run / "meta").mkdir(exist_ok=True)
        (run / "meta" / "run_summary.json").write_text(json.dumps(summary))

    return stamp_dir


def test_axes_parse_standard_name():
    axes = RunAxes.parse(STANDARD_NAME)

    assert axes["model"]     == "resunet"
    assert axes["head"]      == "conv"
    assert axes["matching"]  == "sorted_gt"
    assert axes["k"]         == 5
    assert axes["aug"]       == "hvn"
    assert axes["presence"]  == "none"
    assert axes["loss"]      == "param_l1_1"
    assert axes["losses"]    == [{"name": "param_l1", "weight": 1.0}]
    assert axes["timestamp"] == "20260617_210314"
    assert axes["suffix"]    == ""


def test_axes_parse_multi_loss_and_suffix():
    axes = RunAxes.parse("unet-set_pred-hungarian-K_3-noaug-AB-param_l1_1-coherence_resyn_0.5_20260701_120000_local")

    assert axes["presence"] == "AB"
    assert axes["losses"]   == [{"name": "param_l1", "weight": 1.0}, {"name": "coherence_resyn", "weight": 0.5}]
    assert axes["suffix"]   == "local"


def test_axes_parse_scientific_weight():
    axes = RunAxes.parse("resunet-conv-sorted_gt-K_5-hv-A-covariance_match_1e-05_20260701_120000")

    assert axes["losses"] == [{"name": "covariance_match", "weight": 1e-05}]


def test_axes_reject_non_standard_names():
    assert RunAxes.parse("smoke_image_ae") is None
    assert RunAxes.parse("resunet-conv-sorted_gt-banana-hvn-none-param_l1_1_20260617_210314") is None
    assert RunAxes.parse("resunet-conv-sorted_gt-K_5-hvn-none-notaloss_20260617_210314") is None


def test_table_lists_runs_with_axes_and_metrics(tmp_path):
    _make_run(tmp_path, STANDARD_NAME)
    _make_run(tmp_path, "oddly_named_run")

    board  = RunLeaderboard(WebLogger())
    result = board.table(str(tmp_path))

    assert result["ok"] and len(result["rows"]) == 2
    assert not result["errors"]
    assert {c["key"] for c in result["columns"]} >= {"curve_mse_gt", "overall_r2_gt"}

    by_run = {row["run"]: row for row in result["rows"]}

    standard = by_run[STANDARD_NAME]
    assert standard["axes"]["model"] == "resunet"
    assert standard["metrics"]["curve_mse_gt"] == 0.5
    assert "data_consistency_status" not in standard["metrics"]
    assert "n_pixels" not in standard["metrics"]
    assert "fraction_pred_beats_reduced" not in standard["metrics"]

    assert by_run["oddly_named_run"]["axes"] is None


def test_table_lists_every_inference_stamp(tmp_path):
    _make_run(tmp_path, STANDARD_NAME, stamp="20260701_000000")
    _make_run(tmp_path, STANDARD_NAME, stamp="20260702_000000")

    result = RunLeaderboard(WebLogger()).table(str(tmp_path))

    assert result["ok"]
    assert sorted(row["stamp"] for row in result["rows"]) == ["20260701_000000", "20260702_000000"]


def test_table_reports_unreadable_metrics(tmp_path):
    stamp_dir = _make_run(tmp_path, STANDARD_NAME)
    (stamp_dir / "metrics.json").write_text("{not json")

    result = RunLeaderboard(WebLogger()).table(str(tmp_path))

    assert result["ok"] and not result["rows"]
    assert len(result["errors"]) == 1


def test_table_drops_non_finite_metrics(tmp_path):
    run       = tmp_path / STANDARD_NAME
    stamp_dir = run / "inference" / "20260701_000000"
    stamp_dir.mkdir(parents=True)
    (stamp_dir / "metrics.json").write_text('{"curve_mse_gt": NaN, "overall_r2_gt": 0.8}')

    result = RunLeaderboard(WebLogger()).table(str(tmp_path))

    assert result["ok"]
    assert "curve_mse_gt" not in result["rows"][0]["metrics"]
    assert result["rows"][0]["metrics"]["overall_r2_gt"] == 0.8


def test_table_rejects_bad_base(tmp_path):
    board = RunLeaderboard(WebLogger())

    assert not board.table("")["ok"]
    assert not board.table("relative/path")["ok"]
    assert not board.table(str(tmp_path / "missing"))["ok"]


def test_diff_returns_metrics_configs_and_directions(tmp_path):
    a = _make_run(tmp_path, STANDARD_NAME, metrics={"curve_mse_gt": 0.5, "overall_r2_gt": 0.8}, trainer={"optimizer": {"lr": 0.001}, "epochs": 10}, summary={"model_name": "resunet"})
    b = _make_run(tmp_path, "unet-conv-sorted_gt-K_5-hvn-none-param_l1_1_20260618_210314", metrics={"curve_mse_gt": 0.4, "overall_r2_gt": 0.85}, trainer={"optimizer": {"lr": 0.01}, "epochs": 10}, summary={"model_name": "unet"})

    board = RunLeaderboard(WebLogger())
    assert board.table(str(tmp_path))["ok"]

    result = board.diff([str(a), str(b)])
    assert result["ok"]

    assert result["sides"][0]["metrics"]["curve_mse_gt"] == 0.5
    assert result["sides"][1]["metrics"]["curve_mse_gt"] == 0.4
    assert result["directions"]["curve_mse_gt"]  == -1
    assert result["directions"]["overall_r2_gt"] == 1

    assert result["sides"][0]["config"]["trainer.optimizer.lr"] == 0.001
    assert result["sides"][1]["config"]["trainer.optimizer.lr"] == 0.01
    assert result["sides"][0]["config"]["summary.model_name"]   == "resunet"
    assert result["sides"][0]["config"]["trainer.epochs"]       == 10

    assert result["sections"] == [{"title": "Curve-Level", "keys": ["curve_mse_gt", "overall_r2_gt"]}]


def test_diff_includes_the_family_specific_model_config(tmp_path):
    names = ["dual_unet_skip-set_pred-hungarian-K_2-hv-A-full.full-param_l1_1_20260727_120000",
             "dual_resunet-set_pred-hungarian-K_2-hv-A-full.full-param_l1_1_20260727_130000"]
    stamps = []

    for name, backbone in zip(names, ("unet_skip", "resunet")):
        stamp = _make_run(tmp_path, name, metrics={"curve_mse_gt": 0.5})
        meta  = tmp_path / name / "meta"
        meta.mkdir(exist_ok=True)
        (meta / "dual_model_config.json").write_text(json.dumps({"params_backbone": backbone, "existence_backbone": backbone}))
        stamps.append(stamp)

    board = RunLeaderboard(WebLogger())
    assert board.table(str(tmp_path))["ok"]

    result = board.diff([str(s) for s in stamps])
    assert result["ok"]

    assert result["sides"][0]["config"]["model.params_backbone"] == "unet_skip"
    assert result["sides"][1]["config"]["model.params_backbone"] == "resunet"
    assert not any("model config" in note for note in result["sides"][0]["config_notes"])


def test_diff_notes_a_missing_model_config(tmp_path):
    stamps = [_make_run(tmp_path, name, metrics={"curve_mse_gt": 0.5}) for name in (STANDARD_NAME, "unet-conv-sorted_gt-K_5-hvn-none-param_l1_1_20260618_210314")]

    board = RunLeaderboard(WebLogger())
    assert board.table(str(tmp_path))["ok"]

    result = board.diff([str(s) for s in stamps])

    assert result["ok"]
    assert any("no model config under meta/" in note for note in result["sides"][0]["config_notes"])


def test_diff_compares_many_runs_and_bounds_count(tmp_path):
    stamps = [
        _make_run(tmp_path, name, metrics={"curve_mse_gt": 0.1 * (i + 1)})
        for i, name in enumerate([
            "unet-conv-sorted_gt-K_5-hvn-none-param_l1_1_20260618_210314",
            "unet-conv-sorted_gt-K_5-hvn-none-param_l1_1_20260618_210315",
            "unet-conv-sorted_gt-K_5-hvn-none-param_l1_1_20260618_210316",
        ])
    ]

    board = RunLeaderboard(WebLogger())
    assert board.table(str(tmp_path))["ok"]

    result = board.diff([str(s) for s in stamps])
    assert result["ok"]
    assert [side["metrics"]["curve_mse_gt"] for side in result["sides"]] == [pytest.approx(0.1), pytest.approx(0.2), pytest.approx(0.3)]

    assert not board.diff([str(stamps[0])])["ok"]
    assert not board.diff([str(stamps[0])] * 7)["ok"]


def test_diff_requires_scanned_root(tmp_path):
    stamp = _make_run(tmp_path, STANDARD_NAME)

    board  = RunLeaderboard(WebLogger())
    result = board.diff([str(stamp), str(stamp)])

    assert not result["ok"]


def test_diff_rejects_paths_outside_root(tmp_path):
    inside  = tmp_path / "runs"
    outside = tmp_path / "elsewhere"
    stamp_in  = _make_run(inside, STANDARD_NAME)
    stamp_out = _make_run(outside, STANDARD_NAME)

    board = RunLeaderboard(WebLogger())
    assert board.table(str(inside))["ok"]

    assert board.diff([str(stamp_in), str(stamp_in)])["ok"]
    assert not board.diff([str(stamp_in), str(stamp_out)])["ok"]


def _make_seed_run(base: Path, experiment: str, unit: str, seed: int, stamp: str, metrics: dict) -> Path:
    stamp_dir = base / experiment / unit / f"seed{seed}" / "inference" / stamp
    stamp_dir.mkdir(parents=True)
    (stamp_dir / "metrics.json").write_text(json.dumps(metrics))
    return stamp_dir


def test_table_axes_fall_back_to_ancestor_names(tmp_path):
    unit = "resunet-conv-sorted_gt-K_5-hvn-none-param_l1_1"
    _make_seed_run(tmp_path, "exp", unit, 3, "20260701_000000", {"curve_mse_gt": 0.5})

    nested = tmp_path / "exp2" / unit / "arm_a" / "warm" / "inference" / "20260701_000000"
    nested.mkdir(parents=True)
    (nested / "metrics.json").write_text(json.dumps({"curve_mse_gt": 0.4}))

    board = RunLeaderboard(WebLogger())
    rows  = board.table(str(tmp_path))["rows"]

    for run_name in ("seed3", "warm"):
        row = next(r for r in rows if r["run"] == run_name)
        assert row["axes"]["model"] == "resunet"
        assert row["axes"]["k"]     == 5
        assert row["axes"]["loss"]  == "param_l1_1"


def test_columns_include_ssim_components():
    keys = [c["key"] for c in RunLeaderboard.COLUMNS]
    assert "ssim_gt_azimuth_mean" in keys
    assert "ssim_gt_elev_mean" in keys
    assert "ssim_gt_range_mean" in keys


def test_trials_aggregates_seeds(tmp_path):
    _make_seed_run(tmp_path, "presence_matrix", "unit-A", 0, "20260701_000000", {"curve_mse_gt": 1.0, "overall_r2_gt": 0.5})
    _make_seed_run(tmp_path, "presence_matrix", "unit-A", 1, "20260701_000000", {"curve_mse_gt": 3.0, "overall_r2_gt": 0.7})
    _make_seed_run(tmp_path, "presence_matrix", "unit-B", 0, "20260701_000000", {"curve_mse_gt": 2.0})
    _make_run(tmp_path, STANDARD_NAME)

    board  = RunLeaderboard(WebLogger())
    result = board.trials(str(tmp_path))

    assert result["ok"] and len(result["experiments"]) == 1

    experiment = result["experiments"][0]
    assert experiment["key"] == "presence_matrix"
    assert [u["unit"] for u in experiment["units"]] == ["unit-A", "unit-B"]

    unit_a = experiment["units"][0]
    assert unit_a["seeds"] == [0, 1]
    assert unit_a["metrics"]["curve_mse_gt"]["mean"] == 2.0
    assert abs(unit_a["metrics"]["curve_mse_gt"]["std"] - 1.4142135623730951) < 1e-12
    assert unit_a["metrics"]["curve_mse_gt"]["n"] == 2
    assert unit_a["metrics"]["overall_r2_gt"]["n"] == 2

    unit_b = experiment["units"][1]
    assert unit_b["metrics"]["curve_mse_gt"] == {"mean": 2.0, "std": 0.0, "n": 1}
    assert "overall_r2_gt" not in unit_b["metrics"]


def test_trials_uses_latest_stamp_per_seed(tmp_path):
    older = _make_seed_run(tmp_path, "exp", "unit", 0, "20260701_000000", {"curve_mse_gt": 9.0})
    newer = _make_seed_run(tmp_path, "exp", "unit", 0, "20260702_000000", {"curve_mse_gt": 1.0})

    os.utime(older, (1000, 1000))
    os.utime(newer, (2000, 2000))

    result = RunLeaderboard(WebLogger()).trials(str(tmp_path))

    unit = result["experiments"][0]["units"][0]
    assert unit["metrics"]["curve_mse_gt"] == {"mean": 1.0, "std": 0.0, "n": 1}


def test_trials_ignores_unseeded_runs(tmp_path):
    _make_run(tmp_path, STANDARD_NAME)

    result = RunLeaderboard(WebLogger()).trials(str(tmp_path))

    assert result["ok"] and result["experiments"] == []


def test_table_adds_seed_unit_rows(tmp_path):
    _make_seed_run(tmp_path, "exp", "unit-A", 0, "20260701_000000", {"curve_mse_gt": 1.0, "overall_r2_gt": 0.5})
    _make_seed_run(tmp_path, "exp", "unit-A", 1, "20260701_000000", {"curve_mse_gt": 3.0})
    _make_run(tmp_path, STANDARD_NAME)

    rows = RunLeaderboard(WebLogger()).table(str(tmp_path))["rows"]

    unit = next(row for row in rows if row.get("n_seeds"))
    assert unit["run"]     == "unit-A"
    assert unit["group"]   == "exp"
    assert unit["stamp"]   == "mean of 2 seeds"
    assert unit["n_seeds"] == 2
    assert unit["metrics"]["curve_mse_gt"]  == 2.0
    assert unit["metrics"]["overall_r2_gt"] == 0.5
    assert Path(unit["id"]) == tmp_path / "exp" / "unit-A"

    assert sum(1 for row in rows if row.get("n_seeds")) == 1
    assert len(rows) == 4


def test_table_unit_row_uses_latest_stamp_per_seed(tmp_path):
    older = _make_seed_run(tmp_path, "exp", "unit", 0, "20260701_000000", {"curve_mse_gt": 9.0})
    newer = _make_seed_run(tmp_path, "exp", "unit", 0, "20260702_000000", {"curve_mse_gt": 1.0})

    os.utime(older, (1000, 1000))
    os.utime(newer, (2000, 2000))

    rows = RunLeaderboard(WebLogger()).table(str(tmp_path))["rows"]

    unit = next(row for row in rows if row.get("n_seeds"))
    assert unit["metrics"]["curve_mse_gt"] == 1.0
    assert unit["mtime"] == 2000


def test_diff_accepts_unit_dirs_as_seed_means(tmp_path):
    _make_seed_run(tmp_path, "exp", "unit-A", 0, "20260701_000000", {"curve_mse_gt": 1.0})
    _make_seed_run(tmp_path, "exp", "unit-A", 1, "20260701_000000", {"curve_mse_gt": 3.0})
    stamp = _make_run(tmp_path, STANDARD_NAME, metrics={"curve_mse_gt": 0.5}, trainer={"seed": 99})

    seed0 = tmp_path / "exp" / "unit-A" / "seed0"
    (seed0 / "docs").mkdir()
    (seed0 / "docs" / "trainer_config.json").write_text(json.dumps({"seed": 0}))

    board = RunLeaderboard(WebLogger())
    assert board.table(str(tmp_path))["ok"]

    result = board.diff([str(tmp_path / "exp" / "unit-A"), str(stamp)])
    assert result["ok"]

    unit_side, stamp_side = result["sides"]
    assert unit_side["run"]     == "unit-A"
    assert unit_side["stamp"]   == "mean of 2 seeds"
    assert unit_side["n_seeds"] == 2
    assert unit_side["metrics"]["curve_mse_gt"] == 2.0
    assert unit_side["config"]["trainer.seed"]  == 0

    assert stamp_side["metrics"]["curve_mse_gt"] == 0.5


def test_diff_rejects_dir_without_metrics_or_seeds(tmp_path):
    _make_run(tmp_path, STANDARD_NAME)
    bare = tmp_path / "bare"
    bare.mkdir()

    board = RunLeaderboard(WebLogger())
    assert board.table(str(tmp_path))["ok"]

    result = board.diff([str(bare), str(bare)])
    assert not result["ok"]
    assert "no seeded inference results" in result["error"]


def test_direction_heuristic():
    assert RunLeaderboard._direction("pixel_r2_gt_mean")            == 1
    assert RunLeaderboard._direction("relative_mse_reduction")      == 1
    assert RunLeaderboard._direction("improvement_pixel_mse_mean")  == 1
    assert RunLeaderboard._direction("curve_mse_gt")                == -1
    assert RunLeaderboard._direction("pixel_peak_err_units_p95_gt") == -1
    assert RunLeaderboard._direction("gt_std")                      == 0


def test_table_sorted_by_mtime_desc(tmp_path):
    older = _make_run(tmp_path, STANDARD_NAME, stamp="20260701_000000")
    newer = _make_run(tmp_path, "unet-conv-sorted_gt-K_5-hvn-none-param_l1_1_20260618_210314", stamp="20260702_000000")

    os.utime(older, (1000, 1000))
    os.utime(newer, (2000, 2000))

    result = RunLeaderboard(WebLogger()).table(str(tmp_path))

    assert [row["stamp"] for row in result["rows"]] == ["20260702_000000", "20260701_000000"]


def test_axes_parse_dual_input_token():
    axes = RunAxes.parse("dual_resunet-set_pred-hungarian-K_2-hv-A-full.ifg-param_l1_1_20260726_120000")

    assert axes["model"]  == "dual_resunet"
    assert axes["inputs"] == "full.ifg"
    assert axes["loss"]   == "param_l1_1"
    assert axes["losses"] == [{"name": "param_l1", "weight": 1.0}]


def test_axes_parse_trunk_aware_dual_model_token():
    axes = RunAxes.parse("dual_unet_skip-set_pred-hungarian-K_2-hv-A-full.full-param_l1_1_20260727_120000")

    assert axes["model"]  == "dual_unet_skip"
    assert axes["inputs"] == "full.full"

    mixed = RunAxes.parse("dual_unet_skip.local_cnn-set_pred-hungarian-K_2-hv-A-full.ifg-param_l1_1_20260727_120000")

    assert mixed["model"] == "dual_unet_skip.local_cnn"


def test_axes_parse_input_token_absent_on_single_trunk_names():
    axes = RunAxes.parse(STANDARD_NAME)

    assert axes["inputs"] == ""
