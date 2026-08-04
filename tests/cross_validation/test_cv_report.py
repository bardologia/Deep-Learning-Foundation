from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from configuration.cross_validation               import CrossValidationConfig, FoldConfig
from pipelines.shared.comparison.trial_collection import TrialRecord
from pipelines.cross_validation.cv_report         import CrossValidationReport
from pipelines.cross_validation.folds             import FoldNaming, FoldPlanner
from tools.monitoring.logger                      import Logger


N_FOLDS = 5

CURVE_VALUES = {
    "curve_rmse_gt" : [1.0, 2.0, 3.0, 4.0, 5.0],
    "overall_r2_gt" : [0.5, 0.6, 0.7, 0.8, 0.9],
}


def make_logger(tmp_path: Path) -> Logger:
    return Logger(log_dir=str(tmp_path / "logs"), name="cv_report_test")


def make_planner() -> FoldPlanner:
    config       = CrossValidationConfig()
    config.folds = FoldConfig(n_folds=N_FOLDS, azimuth_start=1000, azimuth_end=2000)
    return FoldPlanner(config, range_start=500, range_end=1000)


def make_records(fold_index_to_metrics: dict[int, dict], with_checkpoint: bool = True) -> list[TrialRecord]:
    records = []

    for index in range(N_FOLDS):
        record = TrialRecord(name=FoldNaming.name(index), run_dir=Path(f"/tmp/folds/{FoldNaming.name(index)}"))

        record.metrics = fold_index_to_metrics.get(index, {})

        if with_checkpoint:
            record.checkpoint      = {"best_val_loss": 0.1 * (index + 1), "best_epoch": index + 1, "n_train_epochs": 10}
            record.training_result = {"duration_s": 60.0 * (index + 1), "status": "DONE"}

        records.append(record)

    return records


def split_metrics() -> dict[int, dict]:
    return {
        index: {key: values[index] for key, values in CURVE_VALUES.items()}
        for index in range(N_FOLDS)
    }


def make_report(tmp_path: Path, base_records=None, records_by_split=None) -> CrossValidationReport:
    metrics = split_metrics()

    base    = base_records     if base_records     is not None else make_records(metrics)
    splits  = records_by_split if records_by_split is not None else {"test": make_records(metrics)}

    return CrossValidationReport(
        base_records     = base,
        records_by_split = splits,
        planner          = make_planner(),
        out_dir          = tmp_path / "reports",
        model_name       = "resunet",
        embed_images     = False,
        logger           = make_logger(tmp_path),
    )


def test_mean_std_matches_numpy_sample_std(tmp_path):
    report     = make_report(tmp_path)
    mean, std  = report._mean_std([1.0, 2.0, 3.0, 4.0, 5.0])

    assert mean == pytest.approx(3.0)
    assert std  == pytest.approx(float(np.std([1.0, 2.0, 3.0, 4.0, 5.0], ddof=1)))


def test_mean_std_single_value_returns_nan_std(tmp_path):
    report    = make_report(tmp_path)
    mean, std = report._mean_std([7.0])

    assert mean == pytest.approx(7.0)
    assert math.isnan(std)


def test_format_std_empty_when_too_few(tmp_path):
    report = make_report(tmp_path)

    assert report._format_std(0.3, 1) == report._format_std(0.3, 1)
    assert report._format_std(0.3, 1) == "—"
    assert report._format_std(float("nan"), 5) == "—"
    assert report._format_std(0.25, 5) != "—"


def test_json_std_none_when_too_few(tmp_path):
    report = make_report(tmp_path)

    assert report._json_std(0.3, 1)            is None
    assert report._json_std(float("nan"), 5)   is None
    assert report._json_std(0.5, 5)            == 0.5


def test_write_all_creates_expected_files(tmp_path):
    report  = make_report(tmp_path)
    written = report.write_all()

    aggregate = report.out_dir / "cv_aggregate_report.md"
    summary   = report.out_dir / "cv_summary.json"

    assert aggregate.exists()
    assert summary.exists()
    assert aggregate in written
    assert summary   in written


def test_summary_json_mean_std_correct(tmp_path):
    report = make_report(tmp_path)
    report.write_all()

    payload = json.loads((report.out_dir / "cv_summary.json").read_text())

    assert payload["model"]   == "resunet"
    assert payload["n_folds"] == N_FOLDS
    assert payload["folds"]   == [FoldNaming.name(i) for i in range(N_FOLDS)]

    rmse = payload["splits"]["test"]["curve_rmse_gt"]
    assert rmse["mean"]    == pytest.approx(3.0)
    assert rmse["std"]     == pytest.approx(float(np.std(CURVE_VALUES["curve_rmse_gt"], ddof=1)))
    assert rmse["n_used"]  == N_FOLDS
    assert rmse["n_total"] == N_FOLDS


def test_summary_json_per_fold_breakdown(tmp_path):
    report = make_report(tmp_path)
    report.write_all()

    payload  = json.loads((report.out_dir / "cv_summary.json").read_text())
    per_fold = payload["splits"]["test"]["curve_rmse_gt"]["per_fold"]

    for index in range(N_FOLDS):
        assert per_fold[FoldNaming.name(index)] == CURVE_VALUES["curve_rmse_gt"][index]


def test_summary_json_best_val_loss_aggregate(tmp_path):
    report = make_report(tmp_path)
    report.write_all()

    payload = json.loads((report.out_dir / "cv_summary.json").read_text())
    losses  = [0.1 * (index + 1) for index in range(N_FOLDS)]

    assert payload["best_val_loss"]["mean"]   == pytest.approx(float(np.mean(losses)))
    assert payload["best_val_loss"]["std"]    == pytest.approx(float(np.std(losses, ddof=1)))
    assert payload["best_val_loss"]["n_used"] == N_FOLDS


def test_aggregate_markdown_has_structure(tmp_path):
    report = make_report(tmp_path)
    report.write_all()

    text = (report.out_dir / "cv_aggregate_report.md").read_text()

    assert text.startswith("# Cross-Validation Aggregate Report")
    assert "## Fold Plan"               in text
    assert "## Training Across Folds"   in text
    assert "## Split `test`"            in text
    assert "`curve_rmse_gt`"            in text


def test_aggregate_markdown_reports_correct_mean(tmp_path):
    report = make_report(tmp_path)
    report.write_all()

    text  = (report.out_dir / "cv_aggregate_report.md").read_text()
    lines = [line for line in text.splitlines() if "`curve_rmse_gt`" in line]

    assert lines
    assert "3" in lines[0]


def test_require_all_folds_rejects_missing_fold(tmp_path):
    metrics = split_metrics()
    short   = make_records(metrics)[:-1]

    report = make_report(tmp_path, base_records=short)

    with pytest.raises(ValueError, match="requires all"):
        report.write_all()


def test_require_all_folds_rejects_missing_checkpoint(tmp_path):
    metrics            = split_metrics()
    base               = make_records(metrics)
    base[2].checkpoint = {}

    report = make_report(tmp_path, base_records=base)

    with pytest.raises(ValueError, match="missing checkpoint"):
        report.write_all()


def test_require_all_folds_rejects_missing_split_metrics(tmp_path):
    metrics            = split_metrics()
    split              = make_records(metrics)
    split[1].metrics   = {}

    report = make_report(tmp_path, records_by_split={"test": split})

    with pytest.raises(ValueError, match="requires metrics for all"):
        report.write_all()


def test_aggregate_table_skips_non_numeric_metrics(tmp_path):
    metrics = {index: {"curve_rmse_gt": float(index + 1), "label_str": "x"} for index in range(N_FOLDS)}
    report  = make_report(tmp_path, base_records=make_records(metrics), records_by_split={"test": make_records(metrics)})

    lines = report._aggregate_table(["curve_rmse_gt", "label_str"], make_records(metrics))
    body  = "\n".join(lines)

    assert "`curve_rmse_gt`" in body
    assert "`label_str`"     not in body


def _seed_dispersion(best_val_loss_std: float, metric_std: float) -> dict:
    return {
        FoldNaming.name(index): {
            "n_seeds"           : 2,
            "best_val_loss_std" : best_val_loss_std,
            "splits"            : {"test": {"curve_rmse_gt": metric_std}},
        }
        for index in range(N_FOLDS)
    }


def test_seed_dispersion_annotates_markdown_and_json(tmp_path):
    metrics = split_metrics()
    report  = CrossValidationReport(
        base_records     = make_records(metrics),
        records_by_split = {"test": make_records(metrics)},
        planner          = make_planner(),
        out_dir          = tmp_path / "reports",
        model_name       = "resunet",
        embed_images     = False,
        logger           = make_logger(tmp_path),
        seed_dispersion  = _seed_dispersion(best_val_loss_std=0.05, metric_std=0.5),
    )

    assert report.has_seed_sweep is True

    report.write_all()

    text = (report.out_dir / "cv_aggregate_report.md").read_text()
    assert "± " in text
    assert "within-fold seed standard deviation" in text

    payload = json.loads((report.out_dir / "cv_summary.json").read_text())
    assert payload["seeds_per_fold"][FoldNaming.name(0)] == 2
    assert payload["splits"]["test"]["curve_rmse_gt"]["per_fold_seed_std"][FoldNaming.name(0)] == pytest.approx(0.5)
    assert payload["best_val_loss"]["per_fold_seed_std"][FoldNaming.name(0)] == pytest.approx(0.05)


def test_no_seed_dispersion_keeps_summary_unchanged(tmp_path):
    report = make_report(tmp_path)
    report.write_all()

    payload = json.loads((report.out_dir / "cv_summary.json").read_text())

    assert report.has_seed_sweep is False
    assert "seeds_per_fold" not in payload
    assert "per_fold_seed_std" not in payload["splits"]["test"]["curve_rmse_gt"]


def test_partial_fold_metrics_counted_in_n_used(tmp_path):
    metrics                     = split_metrics()
    metrics[3]["curve_rmse_gt"] = float("nan")

    base = make_records(metrics)
    rbs  = make_records(metrics)

    report = make_report(tmp_path, base_records=base, records_by_split={"test": rbs})
    report.write_all()

    payload = json.loads((report.out_dir / "cv_summary.json").read_text())
    rmse    = payload["splits"]["test"]["curve_rmse_gt"]

    assert rmse["n_used"]  == N_FOLDS - 1
    assert rmse["n_total"] == N_FOLDS
