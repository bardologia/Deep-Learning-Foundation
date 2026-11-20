
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from configuration.cross_validation               import CrossValidationConfig, FoldConfig
from pipelines.shared.comparison.trial_collection import TrialRecord
from pipelines.cross_validation.cv_report         import CrossValidationReport
from pipelines.cross_validation.folds             import FoldNaming, FoldPlanner
from tools.metrics.scoring                        import SeedAggregation
from tools.reporting.markdown                     import ScalarFormatter

from tests.conftest import make_file_logger


N_FOLDS = 5

METRIC_VALUES = {
    "rmse_mean" : [1.0, 2.0, 3.0, 4.0, 5.0],
    "r2_mean" : [0.5, 0.6, 0.7, 0.8, 0.9],
}


def make_planner() -> FoldPlanner:
    config       = CrossValidationConfig()
    config.folds = FoldConfig(n_folds=N_FOLDS, row_start=1000, row_end=2000)
    return FoldPlanner(config, col_start=500, col_end=1000)


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
        index: {key: values[index] for key, values in METRIC_VALUES.items()}
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
        logger           = make_file_logger(tmp_path, "cv_report_test"),
    )


def test_aggregation_uses_the_shared_sample_std():
    mean, std = SeedAggregation.mean_std([1.0, 2.0, 3.0, 4.0, 5.0])

    assert mean == pytest.approx(3.0)
    assert std  == pytest.approx(float(np.std([1.0, 2.0, 3.0, 4.0, 5.0], ddof=1)))

    mean, std = SeedAggregation.mean_std([7.0])

    assert mean == pytest.approx(7.0)
    assert std  is None


def test_format_std_empty_when_too_few(tmp_path):
    report = make_report(tmp_path)

    assert report._format_std(0.3, 1) == "—"
    assert report._format_std(None, 1) == "—"
    assert report._format_std(float("nan"), 5) == "—"
    assert report._format_std(0.25, 5) != "—"


def test_format_std_applies_the_unit_scale(tmp_path):
    report = make_report(tmp_path)

    assert report._format_std(120.0, 5, 60.0) == report._format_std(2.0, 5)


def test_json_std_none_when_too_few(tmp_path):
    report = make_report(tmp_path)

    assert report._json_std(0.3, 1)            is None
    assert report._json_std(None, 5)           is None
    assert report._json_std(float("nan"), 5)   is None
    assert report._json_std(0.5, 5)            == 0.5


def test_seed_annotation_renders_the_dispersion_once():
    assert ScalarFormatter.with_std("1.5", None)          == "1.5"
    assert ScalarFormatter.with_std("1.5", float("nan"))  == "1.5"
    assert ScalarFormatter.with_std("1.5", 0.25)          == "1.5 ± 0.25"


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

    rmse = payload["splits"]["test"]["rmse_mean"]
    assert rmse["mean"]    == pytest.approx(3.0)
    assert rmse["std"]     == pytest.approx(float(np.std(METRIC_VALUES["rmse_mean"], ddof=1)))
    assert rmse["n_used"]  == N_FOLDS
    assert rmse["n_total"] == N_FOLDS


def test_summary_json_per_fold_breakdown(tmp_path):
    report = make_report(tmp_path)
    report.write_all()

    payload  = json.loads((report.out_dir / "cv_summary.json").read_text())
    per_fold = payload["splits"]["test"]["rmse_mean"]["per_fold"]

    for index in range(N_FOLDS):
        assert per_fold[FoldNaming.name(index)] == METRIC_VALUES["rmse_mean"][index]


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
    assert "`rmse_mean`"            in text


def test_aggregate_markdown_reports_correct_mean(tmp_path):
    report = make_report(tmp_path)
    report.write_all()

    text  = (report.out_dir / "cv_aggregate_report.md").read_text()
    lines = [line for line in text.splitlines() if "`rmse_mean`" in line]

    assert lines
    assert "3" in lines[0]


def test_aggregate_table_skips_non_numeric_metrics(tmp_path):
    metrics = {index: {"rmse_mean": float(index + 1), "label_str": "x"} for index in range(N_FOLDS)}
    report  = make_report(tmp_path, base_records=make_records(metrics), records_by_split={"test": make_records(metrics)})

    lines = report._aggregate_table(["rmse_mean", "label_str"], make_records(metrics))
    body  = "\n".join(lines)

    assert "`rmse_mean`" in body
    assert "`label_str`"     not in body


def _seed_dispersion(best_val_loss_std: float, metric_std: float, n_used: int = 2, missing: tuple = ()) -> dict:
    return {
        FoldNaming.name(index): {
            "n_seeds"           : 2,
            "status"            : "DONE",
            "best_val_loss_std" : best_val_loss_std,
            "splits"            : {"test": {"rmse_mean": metric_std}},
            "n_used"            : {"test": n_used},
            "missing_seeds"     : {"test": list(missing)},
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
        logger           = make_file_logger(tmp_path, "cv_report_test"),
        seed_dispersion  = _seed_dispersion(best_val_loss_std=0.05, metric_std=0.5),
    )

    assert report.has_seed_sweep is True

    report.write_all()

    text = (report.out_dir / "cv_aggregate_report.md").read_text()
    assert "± " in text
    assert "within-fold seed standard deviation" in text

    payload = json.loads((report.out_dir / "cv_summary.json").read_text())
    assert payload["seeds_per_fold"][FoldNaming.name(0)] == 2
    assert payload["splits"]["test"]["rmse_mean"]["per_fold_seed_std"][FoldNaming.name(0)] == pytest.approx(0.5)
    assert payload["best_val_loss"]["per_fold_seed_std"][FoldNaming.name(0)] == pytest.approx(0.05)


def test_no_seed_dispersion_keeps_summary_unchanged(tmp_path):
    report = make_report(tmp_path)
    report.write_all()

    payload = json.loads((report.out_dir / "cv_summary.json").read_text())

    assert report.has_seed_sweep is False
    assert "seeds_per_fold" not in payload
    assert "per_fold_seed_std" not in payload["splits"]["test"]["rmse_mean"]


def test_partial_fold_metrics_counted_in_n_used(tmp_path):
    metrics                     = split_metrics()
    metrics[3]["rmse_mean"] = float("nan")

    base = make_records(metrics)
    rbs  = make_records(metrics)

    report = make_report(tmp_path, base_records=base, records_by_split={"test": rbs})
    report.write_all()

    payload = json.loads((report.out_dir / "cv_summary.json").read_text())
    rmse    = payload["splits"]["test"]["rmse_mean"]

    assert rmse["n_used"]  == N_FOLDS - 1
    assert rmse["n_total"] == N_FOLDS


def test_seed_coverage_reaches_the_fold_headers_and_the_summary_json(tmp_path):
    metrics    = split_metrics()
    dispersion = _seed_dispersion(best_val_loss_std=0.05, metric_std=0.5, n_used=1, missing=("seed1",))
    report     = CrossValidationReport(
        base_records     = make_records(metrics),
        records_by_split = {"test": make_records(metrics)},
        planner          = make_planner(),
        out_dir          = tmp_path / "reports",
        model_name       = "resunet",
        embed_images     = False,
        logger           = make_file_logger(tmp_path, "cv_report_test"),
        seed_dispersion  = dispersion,
    )

    report.write_all()

    text    = (report.out_dir / "cv_aggregate_report.md").read_text()
    payload = json.loads((report.out_dir / "cv_summary.json").read_text())

    assert "`fold_0` (1/2 seeds)" in text
    assert payload["seeds_used_per_fold"][FoldNaming.name(0)] == {"test": 1}
    assert payload["missing_seeds"][FoldNaming.name(0)]       == {"test": ["seed1"]}
    assert payload["fold_status"][FoldNaming.name(0)]         == "DONE"


def test_training_table_carries_the_fold_status_and_seed_count(tmp_path):
    metrics = split_metrics()
    base    = make_records(metrics)
    base[2].training_result = {"duration_s": 180.0, "status": "PARTIAL"}

    swept  = CrossValidationReport(base_records=base, records_by_split={"test": make_records(metrics)}, planner=make_planner(), out_dir=tmp_path / "reports", model_name="resunet", embed_images=False, logger=make_file_logger(tmp_path, "cv_report_test"), seed_dispersion=_seed_dispersion(0.05, 0.5))
    plain  = make_report(tmp_path, base_records=base)

    swept_lines = "\n".join(swept._training_aggregate())
    plain_lines = "\n".join(plain._training_aggregate())

    assert "| Fold" in swept_lines and "| Seeds" in swept_lines and "| Status" in swept_lines
    assert "PARTIAL" in swept_lines
    assert "| Status" in plain_lines
    assert "| Seeds" not in plain_lines


def test_a_metric_non_finite_in_every_fold_is_named_when_it_is_left_out(tmp_path):
    from tests.conftest import RecordingLogger

    metrics = {index: {"rmse_mean": float(index + 1), "r2_mean": float("nan"), "mae_mean": float("nan") if index == 0 else 0.1} for index in range(N_FOLDS)}
    report  = make_report(tmp_path, base_records=make_records(metrics), records_by_split={"test": make_records(metrics)})
    logger  = RecordingLogger()

    report.logger = logger
    report.write_all()

    summary  = json.loads((tmp_path / "reports" / "cv_summary.json").read_text())
    warnings = [message for level, message in logger.messages if level == "warning"]

    assert "split test: ['r2_mean'] non-finite in every fold, omitted from the aggregate" in warnings
    assert "r2_mean" not in summary["splits"]["test"]
    assert summary["splits"]["test"]["mae_mean"]["n_used"] == N_FOLDS - 1


def test_summary_json_names_the_commit_that_wrote_it(tmp_path):
    from tools.runtime.git_state import GitState

    make_report(tmp_path).write_all()

    summary = json.loads((tmp_path / "reports" / "cv_summary.json").read_text())

    assert summary["provenance"]["git_commit"] == GitState.commit()
