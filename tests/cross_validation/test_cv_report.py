"""Tests for the cross-validation aggregate report and its summary JSON."""

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

CURVE_VALUES = {
    "curve_rmse_gt" : [1.0, 2.0, 3.0, 4.0, 5.0],
    "overall_r2_gt" : [0.5, 0.6, 0.7, 0.8, 0.9],
}


def make_planner() -> FoldPlanner:
    """Returns a fold planner over a 1000-line row window split into N_FOLDS blocks."""
    config       = CrossValidationConfig()
    config.folds = FoldConfig(n_folds=N_FOLDS, row_start=1000, row_end=2000)
    return FoldPlanner(config, col_start=500, col_end=1000)


def make_records(fold_index_to_metrics: dict[int, dict], with_checkpoint: bool = True) -> list[TrialRecord]:
    """Builds one TrialRecord per fold with the given metrics.

    Args:
        fold_index_to_metrics : Metric dictionaries keyed by fold index.
        with_checkpoint       : Whether to attach checkpoint and training-result payloads.

    Returns:
        One record per fold in fold-index order.
    """
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
    """Returns the per-fold metric dictionaries derived from CURVE_VALUES."""
    return {
        index: {key: values[index] for key, values in CURVE_VALUES.items()}
        for index in range(N_FOLDS)
    }


def make_report(tmp_path: Path, base_records=None, records_by_split=None) -> CrossValidationReport:
    """Builds a CrossValidationReport writing under a temporary directory.

    Args:
        tmp_path         : Directory that receives the reports/ output tree.
        base_records     : Training records per fold; defaults to synthetic complete ones.
        records_by_split : Per-split records; defaults to a single test split.

    Returns:
        The configured report object.
    """
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
    """Verifies the shared aggregation returns the sample standard deviation and None for one value."""
    mean, std = SeedAggregation.mean_std([1.0, 2.0, 3.0, 4.0, 5.0])

    assert mean == pytest.approx(3.0)
    assert std  == pytest.approx(float(np.std([1.0, 2.0, 3.0, 4.0, 5.0], ddof=1)))

    mean, std = SeedAggregation.mean_std([7.0])

    assert mean == pytest.approx(7.0)
    assert std  is None


def test_format_std_empty_when_too_few(tmp_path):
    """Verifies the dispersion string is blank for too few folds or a non-finite value."""
    report = make_report(tmp_path)

    assert report._format_std(0.3, 1) == "—"
    assert report._format_std(None, 1) == "—"
    assert report._format_std(float("nan"), 5) == "—"
    assert report._format_std(0.25, 5) != "—"


def test_format_std_applies_the_unit_scale(tmp_path):
    """Verifies the dispersion string is divided by the unit scale before formatting."""
    report = make_report(tmp_path)

    assert report._format_std(120.0, 5, 60.0) == report._format_std(2.0, 5)


def test_json_std_none_when_too_few(tmp_path):
    """Verifies the JSON dispersion is None for too few folds or a non-finite value."""
    report = make_report(tmp_path)

    assert report._json_std(0.3, 1)            is None
    assert report._json_std(None, 5)           is None
    assert report._json_std(float("nan"), 5)   is None
    assert report._json_std(0.5, 5)            == 0.5


def test_seed_annotation_renders_the_dispersion_once():
    """Verifies the scalar formatter appends the dispersion only when it is finite."""
    assert ScalarFormatter.with_std("1.5", None)          == "1.5"
    assert ScalarFormatter.with_std("1.5", float("nan"))  == "1.5"
    assert ScalarFormatter.with_std("1.5", 0.25)          == "1.5 ± 0.25"


def test_write_all_creates_expected_files(tmp_path):
    """Verifies write_all emits and reports the aggregate markdown and summary JSON."""
    report  = make_report(tmp_path)
    written = report.write_all()

    aggregate = report.out_dir / "cv_aggregate_report.md"
    summary   = report.out_dir / "cv_summary.json"

    assert aggregate.exists()
    assert summary.exists()
    assert aggregate in written
    assert summary   in written


def test_summary_json_mean_std_correct(tmp_path):
    """Verifies the summary JSON carries the model, fold names and the metric mean and dispersion."""
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
    """Verifies the summary JSON keeps the per-fold metric values."""
    report = make_report(tmp_path)
    report.write_all()

    payload  = json.loads((report.out_dir / "cv_summary.json").read_text())
    per_fold = payload["splits"]["test"]["curve_rmse_gt"]["per_fold"]

    for index in range(N_FOLDS):
        assert per_fold[FoldNaming.name(index)] == CURVE_VALUES["curve_rmse_gt"][index]


def test_summary_json_best_val_loss_aggregate(tmp_path):
    """Verifies the best validation losses are aggregated across folds in the summary JSON."""
    report = make_report(tmp_path)
    report.write_all()

    payload = json.loads((report.out_dir / "cv_summary.json").read_text())
    losses  = [0.1 * (index + 1) for index in range(N_FOLDS)]

    assert payload["best_val_loss"]["mean"]   == pytest.approx(float(np.mean(losses)))
    assert payload["best_val_loss"]["std"]    == pytest.approx(float(np.std(losses, ddof=1)))
    assert payload["best_val_loss"]["n_used"] == N_FOLDS


def test_aggregate_markdown_has_structure(tmp_path):
    """Verifies the aggregate markdown carries the expected section headings."""
    report = make_report(tmp_path)
    report.write_all()

    text = (report.out_dir / "cv_aggregate_report.md").read_text()

    assert text.startswith("# Cross-Validation Aggregate Report")
    assert "## Fold Plan"               in text
    assert "## Training Across Folds"   in text
    assert "## Split `test`"            in text
    assert "`curve_rmse_gt`"            in text


def test_aggregate_markdown_reports_correct_mean(tmp_path):
    """Verifies the metric row of the aggregate markdown shows the mean across folds."""
    report = make_report(tmp_path)
    report.write_all()

    text  = (report.out_dir / "cv_aggregate_report.md").read_text()
    lines = [line for line in text.splitlines() if "`curve_rmse_gt`" in line]

    assert lines
    assert "3" in lines[0]


def test_require_all_folds_rejects_missing_fold(tmp_path):
    """Verifies a missing fold record aborts report writing."""
    metrics = split_metrics()
    short   = make_records(metrics)[:-1]

    report = make_report(tmp_path, base_records=short)

    with pytest.raises(ValueError, match="requires all"):
        report.write_all()


def test_require_all_folds_rejects_missing_checkpoint(tmp_path):
    """Verifies a fold without a checkpoint payload aborts report writing."""
    metrics            = split_metrics()
    base               = make_records(metrics)
    base[2].checkpoint = {}

    report = make_report(tmp_path, base_records=base)

    with pytest.raises(ValueError, match="missing checkpoint"):
        report.write_all()


def test_require_all_folds_rejects_missing_split_metrics(tmp_path):
    """Verifies a split record without metrics aborts report writing."""
    metrics            = split_metrics()
    split              = make_records(metrics)
    split[1].metrics   = {}

    report = make_report(tmp_path, records_by_split={"test": split})

    with pytest.raises(ValueError, match="requires metrics for all"):
        report.write_all()


def test_aggregate_table_skips_non_numeric_metrics(tmp_path):
    """Verifies non-numeric metrics are left out of the aggregate table."""
    metrics = {index: {"curve_rmse_gt": float(index + 1), "label_str": "x"} for index in range(N_FOLDS)}
    report  = make_report(tmp_path, base_records=make_records(metrics), records_by_split={"test": make_records(metrics)})

    lines = report._aggregate_table(["curve_rmse_gt", "label_str"], make_records(metrics))
    body  = "\n".join(lines)

    assert "`curve_rmse_gt`" in body
    assert "`label_str`"     not in body


def _seed_dispersion(best_val_loss_std: float, metric_std: float, n_used: int = 2, missing: tuple = ()) -> dict:
    """Returns a per-fold seed dispersion payload with the given loss and metric deviations and seed coverage.

    Args:
        best_val_loss_std : Within-fold standard deviation of the best validation loss.
        metric_std        : Within-fold standard deviation of the curve RMSE on the test split.
        n_used            : Seeds whose test metrics entered every fold's mean.
        missing           : Seed labels without test metrics in every fold.
    """
    return {
        FoldNaming.name(index): {
            "n_seeds"           : 2,
            "status"            : "DONE",
            "best_val_loss_std" : best_val_loss_std,
            "splits"            : {"test": {"curve_rmse_gt": metric_std}},
            "n_used"            : {"test": n_used},
            "missing_seeds"     : {"test": list(missing)},
        }
        for index in range(N_FOLDS)
    }


def test_seed_dispersion_annotates_markdown_and_json(tmp_path):
    """Verifies seed dispersion adds the annotation to the markdown and the std fields to the JSON."""
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
    assert payload["splits"]["test"]["curve_rmse_gt"]["per_fold_seed_std"][FoldNaming.name(0)] == pytest.approx(0.5)
    assert payload["best_val_loss"]["per_fold_seed_std"][FoldNaming.name(0)] == pytest.approx(0.05)


def test_no_seed_dispersion_keeps_summary_unchanged(tmp_path):
    """Verifies the summary JSON carries no seed fields when no dispersion is supplied."""
    report = make_report(tmp_path)
    report.write_all()

    payload = json.loads((report.out_dir / "cv_summary.json").read_text())

    assert report.has_seed_sweep is False
    assert "seeds_per_fold" not in payload
    assert "per_fold_seed_std" not in payload["splits"]["test"]["curve_rmse_gt"]


def test_partial_fold_metrics_counted_in_n_used(tmp_path):
    """Verifies a NaN fold metric lowers n_used while n_total stays at the fold count."""
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


def test_seed_coverage_reaches_the_fold_headers_and_the_summary_json(tmp_path):
    """A fold that averaged fewer seeds than it holds shows the count in its column header and the JSON lists the used and missing seeds per fold."""
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
    """The training table shows each fold's training status, and its seed count under a sweep."""
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
    """A metric no fold holds as a finite value is left out of the aggregate with a warning naming it and its split; a metric non-finite in some folds only stays in with its count."""
    from tests.conftest import RecordingLogger

    metrics = {index: {"curve_rmse_gt": float(index + 1), "overall_r2_gt": float("nan"), "curve_mae_gt": float("nan") if index == 0 else 0.1} for index in range(N_FOLDS)}
    report  = make_report(tmp_path, base_records=make_records(metrics), records_by_split={"test": make_records(metrics)})
    logger  = RecordingLogger()

    report.logger = logger
    report.write_all()

    summary  = json.loads((tmp_path / "reports" / "cv_summary.json").read_text())
    warnings = [message for level, message in logger.messages if level == "warning"]

    assert "split test: ['overall_r2_gt'] non-finite in every fold, omitted from the aggregate" in warnings
    assert "overall_r2_gt" not in summary["splits"]["test"]
    assert summary["splits"]["test"]["curve_mae_gt"]["n_used"] == N_FOLDS - 1
