from __future__ import annotations

import json
from pathlib import Path

from pipelines.comparison.comparison_report import TrialComparisonReport
from pipelines.comparison.trial_collector   import TrialCollector


class RecordingLogger:
    def __init__(self):
        self.messages = []

    def section(self, msg):
        self.messages.append(("section", msg))

    def subsection(self, msg):
        self.messages.append(("subsection", msg))

    def ok(self, msg):
        self.messages.append(("ok", msg))

    def info(self, msg):
        self.messages.append(("info", msg))

    def error(self, msg):
        self.messages.append(("error", msg))


def _run_with_metrics(run_dir: Path, rmse: float) -> None:
    inference_dir = run_dir / "inference" / "20260101_000000"
    inference_dir.mkdir(parents=True)
    (inference_dir / "metrics.json").write_text(json.dumps({"curve_rmse_gt": rmse}))


def test_trial_tag_expands_and_aggregates_seed_runs(tmp_path):
    _run_with_metrics(tmp_path / "trial_a" / "seed0", 2.0)
    _run_with_metrics(tmp_path / "trial_a" / "seed1", 4.0)

    collector = TrialCollector(runs_dir=tmp_path, run_tags=["trial_a"], logger=RecordingLogger())
    records   = collector.collect()

    assert [record.name for record in records] == ["trial_a"]
    assert records[0].metrics["curve_rmse_gt"] == 3.0
    assert records[0].has_inference
    assert collector.seed_dispersion["trial_a"]["n_seeds"] == 2
    assert collector.seed_dispersion["trial_a"]["metrics"]["curve_rmse_gt"] is not None


def test_explicit_seed_run_tags_aggregate_into_one_trial(tmp_path):
    _run_with_metrics(tmp_path / "trial_a" / "seed0", 2.0)
    _run_with_metrics(tmp_path / "trial_a" / "seed1", 4.0)

    collector = TrialCollector(runs_dir=tmp_path, run_tags=["trial_a/seed0", "trial_a/seed1"], logger=RecordingLogger())
    records   = collector.collect()

    assert [record.name for record in records] == ["trial_a"]
    assert records[0].metrics["curve_rmse_gt"] == 3.0


def test_duplicate_unit_and_seed_tags_collapse_once(tmp_path):
    _run_with_metrics(tmp_path / "trial_a" / "seed0", 2.0)
    _run_with_metrics(tmp_path / "trial_a" / "seed1", 4.0)

    collector = TrialCollector(runs_dir=tmp_path, run_tags=["trial_a", "trial_a/seed0"], logger=RecordingLogger())
    records   = collector.collect()

    assert [record.name for record in records] == ["trial_a"]
    assert records[0].metrics["curve_rmse_gt"] == 3.0
    assert collector.seed_dispersion["trial_a"]["n_seeds"] == 2


def test_flat_runs_pass_through_unchanged(tmp_path):
    _run_with_metrics(tmp_path / "run_x", 1.0)
    _run_with_metrics(tmp_path / "run_y", 2.0)

    collector = TrialCollector(runs_dir=tmp_path, run_tags=["run_x", "run_y"], logger=RecordingLogger())
    records   = collector.collect()

    assert [record.name for record in records] == ["run_x", "run_y"]
    assert collector.seed_dispersion == {}


def test_single_seed_run_tag_stays_identity(tmp_path):
    _run_with_metrics(tmp_path / "trial_a" / "seed0", 2.0)

    collector = TrialCollector(runs_dir=tmp_path, run_tags=["trial_a/seed0"], logger=RecordingLogger())
    records   = collector.collect()

    assert [record.name for record in records] == ["trial_a/seed0"]
    assert collector.seed_dispersion == {}


def test_report_annotates_seed_dispersion(tmp_path):
    _run_with_metrics(tmp_path / "trial_a" / "seed0", 2.0)
    _run_with_metrics(tmp_path / "trial_a" / "seed1", 4.0)
    _run_with_metrics(tmp_path / "run_x", 1.0)

    collector = TrialCollector(runs_dir=tmp_path, run_tags=["trial_a", "run_x"], logger=RecordingLogger())
    records   = collector.collect()

    out_dir = tmp_path / "comparison"
    report  = TrialComparisonReport(records=records, out_dir=out_dir, compare_images=False, compare_gifs=False, embed_images=False, logger=RecordingLogger(), seed_dispersion=collector.seed_dispersion)
    report.write_all()

    overview = (out_dir / "overview.md").read_text()
    metrics  = (out_dir / "metrics_comparison.md").read_text()

    assert "Seeds" in overview
    assert "±" in metrics
