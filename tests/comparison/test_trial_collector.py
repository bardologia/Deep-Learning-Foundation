"""Tests for run comparison collection and reporting across trials.

Covers how TrialCollector expands and aggregates per-seed run tags, how it picks
the inference directory of the requested split, and how TrialComparisonReport
renders dispersion and per-run figure sections.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipelines.comparison.comparison_report import TrialComparisonReport
from pipelines.comparison.trial_collector   import TrialCollector
from tools.runtime.completion               import CompletionMarker

from tests.conftest import RecordingLogger


def _run_with_metrics(run_dir: Path, rmse: float, split: str = "test", stamp: str = "20260101_000000") -> None:
    """Writes a completed inference directory under run_dir carrying the given curve RMSE and split."""
    inference_dir = run_dir / "inference" / stamp
    inference_dir.mkdir(parents=True)
    (inference_dir / "metrics.json").write_text(json.dumps({"curve_rmse_gt": rmse}))
    (inference_dir / CompletionMarker.FILENAME).write_text(json.dumps({"stage": "inference", "split": split}))


def test_trial_tag_expands_and_aggregates_seed_runs(tmp_path):
    """Verifies a bare trial tag expands to its seed runs and aggregates into one record with dispersion."""
    _run_with_metrics(tmp_path / "trial_a" / "seed0", 2.0)
    _run_with_metrics(tmp_path / "trial_a" / "seed1", 4.0)

    collector = TrialCollector(runs_dir=tmp_path, run_tags=["trial_a"], logger=RecordingLogger(), inference_split="test")
    records   = collector.collect()

    assert [record.name for record in records] == ["trial_a"]
    assert records[0].metrics["curve_rmse_gt"] == 3.0
    assert records[0].has_inference
    assert collector.seed_dispersion["trial_a"]["n_seeds"] == 2
    assert collector.seed_dispersion["trial_a"]["metrics"]["curve_rmse_gt"] is not None


def test_explicit_seed_run_tags_aggregate_into_one_trial(tmp_path):
    """Verifies explicitly listed seed tags of one trial aggregate into a single record."""
    _run_with_metrics(tmp_path / "trial_a" / "seed0", 2.0)
    _run_with_metrics(tmp_path / "trial_a" / "seed1", 4.0)

    collector = TrialCollector(runs_dir=tmp_path, run_tags=["trial_a/seed0", "trial_a/seed1"], logger=RecordingLogger(), inference_split="test")
    records   = collector.collect()

    assert [record.name for record in records] == ["trial_a"]
    assert records[0].metrics["curve_rmse_gt"] == 3.0


def test_duplicate_unit_and_seed_tags_collapse_once(tmp_path):
    """Verifies a trial tag repeated alongside one of its seed tags is collected only once."""
    _run_with_metrics(tmp_path / "trial_a" / "seed0", 2.0)
    _run_with_metrics(tmp_path / "trial_a" / "seed1", 4.0)

    collector = TrialCollector(runs_dir=tmp_path, run_tags=["trial_a", "trial_a/seed0"], logger=RecordingLogger(), inference_split="test")
    records   = collector.collect()

    assert [record.name for record in records] == ["trial_a"]
    assert records[0].metrics["curve_rmse_gt"] == 3.0
    assert collector.seed_dispersion["trial_a"]["n_seeds"] == 2


def test_seed_means_disclose_the_seeds_they_cover(tmp_path):
    """Verifies a seed without inference and a seed with a NaN metric are counted out of the mean, warned about, and shown as used/total in the Seeds cell."""
    _run_with_metrics(tmp_path / "trial_a" / "seed0", 2.0)
    _run_with_metrics(tmp_path / "trial_a" / "seed1", float("nan"))
    (tmp_path / "trial_a" / "seed2").mkdir()
    _run_with_metrics(tmp_path / "run_x", 1.0)

    logger    = RecordingLogger()
    collector = TrialCollector(runs_dir=tmp_path, run_tags=["trial_a", "run_x"], logger=logger, inference_split="test")
    records   = collector.collect()

    dispersion = collector.seed_dispersion["trial_a"]
    warnings   = [message for level, message in logger.messages if level == "warning"]
    infos      = [message for level, message in logger.messages if level == "info"]
    oks        = [message for level, message in logger.messages if level == "ok"]

    assert records[0].metrics["curve_rmse_gt"] == 2.0
    assert dispersion["n_seeds"]               == 3
    assert dispersion["n_used"]                == {"curve_rmse_gt": 1}
    assert any("seed2" in message and "curve_rmse_gt" in message and "1 of 3" in message for message in warnings)
    assert any(message == "trial_a: figures and report from trial_a/seed0" for message in infos)
    assert any(message.startswith("Aggregated 3 seed run(s) into 1 trial(s)") for message in oks)

    out_dir = tmp_path / "comparison"
    TrialComparisonReport(records=records, out_dir=out_dir, compare_images=False, compare_gifs=False, embed_images=False, logger=RecordingLogger(), seed_dispersion=collector.seed_dispersion).write_all()

    overview = (out_dir / "overview.md").read_text()

    assert "| 1/3 " in overview.replace("  ", " ")


def test_empty_run_tags_raise(tmp_path):
    """Verifies a comparison without any requested run tag refuses to run, since it discovers no runs on its own."""
    with pytest.raises(ValueError, match="No run tags requested"):
        TrialCollector(runs_dir=tmp_path, run_tags=[], logger=RecordingLogger(), inference_split="test").collect()


def test_missing_run_directory_raises_naming_it(tmp_path):
    """Verifies a requested run tag whose directory does not exist aborts the comparison and is named."""
    _run_with_metrics(tmp_path / "run_x", 1.0)

    with pytest.raises(FileNotFoundError, match="gone"):
        TrialCollector(runs_dir=tmp_path, run_tags=["run_x", "gone"], logger=RecordingLogger(), inference_split="test").collect()


def test_trial_without_inference_on_the_split_warns_and_names_the_family_subdir(tmp_path):
    """Verifies a trial inferred on another split, with stamps in a family sub-directory, is warned about but still listed."""
    _run_with_metrics(tmp_path / "run_x", 1.0)
    _run_with_metrics(tmp_path / "run_y", 2.0, split="val")
    (tmp_path / "run_y" / "inference" / "profile_ae").mkdir()

    logger    = RecordingLogger()
    collector = TrialCollector(runs_dir=tmp_path, run_tags=["run_x", "run_y"], logger=logger, inference_split="test")
    records   = collector.collect()

    warnings = [message for level, message in logger.messages if level == "warning"]
    oks      = [message for level, message in logger.messages if level == "ok"]

    assert [record.name for record in records] == ["run_x", "run_y"]
    assert len(warnings) == 1
    assert "run_y: no completed inference on split 'test'" in warnings[0]
    assert "inference/profile_ae/" in warnings[0]
    assert "2 trial(s) collected, 1 with inference metrics on split 'test'" in oks


def test_comparison_without_any_scored_trial_raises(tmp_path):
    """Verifies a comparison in which no trial carries metrics on the split aborts instead of writing empty pages."""
    _run_with_metrics(tmp_path / "run_x", 1.0, split="val")
    _run_with_metrics(tmp_path / "run_y", 2.0, split="val")

    with pytest.raises(RuntimeError, match="split 'test'"):
        TrialCollector(runs_dir=tmp_path, run_tags=["run_x", "run_y"], logger=RecordingLogger(), inference_split="test").collect()


def test_flat_runs_pass_through_unchanged(tmp_path):
    """Verifies runs without a seed level are kept separate and report no dispersion."""
    _run_with_metrics(tmp_path / "run_x", 1.0)
    _run_with_metrics(tmp_path / "run_y", 2.0)

    collector = TrialCollector(runs_dir=tmp_path, run_tags=["run_x", "run_y"], logger=RecordingLogger(), inference_split="test")
    records   = collector.collect()

    assert [record.name for record in records] == ["run_x", "run_y"]
    assert collector.seed_dispersion == {}


def test_named_output_subdir_of_other_split_does_not_win(tmp_path):
    """Verifies an inference directory of another split is ignored even when named."""
    _run_with_metrics(tmp_path / "run_x", 1.0)
    _run_with_metrics(tmp_path / "run_x", 9.9, split="val", stamp="region_override")

    collector = TrialCollector(runs_dir=tmp_path, run_tags=["run_x"], logger=RecordingLogger(), inference_split="test")
    records   = collector.collect()

    assert records[0].inference_dir.name       == "20260101_000000"
    assert records[0].metrics["curve_rmse_gt"] == 1.0


def test_single_seed_run_tag_stays_identity(tmp_path):
    """Verifies a lone seed tag keeps its full name and reports no dispersion."""
    _run_with_metrics(tmp_path / "trial_a" / "seed0", 2.0)

    collector = TrialCollector(runs_dir=tmp_path, run_tags=["trial_a/seed0"], logger=RecordingLogger(), inference_split="test")
    records   = collector.collect()

    assert [record.name for record in records] == ["trial_a/seed0"]
    assert collector.seed_dispersion == {}


def test_report_annotates_seed_dispersion(tmp_path):
    """Verifies the overview gains a Seeds column and the metric table a plus-minus dispersion term."""
    _run_with_metrics(tmp_path / "trial_a" / "seed0", 2.0)
    _run_with_metrics(tmp_path / "trial_a" / "seed1", 4.0)
    _run_with_metrics(tmp_path / "run_x", 1.0)

    collector = TrialCollector(runs_dir=tmp_path, run_tags=["trial_a", "run_x"], logger=RecordingLogger(), inference_split="test")
    records   = collector.collect()

    out_dir = tmp_path / "comparison"
    report  = TrialComparisonReport(records=records, out_dir=out_dir, compare_images=False, compare_gifs=False, embed_images=False, logger=RecordingLogger(), seed_dispersion=collector.seed_dispersion)
    report.write_all()

    overview = (out_dir / "overview.md").read_text()
    metrics  = (out_dir / "metrics_comparison.md").read_text()

    assert "Seeds" in overview
    assert "±" in metrics


def test_figure_section_marks_figures_absent_from_a_trial(tmp_path):
    """Verifies the figure section lists the union of figure names and marks the ones a run lacks."""
    _run_with_metrics(tmp_path / "run_x", 1.0)
    _run_with_metrics(tmp_path / "run_y", 2.0)

    for tag, names in (("run_x", ["a.png", "b.png"]), ("run_y", ["a.png"])):
        profiles = tmp_path / tag / "inference" / "20260101_000000" / "figures" / "profiles"
        profiles.mkdir(parents=True)
        for name in names:
            (profiles / name).write_bytes(b"")

    out_dir = tmp_path / "comparison"
    out_dir.mkdir()

    collector = TrialCollector(runs_dir=tmp_path, run_tags=["run_x", "run_y"], logger=RecordingLogger(), inference_split="test")
    report    = TrialComparisonReport(records=collector.collect(), out_dir=out_dir, compare_images=True, compare_gifs=False, embed_images=False, logger=RecordingLogger())

    path = report._write_figure_section("profiles", "Profile reconstructions")
    text = path.read_text()

    assert "## `a.png`" in text
    assert "## `b.png`" in text
    assert text.count("_(not in this run)_") == 1
