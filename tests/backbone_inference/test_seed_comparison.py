"""Tests covering seed-run inference resolution and the aggregated seed comparison report."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types   import SimpleNamespace

import numpy as np
import pytest

from pipelines.backbone.inference.analysis.seed_comparison import SeedActivityThreshold, SeedComparison, SeedComparisonReport, SeedEnsembleEvaluation, SeedInferenceResolver
from tools.runtime.completion                              import CompletionMarker

from tests.conftest import RecordingLogger, SilentLogger


def _stamp(run_dir: Path, stamp: str, metrics: dict, split: str | None = "test") -> Path:
    """Writes one inference stamp with metrics and a report under a run, completed unless split is None.

    Args:
        run_dir : Run directory receiving the inference/<stamp> subtree.
        stamp   : Name of the inference output subdirectory.
        metrics : Metric payload written to the inference metrics file.
        split   : Split recorded in the completion marker, or None to leave the stamp unfinished.

    Returns:
        Path of the created stamp directory.
    """
    out_dir = run_dir / "inference" / stamp
    out_dir.mkdir(parents=True)

    (out_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (out_dir / "report.md").write_text("# per-seed report", encoding="utf-8")

    if split is not None:
        CompletionMarker.stamp(out_dir, {"stage": "inference", "split": split})

    return out_dir


def _seed_run(group: Path, seed_name: str, metrics: dict, stamp: str = "stamp", activity_threshold: float | None = 1e-3) -> Path:
    """Writes a seed run directory holding one completed test-split inference stamp.

    Args:
        group              : Directory holding the seed runs.
        seed_name          : Name of the seed run directory.
        metrics            : Metric payload written to the inference metrics file.
        stamp              : Name of the inference output subdirectory.
        activity_threshold : Activity threshold recorded in the metrics file, or None to leave it out as inference wrote it before the field existed.

    Returns:
        Path of the created run directory.
    """
    run_dir = group / seed_name
    (run_dir / "meta").mkdir(parents=True)

    recorded = dict(metrics)
    if activity_threshold is not None:
        recorded["activity_threshold"] = activity_threshold

    _stamp(run_dir, stamp, recorded)

    return run_dir


def _report(group: Path, runs: list[Path], stamp: str = "stamp") -> SeedComparisonReport:
    """Returns a seed comparison report over the given run directories and inference stamp."""
    return SeedComparisonReport(
        group_dir          = group,
        run_dirs           = runs,
        inference_dirs     = [run / "inference" / stamp for run in runs],
        activity_threshold = 1e-3,
        output_subdir      = stamp,
        metrics_filename   = "metrics.json",
        report_filename    = "report.md",
        logger             = SilentLogger(),
    )


def _resolver(inference_subdir: str = "", split: str = "test", logger=None) -> SeedInferenceResolver:
    """Returns a resolver over metrics.json stamps with the given explicit stamp, split and logger."""
    return SeedInferenceResolver(inference_subdir, split, "metrics.json", logger if logger is not None else SilentLogger())


def test_resolver_picks_the_explicit_inference_subdir(tmp_path):
    """An explicit, completed inference subdirectory of the split is resolved directly."""
    run = _seed_run(tmp_path / "group", "seed0", {"m": 1.0}, stamp="20240101_000000")

    resolved = _resolver("20240101_000000").resolve(run)

    assert resolved == run / "inference" / "20240101_000000"


def test_resolver_rejects_an_explicit_subdir_without_a_completion_marker(tmp_path):
    """An explicit subdirectory holding metrics but no completion marker is refused as unfinished."""
    run = _seed_run(tmp_path / "group", "seed0", {"m": 1.0}, stamp="20240101_000000")
    _stamp(run, "20240202_000000", {"m": 2.0}, split=None)

    with pytest.raises(FileNotFoundError, match="completion marker"):
        _resolver("20240202_000000").resolve(run)

    with pytest.raises(FileNotFoundError):
        _resolver("20240303_000000").resolve(run)


def test_resolver_rejects_an_explicit_subdir_of_another_split(tmp_path):
    """An explicit subdirectory completed on another split is refused, naming both splits."""
    run = _seed_run(tmp_path / "group", "seed0", {"m": 1.0}, stamp="20240101_000000")
    _stamp(run, "20240202_000000", {"m": 2.0}, split="val")

    with pytest.raises(ValueError, match="scored on split 'val', not on the configured split 'test'"):
        _resolver("20240202_000000").resolve(run)


def test_resolver_picks_the_newest_completed_stamp_of_the_split(tmp_path):
    """Without an explicit subdirectory the newest stamp completed on the split wins, skipping newer unfinished and other-split stamps."""
    run    = _seed_run(tmp_path / "group", "seed0", {"m": 1.0}, stamp="20240101_000000")
    chosen = _stamp(run, "20240301_000000", {"m": 2.0})
    _stamp(run, "20240401_000000", {"m": 3.0}, split=None)
    _stamp(run, "20240501_000000", {"m": 4.0}, split="val")

    logger   = RecordingLogger()
    resolver = _resolver(logger=logger)
    resolved = resolver.resolve(run)

    assert resolved == chosen
    assert resolver.unfinished == {run: ["20240401_000000"]}
    assert any(level == "warning" and "20240401_000000" in message and "unfinished" in message for level, message in logger.messages)


def test_resolver_raises_when_a_run_has_no_inference(tmp_path):
    """A run without any inference directory is rejected."""
    run = tmp_path / "group" / "seed0"
    run.mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        _resolver().resolve(run)


def test_resolver_raises_when_only_unfinished_or_other_split_stamps_exist(tmp_path):
    """Metrics without a completion marker, or completed on another split, never qualify."""
    run = tmp_path / "group" / "seed0"
    (run / "meta").mkdir(parents=True)
    _stamp(run, "20240101_000000", {"m": 1.0}, split=None)
    _stamp(run, "20240201_000000", {"m": 1.0}, split="val")

    with pytest.raises(FileNotFoundError, match="No completed 'test' inference"):
        _resolver().resolve(run)


def test_resolver_record_reads_the_completion_marker(tmp_path):
    """The record of a resolved stamp carries its name, split and completion time."""
    run    = _seed_run(tmp_path / "group", "seed0", {"m": 1.0}, stamp="20240101_000000")
    record = SeedInferenceResolver.record(run / "inference" / "20240101_000000")

    assert record["stamp"] == "20240101_000000"
    assert record["split"] == "test"
    assert record["completed_at"]


def test_report_requires_at_least_two_runs(tmp_path):
    """Aggregating a single seed run is rejected."""
    group = tmp_path / "group"
    run   = _seed_run(group, "seed0", {"m": 1.0})

    with pytest.raises(ValueError):
        _report(group, [run])


def test_report_aggregates_mean_and_sample_std(tmp_path):
    """Numeric metrics are aggregated as mean and sample standard deviation, and non-numeric keys are dropped."""
    group = tmp_path / "group"
    runs  = [
        _seed_run(group, "seed0", {"curve_mse_gt": 1.0, "overall_r2_gt": 0.5, "split": "test"}),
        _seed_run(group, "seed1", {"curve_mse_gt": 3.0, "overall_r2_gt": 0.7, "split": "test"}),
    ]

    _report(group, runs).run()

    payload = json.loads((group / "inference" / "stamp" / "metrics.json").read_text(encoding="utf-8"))

    assert payload["n_seeds"]               == 2
    assert payload["split"]                 == "test"
    assert payload["activity_threshold"]    == 1e-3
    assert list(payload["seed_runs"])       == ["seed0", "seed1"]
    assert payload["seed_inference"]["seed0"]["stamp"] == "stamp"
    assert payload["seed_inference"]["seed1"]["split"] == "test"
    assert payload["seed_inference"]["seed1"]["completed_at"]
    assert payload["mean"]["curve_mse_gt"]  == pytest.approx(2.0)
    assert payload["std"]["curve_mse_gt"]   == pytest.approx(math.sqrt(2.0))
    assert payload["mean"]["overall_r2_gt"] == pytest.approx(0.6)
    assert "split" not in payload["mean"]


def test_report_records_confidence_intervals_and_counts(tmp_path):
    """The metrics payload carries a symmetric t-based 95% interval around the mean and the finite-sample count."""
    group = tmp_path / "group"
    runs  = [
        _seed_run(group, "seed0", {"curve_mse_gt": 1.0}),
        _seed_run(group, "seed1", {"curve_mse_gt": 3.0}),
    ]

    _report(group, runs).run()

    payload  = json.loads((group / "inference" / "stamp" / "metrics.json").read_text(encoding="utf-8"))
    low, high = payload["ci95"]["curve_mse_gt"]

    assert payload["n"]["curve_mse_gt"] == 2
    assert low < 2.0 < high
    assert (low + high) / 2.0 == pytest.approx(2.0)
    assert high - low == pytest.approx(2.0 * 12.7062, rel=1e-3)


def test_report_refuses_seeds_scored_on_different_splits(tmp_path):
    """Seeds whose stamps were completed on different splits are never aggregated together."""
    group = tmp_path / "group"
    runs  = [_seed_run(group, "seed0", {"curve_mse_gt": 1.0})]

    other = group / "seed1"
    (other / "meta").mkdir(parents=True)
    _stamp(other, "stamp", {"curve_mse_gt": 3.0}, split="val")
    runs.append(other)

    with pytest.raises(ValueError, match="different splits"):
        _report(group, runs)


def test_report_markdown_has_per_seed_columns_stamps_and_links(tmp_path):
    """The markdown table lists per-seed columns, the inference stamp and relative links to per-seed reports."""
    group = tmp_path / "group"
    runs  = [
        _seed_run(group, "seed0", {"curve_mse_gt": 1.0}),
        _seed_run(group, "seed1", {"curve_mse_gt": 3.0}),
    ]

    report_path = _report(group, runs).run()

    text = report_path.read_text(encoding="utf-8")

    assert report_path == group / "inference" / "stamp" / "report.md"
    assert "seed0" in text and "seed1" in text
    assert "±" in text
    assert "`stamp`" in text
    assert "| Split" in text and "| test " in text
    assert "../../seed0/inference/stamp/report.md" in text
    assert "`curve_mse_gt`" in text


def test_report_excludes_per_slice_metrics_from_the_tables(tmp_path):
    """Per-slice metrics are aggregated into the payload but kept out of the tables."""
    group = tmp_path / "group"
    runs  = [
        _seed_run(group, "seed0", {"curve_mse_gt": 1.0, "ssim_gt_z_3": 0.9}),
        _seed_run(group, "seed1", {"curve_mse_gt": 3.0, "ssim_gt_z_3": 0.8}),
    ]

    report_path = _report(group, runs).run()

    payload = json.loads((group / "inference" / "stamp" / "metrics.json").read_text(encoding="utf-8"))
    text    = report_path.read_text(encoding="utf-8")

    assert payload["mean"]["ssim_gt_z_3"] == pytest.approx(0.85)
    assert "ssim_gt_z_3" not in text


def _config(runs_dir: Path, group_tags: list[str], inference_subdir: str = "", output_subdir: str = "") -> SimpleNamespace:
    """Returns a seed comparison config stub for the given runs directory and group tags."""
    return SimpleNamespace(
        runs_dir             = str(runs_dir),
        group_tags           = group_tags,
        split                = "test",
        inference_subdir     = inference_subdir,
        output_subdir        = output_subdir,
        metrics_filename     = "metrics.json",
        report_filename      = "report.md",
        compute_disagreement = False,
        compute_ensemble     = False,
        cubes_subdir         = "cubes",
    )


def test_activity_threshold_is_the_value_every_seed_recorded(tmp_path):
    """Seeds scored at one recorded threshold resolve to that value."""
    group = tmp_path / "group"
    runs  = [_seed_run(group, f"seed{i}", {"curve_mse_gt": 1.0}, activity_threshold=1e-4) for i in range(2)]

    threshold = SeedActivityThreshold("metrics.json").resolve(["seed0", "seed1"], [run / "inference" / "stamp" for run in runs])

    assert threshold == 1e-4


def test_seeds_scored_at_different_thresholds_are_refused_on_the_default_path(tmp_path):
    """Seeds whose metrics record different activity thresholds are refused before any metric is averaged."""
    group = tmp_path / "runs" / "group"
    _seed_run(group, "seed0", {"curve_mse_gt": 1.0}, activity_threshold=1e-4)
    _seed_run(group, "seed1", {"curve_mse_gt": 3.0}, activity_threshold=1e-3)

    with pytest.raises(ValueError, match="different activity thresholds"):
        SeedComparison(_config(group, [], output_subdir="agg"), SilentLogger()).run()

    assert not (group / "inference" / "agg" / "metrics.json").exists()


def test_metrics_without_a_recorded_threshold_are_refused(tmp_path):
    """Metrics written before the threshold was recorded are refused with a request to re-run inference."""
    group = tmp_path / "runs" / "group"
    _seed_run(group, "seed0", {"curve_mse_gt": 1.0}, activity_threshold=None)
    _seed_run(group, "seed1", {"curve_mse_gt": 3.0}, activity_threshold=None)

    with pytest.raises(KeyError, match="re-run inference for seed seed0"):
        SeedComparison(_config(group, [], output_subdir="agg"), SilentLogger()).run()


def test_comparison_compares_each_selected_group_in_isolation(tmp_path):
    """Each selected group is aggregated into its own report and metrics payload."""
    group_a = tmp_path / "runs" / "group_a"
    _seed_run(group_a, "seed0", {"curve_mse_gt": 1.0})
    _seed_run(group_a, "seed1", {"curve_mse_gt": 3.0})

    group_b = tmp_path / "runs" / "group_b"
    _seed_run(group_b, "seed0", {"curve_mse_gt": 10.0})
    _seed_run(group_b, "seed1", {"curve_mse_gt": 30.0})

    reports = SeedComparison(_config(tmp_path / "runs", ["group_a", "group_b"], output_subdir="agg"), SilentLogger()).run()

    payload_a = json.loads((group_a / "inference" / "agg" / "metrics.json").read_text(encoding="utf-8"))
    payload_b = json.loads((group_b / "inference" / "agg" / "metrics.json").read_text(encoding="utf-8"))

    assert reports == [group_a / "inference" / "agg" / "report.md", group_b / "inference" / "agg" / "report.md"]
    assert payload_a["mean"]["curve_mse_gt"] == pytest.approx(2.0)
    assert payload_b["mean"]["curve_mse_gt"] == pytest.approx(20.0)


def test_comparison_raises_on_unknown_group_tag(tmp_path):
    """A group tag with no matching directory is rejected."""
    group = tmp_path / "runs" / "group"
    _seed_run(group, "seed0", {"curve_mse_gt": 1.0})

    with pytest.raises(FileNotFoundError):
        SeedComparison(_config(tmp_path / "runs", ["group", "missing"]), SilentLogger()).run()


def test_comparison_raises_when_a_group_holds_fewer_than_two_runs(tmp_path):
    """A group holding a single run is rejected."""
    group = tmp_path / "runs" / "group"
    _seed_run(group, "seed0", {"curve_mse_gt": 1.0})

    with pytest.raises(ValueError):
        SeedComparison(_config(tmp_path / "runs", ["group"]), SilentLogger()).run()


def test_comparison_treats_runs_dir_as_the_group_when_no_tags_are_given(tmp_path):
    """With no group tags the runs directory itself is the group."""
    group = tmp_path / "runs" / "group"
    _seed_run(group, "seed0", {"curve_mse_gt": 1.0})
    _seed_run(group, "seed1", {"curve_mse_gt": 3.0})

    reports = SeedComparison(_config(group, [], output_subdir="agg"), SilentLogger()).run()

    payload = json.loads((group / "inference" / "agg" / "metrics.json").read_text(encoding="utf-8"))

    assert reports == [group / "inference" / "agg" / "report.md"]
    assert payload["mean"]["curve_mse_gt"] == pytest.approx(2.0)


def test_comparison_output_falls_back_to_the_inference_stamp(tmp_path):
    """Without an output subdirectory the inference stamp names the aggregate directory."""
    group = tmp_path / "runs" / "group"
    _seed_run(group, "seed0", {"curve_mse_gt": 1.0})
    _seed_run(group, "seed1", {"curve_mse_gt": 3.0})

    SeedComparison(_config(group, [], inference_subdir="stamp"), SilentLogger()).run()

    assert (group / "inference" / "stamp" / "metrics.json").is_file()


def test_comparison_output_gets_a_fresh_stamp_when_nothing_is_set(tmp_path):
    """With neither subdirectory set the aggregate lands in a freshly stamped directory."""
    group = tmp_path / "runs" / "group"
    _seed_run(group, "seed0", {"curve_mse_gt": 1.0})
    _seed_run(group, "seed1", {"curve_mse_gt": 3.0})

    reports = SeedComparison(_config(group, []), SilentLogger()).run()

    assert reports[0].parent.parent == group / "inference"
    assert reports[0].parent.name not in ("stamp", "")


def _seed_cubes(run: Path, pred: "np.ndarray", gt: "np.ndarray", stamp: str = "stamp") -> None:
    """Writes prediction and ground-truth cubes into a seed run's inference stamp."""
    cube_dir = run / "inference" / stamp / "cubes"
    cube_dir.mkdir(parents=True, exist_ok=True)
    np.save(cube_dir / "pred_curves.npy", pred)
    np.save(cube_dir / "gt_curves.npy", gt)


def test_seed_ensemble_scores_the_mean_prediction(tmp_path):
    """The ensemble scalars score the across-seed mean cube, beating two biased seeds."""
    group = tmp_path / "group"
    gt    = np.zeros((2, 4, 4), dtype=np.float32)

    run_a = _seed_run(group, "seed0", {"curve_mse_gt": 1.0})
    run_b = _seed_run(group, "seed1", {"curve_mse_gt": 1.0})
    _seed_cubes(run_a, gt + 1.0, gt)
    _seed_cubes(run_b, gt - 1.0, gt)

    scalars = SeedEnsembleEvaluation([run / "inference" / "stamp" / "cubes" for run in (run_a, run_b)], SilentLogger()).run()

    assert scalars["n_seeds"]      == 2
    assert scalars["curve_mse_gt"] == pytest.approx(0.0)
    assert scalars["curve_mae_gt"] == pytest.approx(0.0)


def test_seed_ensemble_raises_without_saved_cubes(tmp_path):
    """A seed without saved prediction cubes fails loudly, naming the fix."""
    group = tmp_path / "group"
    run   = _seed_run(group, "seed0", {"curve_mse_gt": 1.0})

    with pytest.raises(FileNotFoundError, match="save_cubes"):
        SeedEnsembleEvaluation([run / "inference" / "stamp" / "cubes"], SilentLogger()).run()


def test_report_renders_the_ensemble_section(tmp_path):
    """With an ensemble payload the report carries the section and the metrics JSON the scalars."""
    group = tmp_path / "group"
    runs  = [
        _seed_run(group, "seed0", {"curve_mse_gt": 1.0}),
        _seed_run(group, "seed1", {"curve_mse_gt": 3.0}),
    ]

    report = SeedComparisonReport(
        group_dir          = group,
        run_dirs           = runs,
        inference_dirs     = [run / "inference" / "stamp" for run in runs],
        activity_threshold = 1e-3,
        output_subdir      = "stamp",
        metrics_filename   = "metrics.json",
        report_filename    = "report.md",
        logger             = SilentLogger(),
        ensemble           = {"n_seeds": 2, "curve_mse_gt": 0.5},
    )

    text    = report.run().read_text(encoding="utf-8")
    payload = json.loads((group / "inference" / "stamp" / "metrics.json").read_text(encoding="utf-8"))

    assert "Seed ensemble" in text
    assert payload["ensemble"]["curve_mse_gt"] == pytest.approx(0.5)


def test_comparison_logs_through_the_given_logger(tmp_path):
    """Every message of the comparison goes to the caller's logger instead of a private log file."""
    group = tmp_path / "runs" / "group"
    _seed_run(group, "seed0", {"curve_mse_gt": 1.0})
    _seed_run(group, "seed1", {"curve_mse_gt": 3.0})

    logger = RecordingLogger()
    SeedComparison(_config(group, [], output_subdir="agg"), logger).run()

    assert ("section", "Seed comparison") in logger.messages
    assert ("metrics_table", "Seed inferences") in logger.messages
    assert any(level == "subsection" and message.startswith("Group: ") for level, message in logger.messages)


def test_comparison_counts_the_unfinished_stamps_it_skipped(tmp_path):
    """Unfinished stamps are skipped per seed and summed up in a closing warning."""
    group = tmp_path / "runs" / "group"
    runs  = [_seed_run(group, "seed0", {"curve_mse_gt": 1.0}), _seed_run(group, "seed1", {"curve_mse_gt": 3.0})]
    _stamp(runs[0], "zz_crashed", {"curve_mse_gt": 9.0}, split=None)

    logger = RecordingLogger()
    SeedComparison(_config(group, [], output_subdir="agg"), logger).run()

    payload  = json.loads((group / "inference" / "agg" / "metrics.json").read_text(encoding="utf-8"))
    warnings = [message for level, message in logger.messages if level == "warning"]

    assert payload["seed_inference"]["seed0"]["stamp"] == "stamp"
    assert any("1 unfinished inference stamp(s) skipped across 1 seed run(s)" == message for message in warnings)
