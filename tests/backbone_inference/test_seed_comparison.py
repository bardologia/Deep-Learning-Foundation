
from __future__ import annotations

import json
import math
from pathlib import Path
from types   import SimpleNamespace

import numpy as np
import pytest

from pipelines.backbone.inference.analysis.seed_comparison   import SeedComparison, SeedComparisonReport, SeedEnsembleEvaluation, SeedInferenceResolver
from pipelines.backbone.inference.analysis.seed_disagreement import SeedMaps
from tools.runtime.completion                                import CompletionMarker

from tests.conftest import RecordingLogger, SilentLogger


def _stamp(run_dir: Path, stamp: str, metrics: dict, split: str | None = "test") -> Path:
    out_dir = run_dir / "inference" / stamp
    out_dir.mkdir(parents=True)

    (out_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (out_dir / "report.md").write_text("# per-seed report", encoding="utf-8")

    if split is not None:
        CompletionMarker.stamp(out_dir, {"stage": "inference", "split": split})

    return out_dir


def _seed_run(group: Path, seed_name: str, metrics: dict, stamp: str = "stamp") -> Path:
    run_dir = group / seed_name
    (run_dir / "meta").mkdir(parents=True)
    (run_dir / "meta" / "run_summary.json").write_text(json.dumps({"target_channels": ["a", "b"]}), encoding="utf-8")

    _stamp(run_dir, stamp, metrics)

    return run_dir


def _report(group: Path, runs: list[Path], stamp: str = "stamp") -> SeedComparisonReport:
    return SeedComparisonReport(
        group_dir          = group,
        run_dirs           = runs,
        inference_dirs   = [run / "inference" / stamp for run in runs],
        output_subdir    = stamp,
        metrics_filename = "metrics.json",
        report_filename  = "report.md",
        logger           = SilentLogger(),
    )


def _resolver(inference_subdir: str = "", split: str = "test", logger=None) -> SeedInferenceResolver:
    return SeedInferenceResolver(inference_subdir, split, "metrics.json", logger if logger is not None else SilentLogger())


def test_resolver_picks_the_explicit_inference_subdir(tmp_path):
    run = _seed_run(tmp_path / "group", "seed0", {"m": 1.0}, stamp="20240101_000000")

    resolved = _resolver("20240101_000000").resolve(run)

    assert resolved == run / "inference" / "20240101_000000"


def test_resolver_picks_the_newest_completed_stamp_of_the_split(tmp_path):
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
    run = tmp_path / "group" / "seed0"
    run.mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        _resolver().resolve(run)


def test_resolver_record_reads_the_completion_marker(tmp_path):
    run    = _seed_run(tmp_path / "group", "seed0", {"m": 1.0}, stamp="20240101_000000")
    record = SeedInferenceResolver.record(run / "inference" / "20240101_000000")

    assert record["stamp"] == "20240101_000000"
    assert record["split"] == "test"
    assert record["completed_at"]


def test_report_aggregates_mean_and_sample_std(tmp_path):
    group = tmp_path / "group"
    runs  = [
        _seed_run(group, "seed0", {"rmse_mean": 1.0, "r2_mean": 0.5, "split": "test"}),
        _seed_run(group, "seed1", {"rmse_mean": 3.0, "r2_mean": 0.7, "split": "test"}),
    ]

    _report(group, runs).run()

    payload = json.loads((group / "inference" / "stamp" / "metrics.json").read_text(encoding="utf-8"))

    assert payload["n_seeds"]               == 2
    assert payload["split"]                 == "test"
    assert list(payload["seed_runs"])       == ["seed0", "seed1"]
    assert payload["seed_inference"]["seed0"]["stamp"] == "stamp"
    assert payload["seed_inference"]["seed1"]["split"] == "test"
    assert payload["seed_inference"]["seed1"]["completed_at"]
    assert payload["mean"]["rmse_mean"]  == pytest.approx(2.0)
    assert payload["std"]["rmse_mean"]   == pytest.approx(math.sqrt(2.0))
    assert payload["mean"]["r2_mean"] == pytest.approx(0.6)
    assert "split" not in payload["mean"]


def test_report_records_confidence_intervals_and_counts(tmp_path):
    group = tmp_path / "group"
    runs  = [
        _seed_run(group, "seed0", {"rmse_mean": 1.0}),
        _seed_run(group, "seed1", {"rmse_mean": 3.0}),
    ]

    _report(group, runs).run()

    payload  = json.loads((group / "inference" / "stamp" / "metrics.json").read_text(encoding="utf-8"))
    low, high = payload["ci95"]["rmse_mean"]

    assert payload["n"]["rmse_mean"] == 2
    assert low < 2.0 < high
    assert (low + high) / 2.0 == pytest.approx(2.0)
    assert high - low == pytest.approx(2.0 * 12.7062, rel=1e-3)


def test_report_counts_the_seeds_behind_every_aggregate(tmp_path):
    group = tmp_path / "group"
    runs  = [
        _seed_run(group, "seed0", {"rmse_mean": 1.0, "r2_mean": 0.5, "mae_mean": float("nan")}),
        _seed_run(group, "seed1", {"rmse_mean": 3.0, "mae_mean": float("nan")}),
        _seed_run(group, "seed2", {"rmse_mean": 5.0, "r2_mean": float("nan"), "mae_mean": float("nan")}),
    ]

    logger = RecordingLogger()
    report = SeedComparisonReport(group, runs, [run / "inference" / "stamp" for run in runs], "stamp", "metrics.json", "report.md", logger)
    report.run()

    payload  = json.loads((group / "inference" / "stamp" / "metrics.json").read_text(encoding="utf-8"))
    warnings = [message for level, message in logger.messages if level == "warning"]

    assert payload["n"]["rmse_mean"]  == 3
    assert payload["n"]["r2_mean"] == 1
    assert "mae_mean" not in payload["mean"]
    assert any(message.startswith("1 metric(s) aggregated over fewer than 3 seeds: r2_mean (missing seed1,seed2)") for message in warnings)
    assert any("non-finite in every seed" in message and "mae_mean" in message for message in warnings)
    assert ("info", "2 metric(s) aggregated over 3 seeds, 1 with every seed contributing") in logger.messages


def test_report_markdown_has_per_seed_columns_stamps_and_links(tmp_path):
    group = tmp_path / "group"
    runs  = [
        _seed_run(group, "seed0", {"rmse_mean": 1.0}),
        _seed_run(group, "seed1", {"rmse_mean": 3.0}),
    ]

    report_path = _report(group, runs).run()

    text = report_path.read_text(encoding="utf-8")

    assert report_path == group / "inference" / "stamp" / "report.md"
    assert "seed0" in text and "seed1" in text
    assert "±" in text
    assert "`stamp`" in text
    assert "| Split" in text and "| test " in text
    assert "../../seed0/inference/stamp/report.md" in text
    assert "`rmse_mean`" in text


def _config(runs_dir: Path, group_tags: list[str], inference_subdir: str = "", output_subdir: str = "") -> SimpleNamespace:
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
        maps_subdir          = "maps",
    )


def test_comparison_compares_each_selected_group_in_isolation(tmp_path):
    group_a = tmp_path / "runs" / "group_a"
    _seed_run(group_a, "seed0", {"rmse_mean": 1.0})
    _seed_run(group_a, "seed1", {"rmse_mean": 3.0})

    group_b = tmp_path / "runs" / "group_b"
    _seed_run(group_b, "seed0", {"rmse_mean": 10.0})
    _seed_run(group_b, "seed1", {"rmse_mean": 30.0})

    reports = SeedComparison(_config(tmp_path / "runs", ["group_a", "group_b"], output_subdir="agg"), SilentLogger()).run()

    payload_a = json.loads((group_a / "inference" / "agg" / "metrics.json").read_text(encoding="utf-8"))
    payload_b = json.loads((group_b / "inference" / "agg" / "metrics.json").read_text(encoding="utf-8"))

    assert reports == [group_a / "inference" / "agg" / "report.md", group_b / "inference" / "agg" / "report.md"]
    assert payload_a["mean"]["rmse_mean"] == pytest.approx(2.0)
    assert payload_b["mean"]["rmse_mean"] == pytest.approx(20.0)


def test_comparison_raises_on_unknown_group_tag(tmp_path):
    group = tmp_path / "runs" / "group"
    _seed_run(group, "seed0", {"rmse_mean": 1.0})

    with pytest.raises(FileNotFoundError):
        SeedComparison(_config(tmp_path / "runs", ["group", "missing"]), SilentLogger()).run()


def test_comparison_treats_runs_dir_as_the_group_when_no_tags_are_given(tmp_path):
    group = tmp_path / "runs" / "group"
    _seed_run(group, "seed0", {"rmse_mean": 1.0})
    _seed_run(group, "seed1", {"rmse_mean": 3.0})

    reports = SeedComparison(_config(group, [], output_subdir="agg"), SilentLogger()).run()

    payload = json.loads((group / "inference" / "agg" / "metrics.json").read_text(encoding="utf-8"))

    assert reports == [group / "inference" / "agg" / "report.md"]
    assert payload["mean"]["rmse_mean"] == pytest.approx(2.0)


def test_comparison_output_falls_back_to_the_inference_stamp(tmp_path):
    group = tmp_path / "runs" / "group"
    _seed_run(group, "seed0", {"rmse_mean": 1.0})
    _seed_run(group, "seed1", {"rmse_mean": 3.0})

    SeedComparison(_config(group, [], inference_subdir="stamp"), SilentLogger()).run()

    assert (group / "inference" / "stamp" / "metrics.json").is_file()


def test_comparison_output_gets_a_fresh_stamp_when_nothing_is_set(tmp_path):
    group = tmp_path / "runs" / "group"
    _seed_run(group, "seed0", {"rmse_mean": 1.0})
    _seed_run(group, "seed1", {"rmse_mean": 3.0})

    reports = SeedComparison(_config(group, []), SilentLogger()).run()

    assert reports[0].parent.parent == group / "inference"
    assert reports[0].parent.name not in ("stamp", "")


def _seed_maps(run: Path, predictions: np.ndarray, targets: np.ndarray, stamp: str = "stamp") -> None:
    maps_dir = run / "inference" / stamp / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    np.save(maps_dir / "predictions.npy", predictions)
    np.save(maps_dir / "targets.npy", targets)


def test_seed_ensemble_scores_the_mean_prediction(tmp_path):
    group   = tmp_path / "group"
    targets = np.arange(32, dtype=np.float32).reshape(2, 4, 4)

    run_a = _seed_run(group, "seed0", {"rmse_mean": 1.0})
    run_b = _seed_run(group, "seed1", {"rmse_mean": 1.0})
    _seed_maps(run_a, targets + 1.0, targets)
    _seed_maps(run_b, targets - 1.0, targets)

    maps    = SeedMaps([run_a, run_b], [run / "inference" / "stamp" for run in (run_a, run_b)], "maps")
    scalars = SeedEnsembleEvaluation(maps, SilentLogger()).run()

    assert scalars["n_seeds"]   == 2
    assert scalars["rmse_mean"] == pytest.approx(0.0)
    assert scalars["mae_a"]     == pytest.approx(0.0)
    assert scalars["r2_b"]      == pytest.approx(1.0)


def test_report_renders_the_ensemble_section(tmp_path):
    group = tmp_path / "group"
    runs  = [
        _seed_run(group, "seed0", {"rmse_mean": 1.0}),
        _seed_run(group, "seed1", {"rmse_mean": 3.0}),
    ]

    report = SeedComparisonReport(
        group_dir        = group,
        run_dirs         = runs,
        inference_dirs   = [run / "inference" / "stamp" for run in runs],
        output_subdir    = "stamp",
        metrics_filename = "metrics.json",
        report_filename  = "report.md",
        logger           = SilentLogger(),
        ensemble         = {"n_seeds": 2, "rmse_mean": 0.5},
    )

    text    = report.run().read_text(encoding="utf-8")
    payload = json.loads((group / "inference" / "stamp" / "metrics.json").read_text(encoding="utf-8"))

    assert "Seed ensemble" in text
    assert payload["ensemble"]["rmse_mean"] == pytest.approx(0.5)


def test_comparison_logs_through_the_given_logger(tmp_path):
    group = tmp_path / "runs" / "group"
    _seed_run(group, "seed0", {"rmse_mean": 1.0})
    _seed_run(group, "seed1", {"rmse_mean": 3.0})

    logger = RecordingLogger()
    SeedComparison(_config(group, [], output_subdir="agg"), logger).run()

    assert ("section", "Seed comparison") in logger.messages
    assert ("metrics_table", "Seed inferences") in logger.messages
    assert any(level == "subsection" and message.startswith("Group: ") for level, message in logger.messages)


def test_comparison_counts_the_unfinished_stamps_it_skipped(tmp_path):
    group = tmp_path / "runs" / "group"
    runs  = [_seed_run(group, "seed0", {"rmse_mean": 1.0}), _seed_run(group, "seed1", {"rmse_mean": 3.0})]
    _stamp(runs[0], "zz_crashed", {"rmse_mean": 9.0}, split=None)

    logger = RecordingLogger()
    SeedComparison(_config(group, [], output_subdir="agg"), logger).run()

    payload  = json.loads((group / "inference" / "agg" / "metrics.json").read_text(encoding="utf-8"))
    warnings = [message for level, message in logger.messages if level == "warning"]

    assert payload["seed_inference"]["seed0"]["stamp"] == "stamp"
    assert any("1 unfinished inference stamp(s) skipped across 1 seed run(s)" == message for message in warnings)
