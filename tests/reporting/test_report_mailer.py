"""Tests covering the mailing of experiment report bundles.

Covers finding experiments by the trial runs they hold (trial sweeps,
cross-validation runs and benchmark runs, wherever they sit), gathering the
seed-aggregate report of every trial, the comparison reports, a
cross-validation run's cross-fold report and the configuration files into a
staging directory, writing the seed aggregates and comparisons that are
missing, assembling the email with attachments, and the batch loop over every
experiment.
"""

from __future__ import annotations

import base64
import os
import json
from pathlib import Path

import numpy as np
import pytest

from configuration.benchmark       import BenchmarkConfig
from configuration.diagnostics     import ReportMailEntryConfig
from tools.reporting.report_mailer import BenchmarkExperiment, CrossValidationExperiment, Experiment, ExperimentReportBundle, ExperimentScanner, ExperimentSelector, ReportMailBatch, ReportMailer
from tools.runtime.completion      import CompletionMarker
from tools.runtime.config_cli      import ConfigCli

from tests.conftest                     import RecordingLogger
from tests.cross_validation.test_stages import build_finished_run


PNG_BYTES = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")


def _write_report(run_dir: Path, tag: str, metrics: dict | None = None, split: str = "test") -> Path:
    """Writes a completed inference stamp with a report.md, one PNG figure, metrics and the prediction cubes under the run.

    The cubes are what the seed comparison's deep-ensemble evaluation reads
    when the mailer writes a missing seed aggregate.

    Args:
        run_dir : Run directory that receives the inference/<tag> subtree.
        tag     : Inference stamp directory name.
        metrics : Scalar metrics written into the stamp, a fixed pair when omitted.
        split   : Split the completion marker records.

    Returns:
        Path of the written report.md.
    """
    output_dir  = run_dir / "inference" / tag
    figures_dir = output_dir / "figures"
    cubes_dir   = output_dir / "cubes"
    figures_dir.mkdir(parents=True)
    cubes_dir.mkdir()

    (figures_dir / "map.png").write_bytes(PNG_BYTES)
    (output_dir / "report.md").write_text(f"# Report ({split})\n\n![map](figures/map.png)\n", encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics or {"curve_rmse_gt": 0.5, "overall_r2_gt": 0.8}), encoding="utf-8")
    np.save(cubes_dir / "pred_curves.npy", np.full((2, 2, 4), 0.5, dtype=np.float32))
    np.save(cubes_dir / "gt_curves.npy", np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(2, 2, 4))
    CompletionMarker.stamp(output_dir, {"stage": "inference", "split": split})

    return output_dir / "report.md"


def _write_aggregate(trial_dir: Path, tag: str, inference: dict) -> Path:
    """Writes a seed-aggregate stamp as the seed comparison does: a report and a metrics file naming the averaged seeds.

    Args:
        trial_dir : Seeded trial directory receiving the inference/<tag> stamp.
        tag       : Aggregate stamp directory name.
        inference : Inference stamp name read for each averaged seed, by seed label.

    Returns:
        Path of the written report.md.
    """
    output_dir = trial_dir / "inference" / tag
    output_dir.mkdir(parents=True)

    (output_dir / "report.md").write_text("# Seed aggregate\n\ncurve_rmse_gt 0.5\n", encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps({
        "n_seeds"        : len(inference),
        "seed_runs"      : {seed: str(trial_dir / seed) for seed in inference},
        "seed_inference" : dict(inference),
        "mean"           : {"curve_rmse_gt": 0.5},
        "std"            : {"curve_rmse_gt": 0.1},
    }), encoding="utf-8")

    return output_dir / "report.md"


def _make_run(run_dir: Path) -> Path:
    """Marks a directory as a training run by giving it a meta directory."""
    (run_dir / "meta").mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_experiment(root: Path, name: str, aggregate: bool = True, comparison: bool = True) -> Path:
    """Builds one trial sweep with a seeded trial, a single-run trial and, by default, an aggregate and a comparison.

    Args:
        root       : Directory the sweep folder is created in.
        name       : Name of the sweep folder.
        aggregate  : Whether the seeded trial already holds its seed-aggregate report.
        comparison : Whether a trial-comparison stamp is written.

    Returns:
        Path of the sweep directory.
    """
    experiment_dir = root / name

    seeded = experiment_dir / "trial_a"
    for seed, metrics in (("seed0", {"curve_rmse_gt": 0.4, "activity_threshold": 1e-3}), ("seed1", {"curve_rmse_gt": 0.6, "activity_threshold": 1e-3})):
        _write_report(_make_run(seeded / seed), "20260101_000000", metrics)
    if aggregate:
        _write_aggregate(seeded, "seed_comparison", {"seed0": "20260101_000000", "seed1": "20260101_000000"})

    resolved = seeded / "seed0" / "docs"
    resolved.mkdir(parents=True)
    (resolved / "resolved_entry_config.json").write_text(json.dumps({"backbone_name": "resunet"}), encoding="utf-8")

    _write_report(_make_run(experiment_dir / "trial_b"), "20260102_000000")

    if comparison:
        stamp = experiment_dir / "_comparison" / "20260103_000000"
        stamp.mkdir(parents=True)
        (stamp / "overview.md").write_text("# Overview\n", encoding="utf-8")
        (stamp / "metrics_comparison.md").write_text("# Metrics\n", encoding="utf-8")
        (stamp / "comparison_summary.json").write_text(json.dumps({"inference_split": "test", "trials": [{"name": "trial_a"}, {"name": "trial_b"}]}), encoding="utf-8")

    logs = experiment_dir / "batch_train_logs"
    logs.mkdir()
    (logs / "resolved_config.json").write_text(json.dumps({"trials_mode": "curriculum"}), encoding="utf-8")
    (logs / "train_scheduler_results.json").write_text(json.dumps({"units": 2}), encoding="utf-8")

    return experiment_dir


def _write_cv_run(root: Path, run_tag: str) -> Path:
    """Builds a finished cross-validation run whose fold stamps also carry a report."""
    run_dir = build_finished_run(root, run_tag, n_folds=3, splits=("test",))

    for stamp in sorted(run_dir.glob("folds/*/inference/*")):
        (stamp / "report.md").write_text(f"# Fold report {stamp.parent.parent.name}\n", encoding="utf-8")

    return run_dir


def _write_benchmark_run(root: Path, run_tag: str) -> Path:
    """Builds a benchmark run with two unit runs, one comparison stamp and its pipeline files."""
    run_dir = root / run_tag

    for unit in ("unet-conv", "resunet-conv"):
        _write_report(_make_run(run_dir / "training" / unit), "20260105_000000")

    stamp = run_dir / "comparison" / "20260106_000000"
    stamp.mkdir(parents=True)
    (stamp / "benchmark_overview.md").write_text("# Benchmark\n", encoding="utf-8")
    (stamp / "metrics_comparison.md").write_text("# Metrics\n", encoding="utf-8")
    (stamp / "comparison_summary.json").write_text(json.dumps([{"name": "resunet-conv"}, {"name": "unet-conv"}]), encoding="utf-8")

    pipeline = run_dir / "pipeline"
    pipeline.mkdir()
    ConfigCli.save_resolved(BenchmarkConfig(), pipeline / "resolved_config.json")
    (pipeline / "state.json").write_text(json.dumps({"stages": {}}), encoding="utf-8")

    return run_dir


@pytest.fixture
def workspace(tmp_path):
    """Returns a sequence workspace holding two trial sweeps, a cross-validation run and a benchmark run."""
    root = tmp_path / "sequences" / "plan_20260901"
    _write_experiment(root, "exp_alpha")
    _write_experiment(root, "exp_beta")
    _write_cv_run(root, "cv1")
    _write_benchmark_run(root, "bench")
    (root / "_mailed_reports").mkdir()
    (root / "logs").mkdir()
    return root


@pytest.fixture
def entry_config(workspace, tmp_path):
    """Returns an entry config pointing at the workspace and a staging directory."""
    return ReportMailEntryConfig(
        experiments_dir = workspace,
        staging_dir     = tmp_path / "staging",
        recipient       = "victor@example.com",
        sender          = "victor@example.com",
        password_env    = "TEST_REPORT_MAIL_PASSWORD",
    )


def _bundle(entry_config, experiment):
    """Returns the report bundle of one experiment under the entry config's staging directory."""
    return ExperimentReportBundle(experiment, entry_config.to_config(experiment.label), RecordingLogger())


def test_scanner_finds_every_kind_of_experiment_under_a_workspace(workspace):
    """Verifies the scan finds the sweeps, the cross-validation run and the benchmark run, classified by their contents."""
    found = {experiment.root.name: experiment for experiment in ExperimentScanner(workspace).scan()}

    assert sorted(found) == ["bench", "cv1", "exp_alpha", "exp_beta"]
    assert isinstance(found["cv1"], CrossValidationExperiment)
    assert isinstance(found["bench"], BenchmarkExperiment)
    assert found["exp_alpha"].KIND == "sweep"
    assert found["cv1"].trials_dir  == workspace / "cv1" / "folds"
    assert found["bench"].trials_dir == workspace / "bench" / "training"


def test_scanner_finds_experiments_at_any_depth_without_entering_runs(tmp_path):
    """Verifies nested workspaces are searched, hidden and underscore directories skipped, and trial trees never entered."""
    runs = tmp_path / "runs"
    _write_experiment(runs / "sequences" / "ws_a", "curriculum")
    _write_experiment(runs / "sequences" / "ws_b", "patch")
    _write_experiment(runs / "sequences" / "_archive", "old")
    _write_experiment(runs / "sequences" / ".hidden", "old")
    _write_experiment(runs / "a" / "b" / "c" / "d" / "too_deep", "deep")

    scanner = ExperimentScanner(runs)
    names   = [scanner.name(experiment) for experiment in scanner.scan()]

    assert names == ["sequences/ws_a/curriculum", "sequences/ws_b/patch"]


def test_scanner_goes_on_below_a_sweep_of_standalone_runs(tmp_path):
    """Verifies a directory holding a standalone run beside a seeded sweep yields both the sweep of that run and the nested sweep."""
    family = tmp_path / "backbone"
    _write_report(_make_run(family / "smoke"), "20260101_000000")
    _write_experiment(family, "augmentation")

    scanner = ExperimentScanner(family)
    found   = {scanner.name(experiment): experiment for experiment in scanner.scan()}

    assert sorted(found) == ["augmentation", "backbone"]
    assert found["backbone"].trials()     == [family / "smoke"]
    assert found["augmentation"].trials() == [family / "augmentation" / "trial_a", family / "augmentation" / "trial_b"]


def test_scanner_accepts_one_experiment_its_folds_directory_and_one_trial_as_the_root(workspace):
    """Verifies the root may be a sweep, a run's folds directory or a single trial, each resolving to one experiment."""
    sweep = ExperimentScanner(workspace / "exp_alpha").scan()
    folds = ExperimentScanner(workspace / "cv1" / "folds").scan()
    trial = ExperimentScanner(workspace / "exp_alpha" / "trial_a").scan()

    assert [experiment.root for experiment in sweep] == [workspace / "exp_alpha"]
    assert [experiment.root for experiment in folds] == [workspace / "cv1"]
    assert isinstance(folds[0], CrossValidationExperiment)
    assert trial[0].root   == workspace / "exp_alpha" / "trial_a"
    assert trial[0].trials() == [workspace / "exp_alpha" / "trial_a"]


def test_scanner_names_the_root_experiment_by_its_own_name(workspace):
    """Verifies an experiment that is the root is named by its directory name and labelled by its parent."""
    scanner    = ExperimentScanner(workspace / "exp_alpha")
    experiment = scanner.scan()[0]

    assert scanner.name(experiment) == "exp_alpha"
    assert experiment.label         == "plan_20260901/exp_alpha"


def test_selector_lists_and_filters_experiments(workspace):
    """Verifies the selector discovers every experiment and resolves a filter by relative name."""
    selector = ExperimentSelector(workspace, "_comparison", RecordingLogger())

    assert [experiment.root.name for experiment in selector.select_experiments([])]          == ["bench", "cv1", "exp_alpha", "exp_beta"]
    assert [experiment.root.name for experiment in selector.select_experiments(["exp_beta"])] == ["exp_beta"]


def test_selector_raises_without_any_experiment(tmp_path):
    """Verifies a directory without any trial run raises."""
    (tmp_path / "empty" / "notes").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="No experiment found"):
        ExperimentSelector(tmp_path / "empty", "_comparison", RecordingLogger()).all()


def test_bundle_stages_trial_reports_comparison_and_config_files(entry_config, workspace):
    """Verifies the bundle stages one report per trial, the comparison reports and the sweep's configuration files."""
    staged = _bundle(entry_config, Experiment(workspace / "exp_alpha", workspace / "exp_alpha")).stage()

    assert [path.name for path in staged] == [
        "trial_a.md",
        "trial_b.md",
        "comparison_overview.md",
        "comparison_metrics_comparison.md",
        "resolved_config.json",
        "train_scheduler_results.json",
        "resolved_entry_config.json",
    ]
    assert all(path.is_file() for path in staged)
    assert staged[0].parent == entry_config.staging_dir / "plan_20260901" / "exp_alpha"


def test_bundle_rewrites_image_links_to_absolute_paths(entry_config, workspace):
    """Verifies a staged trial report points at the original figure by absolute path."""
    staged = _bundle(entry_config, Experiment(workspace / "exp_alpha", workspace / "exp_alpha")).stage()

    original = (workspace / "exp_alpha" / "trial_b" / "inference" / "20260102_000000" / "figures" / "map.png").resolve()

    assert original.as_posix() in staged[1].read_text(encoding="utf-8")


def test_bundle_writes_the_missing_seed_aggregate_first(entry_config, tmp_path):
    """Verifies a seeded trial without its aggregate report gets one written from its seed runs before staging."""
    experiment_dir = _write_experiment(tmp_path / "ws", "exp_bare", aggregate=False)

    staged = _bundle(entry_config, Experiment(experiment_dir, experiment_dir)).stage()

    reports = sorted((experiment_dir / "trial_a" / "inference").glob("*/report.md"))
    assert len(reports) == 1
    assert staged[0].name == "trial_a.md"
    assert "curve_rmse_gt" in staged[0].read_text(encoding="utf-8")


def test_bundle_writes_the_missing_trial_comparison_first(entry_config, tmp_path):
    """Verifies a sweep without a comparison stamp gets one written over its trials before staging."""
    experiment_dir = _write_experiment(tmp_path / "ws", "exp_bare", comparison=False)

    staged = _bundle(entry_config, Experiment(experiment_dir, experiment_dir)).stage()

    stamps = sorted(child for child in (experiment_dir / "_comparison").iterdir() if child.is_dir())
    assert len(stamps) == 1
    assert (stamps[0] / "overview.md").is_file()
    assert [path.name for path in staged][2:4] == ["comparison_overview.md", "comparison_metrics_comparison.md"]


def test_bundle_regenerates_what_exists_when_asked(entry_config, workspace):
    """Verifies the regenerate switch rewrites the seed aggregate and the comparison even when present."""
    entry_config.regenerate_reports = True
    experiment_dir = workspace / "exp_alpha"

    _bundle(entry_config, Experiment(experiment_dir, experiment_dir)).stage()

    assert len(list((experiment_dir / "trial_a" / "inference").glob("*/report.md"))) == 2
    assert len([child for child in (experiment_dir / "_comparison").iterdir() if child.is_dir()]) == 2


def _stage_recorded(entry_config, experiment) -> RecordingLogger:
    """Stages one experiment's bundle and returns the logger that recorded it."""
    logger = RecordingLogger()
    ExperimentReportBundle(experiment, entry_config.to_config(experiment.label), logger).stage()
    return logger


def _warnings(logger: RecordingLogger) -> list[str]:
    """Returns the warning lines a recording logger holds."""
    return [message for level, message in logger.messages if level == "warning"]


def _age(path: Path, seconds: float) -> None:
    """Moves the modification time of a file into the past by the given number of seconds."""
    mtime = path.stat().st_mtime - seconds
    os.utime(path, (mtime, mtime))


def test_bundle_names_reused_reports_and_finds_them_current(entry_config, workspace):
    """Verifies a reused comparison and seed aggregate covering every trial and seed are named in the log without a warning."""
    experiment_dir = workspace / "exp_alpha"
    logger         = _stage_recorded(entry_config, Experiment(experiment_dir, experiment_dir))

    assert ("info", f"Reusing trial comparison {experiment_dir / '_comparison' / '20260103_000000'}") in logger.messages
    assert ("info", f"Reusing seed aggregate of trial_a {experiment_dir / 'trial_a' / 'inference' / 'seed_comparison'}") in logger.messages
    assert _warnings(logger) == []


def test_bundle_warns_when_a_reused_comparison_predates_an_inference(entry_config, workspace):
    """Verifies a trial inferred after the reused comparison was written is named in a staleness warning."""
    experiment_dir = workspace / "exp_alpha"
    stamp          = experiment_dir / "_comparison" / "20260103_000000"
    for path in stamp.iterdir():
        _age(path, 3600)

    warnings = _warnings(_stage_recorded(entry_config, Experiment(experiment_dir, experiment_dir)))

    assert any("Reused trial comparison 20260103_000000 predates 3 completed inference(s)" in warning and "regenerate_reports=True" in warning for warning in warnings)


def test_bundle_warns_when_a_reused_comparison_misses_a_trial_or_scored_another_split(entry_config, workspace):
    """Verifies a reused comparison written over one trial on the val split warns about the missing trial and the split."""
    experiment_dir = workspace / "exp_alpha"
    summary        = experiment_dir / "_comparison" / "20260103_000000" / "comparison_summary.json"
    summary.write_text(json.dumps({"inference_split": "val", "trials": [{"name": "trial_a"}]}), encoding="utf-8")

    warnings = _warnings(_stage_recorded(entry_config, Experiment(experiment_dir, experiment_dir)))

    assert any("does not cover trial_b" in warning for warning in warnings)
    assert any("scored split 'val' while the trial reports are taken from 'test'" in warning for warning in warnings)


def test_bundle_warns_when_a_reused_comparison_carries_no_summary(entry_config, workspace):
    """Verifies a comparison stamp written before summaries existed is reused with a warning that its coverage cannot be checked."""
    experiment_dir = workspace / "exp_alpha"
    (experiment_dir / "_comparison" / "20260103_000000" / "comparison_summary.json").unlink()

    warnings = _warnings(_stage_recorded(entry_config, Experiment(experiment_dir, experiment_dir)))

    assert any("holds no comparison_summary.json" in warning for warning in warnings)


def test_bundle_warns_when_a_reused_seed_aggregate_misses_a_seed(entry_config, workspace):
    """Verifies a seed run added after the reused aggregate was written is named in a warning."""
    experiment_dir = workspace / "exp_alpha"
    seed2          = _make_run(experiment_dir / "trial_a" / "seed2")
    _write_report(seed2, "20260101_000000")
    for path in (experiment_dir / "trial_a" / "inference" / "seed_comparison").iterdir():
        _age(path, -3600)
    for path in (experiment_dir / "_comparison" / "20260103_000000").iterdir():
        _age(path, -3600)

    warnings = _warnings(_stage_recorded(entry_config, Experiment(experiment_dir, experiment_dir)))

    assert warnings == ["Reused seed aggregate of trial_a seed_comparison does not cover trial_a/seed2; set regenerate_reports=True to rewrite it"]


def test_bundle_warns_when_a_seed_aggregate_averages_another_split(entry_config, workspace):
    """Verifies a seed aggregate built on one seed's val inference is mailed with a warning naming that seed and split."""
    experiment_dir = workspace / "exp_alpha"
    _write_report(experiment_dir / "trial_a" / "seed1", "20260108_000000", split="val")
    record = experiment_dir / "trial_a" / "inference" / "seed_comparison" / "metrics.json"
    record.write_text(json.dumps({**json.loads(record.read_text(encoding="utf-8")), "seed_inference": {"seed0": "20260101_000000", "seed1": "20260108_000000"}}), encoding="utf-8")

    warnings = _warnings(_stage_recorded(entry_config, Experiment(experiment_dir, experiment_dir)))

    assert any("averages inference outside the 'test' split the comparison reads: seed1 (20260108_000000, split val)" in warning for warning in warnings)


def test_cross_validation_bundle_warns_when_a_reused_cross_fold_report_predates_a_fold_inference(entry_config, workspace):
    """Verifies a reused cross-fold report older than a fold's completed inference is named and warned about."""
    experiment = CrossValidationExperiment(workspace / "cv1")
    _bundle(entry_config, experiment).stage()

    stamp = sorted((workspace / "cv1" / "reports").iterdir())[-1]
    for path in stamp.iterdir():
        if path.is_file():
            _age(path, 3600)

    logger = _stage_recorded(entry_config, experiment)

    assert ("info", f"Reusing cross-fold report {stamp}") in logger.messages
    assert any(warning.startswith(f"Reused cross-fold report {stamp.name} predates 3 completed inference(s)") for warning in _warnings(logger))


def test_bundle_requires_an_inference_report_on_a_single_run_trial(entry_config, workspace):
    """Verifies a single-run trial without any inference report aborts, since nothing can be written for it."""
    experiment_dir = workspace / "exp_alpha"
    for path in sorted((experiment_dir / "trial_b" / "inference").rglob("*"), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()

    with pytest.raises(FileNotFoundError, match="run inference on the 'test' split of this trial"):
        _bundle(entry_config, Experiment(experiment_dir, experiment_dir)).stage()


def test_bundle_mails_the_report_of_the_comparison_split_and_names_its_source(entry_config, workspace):
    """Verifies a single-run trial inferred on test and later on val mails its test report and logs the stamp it came from."""
    experiment_dir = workspace / "exp_alpha"
    _write_report(experiment_dir / "trial_b", "20260109_000000", split="val")

    logger = RecordingLogger()
    staged = ExperimentReportBundle(Experiment(experiment_dir, experiment_dir), entry_config.to_config("ws/exp_alpha"), logger).stage()

    source = experiment_dir / "trial_b" / "inference" / "20260102_000000" / "report.md"
    assert staged[1].read_text(encoding="utf-8").startswith("# Report (test)")
    assert ("ok", f"Staged trial_b.md from {source}") in logger.messages


def test_bundle_refuses_a_single_run_trial_inferred_only_on_another_split(entry_config, workspace):
    """Verifies a single-run trial whose only report scored val raises, naming the split it has."""
    experiment_dir = workspace / "exp_alpha"
    stamp          = experiment_dir / "trial_b" / "inference" / "20260102_000000"
    CompletionMarker.stamp(stamp, {"stage": "inference", "split": "val"})

    with pytest.raises(FileNotFoundError, match=r"reports exist for split\(s\) val"):
        _bundle(entry_config, Experiment(experiment_dir, experiment_dir)).stage()


def test_bundle_finds_a_report_under_a_family_sub_directory(entry_config, workspace):
    """Verifies a single-run trial whose stamps nest under a family sub-directory mails that report."""
    experiment_dir = workspace / "exp_alpha"
    family_run     = experiment_dir / "trial_b"
    nested         = family_run / "inference" / "profile_ae"
    nested.mkdir()
    (family_run / "inference" / "20260102_000000").rename(nested / "20260102_000000")

    staged = _bundle(entry_config, Experiment(experiment_dir, experiment_dir)).stage()

    assert staged[1].name == "trial_b.md"


def test_bundle_of_a_single_trial_skips_the_comparison(entry_config, workspace):
    """Verifies an experiment of one trial stages its report and configuration without a comparison."""
    trial  = workspace / "exp_alpha" / "trial_a"
    staged = _bundle(entry_config, Experiment(trial, trial.parent, selected=[trial])).stage()

    assert [path.name for path in staged] == ["trial_a.md", "resolved_config.json", "train_scheduler_results.json", "resolved_entry_config.json"]


def test_bundle_warns_when_config_files_are_absent(entry_config, tmp_path):
    """Verifies missing configuration files only warn, one line each."""
    experiment_dir = _write_experiment(tmp_path / "ws", "exp_alpha")
    for name in ("resolved_config.json", "train_scheduler_results.json"):
        (experiment_dir / "batch_train_logs" / name).unlink()

    logger = RecordingLogger()
    staged = ExperimentReportBundle(Experiment(experiment_dir, experiment_dir), entry_config.to_config("ws/exp_alpha"), logger).stage()

    warnings = [message for level, message in logger.messages if level == "warning"]
    assert [path.name for path in staged][-1] == "resolved_entry_config.json"
    assert len(warnings) == 2


def test_cross_validation_bundle_stages_folds_comparison_cross_fold_report_and_pipeline_files(entry_config, workspace):
    """Verifies a cross-validation run stages one report per fold, the fold comparison, the cross-fold report with its split pages, and the pipeline files."""
    staged = _bundle(entry_config, CrossValidationExperiment(workspace / "cv1")).stage()
    names  = [path.name for path in staged]

    assert names[:3]  == ["resunet_fold_0.md", "resunet_fold_1.md", "resunet_fold_2.md"]
    assert names[3:5] == ["comparison_overview.md", "comparison_metrics_comparison.md"]
    assert names[5:9] == ["cv_aggregate_report.md", "cv_summary.json", "test_benchmark_overview.md", "test_metrics_comparison.md"]
    assert names[9:]  == ["state.json"]
    assert (workspace / "cv1" / "folds" / "_comparison").is_dir()
    assert (workspace / "cv1" / "reports").is_dir()


def test_cross_validation_bundle_keeps_an_existing_cross_fold_report(entry_config, workspace, monkeypatch):
    """Verifies an existing cross-fold report stamp is gathered as it is unless regeneration is asked for."""
    from itertools import count

    from tools.runtime.run_tag import RunTag

    stamps = count()
    monkeypatch.setattr(RunTag, "now", classmethod(lambda cls: f"stamp_{next(stamps):03d}"))

    experiment = CrossValidationExperiment(workspace / "cv1")
    _bundle(entry_config, experiment).stage()
    first = sorted((workspace / "cv1" / "reports").iterdir())

    _bundle(entry_config, experiment).stage()
    assert sorted((workspace / "cv1" / "reports").iterdir()) == first

    entry_config.regenerate_reports = True
    _bundle(entry_config, experiment).stage()
    assert len(sorted((workspace / "cv1" / "reports").iterdir())) == 2


def test_benchmark_bundle_stages_units_its_own_comparison_and_pipeline_files(entry_config, workspace):
    """Verifies a benchmark run stages one report per unit, its own comparison pages and the pipeline files."""
    staged = _bundle(entry_config, BenchmarkExperiment(workspace / "bench")).stage()

    assert [path.name for path in staged] == [
        "resunet-conv.md",
        "unet-conv.md",
        "comparison_benchmark_overview.md",
        "comparison_metrics_comparison.md",
        "resolved_config.json",
        "state.json",
    ]


def test_benchmark_bundle_regenerates_its_comparison_on_the_recorded_split(entry_config, workspace):
    """Verifies regenerating a benchmark run's comparison runs its comparison stage on the split its configuration recorded."""
    entry_config.regenerate_reports = True
    experiment = BenchmarkExperiment(workspace / "bench")

    staged = _bundle(entry_config, experiment).stage()

    stamps = sorted(child for child in (workspace / "bench" / "comparison").iterdir() if child.is_dir())
    assert experiment.inference_split() == BenchmarkConfig().inference.split
    assert len(stamps) == 2
    assert [path.name for path in staged][2:4] == ["comparison_benchmark_overview.md", "comparison_metrics_comparison.md"]


def test_mailer_raises_without_the_password_variable(entry_config, monkeypatch, tmp_path):
    """Verifies sending without the password variable raises with the variable name."""
    monkeypatch.delenv(entry_config.password_env, raising=False)
    attachment = tmp_path / "a.md"
    attachment.write_text("x", encoding="utf-8")

    with pytest.raises(EnvironmentError, match=entry_config.password_env):
        ReportMailer(entry_config, RecordingLogger()).send("subject", "body", [attachment])


def test_mailer_dry_run_needs_no_password_and_sends_nothing(entry_config, monkeypatch, fake_smtp, tmp_path):
    """Verifies a dry run stages the message without touching SMTP."""
    monkeypatch.delenv(entry_config.password_env, raising=False)
    entry_config.dry_run = True
    attachment = tmp_path / "a.md"
    attachment.write_text("x", encoding="utf-8")

    ReportMailer(entry_config, RecordingLogger()).send("subject", "body", [attachment])

    assert fake_smtp.instances == []


def test_batch_sends_one_message_per_experiment(entry_config, monkeypatch, fake_smtp):
    """Verifies the batch mails every experiment of the workspace with its staged files attached and labelled by kind."""
    monkeypatch.setenv(entry_config.password_env, "app-password")

    results = ReportMailBatch(entry_config, RecordingLogger()).run()

    assert [result["label"] for result in results] == ["plan_20260901/bench", "plan_20260901/cv1", "plan_20260901/exp_alpha", "plan_20260901/exp_beta"]
    assert len(fake_smtp.instances) == 4

    message     = fake_smtp.instances[2].messages[0]
    attachments = [part.get_filename() for part in message.iter_attachments()]

    assert message["To"] == "victor@example.com"
    assert message["Subject"] == f"{entry_config.subject_prefix}: plan_20260901/exp_alpha"
    assert attachments == ["trial_a.md", "trial_b.md", "comparison_overview.md", "comparison_metrics_comparison.md", "resolved_config.json", "train_scheduler_results.json", "resolved_entry_config.json"]
    assert fake_smtp.instances[2].logins == [("victor@example.com", "app-password")]
    assert "Report bundle of experiment" in message.get_body(preferencelist=("plain",)).get_content()


def test_batch_respects_the_experiment_filter(entry_config, monkeypatch, fake_smtp):
    """Verifies the filter restricts the batch to the named experiment."""
    monkeypatch.setenv(entry_config.password_env, "app-password")
    entry_config.experiment_filter = ["exp_beta"]

    results = ReportMailBatch(entry_config, RecordingLogger()).run()

    assert len(results) == 1
    assert results[0]["subjects"][0].endswith("exp_beta")


def test_batch_mails_the_run_whose_folds_directory_is_the_root(entry_config, workspace, monkeypatch, fake_smtp):
    """Verifies pointing the path at a run's folds directory mails that cross-validation run under its own label."""
    monkeypatch.setenv(entry_config.password_env, "app-password")
    entry_config.experiments_dir = workspace / "cv1" / "folds"

    results = ReportMailBatch(entry_config, RecordingLogger()).run()

    assert [result["label"] for result in results] == ["plan_20260901/cv1"]
    assert results[0]["n_attachments"] == 10


def test_same_named_sweeps_under_different_parents_do_not_collide(entry_config, tmp_path):
    """Two 'curriculum' sweeps of different workspaces get distinct labels and staging trees."""
    first  = Experiment(tmp_path / "ws_a" / "curriculum", tmp_path / "ws_a" / "curriculum")
    second = Experiment(tmp_path / "ws_b" / "curriculum", tmp_path / "ws_b" / "curriculum")

    assert first.label  == "ws_a/curriculum"
    assert second.label == "ws_b/curriculum"
    assert entry_config.to_config(first.label).staging_dir  == entry_config.staging_dir / "ws_a" / "curriculum"
    assert entry_config.to_config(second.label).staging_dir == entry_config.staging_dir / "ws_b" / "curriculum"
