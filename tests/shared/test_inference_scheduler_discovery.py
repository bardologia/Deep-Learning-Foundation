"""Tests how the shared inference scheduler discovers and selects run directories.

Covers recursive discovery filtered by run type, nested relative run filters,
unique job names and log paths for equally named leaves, the GPU pool file
location, and the errors raised for a missing runs directory or no match."""

from __future__ import annotations

from pathlib import Path
from types   import SimpleNamespace

import pytest

from pipelines.shared.inference.inference_scheduler import InferenceScheduler
from pipelines.shared.inference.run_classifier      import RunArtifacts, RunType

from tests.conftest import SilentLogger


def _make_run(directory: Path, config_name: str) -> None:
    """Creates a run directory identified by the given meta config filename."""
    (directory / "meta").mkdir(parents=True)
    (directory / "meta" / config_name).write_text("{}")


def _scheduler(runs: Path, run_filter, run_type: str, gpus_file: str = "") -> InferenceScheduler:
    """Builds an InferenceScheduler over a runs directory.

    Args:
        runs       : Root directory holding the run directories.
        run_filter : Run names to restrict discovery to; empty selects all.
        run_type   : Run type the scheduler accepts.
        gpus_file  : Explicit GPU pool file path, or empty for the default.

    Returns:
        The configured scheduler.
    """
    config = SimpleNamespace(runs_dir=str(runs), run_filter=run_filter, gpus=[0], gpus_file=gpus_file, poll_interval_s=1)
    return InferenceScheduler(config, Path("main/x.py"), run_type)


def test_no_filter_discovers_runs_at_any_depth_and_respects_type(tmp_path):
    """Verifies unfiltered discovery walks all depths and keeps only the requested run type."""
    runs = tmp_path / "runs"
    _make_run(runs / "run_top", RunArtifacts.BACKBONE_CONFIG)
    _make_run(runs / "group_a" / "run_a1", RunArtifacts.BACKBONE_CONFIG)
    _make_run(runs / "group_a" / "deep" / "run_a2", RunArtifacts.BACKBONE_CONFIG)
    _make_run(runs / "group_b" / "ae_run", RunArtifacts.PROFILE_AE_CONFIG)
    (runs / "empty").mkdir(parents=True)

    scheduler = _scheduler(runs, [], RunType.BACKBONE)
    matched   = sorted(str(directory.relative_to(runs)) for directory in scheduler._run_dirs(SilentLogger()))

    assert matched == ["group_a/deep/run_a2", "group_a/run_a1", "run_top"]


def test_filter_resolves_nested_relative_names(tmp_path):
    """Verifies a filter given as a nested relative path resolves to that run."""
    runs = tmp_path / "runs"
    _make_run(runs / "group_a" / "run_a1", RunArtifacts.BACKBONE_CONFIG)

    scheduler = _scheduler(runs, ["group_a/run_a1"], RunType.BACKBONE)
    selected  = [str(directory.relative_to(runs)) for directory in scheduler._candidate_dirs()]

    assert selected == ["group_a/run_a1"]


def test_job_names_and_logs_stay_unique_for_equally_named_leaves(tmp_path):
    """Verifies equally named fold directories keep distinct job names and log paths."""
    runs = tmp_path / "runs"
    _make_run(runs / "cv_a" / "folds" / "fold_0", RunArtifacts.BACKBONE_CONFIG)
    _make_run(runs / "cv_b" / "folds" / "fold_0", RunArtifacts.BACKBONE_CONFIG)

    scheduler = _scheduler(runs, [], RunType.BACKBONE)
    jobs      = scheduler._jobs(scheduler._run_dirs(SilentLogger()), tmp_path / "resolved_config.json")

    assert [job.name for job in jobs]          == ["cv_a/folds/fold_0", "cv_b/folds/fold_0"]
    assert len({job.log_path for job in jobs}) == 2


def test_missing_runs_directory_raises(tmp_path):
    """Verifies a missing runs directory is reported instead of yielding no runs."""
    scheduler = _scheduler(tmp_path / "absent", [], RunType.BACKBONE)

    with pytest.raises(NotADirectoryError, match="runs_dir does not exist"):
        scheduler._run_dirs(SilentLogger())


def test_pool_file_defaults_into_the_work_directory(tmp_path):
    """Verifies the GPU pool file defaults into the scheduler work directory."""
    scheduler = _scheduler(tmp_path / "runs", [], RunType.BACKBONE)

    assert scheduler.pool_file == scheduler.work_dir / "gpu_pool.json"


def test_pool_file_honors_configured_gpus_file(tmp_path):
    """Verifies an explicit gpus_file path overrides the default pool file location."""
    configured = tmp_path / "pools" / "job42.json"
    scheduler  = _scheduler(tmp_path / "runs", [], RunType.BACKBONE, gpus_file=str(configured))

    assert scheduler.pool_file == configured


def test_zero_matching_runs_raises(tmp_path):
    """Verifies discovering no run of the requested type raises."""
    runs = tmp_path / "runs"
    _make_run(runs / "ae_run", RunArtifacts.PROFILE_AE_CONFIG)

    scheduler = _scheduler(runs, [], RunType.BACKBONE)

    with pytest.raises(FileNotFoundError, match="No run of type 'backbone'"):
        scheduler._run_dirs(SilentLogger())
