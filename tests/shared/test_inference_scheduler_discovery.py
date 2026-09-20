"""Tests how the shared inference scheduler discovers, selects and skips run directories.

Covers recursive discovery filtered by run type, a root that is itself a run,
filters naming runs by relative path, parent directory or run name, the skip
of runs already scored on the split, unique job names and log paths for
equally named leaves, the GPU pool file location, and the errors raised for a
missing runs directory, an unmatched filter entry or no match.
"""

from __future__ import annotations

from pathlib import Path
from types   import SimpleNamespace

import pytest

from pipelines.shared.inference.inference_scheduler import InferenceScheduler, RunSelection
from pipelines.shared.inference.run_classifier      import InferenceLayout, RunArtifacts, RunType
from tools.runtime.completion                       import CompletionMarker

from tests.conftest import RecordingLogger, SilentLogger


def _make_run(directory: Path, config_name: str) -> Path:
    """Creates a run directory identified by the given meta config filename."""
    (directory / "meta").mkdir(parents=True)
    (directory / "meta" / config_name).write_text("{}")
    return directory


def _stamp(run_dir: Path, run_type: str, split: str, name: str = "20260101_000000") -> Path:
    """Writes a completed inference stamp of the family on the split under the run."""
    stamp = InferenceLayout.base(run_dir, run_type) / name
    stamp.mkdir(parents=True)
    CompletionMarker.stamp(stamp, {"split": split})
    return stamp


def _scheduler(runs: Path, run_filter, run_type: str, gpus_file: str = "", skip_completed: bool = True, split: str = "test") -> InferenceScheduler:
    """Builds an InferenceScheduler over a runs directory.

    Args:
        runs           : Root directory holding the run directories.
        run_filter     : Run names to restrict discovery to; empty selects all.
        run_type       : Run type the scheduler accepts.
        gpus_file      : Explicit GPU pool file path, or empty for the default.
        skip_completed : Whether runs already scored on the split are left alone.
        split          : Split the scheduler scores.

    Returns:
        The configured scheduler.
    """
    config = SimpleNamespace(runs_dir=str(runs), run_filter=run_filter, gpus=[0], gpus_file=gpus_file, poll_interval_s=1, skip_completed=skip_completed)
    return InferenceScheduler(config, Path("main/x.py"), run_type, split=split)


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


def test_a_root_that_is_a_run_is_scored_itself(tmp_path):
    """Verifies pointing runs_dir at one run directory scores that run under its own name."""
    run = _make_run(tmp_path / "runs" / "single_run", RunArtifacts.BACKBONE_CONFIG)

    scheduler = _scheduler(run, [], RunType.BACKBONE)
    matched   = scheduler._run_dirs(SilentLogger())
    jobs      = scheduler._jobs(matched, tmp_path / "resolved_config.json")

    assert matched                    == [run]
    assert [job.name for job in jobs] == ["single_run"]


def test_filter_resolves_nested_relative_names(tmp_path):
    """Verifies a filter given as a nested relative path resolves to that run."""
    runs = tmp_path / "runs"
    _make_run(runs / "group_a" / "run_a1", RunArtifacts.BACKBONE_CONFIG)

    scheduler = _scheduler(runs, ["group_a/run_a1"], RunType.BACKBONE)
    selected  = [str(directory.relative_to(runs)) for directory in scheduler._candidate_dirs()]

    assert selected == ["group_a/run_a1"]


def test_filter_takes_every_run_under_a_named_directory_and_matches_bare_run_names(tmp_path):
    """Verifies a directory name selects all runs below it and a bare run name selects the run wherever it sits."""
    runs = tmp_path / "runs"
    _make_run(runs / "exp" / "trial_a" / "seed0", RunArtifacts.BACKBONE_CONFIG)
    _make_run(runs / "exp" / "trial_a" / "seed1", RunArtifacts.BACKBONE_CONFIG)
    _make_run(runs / "exp" / "trial_b" / "seed0", RunArtifacts.BACKBONE_CONFIG)
    _make_run(runs / "other" / "lonely", RunArtifacts.BACKBONE_CONFIG)

    by_parent = [str(directory.relative_to(runs)) for directory in _scheduler(runs, ["exp/trial_a"], RunType.BACKBONE)._candidate_dirs()]
    by_name   = [str(directory.relative_to(runs)) for directory in _scheduler(runs, ["lonely", "seed1"], RunType.BACKBONE)._candidate_dirs()]

    assert by_parent == ["exp/trial_a/seed0", "exp/trial_a/seed1"]
    assert by_name   == ["other/lonely", "exp/trial_a/seed1"]


def test_an_unmatched_filter_entry_lists_the_runs_found(tmp_path):
    """Verifies a filter entry naming nothing raises and names the runs that were found."""
    runs = tmp_path / "runs"
    _make_run(runs / "group_a" / "run_a1", RunArtifacts.BACKBONE_CONFIG)

    with pytest.raises(FileNotFoundError, match="names no run under .*runs found there: group_a/run_a1"):
        _scheduler(runs, ["missing"], RunType.BACKBONE)._candidate_dirs()


def test_runs_already_scored_on_the_split_are_skipped(tmp_path):
    """Verifies runs holding a completed inference on the split are left alone and reported as skipped."""
    runs    = tmp_path / "runs"
    done    = _make_run(runs / "done", RunArtifacts.BACKBONE_CONFIG)
    pending = _make_run(runs / "pending", RunArtifacts.BACKBONE_CONFIG)
    other   = _make_run(runs / "other_split", RunArtifacts.BACKBONE_CONFIG)
    _stamp(done, RunType.BACKBONE, "test")
    _stamp(other, RunType.BACKBONE, "val")

    scheduler = _scheduler(runs, [], RunType.BACKBONE, split="test")
    matched   = scheduler._run_dirs(SilentLogger())

    assert matched           == [other, pending]
    assert scheduler.skipped == [done]


def test_skipping_can_be_switched_off(tmp_path):
    """Verifies skip_completed off re-scores runs that already hold their inference."""
    runs = tmp_path / "runs"
    done = _make_run(runs / "done", RunArtifacts.BACKBONE_CONFIG)
    _stamp(done, RunType.BACKBONE, "test")

    scheduler = _scheduler(runs, [], RunType.BACKBONE, skip_completed=False)

    assert scheduler._run_dirs(SilentLogger()) == [done]
    assert scheduler.skipped                   == []


def test_nested_families_look_for_their_stamps_under_their_own_subdirectory(tmp_path):
    """Verifies a profile-autoencoder run counts as scored only through the stamps under inference/profile_ae."""
    runs      = tmp_path / "runs"
    done      = _make_run(runs / "ae_done", RunArtifacts.PROFILE_AE_CONFIG)
    misplaced = _make_run(runs / "ae_misplaced", RunArtifacts.PROFILE_AE_CONFIG)
    _stamp(done, RunType.PROFILE_AE, "test")
    _stamp(misplaced, RunType.BACKBONE, "test")

    scheduler = _scheduler(runs, [], RunType.PROFILE_AE)

    assert scheduler._run_dirs(SilentLogger()) == [misplaced]
    assert scheduler.skipped                   == [done]


def test_the_run_table_names_every_matched_run_with_its_action(tmp_path):
    """Verifies the scheduler logs one table over the matched runs before deciding what to score."""
    runs = tmp_path / "runs"
    _make_run(runs / "a", RunArtifacts.BACKBONE_CONFIG)

    logger = RecordingLogger()
    _scheduler(runs, [], RunType.BACKBONE)._run_dirs(logger)

    assert ("metrics_table", f"1 backbone run(s) under {runs}") in logger.messages


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


def test_run_selection_relative_name_of_the_root_is_its_own_name(tmp_path):
    """Verifies the root run is named by its directory name rather than an empty relative path."""
    selection = RunSelection(tmp_path / "run", [tmp_path / "run"])

    assert selection.relative(tmp_path / "run") == "run"
    assert selection.resolve(["run"])           == [tmp_path / "run"]
