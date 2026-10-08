"""Tests how the shared inference scheduler discovers, selects and skips run directories.

Covers recursive discovery filtered by run type, a root that is itself a run,
filters naming runs by relative path, parent directory or run name, the skip
of runs already scored on the split with the requested checkpoint, partly
stamped region groups and other-checkpoint stamps that are scored again, the
table of rejected runs, unique job names and log paths for equally named
leaves, the GPU pool file location, and the errors raised for a missing runs
directory, an unmatched filter entry, no match or a family block setting a
field the fan-out overwrites.
"""

from __future__ import annotations

from pathlib import Path
from types   import SimpleNamespace

import pytest

from pipelines.shared.inference.inference_scheduler import InferenceScheduler, RunSelection
from pipelines.shared.inference.metadata            import CompletedInference, RegionPass
from pipelines.shared.inference.run_classifier      import InferenceLayout, RunArtifacts, RunType
from tools.runtime.completion                       import CompletionMarker

from tests.conftest import RecordingLogger, SilentLogger


def _make_run(directory: Path, config_name: str) -> Path:
    """Creates a run directory identified by the given meta config filename."""
    (directory / "meta").mkdir(parents=True)
    (directory / "meta" / config_name).write_text("{}")
    return directory


def _stamp(run_dir: Path, run_type: str, split: str, name: str = "20260101_000000", checkpoint: str | None = None) -> Path:
    """Writes a completed inference stamp of the family on the split under the run, recording the checkpoint when one is named."""
    stamp   = InferenceLayout.base(run_dir, run_type) / name
    payload = {"split": split}

    if checkpoint is not None:
        payload[CompletedInference.CHECKPOINT_KEY] = {"name": checkpoint, "epoch": 10, "best_epoch": 10, "best_val_loss": 0.1}

    stamp.mkdir(parents=True)
    CompletionMarker.stamp(stamp, payload)
    return stamp


def _scheduler(runs: Path, run_filter, run_type: str, gpus_file: str = "", skip_completed: bool = True, split: str = "test", checkpoint_name: str = "best_model.pt") -> InferenceScheduler:
    """Builds an InferenceScheduler over a runs directory.

    Args:
        runs            : Root directory holding the run directories.
        run_filter      : Run names to restrict discovery to; empty selects all.
        run_type        : Run type the scheduler accepts.
        gpus_file       : Explicit GPU pool file path, or empty for the default.
        skip_completed  : Whether runs already scored on the split are left alone.
        split           : Split the scheduler scores.
        checkpoint_name : Checkpoint the scheduler scores.

    Returns:
        The configured scheduler.
    """
    config = SimpleNamespace(runs_dir=str(runs), run_filter=run_filter, gpus=[0], gpus_file=gpus_file, poll_interval_s=1, skip_completed=skip_completed)
    block  = SimpleNamespace(split=split, checkpoint_name=checkpoint_name, run_directory=Path("."), output_subdir=None)
    return InferenceScheduler(config, Path("main/x.py"), run_type, block=block)


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


def test_a_partly_stamped_region_group_is_shown_as_partial_and_scored_again(tmp_path):
    """Verifies a run whose region group stamped one of two regions is listed as partial, warned about with the missing region and scored again."""
    runs    = tmp_path / "runs"
    partial = _make_run(runs / "partial", RunArtifacts.BACKBONE_CONFIG)
    stamp   = InferenceLayout.base(partial, RunType.BACKBONE) / "20260101_000000_region0"

    stamp.mkdir(parents=True)
    CompletionMarker.stamp(stamp, {"split": "test", RegionPass.KEY: RegionPass(index=0, count=2, group="20260101_000000").payload()})

    logger    = RecordingLogger()
    scheduler = _scheduler(runs, [], RunType.BACKBONE)

    assert scheduler._run_dirs(logger) == [partial]
    assert scheduler.skipped           == []
    assert ("warning", "partial: the region group 20260101_000000 on split 'test' stamped regions [0] of 2; regions [1] were never scored, so the run is scored again") in logger.messages
    assert scheduler._inference_state(partial, SilentLogger()) == ("partial (1/2 regions)", False)


def test_a_run_scored_only_with_another_checkpoint_is_scored_again(tmp_path):
    """Verifies stamps recording another checkpoint do not count as scored, are warned about and shown as done with the other checkpoint."""
    runs  = tmp_path / "runs"
    other = _make_run(runs / "other", RunArtifacts.BACKBONE_CONFIG)
    same  = _make_run(runs / "same", RunArtifacts.BACKBONE_CONFIG)
    _stamp(other, RunType.BACKBONE, "test", checkpoint="best_model.pt")
    _stamp(same,  RunType.BACKBONE, "test", checkpoint="best_model.pt")
    _stamp(same,  RunType.BACKBONE, "test", name="20260102_000000", checkpoint="last_model.pt")

    logger    = RecordingLogger()
    scheduler = _scheduler(runs, [], RunType.BACKBONE, checkpoint_name="last_model.pt")

    assert scheduler._run_dirs(logger) == [other]
    assert scheduler.skipped           == [same]
    assert ("warning", "other: its stamps on split 'test' were scored with best_model.pt, not the requested last_model.pt; the run is scored again") in logger.messages
    assert scheduler._inference_state(other, SilentLogger()) == ("done (other checkpoint: best_model.pt)", False)


def test_a_stamp_without_a_recorded_checkpoint_still_counts_as_scored(tmp_path):
    """Verifies a marker that records no checkpoint keeps the run skipped and says so in the table state."""
    runs = tmp_path / "runs"
    done = _make_run(runs / "done", RunArtifacts.BACKBONE_CONFIG)
    _stamp(done, RunType.BACKBONE, "test")

    scheduler = _scheduler(runs, [], RunType.BACKBONE, checkpoint_name="last_model.pt")

    assert scheduler._inference_state(done, SilentLogger()) == ("done (checkpoint not recorded)", True)


def test_rejected_runs_are_listed_with_their_family_and_unclassifiable_ones_warned_about(tmp_path):
    """Verifies other-family runs and unclassifiable directories are tabled by name, and the unclassifiable ones warned about with the classifier's reason."""
    runs = tmp_path / "runs"
    _make_run(runs / "backbone_run", RunArtifacts.BACKBONE_CONFIG)
    _make_run(runs / "ae_run", RunArtifacts.PROFILE_AE_CONFIG)
    (runs / "interrupted" / "meta").mkdir(parents=True)

    logger    = RecordingLogger()
    scheduler = _scheduler(runs, [], RunType.BACKBONE)

    assert scheduler._run_dirs(logger) == [runs / "backbone_run"]
    assert ("metrics_table", "Skipping 2 run(s) not of type 'backbone'") in logger.messages

    warnings = [msg for level, msg in logger.messages if level == "warning"]

    assert len(warnings) == 1
    assert warnings[0].startswith("interrupted is left out: Run ") and "has no recognized model config under meta/" in warnings[0]


@pytest.mark.parametrize("field, value, message", [
    ("output_subdir", "final_test",     "output_subdir='final_test'"),
    ("run_directory", Path("runs/one"), "run_directory='runs/one'"),
])
def test_a_block_setting_a_field_the_fan_out_overwrites_is_refused(tmp_path, field, value, message):
    """Verifies the scheduler refuses an output_subdir or run_directory it would silently replace, before walking any run."""
    scheduler = _scheduler(tmp_path / "absent", [], RunType.BACKBONE)
    setattr(scheduler.block, field, value)

    with pytest.raises(ValueError, match=message):
        scheduler.run()
