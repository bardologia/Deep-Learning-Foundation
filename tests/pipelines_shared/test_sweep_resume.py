"""Tests for sweep resume root resolution and the completed, unfinished, and missing run plan."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipelines.shared.training.sweep_resume import SweepResumePlan, SweepResumeRoot
from tools.runtime.completion               import CompletionMarker


def _run_dir(root: Path, name: str, complete: bool = False) -> Path:
    """Creates a run directory under root, stamping the completion marker when complete."""
    run = root / name
    run.mkdir(parents=True)
    if complete:
        CompletionMarker.stamp(run, {})
    return run


def test_root_resolution_accepts_the_sweep_folder(tmp_path):
    """Pointing at the sweep folder itself resolves to that folder."""
    _run_dir(tmp_path, "a")

    assert SweepResumeRoot.resolve(str(tmp_path), ["a", "b"]) == tmp_path


def test_root_resolution_climbs_out_of_a_pasted_run_dir(tmp_path):
    """Pointing at a single run directory resolves to its sweep folder."""
    run = _run_dir(tmp_path, "a")

    assert SweepResumeRoot.resolve(str(run), ["a", "b"]) == tmp_path


def test_root_resolution_climbs_out_of_a_nested_seed_run(tmp_path):
    """Pointing at a nested seed run climbs two levels to the sweep folder."""
    run = _run_dir(tmp_path, "a/seed0")

    assert SweepResumeRoot.resolve(str(run), ["a/seed0", "a/seed1"]) == tmp_path


def test_root_resolution_rejects_a_missing_path(tmp_path):
    """A non-existent resume path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        SweepResumeRoot.resolve(str(tmp_path / "nope"), ["a"])


def test_root_resolution_rejects_a_folder_without_planned_runs(tmp_path):
    """A folder holding none of the planned runs exits with an error."""
    with pytest.raises(SystemExit, match="holds no run directory matching"):
        SweepResumeRoot.resolve(str(tmp_path), ["a", "b"])


def test_plan_classifies_complete_unfinished_and_missing(tmp_path):
    """The plan separates completed, unfinished, and never-created runs."""
    _run_dir(tmp_path, "a", complete=True)
    _run_dir(tmp_path, "b")

    plan = SweepResumePlan(tmp_path, ["a", "b", "c"]).build()

    assert plan.completed  == ["a"]
    assert plan.unfinished == ["b"]
    assert plan.missing    == ["c"]


def test_plan_tracks_missing_seeds_inside_a_base_run(tmp_path):
    """A missing seed inside an existing base run is listed as missing."""
    _run_dir(tmp_path, "base/seed0", complete=True)

    plan = SweepResumePlan(tmp_path, ["base/seed0", "base/seed1"]).build()

    assert plan.completed == ["base/seed0"]
    assert plan.missing   == ["base/seed1"]


def test_plan_flags_unplanned_directories_without_touching_them(tmp_path):
    """Directories outside the plan are flagged and left on disk."""
    _run_dir(tmp_path, "a", complete=True)
    _run_dir(tmp_path, "stray")
    (tmp_path / "batch_train_logs").mkdir()

    plan = SweepResumePlan(tmp_path, ["a"]).build()

    assert plan.unplanned == ["stray"]
    assert (tmp_path / "stray").is_dir()


def test_root_resolution_matches_label_prefixed_run_dirs(tmp_path):
    """Run directories carrying a model-label prefix still match the planned names."""
    _run_dir(tmp_path, "resunet-conv_t-a/seed0")

    assert SweepResumeRoot.resolve(str(tmp_path), ["t-a/seed0", "t-a/seed1"]) == tmp_path


def test_root_resolution_climbs_out_of_a_label_prefixed_run(tmp_path):
    """Pointing at a label-prefixed run directory resolves to its sweep folder."""
    run = _run_dir(tmp_path, "resunet-conv_t-a")

    assert SweepResumeRoot.resolve(str(run), ["t-a", "t-b"]) == tmp_path


def test_plan_fills_missing_seeds_inside_label_prefixed_bases(tmp_path):
    """Completed and missing seeds are resolved inside a label-prefixed base run."""
    for seed in (0, 1, 2):
        _run_dir(tmp_path, f"resunet-conv_t-a/seed{seed}", complete=True)

    names = [f"t-a/seed{seed}" for seed in range(5)]
    plan  = SweepResumePlan(tmp_path, names).build()

    assert plan.completed == names[:3]
    assert plan.missing   == names[3:]
    assert plan.unplanned == []


def test_plan_marks_a_prefixed_base_without_marker_as_unfinished(tmp_path):
    """A label-prefixed base run without a completion marker counts as unfinished."""
    _run_dir(tmp_path, "resunet-conv_t-a")

    plan = SweepResumePlan(tmp_path, ["t-a"]).build()

    assert plan.unfinished == ["t-a"]
