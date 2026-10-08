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


def test_plan_tells_unfinished_runs_with_a_trainer_state_from_those_without(tmp_path):
    """An unfinished run holding last.pt resumes from it; one without restarts, and the table and summary say which is which."""
    _run_dir(tmp_path, "a")
    _run_dir(tmp_path, "b")
    (tmp_path / "a" / "last.pt").write_bytes(b"\0")

    plan = SweepResumePlan(tmp_path, ["a", "b"]).build()

    assert plan.resumable  == ["a"]
    assert plan.restarting == ["b"]
    assert plan.unfinished == ["a", "b"]
    assert plan.summary()["Unfinished"] == "1 with a trainer state (resume from last.pt), 1 without (restart from scratch)"

    status = {row["Run"]: row["Status"] for row in plan.rows()}

    assert status["a"] == "unfinished, trainer state saved (resumes from last.pt)"
    assert status["b"] == "unfinished, no trainer state (restarts from scratch)"


def test_plan_reports_which_completed_runs_lack_inference(tmp_path):
    """With a scored split the plan lists per completed run whether a completed inference stamp for that split exists."""
    _run_dir(tmp_path, "a", complete=True)
    _run_dir(tmp_path, "b", complete=True)
    _run_dir(tmp_path, "c")

    stamp = tmp_path / "a" / "inference" / "20260101_000000"
    stamp.mkdir(parents=True)
    CompletionMarker.stamp(stamp, {"split": "test"})

    plan = SweepResumePlan(tmp_path, ["a", "b", "c"], split="test").build()
    rows = {row["Run"]: row["Inference"] for row in plan.rows()}

    assert plan.inferred == ["a"]
    assert plan.columns() == ["Run", "Status", "Inference"]
    assert rows == {"a": "test done", "b": "test missing", "c": "-"}
    assert plan.summary()["Complete"] == "2 (reused), 1 of them without 'test' inference"


def test_plan_without_a_split_has_no_inference_column(tmp_path):
    """A sweep that infers nothing reports no inference column at all."""
    _run_dir(tmp_path, "a", complete=True)

    plan = SweepResumePlan(tmp_path, ["a"]).build()

    assert plan.columns() == ["Run", "Status"]
    assert "Inference" not in plan.rows()[0]


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
