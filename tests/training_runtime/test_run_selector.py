"""Tests the run selector's resolution of names given on the command line.

Covers unique names and root-relative paths, the expansion of a parent directory
name to the runs beneath it, the refusal of a name several runs share with the
candidates listed, the missing-run error for a name nothing matches, and the
report selector finding stamps under the inference family sub-directories."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest             import RecordingLogger
from tools.runtime.run_selector import ReportRunSelector, RunSelector

MARKER = "best_model.pt"


def _runs(tmp_path: Path) -> Path:
    """Writes two trials with a seed0 run each plus a stand-alone run, and returns the runs root.

    Args:
        tmp_path : Directory the runs root is created in.

    Returns:
        The runs root.
    """
    root = tmp_path / "runs"

    for relative in ("trial_a/seed0", "trial_b/seed0", "solo"):
        (root / relative).mkdir(parents=True)
        (root / relative / MARKER).write_bytes(b"x")

    return root


def test_a_name_shared_by_several_runs_is_refused_with_the_candidates(tmp_path):
    """Verifies a seed directory name every trial carries is reported as ambiguous, naming each candidate, not as missing."""
    root = _runs(tmp_path)

    with pytest.raises(ValueError, match="run name 'seed0' is ambiguous") as refusal:
        RunSelector(root, MARKER, RecordingLogger()).filter(["seed0"])

    assert "trial_a/seed0" in str(refusal.value)
    assert "trial_b/seed0" in str(refusal.value)
    assert f"pass the path relative to {root}" in str(refusal.value)


def test_a_root_relative_path_resolves_a_shared_name(tmp_path):
    """Verifies the path relative to the runs root picks one of the runs sharing a name."""
    root = _runs(tmp_path)

    assert RunSelector(root, MARKER, RecordingLogger()).filter(["trial_a/seed0"]) == [root / "trial_a" / "seed0"]


def test_a_unique_name_and_a_parent_name_resolve_as_before(tmp_path):
    """Verifies a unique run name matches its run and a parent directory name expands to the runs beneath it."""
    root = _runs(tmp_path)

    assert RunSelector(root, MARKER, RecordingLogger()).filter(["solo"])    == [root / "solo"]
    assert RunSelector(root, MARKER, RecordingLogger()).filter(["trial_b"]) == [root / "trial_b" / "seed0"]


def test_a_name_nothing_matches_is_still_reported_as_missing(tmp_path):
    """Verifies a name that is neither a run nor a parent directory keeps the missing-run error."""
    root = _runs(tmp_path)

    with pytest.raises(FileNotFoundError, match="No run 'nope'"):
        RunSelector(root, MARKER, RecordingLogger()).filter(["nope"])


def test_the_report_selector_finds_reports_under_a_family_subdirectory(tmp_path):
    """Verifies a run whose stamps sit under inference/<family>/ is offered beside a run with direct stamps, its reports ordered by stamp name."""
    for relative in ("run_profile/inference/profile_ae/20260102_000000", "run_profile/inference/profile_ae/20260101_000000", "run_direct/inference/20260103_000000"):
        (tmp_path / relative).mkdir(parents=True)
        (tmp_path / relative / "report.md").write_text("# report\n")

    selector = ReportRunSelector(tmp_path, "inference", "report.md", RecordingLogger())
    reports  = selector._reports_of(tmp_path / "run_profile" / "inference")

    assert selector.all() == [tmp_path / "run_direct", tmp_path / "run_profile"]
    assert [path.parent.name for path in reports] == ["20260101_000000", "20260102_000000"]
