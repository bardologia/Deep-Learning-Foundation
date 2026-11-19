
from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest             import RecordingLogger
from tools.runtime.run_selector import ReportRunSelector, RunSelector

MARKER = "best_model.pt"


def _runs(tmp_path: Path) -> Path:
    root = tmp_path / "runs"

    for relative in ("trial_a/seed0", "trial_b/seed0", "solo"):
        (root / relative).mkdir(parents=True)
        (root / relative / MARKER).write_bytes(b"x")

    return root


def test_a_name_shared_by_several_runs_is_refused_with_the_candidates(tmp_path):
    root = _runs(tmp_path)

    with pytest.raises(ValueError, match="run name 'seed0' is ambiguous") as refusal:
        RunSelector(root, MARKER, RecordingLogger()).filter(["seed0"])

    assert "trial_a/seed0" in str(refusal.value)
    assert "trial_b/seed0" in str(refusal.value)
    assert f"pass the path relative to {root}" in str(refusal.value)


def test_a_root_relative_path_resolves_a_shared_name(tmp_path):
    root = _runs(tmp_path)

    assert RunSelector(root, MARKER, RecordingLogger()).filter(["trial_a/seed0"]) == [root / "trial_a" / "seed0"]


def test_a_unique_name_and_a_parent_name_resolve_as_before(tmp_path):
    root = _runs(tmp_path)

    assert RunSelector(root, MARKER, RecordingLogger()).filter(["solo"])    == [root / "solo"]
    assert RunSelector(root, MARKER, RecordingLogger()).filter(["trial_b"]) == [root / "trial_b" / "seed0"]


def test_a_name_nothing_matches_is_still_reported_as_missing(tmp_path):
    root = _runs(tmp_path)

    with pytest.raises(FileNotFoundError, match="No run 'nope'"):
        RunSelector(root, MARKER, RecordingLogger()).filter(["nope"])


def test_the_report_selector_finds_reports_under_a_family_subdirectory(tmp_path):
    for relative in ("run_profile/inference/profile_ae/20260102_000000", "run_profile/inference/profile_ae/20260101_000000", "run_direct/inference/20260103_000000"):
        (tmp_path / relative).mkdir(parents=True)
        (tmp_path / relative / "report.md").write_text("# report\n")

    selector = ReportRunSelector(tmp_path, "inference", "report.md", RecordingLogger())
    reports  = selector._reports_of(tmp_path / "run_profile" / "inference")

    assert selector.all() == [tmp_path / "run_direct", tmp_path / "run_profile"]
    assert [path.parent.name for path in reports] == ["20260101_000000", "20260102_000000"]
