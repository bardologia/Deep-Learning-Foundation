"""Tests the completion marker written into finished run directories.

Covers the marker path, the stamped payload, clearing, and the states that must
not count as complete."""

from __future__ import annotations

import json

from tools.runtime.completion import CompletionMarker


def test_path_points_at_marker_filename(tmp_path):
    """Verifies the marker path is complete.json inside the run directory."""
    assert CompletionMarker.path(tmp_path) == tmp_path / "complete.json"


def test_fresh_directory_is_not_complete(tmp_path):
    """Verifies an unstamped directory is not reported complete."""
    assert not CompletionMarker.is_complete(tmp_path)


def test_stamp_makes_directory_complete_and_stores_payload(tmp_path):
    """Verifies stamping stores the given payload plus a completion timestamp."""
    CompletionMarker.stamp(tmp_path, {"stage": "training", "epochs_total": 5})

    assert CompletionMarker.is_complete(tmp_path)

    payload = json.loads(CompletionMarker.path(tmp_path).read_text())
    assert payload["stage"]        == "training"
    assert payload["epochs_total"] == 5
    assert payload["completed_at"]


def test_clear_removes_marker(tmp_path):
    """Verifies clearing removes the marker so the directory is incomplete again."""
    CompletionMarker.stamp(tmp_path, {"stage": "training"})
    CompletionMarker.clear(tmp_path)

    assert not CompletionMarker.is_complete(tmp_path)


def test_clear_on_missing_marker_is_silent(tmp_path):
    """Verifies clearing a directory without a marker is a no-op."""
    CompletionMarker.clear(tmp_path)

    assert not CompletionMarker.is_complete(tmp_path)


def test_directory_file_is_not_a_marker(tmp_path):
    """Verifies a directory named complete.json does not count as a marker."""
    CompletionMarker.path(tmp_path).mkdir()

    assert not CompletionMarker.is_complete(tmp_path)
