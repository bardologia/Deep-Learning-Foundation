
from __future__ import annotations

from pathlib import Path

from tests.conftest             import RecordingLogger
from tools.runtime.run_selector import RunSelector

MARKER = "best_model.pt"


def _runs(tmp_path: Path) -> Path:
    root = tmp_path / "runs"

    for relative in ("trial_a/seed0", "trial_b/seed0", "solo"):
        (root / relative).mkdir(parents=True)
        (root / relative / MARKER).write_bytes(b"x")

    return root


def test_a_root_relative_path_resolves_a_shared_name(tmp_path):
    root = _runs(tmp_path)

    assert RunSelector(root, MARKER, RecordingLogger()).filter(["trial_a/seed0"]) == [root / "trial_a" / "seed0"]


def test_a_unique_name_and_a_parent_name_resolve_as_before(tmp_path):
    root = _runs(tmp_path)

    assert RunSelector(root, MARKER, RecordingLogger()).filter(["solo"])    == [root / "solo"]
    assert RunSelector(root, MARKER, RecordingLogger()).filter(["trial_b"]) == [root / "trial_b" / "seed0"]
