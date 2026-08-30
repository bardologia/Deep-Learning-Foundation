"""Tests that the completion-marker backfill script stamps legacy run directories.

Covers the split recovered for inference outputs, the resulting marker being
honoured by the resume reader, unstampable outputs being reported instead of
guessed, and training runs still being stamped."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from backfill_completion_markers import CompletionMarkerBackfiller

from pipelines.shared.training.unit_resume import UnitResume

from tools.data.io            import FileIO
from tools.runtime.completion import CompletionMarker


class RecordingLogger:

    """Logger stub that stores every emitted message.

    Attributes:
        messages : Messages recorded so far, in emission order.
    """
    def __init__(self) -> None:
        """Initialises an empty message list."""
        self.messages : list[str] = []

    @property
    def text(self) -> str:
        """Returns every recorded message joined by newlines."""
        return "\n".join(self.messages)

    def info(self, message) -> None:
        """Records an informational message."""
        self.messages.append(str(message))

    def warning(self, message) -> None:
        """Records a warning message."""
        self.messages.append(str(message))

    def error(self, message) -> None:
        """Records an error message."""
        self.messages.append(str(message))


def _inference_output(run_dir: Path, tag: str, split: str | None) -> Path:
    """Creates a legacy inference output directory without a completion marker.

    Args:
        run_dir : Run directory that receives the inference subtree.
        tag     : Timestamp tag naming the inference output directory.
        split   : Split recorded in metrics.json, or None to omit the metrics file.

    Returns:
        The created inference output directory.
    """
    output_dir = run_dir / "inference" / tag
    output_dir.mkdir(parents=True)
    (output_dir / "report.md").write_text("report", encoding="utf-8")

    if split is not None:
        FileIO.save_json({"split": split, "curve_mse_gt": 0.1}, output_dir / "metrics.json")

    return output_dir


def test_backfilled_inference_marker_records_the_split(tmp_path):
    """Verifies the split is recovered from metrics.json into the marker payload."""
    run_dir    = tmp_path / "run"
    output_dir = _inference_output(run_dir, "20260617_210444", "test")

    CompletionMarkerBackfiller(tmp_path, RecordingLogger()).run()

    assert CompletionMarker.payload(output_dir)["split"] == "test"


def test_backfilled_marker_is_reused_by_the_resume_reader(tmp_path):
    """Verifies the backfilled marker makes resume skip only the completed split."""
    run_dir = tmp_path / "run"
    _inference_output(run_dir, "20260617_210444", "val")

    CompletionMarkerBackfiller(tmp_path, RecordingLogger()).run()

    assert UnitResume(run_dir, True).skip_inference("val")
    assert not UnitResume(run_dir, True).skip_inference("test")


def test_inference_without_recoverable_split_is_left_unstamped(tmp_path):
    """Verifies an output with no recoverable split stays unstamped and is logged."""
    run_dir    = tmp_path / "run"
    output_dir = _inference_output(run_dir, "20260617_210444", None)
    logger     = RecordingLogger()

    CompletionMarkerBackfiller(tmp_path, logger).run()

    assert not CompletionMarker.is_complete(output_dir)
    assert str(output_dir) in logger.text


def test_training_runs_are_still_stamped(tmp_path):
    """Verifies a run with test metrics is stamped as a completed training stage."""
    run_dir = tmp_path / "run"
    FileIO.save_json({"curve_mse_gt": 0.1}, run_dir / "meta" / "test_metrics.json")

    CompletionMarkerBackfiller(tmp_path, RecordingLogger()).run()

    assert CompletionMarker.payload(run_dir)["stage"] == "training"
