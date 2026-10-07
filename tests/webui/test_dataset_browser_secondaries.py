"""Tests for the listing of a dataset's secondary channels behind the launch-form picker.

Covers the channel order and the reference channel read from the dataset layout,
the standard flag taken from the training paths default, and the refusals for a
missing directory, a missing layout and a layout without channel labels.
"""

from __future__ import annotations

import json
from pathlib import Path

from dataset_browser import DatasetBrowser
from web_logger      import WebLogger

from configuration.training.general.run import TrainingPathsConfig


def _dataset(root: Path, labels: list | None) -> Path:
    """Writes a dataset directory whose layout records the given channel labels, or none when ``labels`` is None."""
    data = root / "data"
    data.mkdir(parents=True)
    layout = {"global_crop": [0, 8, 0, 6], "artifacts": {}}
    if labels is not None:
        layout["channel_labels"] = labels
    (data / "dataset.json").write_text(json.dumps(layout))
    return root


def test_secondaries_follow_the_stack_order_after_the_reference(tmp_path):
    """The first channel is the reference and every later one a secondary, flagged when the training default uses it."""
    standard = list(TrainingPathsConfig().secondary_labels)
    dataset  = _dataset(tmp_path / "set", ["CH00", standard[0], "CH99", standard[1]])

    result = DatasetBrowser(WebLogger()).secondaries(str(dataset))

    assert result["ok"], result
    assert result["reference"] == "CH00"
    assert result["standard"] == standard
    assert [entry["name"] for entry in result["secondaries"]] == [standard[0], "CH99", standard[1]]
    assert [entry["standard"] for entry in result["secondaries"]] == [True, False, True]


def test_secondaries_refuse_what_they_cannot_read(tmp_path):
    """A missing directory, a missing layout and an empty channel list each answer ok False with the reason."""
    browser = DatasetBrowser(WebLogger())

    assert "not a directory" in browser.secondaries(str(tmp_path / "nowhere"))["error"]

    (tmp_path / "bare").mkdir()
    assert "no data/dataset.json" in browser.secondaries(str(tmp_path / "bare"))["error"]

    empty = _dataset(tmp_path / "empty", [])
    assert "records no channel labels" in browser.secondaries(str(empty))["error"]
