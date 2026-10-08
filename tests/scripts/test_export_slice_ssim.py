"""Tests that the slice SSIM export writes one curve per image axis from an inference stamp's metrics.

Covers the column and row curves written in slice order with non-finite slices kept as
NaN, and the refusals when the per-slice keys do not match the recorded split region.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT   = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from export_slice_ssim import SliceSsimExporter

from tests.conftest import RecordingLogger


def _metrics(stamp: Path, col_ssim: list, row_ssim: list) -> Path:
    """Writes a metrics.json over a three-row, four-column split region with the given per-slice SSIM.

    Args:
        stamp    : Inference stamp directory receiving metrics.json.
        col_ssim : SSIM per column slice, stored as ssim_gt_col_<i>.
        row_ssim : SSIM per row slice, stored as ssim_gt_row_<i>.

    Returns:
        Path of the written metrics file.
    """
    payload = {"split_region": [10, 13, 20, 24], "ssim_gt_col_mean": 0.5, "ssim_gt_row_mean": 0.5}
    payload.update({f"ssim_gt_col_{i}": value for i, value in enumerate(col_ssim)})
    payload.update({f"ssim_gt_row_{i}": value for i, value in enumerate(row_ssim)})

    stamp.mkdir(parents=True)
    path = stamp / "metrics.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    return path


def test_the_curves_are_written_in_slice_order_beside_the_metrics(tmp_path):
    """The column and row curves land next to metrics.json in slice order, keeping a non-finite slice as NaN, and each file is logged."""
    path   = _metrics(tmp_path / "stamp", [0.1, 0.2, 0.3, 0.4], [0.7, float("nan"), 0.9])
    logger = RecordingLogger()

    SliceSsimExporter(path, logger).run()

    cols = np.load(tmp_path / "stamp" / "ssim_gt_col.npy")
    rows = np.load(tmp_path / "stamp" / "ssim_gt_row.npy")

    assert cols.tolist() == [0.1, 0.2, 0.3, 0.4]
    assert rows[0] == 0.7 and np.isnan(rows[1]) and rows[2] == 0.9
    assert [level for level, _message in logger.messages] == ["info", "info"]
    assert "1 non-finite" in logger.messages[1][1]


def test_more_stored_slices_than_the_region_holds_are_refused(tmp_path):
    """A column key past the region's width means the keys belong to another region, which is refused."""
    path = _metrics(tmp_path / "stamp", [0.1, 0.2, 0.3, 0.4, 0.5], [0.7, 0.8, 0.9])

    with pytest.raises(ValueError, match="past the 4 col slices"):
        SliceSsimExporter(path, RecordingLogger()).run()


def test_a_missing_slice_inside_the_region_is_refused(tmp_path):
    """A row slice of the region without a stored SSIM fails the export."""
    path = _metrics(tmp_path / "stamp", [0.1, 0.2, 0.3, 0.4], [0.7, 0.8])

    with pytest.raises(KeyError, match="ssim_gt_row_2"):
        SliceSsimExporter(path, RecordingLogger()).run()
