"""Tests for the result-table exporter: discovery, column selection and the three outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from configuration.diagnostics          import ResultTableEntryConfig
from pipelines.comparison.result_tables import ResultTableExporter
from tools.runtime.completion           import CompletionMarker
from tests.conftest                     import RecordingLogger, SilentLogger


def _run(root: Path, name: str, stamp: str, metrics: dict, split: str | None = "test") -> Path:
    """Creates one fake run with an inference stamp holding a metrics file, completed on the split unless it is None."""
    stamp_dir = root / name / "inference" / stamp
    stamp_dir.mkdir(parents=True)
    (stamp_dir / "metrics.json").write_text(json.dumps(metrics))

    if split is not None:
        CompletionMarker.stamp(stamp_dir, {"stage": "inference", "split": split})

    return root / name


def _entry(root: Path, **overrides) -> ResultTableEntryConfig:
    """Returns an entry config pointed at the fake runs root and a tables dir beside it."""
    entry            = ResultTableEntryConfig(**overrides)
    entry.runs_dir   = root
    entry.output_dir = root.parent / "tables"

    return entry


def test_exporter_writes_csv_tex_and_md(tmp_path):
    """One row per run lands in all three outputs, with the metric columns shared."""
    root = tmp_path / "runs"
    _run(root, "run_a", "s1", {"curve_mse_gt": 1.25, "overall_r2_gt": 0.5, "split": "test"})
    _run(root, "run_b", "s1", {"curve_mse_gt": 2.5, "overall_r2_gt": 0.25, "split": "test"})

    paths = ResultTableExporter(_entry(root), SilentLogger()).run()

    csv = paths["csv"].read_text().splitlines()
    tex = paths["tex"].read_text()
    md  = paths["md"].read_text()

    assert csv[0] == "run,curve_mse_gt,overall_r2_gt"
    assert csv[1] == "run_a,1.25,0.5"
    assert "\\toprule" in tex and "run\\_a & 1.25 & 0.5" in tex
    assert "`run_b`" in md and "2.5" in md


def test_exporter_uses_latest_stamp_and_nested_runs(tmp_path):
    """The newest stamp of each run wins and nested run directories are discovered."""
    root = tmp_path / "runs"
    _run(root, "group/run_a", "s1", {"curve_mse_gt": 9.0})
    _run(root, "group/run_a", "s2", {"curve_mse_gt": 1.0})

    paths = ResultTableExporter(_entry(root), SilentLogger()).run()

    csv = paths["csv"].read_text()

    assert "group/run_a,1.0" in csv
    assert "9.0" not in csv


def test_exporter_selected_columns_and_sorting(tmp_path):
    """Configured metrics restrict the LaTeX columns and sort_by orders the rows."""
    root = tmp_path / "runs"
    _run(root, "worse",  "s1", {"curve_mse_gt": 3.0, "other": 1.0})
    _run(root, "better", "s1", {"curve_mse_gt": 1.0, "other": 2.0})

    entry = _entry(root, metrics=("curve_mse_gt",), sort_by="curve_mse_gt", ascending=True)
    paths = ResultTableExporter(entry, SilentLogger()).run()

    tex = paths["tex"].read_text()

    assert "other" not in tex
    assert tex.index("better") < tex.index("worse")


def test_exporter_rejects_unknown_metric_keys(tmp_path):
    """A configured column no run reports fails loudly instead of rendering blanks."""
    root = tmp_path / "runs"
    _run(root, "run_a", "s1", {"curve_mse_gt": 1.0})

    entry = _entry(root, metrics=("not_a_metric",))

    with pytest.raises(KeyError, match="not_a_metric"):
        ResultTableExporter(entry, SilentLogger()).run()


def test_exporter_raises_without_finished_inference(tmp_path):
    """A runs root with no finished inference is rejected."""
    root = tmp_path / "runs"
    (root / "run_a" / "inference").mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        ResultTableExporter(_entry(root), SilentLogger()).run()


def test_exporter_reads_only_completed_stamps_of_the_split(tmp_path):
    """A newer unfinished or other-split stamp is skipped and the completed stamp of the split is exported."""
    root = tmp_path / "runs"
    _run(root, "run_a", "s1", {"curve_mse_gt": 1.0})
    _run(root, "run_a", "s2", {"curve_mse_gt": 5.0}, split="val")
    _run(root, "run_a", "s3", {"curve_mse_gt": 9.0}, split=None)

    logger = RecordingLogger()
    paths  = ResultTableExporter(_entry(root), logger).run()

    csv = paths["csv"].read_text()

    assert "run_a,1.0" in csv
    assert any(level == "warning" and "s3" in message for level, message in logger.messages)
