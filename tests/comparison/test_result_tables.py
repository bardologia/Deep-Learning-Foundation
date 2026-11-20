
from __future__ import annotations

import json
from pathlib import Path


from configuration.diagnostics          import ResultTableEntryConfig
from pipelines.comparison.result_tables import ResultTableExporter
from tools.runtime.completion           import CompletionMarker
from tests.conftest                     import RecordingLogger, SilentLogger


def _run(root: Path, name: str, stamp: str, metrics: dict, split: str | None = "test") -> Path:
    stamp_dir = root / name / "inference" / stamp
    stamp_dir.mkdir(parents=True)
    (stamp_dir / "metrics.json").write_text(json.dumps(metrics))

    if split is not None:
        CompletionMarker.stamp(stamp_dir, {"stage": "inference", "split": split})

    return root / name


def _entry(root: Path, **overrides) -> ResultTableEntryConfig:
    entry            = ResultTableEntryConfig(**overrides)
    entry.runs_dir   = root
    entry.output_dir = root.parent / "tables"

    return entry


def test_exporter_writes_csv_tex_and_md(tmp_path):
    root = tmp_path / "runs"
    _run(root, "run_a", "s1", {"rmse_mean": 1.25, "r2_mean": 0.5, "split": "test"})
    _run(root, "run_b", "s1", {"rmse_mean": 2.5, "r2_mean": 0.25, "split": "test"})

    paths = ResultTableExporter(_entry(root), SilentLogger()).run()

    csv = paths["csv"].read_text().splitlines()
    tex = paths["tex"].read_text()
    md  = paths["md"].read_text()

    assert csv[0] == "run,r2_mean,rmse_mean"
    assert csv[1] == "run_a,0.5,1.25"
    assert "\\toprule" in tex and "run\\_a & 0.5 & 1.25" in tex
    assert "`run_b`" in md and "2.5" in md


def test_exporter_uses_latest_stamp_and_nested_runs(tmp_path):
    root = tmp_path / "runs"
    _run(root, "group/run_a", "s1", {"rmse_mean": 9.0})
    _run(root, "group/run_a", "s2", {"rmse_mean": 1.0})

    paths = ResultTableExporter(_entry(root), SilentLogger()).run()

    csv = paths["csv"].read_text()

    assert "group/run_a,1.0" in csv
    assert "9.0" not in csv


def test_exporter_selected_columns_and_sorting(tmp_path):
    root = tmp_path / "runs"
    _run(root, "worse",  "s1", {"rmse_mean": 3.0, "other": 1.0})
    _run(root, "better", "s1", {"rmse_mean": 1.0, "other": 2.0})

    entry = _entry(root, metrics=("rmse_mean",), sort_by="rmse_mean", ascending=True)
    paths = ResultTableExporter(entry, SilentLogger()).run()

    tex = paths["tex"].read_text()

    assert "other" not in tex
    assert tex.index("better") < tex.index("worse")


def test_exporter_reads_only_completed_stamps_of_the_split(tmp_path):
    root = tmp_path / "runs"
    _run(root, "run_a", "s1", {"rmse_mean": 1.0})
    _run(root, "run_a", "s2", {"rmse_mean": 5.0}, split="val")
    _run(root, "run_a", "s3", {"rmse_mean": 9.0}, split=None)

    logger = RecordingLogger()
    paths  = ResultTableExporter(_entry(root), logger).run()

    csv = paths["csv"].read_text()

    assert "run_a,1.0" in csv
    assert any(level == "warning" and "s3" in message for level, message in logger.messages)
