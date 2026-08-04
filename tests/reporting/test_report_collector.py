from __future__ import annotations

import base64

import pytest

from configuration.diagnostics        import ReportCollectionConfig, ReportCollectionEntryConfig
from tools.reporting.report_collector import ReportCollection, ReportCollectionBatch, ReportImageRewriter, RunReportLocator
from tools.runtime.run_selector       import ReportRunSelector


class RecordingLogger:
    def __init__(self):
        self.messages = []

    def section(self, msg):
        self.messages.append(("section", msg))

    def subsection(self, msg):
        self.messages.append(("subsection", msg))

    def ok(self, msg):
        self.messages.append(("ok", msg))

    def info(self, msg):
        self.messages.append(("info", msg))

    def metrics_table(self, rows, columns=None, title=None):
        self.messages.append(("metrics_table", title))


PNG_BYTES = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")


def _write_report(run_dir, tag, body=None):
    output_dir  = run_dir / "inference" / tag
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True)

    (figures_dir / "map.png").write_bytes(PNG_BYTES)
    (output_dir / "report.md").write_text(body if body is not None else "# Report\n\n![map](figures/map.png)\n", encoding="utf-8")
    return output_dir / "report.md"


@pytest.fixture
def run_directory(tmp_path):
    run_dir = tmp_path / "run_demo"
    _write_report(run_dir, "20260101_000000")
    _write_report(run_dir, "20260102_000000")
    return run_dir


def test_locator_picks_the_latest_report(run_directory):
    config  = ReportCollectionConfig(run_directory=run_directory)
    reports = RunReportLocator(config).locate()

    assert [path.parent.name for path in reports] == ["20260102_000000"]


def test_locator_returns_every_report_when_latest_only_is_off(run_directory):
    config  = ReportCollectionConfig(run_directory=run_directory, latest_only=False)
    reports = RunReportLocator(config).locate()

    assert [path.parent.name for path in reports] == ["20260101_000000", "20260102_000000"]


def test_locator_raises_without_inference_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="inference"):
        RunReportLocator(ReportCollectionConfig(run_directory=tmp_path)).locate()


def test_locator_raises_without_any_report(tmp_path):
    (tmp_path / "inference" / "20260101_000000").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="report.md"):
        RunReportLocator(ReportCollectionConfig(run_directory=tmp_path)).locate()


def test_rewriter_points_relative_links_at_the_original_figures(run_directory):
    report_path = run_directory / "inference" / "20260102_000000" / "report.md"
    rewritten   = ReportImageRewriter(report_path, embed_images=False).rewrite(report_path.read_text())

    expected = (report_path.parent / "figures" / "map.png").resolve().as_posix()
    assert f"![map]({expected})" in rewritten


def test_rewriter_embeds_images_as_data_uris(run_directory):
    report_path = run_directory / "inference" / "20260102_000000" / "report.md"
    rewritten   = ReportImageRewriter(report_path, embed_images=True).rewrite(report_path.read_text())

    payload = base64.b64encode(PNG_BYTES).decode("ascii")
    assert f"![map](data:image/png;base64,{payload})" in rewritten


def test_rewriter_leaves_remote_and_data_links_untouched(run_directory):
    report_path = run_directory / "inference" / "20260102_000000" / "report.md"
    body        = "![a](https://example.org/x.png)\n![b](data:image/png;base64,AAAA)\n"

    assert ReportImageRewriter(report_path, embed_images=False).rewrite(body) == body


def test_rewriter_raises_on_a_missing_image(run_directory):
    report_path = run_directory / "inference" / "20260102_000000" / "report.md"

    with pytest.raises(FileNotFoundError, match="missing image"):
        ReportImageRewriter(report_path, embed_images=False).rewrite("![gone](figures/gone.png)")


def test_collection_writes_the_report_renamed_after_the_run(run_directory, tmp_path):
    collector = tmp_path / "collected"
    collector.mkdir()
    config = ReportCollectionConfig(run_directory=run_directory, collector_dir=collector)

    result = ReportCollection(config, RecordingLogger()).run()

    assert result["n_reports"] == 1
    assert (collector / "run_demo.md").is_file()


def test_collection_suffixes_the_inference_tag_when_collecting_all(run_directory, tmp_path):
    collector = tmp_path / "collected"
    collector.mkdir()
    config = ReportCollectionConfig(run_directory=run_directory, collector_dir=collector, latest_only=False)

    result = ReportCollection(config, RecordingLogger()).run()

    assert result["n_reports"] == 2
    assert (collector / "run_demo_20260101_000000.md").is_file()
    assert (collector / "run_demo_20260102_000000.md").is_file()


def test_selector_discovers_runs_by_inference_report(run_directory, tmp_path):
    selector = ReportRunSelector(tmp_path, "inference", "report.md", RecordingLogger())

    assert selector.all() == [run_directory]


def test_batch_collects_filtered_runs_into_the_collector(run_directory, tmp_path):
    collector = tmp_path / "collected"
    entry     = ReportCollectionEntryConfig(runs_dir=tmp_path, run_filter=["run_demo"], collector_dir=collector)

    results = ReportCollectionBatch(entry, RecordingLogger()).run()

    assert len(results) == 1
    assert (collector / "run_demo.md").is_file()


def test_batch_rejects_selected_runs_sharing_a_name(tmp_path):
    _write_report(tmp_path / "a" / "run_x", "20260101_000000")
    _write_report(tmp_path / "b" / "run_x", "20260101_000000")
    entry = ReportCollectionEntryConfig(runs_dir=tmp_path, run_filter=["a/run_x", "b/run_x"], collector_dir=tmp_path / "collected")

    with pytest.raises(ValueError, match="collide"):
        ReportCollectionBatch(entry, RecordingLogger()).run()


def test_recollection_overwrites_the_previous_copy(run_directory, tmp_path):
    collector = tmp_path / "collected"
    entry     = ReportCollectionEntryConfig(runs_dir=tmp_path, run_filter=["run_demo"], collector_dir=collector)

    ReportCollectionBatch(entry, RecordingLogger()).run()
    ReportCollectionBatch(entry, RecordingLogger()).run()

    assert (collector / "run_demo.md").is_file()


def test_seed_run_labels_include_the_trial_directory(tmp_path):
    _write_report(tmp_path / "trial_a" / "seed0", "20260101_000000")
    _write_report(tmp_path / "trial_b" / "seed0", "20260101_000000")
    collector = tmp_path / "collected"
    entry     = ReportCollectionEntryConfig(runs_dir=tmp_path, run_filter=["trial_a/seed0", "trial_b/seed0"], collector_dir=collector)

    ReportCollectionBatch(entry, RecordingLogger()).run()

    assert (collector / "trial_a_seed0.md").is_file()
    assert (collector / "trial_b_seed0.md").is_file()


def test_trial_name_filter_expands_to_the_seed_runs(tmp_path):
    _write_report(tmp_path / "trial_a" / "seed0", "20260101_000000")
    _write_report(tmp_path / "trial_a" / "seed1", "20260101_000000")
    collector = tmp_path / "collected"
    entry     = ReportCollectionEntryConfig(runs_dir=tmp_path, run_filter=["trial_a"], collector_dir=collector)

    ReportCollectionBatch(entry, RecordingLogger()).run()

    assert (collector / "trial_a_seed0.md").is_file()
    assert (collector / "trial_a_seed1.md").is_file()


def test_trial_name_filter_collects_the_seed_comparison_report_when_present(tmp_path):
    _write_report(tmp_path / "trial_a" / "seed0", "20260101_000000")
    _write_report(tmp_path / "trial_a" / "seed1", "20260101_000000")
    _write_report(tmp_path / "trial_a", "seed_comparison", body="# Seed-Comparison Report\n")
    collector = tmp_path / "collected"
    entry     = ReportCollectionEntryConfig(runs_dir=tmp_path, run_filter=["trial_a"], collector_dir=collector)

    ReportCollectionBatch(entry, RecordingLogger()).run()

    assert (collector / "trial_a.md").read_text(encoding="utf-8") == "# Seed-Comparison Report\n"
    assert not (collector / "trial_a_seed0.md").is_file()
