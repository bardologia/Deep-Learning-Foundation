"""End-to-end test of the activation x-ray on the real-data run directory.

Covers the summary the run persists beside its verdict: the capture settings,
the thresholds, the hooked-versus-recorded layer counts and the provenance,
and the report header that renders them.
"""

from __future__ import annotations

import json

import pytest

from configuration.diagnostics                             import ActivationXrayConfig
from pipelines.backbone.inference.analysis.activation_xray import ActivationXrayRun
from pipelines.backbone.inference.analysis.run_batch       import AnalysisOutcome
from tools.diagnostics.activation_xray_analysis            import ActivationXraySummarizer

from tests.backbone_inference._helpers import build_run_directory
from tests.conftest                    import RecordingLogger


@pytest.mark.dataset
@pytest.mark.slow
def test_activation_xray_run_persists_capture_settings_layer_counts_and_thresholds(tmp_path, test_data_dir, params_dir):
    """One x-ray of the miniature U-Net run writes a summary with its settings, counts and thresholds, and a report rendering them."""
    run_dir = build_run_directory(tmp_path, test_data_dir, params_dir)
    config  = ActivationXrayConfig(runs_dir=tmp_path, split="test", device="cpu", batch_size=2, max_batches=1, make_plots=False)
    logger  = RecordingLogger()

    outcome = ActivationXrayRun(run_dir, config, logger).run()
    summary = outcome.payload["summary"]
    stored  = json.loads((run_dir / "activation_xray" / "activation_xray.json").read_text())
    report  = (run_dir / "activation_xray" / "activation_xray.md").read_text()

    assert summary["checkpoint"]  == "best_model.pt"
    assert summary["split"]       == "test"
    assert summary["batches"]     == 1
    assert summary["batch_size"]  == 2
    assert summary["hooked_layers"] >= summary["layers"] > 0
    assert set(summary["thresholds"]) == set(ActivationXraySummarizer.THRESHOLDS)
    assert summary["provenance"]["git_commit"]
    assert stored["summary"]["verdict"] == summary["verdict"]

    assert "Thresholds"    in report
    assert "Hooked layers" in report
    assert any(message.startswith(f"{run_dir.name}: {summary['verdict']}") for recorded, message in logger.messages if recorded in ("ok", "warning", "error"))


@pytest.mark.parametrize(("counts", "status"), [
    ({"critical": 1, "warning": 2, "info": 0, "ok": 5}, "critical"),
    ({"critical": 0, "warning": 2, "info": 1, "ok": 5}, "warning"),
    ({"critical": 0, "warning": 0, "info": 3, "ok": 5}, "ok"),
])
def test_status_follows_the_worst_flagged_severity(counts, status):
    """Critical layers make the result line an error, warning layers a warning, and minor observations stay ok."""
    assert ActivationXrayRun._status({"severity_counts": counts}) == status


def test_critical_result_line_is_logged_as_an_error(tmp_path):
    """The shared result line of a critical outcome goes out at error level, not through ok."""
    logger = RecordingLogger()
    xray   = ActivationXrayRun(tmp_path, ActivationXrayConfig(), logger)

    xray._report(AnalysisOutcome({}, "critical issues detected (1/3 layers flagged)", tmp_path / "activation_xray.md", "critical"))

    assert logger.messages == [("error", f"{tmp_path.name}: critical issues detected (1/3 layers flagged) -> {tmp_path / 'activation_xray.md'}")]

