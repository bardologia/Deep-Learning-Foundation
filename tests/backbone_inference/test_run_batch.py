"""Tests for the shared per-run analysis orchestrator and its batch driver.

Covers the provenance record and completion marker every analysis pass writes, the
warning and figure cleanup when a pass replaces an earlier result, and the refusal of
an output subdirectory that resolves to the run directory itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from types   import SimpleNamespace

import pytest

from configuration.diagnostics                             import ActivationXrayConfig
from pipelines.backbone.inference.analysis.activation_xray import ActivationXrayRun
from pipelines.backbone.inference.analysis.run_batch       import AnalysisOutcome, AnalysisRun
from tools.data.io                                         import FileIO
from tools.runtime.completion                              import CompletionMarker

from tests.backbone_inference._helpers import build_run_directory
from tests.conftest                    import RecordingLogger


CHECKPOINT_META = {"epoch": 41, "best_val_loss": 0.125, "best_epoch": 37}


class _StubRun(AnalysisRun):
    """Analysis stub that loads no model and writes one summary, one figure and one report.

    Attributes:
        fail : Whether the analysis step raises after writing its figure.
    """

    SUMMARY_FILENAME = "stub.json"

    def __init__(self, run_dir: Path, config, logger, fail: bool = False) -> None:
        """Binds the stub to a run directory.

        Args:
            run_dir : Directory of the run under analysis.
            config  : Analysis configuration.
            logger  : Logger receiving the result line.
            fail    : Whether the analysis step raises after writing its figure.
        """
        super().__init__(run_dir, config, logger)
        self.fail = fail

    def _load_run(self):
        """Returns a run stand-in carrying only the checkpoint metadata."""
        return SimpleNamespace(checkpoint_meta=dict(CHECKPOINT_META))

    def _analyse(self, run) -> AnalysisOutcome:
        """Writes the figure, the summary and the report, raising in between when asked to fail."""
        FileIO.ensure_dirs(self.output_dir / "plots")
        (self.output_dir / "plots" / "fresh.png").write_bytes(b"png")

        if self.fail:
            raise RuntimeError("analysis failed")

        FileIO.save_json({"value": 1}, self.output_dir / self.SUMMARY_FILENAME)
        report = self.output_dir / "stub.md"
        report.write_text("# stub", encoding="utf-8")

        return AnalysisOutcome({"value": 1}, "value 1", report)


def _config() -> ActivationXrayConfig:
    """Returns an analysis configuration writing into the 'stub' subdirectory."""
    return ActivationXrayConfig(output_subdir="stub")


def test_pass_records_config_provenance_and_completion(tmp_path):
    """A finished pass leaves the resolved config, the checkpoint provenance and a completion marker naming the analysis."""
    logger  = RecordingLogger()
    outcome = _StubRun(tmp_path, _config(), logger).run()

    output     = tmp_path / "stub"
    config     = json.loads((output / "analysis_config.json").read_text())
    provenance = json.loads((output / "analysis_provenance.json").read_text())
    marker     = CompletionMarker.payload(output)

    assert config["output_subdir"] == "stub" and config["max_batches"] == 8
    assert provenance["checkpoint"] == {"name": "best_model.pt", **CHECKPOINT_META}
    assert provenance["split"] == "test" and provenance["git_commit"]
    assert marker["stage"] == "stub" and marker["epoch"] == 41 and marker["status"] == "ok"
    assert outcome.replaced is False
    assert ("ok", f"{tmp_path.name}: value 1 -> {output / 'stub.md'}") in logger.messages
    assert not [message for level, message in logger.messages if level == "warning"]


def test_replacing_a_complete_result_warns_and_clears_its_figures(tmp_path):
    """A second pass names the earlier result's checkpoint in a warning and removes the figures it left."""
    _StubRun(tmp_path, _config(), RecordingLogger()).run()
    stale = tmp_path / "stub" / "plots" / "stale.png"
    stale.write_bytes(b"old")

    logger  = RecordingLogger()
    outcome = _StubRun(tmp_path, _config(), logger).run()

    warnings = [message for level, message in logger.messages if level == "warning"]

    assert outcome.replaced is True
    assert len(warnings) == 1 and "complete stub result of checkpoint 'best_model.pt' (epoch 41" in warnings[0]
    assert not stale.exists()
    assert (tmp_path / "stub" / "plots" / "fresh.png").is_file()


def test_failed_pass_leaves_no_completion_marker(tmp_path):
    """A pass that raises clears the earlier marker, so the directory reads as unfinished."""
    _StubRun(tmp_path, _config(), RecordingLogger()).run()

    with pytest.raises(RuntimeError, match="analysis failed"):
        _StubRun(tmp_path, _config(), RecordingLogger(), fail=True).run()

    assert not CompletionMarker.is_complete(tmp_path / "stub")


def test_replacing_an_unfinished_result_says_so(tmp_path):
    """A summary left without a completion marker is reported as an unfinished earlier result."""
    output = tmp_path / "stub"
    output.mkdir()
    FileIO.save_json({"value": 0}, output / "stub.json")

    logger = RecordingLogger()
    _StubRun(tmp_path, _config(), logger).run()

    warnings = [message for level, message in logger.messages if level == "warning"]

    assert len(warnings) == 1 and "unfinished stub result without a completion marker" in warnings[0]


def test_output_subdir_resolving_to_the_run_is_refused(tmp_path):
    """An empty output subdirectory would clear the run's own completion marker and is refused."""
    with pytest.raises(ValueError, match="resolves to the run directory"):
        _StubRun(tmp_path, ActivationXrayConfig(output_subdir=""), RecordingLogger())


@pytest.mark.dataset
@pytest.mark.slow
def test_activation_xray_pass_on_a_real_run_is_stamped(tmp_path, test_data_dir, params_dir):
    """An x-ray pass over a real run writes its summary next to the provenance of the loaded checkpoint and a completion marker."""
    run_dir = build_run_directory(tmp_path, test_data_dir, params_dir)
    config  = ActivationXrayConfig(batch_size=2, max_batches=1, make_plots=False)

    outcome = ActivationXrayRun(run_dir, config, RecordingLogger()).run()

    output     = run_dir / "activation_xray"
    provenance = json.loads((output / "analysis_provenance.json").read_text())

    assert (output / "activation_xray.json").is_file()
    assert provenance["checkpoint"] == {"name": "best_model.pt", "epoch": 1, "best_val_loss": 0.1, "best_epoch": 1}
    assert CompletionMarker.payload(output)["stage"] == "activation_xray"
    assert outcome.report_path == output / "activation_xray.md"
