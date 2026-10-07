"""Tests the staged pipeline base: its launch record, the configuration changes a relaunch reports, and how a run ends.

Covers the provenance line every launch appends, the archived recording and
the warning a relaunch with other settings logs, the resumed-run report that
replaces that warning, the stage an exception is attributed to, the partial
exit of a run whose units failed, and the clean exit that is not an abort."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib     import Path

import pytest

from pipelines.shared.orchestration.staged_pipeline import PartialRun, StagedPipeline
from tools.data.io                                  import FileIO
from tools.runtime.git_state                        import GitState


@dataclass
class PathsSection:
    """Run root of the stand-in configuration."""
    log_base_dir : Path = Path("runs")


@dataclass
class StagedConfig:
    """Entry configuration carrying the fields a staged pipeline reads, plus one setting a relaunch may change."""
    paths   : PathsSection = field(default_factory=PathsSection)
    run_tag : str          = "unit_run"
    stage   : str          = "full"
    gpus    : list         = field(default_factory=lambda: [0, 1])
    epochs  : int          = 10


class StubPipeline(StagedPipeline):
    """Staged pipeline whose stage sequence is a callable the test supplies."""

    LOGGER_NAME = "stub_pipeline"

    def __init__(self, config, stages=None, resumed=None) -> None:
        """Binds the stage sequence and prepares the run directory."""
        self.stages = stages
        super().__init__(config, Path("main/stub.py"), resumed=resumed)

    def _execute(self) -> None:
        """Runs the supplied stage sequence."""
        self.stages(self)


def _config(tmp_path, **kwargs) -> StagedConfig:
    """Returns a stand-in configuration rooted at tmp_path with the given field overrides."""
    config                    = StagedConfig(**kwargs)
    config.paths.log_base_dir = tmp_path

    return config


def _log(pipeline) -> str:
    """Returns the text of the pipeline log."""
    return (pipeline.pipeline_dir / f"{pipeline.LOGGER_NAME}.log").read_text(encoding="utf-8")


def test_every_launch_appends_its_provenance_to_the_launch_record(tmp_path):
    """Each launch into a run tag adds one line naming its stage, the recording it archived and the commit it ran from, and the log names the commit."""
    StubPipeline(_config(tmp_path)).logger.close()

    second = StubPipeline(_config(tmp_path, stage="compare", epochs=11))
    second._log_launch()
    second.logger.close()

    lines = [json.loads(line) for line in second.launches_path.read_text(encoding="utf-8").splitlines()]

    assert [line["stage"] for line in lines]    == ["full", "compare"]
    assert lines[0]["archived_config"]          is None
    assert lines[1]["archived_config"]          == str(second.archived_config)
    assert lines[1]["resumed_from"]             is None
    assert lines[1]["provenance"]["git_commit"] == GitState.commit()
    assert f"Commit        : {GitState.commit()}" in _log(second)


def test_a_relaunch_with_a_changed_config_warns_and_names_the_archived_recording(tmp_path):
    """Relaunching a run tag with other settings archives the recording and the log lists the changed leaves with the archive path."""
    StubPipeline(_config(tmp_path)).logger.close()

    second = StubPipeline(_config(tmp_path, epochs=13))
    second._log_launch()
    second.logger.close()

    text = _log(second)

    assert second.config_changes == {"epochs": (10, 13)}
    assert second.archived_config.parent == second.history_dir
    assert second.archived_config.is_file()
    assert "1 leaf(s) differ from the configuration this run recorded" in text
    assert str(second.archived_config) in text


def test_a_relaunch_changing_only_launch_owned_fields_logs_no_config_warning(tmp_path):
    """A compare launch following a prepare launch changes only the stage and the hardware, which are not reported as configuration changes."""
    StubPipeline(_config(tmp_path, stage="prepare")).logger.close()

    second = StubPipeline(_config(tmp_path, stage="compare", gpus=[0]))
    second._log_launch()
    second.logger.close()

    assert set(second.config_changes) == {"stage", "gpus"}
    assert "differ from the configuration this run recorded" not in _log(second)


def test_a_first_launch_records_no_config_change(tmp_path):
    """A fresh run tag has nothing to compare against."""
    pipeline = StubPipeline(_config(tmp_path))
    pipeline.logger.close()

    assert pipeline.config_changes  == {}
    assert pipeline.archived_config is None


def test_a_resumed_launch_refreshes_and_logs_the_resumed_report_instead_of_the_relaunch_warning(tmp_path):
    """With a resumed-run description the pipeline refreshes its change report with the final configuration and logs it in place of the relaunch warning."""
    StubPipeline(_config(tmp_path)).logger.close()

    class FakeResumed:
        """Resumed-run stand-in recording the refresh and the report call."""

        def __init__(self) -> None:
            """Starts with no report logged and no refresh."""
            self.root      = tmp_path / "unit_run"
            self.logged    = False
            self.refreshed = None

        def refresh(self, config) -> None:
            """Records the configuration the pipeline refreshed the change report with."""
            self.refreshed = config

        def log(self, logger) -> None:
            """Records the report call."""
            self.logged = True

    changed = _config(tmp_path, epochs=13)
    resumed = FakeResumed()
    second  = StubPipeline(changed, resumed=resumed)

    second._log_launch()
    second.logger.close()

    assert resumed.logged    is True
    assert resumed.refreshed is changed
    assert "differ from the configuration this run recorded" not in _log(second)


def test_an_exception_in_a_stage_is_logged_with_its_stage_and_marked_failed(tmp_path):
    """An exception escaping a stage is re-raised after the pipeline log names the stage and state.json marks it and the pipeline failed."""
    def stages(pipeline):
        """Fails inside the training stage."""
        pipeline._begin_stage("training")
        raise RuntimeError("queue died")

    pipeline = StubPipeline(_config(tmp_path), stages)

    with pytest.raises(RuntimeError, match="queue died"):
        pipeline.run()

    state = FileIO.load_json(pipeline.state_path)
    text  = _log(pipeline)

    assert state["stages"]["training"]["status"] == "failed"
    assert state["stages"]["pipeline"]["status"] == "failed"
    assert "Aborted in stage 'training' by RuntimeError: queue died" in text
    assert ">>> ABORTED stub_pipeline" in text


def test_a_clean_system_exit_is_not_recorded_as_an_abort(tmp_path):
    """A SystemExit with code zero ends the run with the END banner and leaves the stage state untouched."""
    def stages(pipeline):
        """Ends the stage sequence deliberately."""
        raise SystemExit(0)

    pipeline = StubPipeline(_config(tmp_path), stages)

    with pytest.raises(SystemExit):
        pipeline.run()

    assert not pipeline.state_path.exists()
    assert ">>> END stub_pipeline" in _log(pipeline)
    assert "Aborted" not in _log(pipeline)


def test_failed_units_end_the_run_as_partial_with_a_non_zero_exit(tmp_path):
    """A run whose training and inference units partly failed is marked partial, names each unit and exits non-zero after the END banner."""
    def stages(pipeline):
        """Finishes with one failed training unit and one failed inference unit."""
        training  = [{"name": "a", "status": "DONE"}, {"name": "b", "status": "FAILED"}, {"name": "c", "status": "DONE"}]
        inference = [{"name": "a", "status": "DONE"}, {"name": "b", "status": "SKIPPED"}, {"name": "c", "status": "FAILED"}]
        pipeline._finish(training, inference, {"Reports": "reports/"})

    pipeline = StubPipeline(_config(tmp_path), stages)

    with pytest.raises(PartialRun, match="Partial: 2 of 3 units failed"):
        pipeline.run()

    text = _log(pipeline)

    assert FileIO.load_json(pipeline.state_path)["stages"]["pipeline"]["status"] == "partial"
    assert "training failed: b" in text
    assert "inference failed: c" in text
    assert ">>> END stub_pipeline" in text
    assert ">>> ABORTED" not in text


def test_a_run_whose_units_all_finished_ends_as_done_and_says_when_inference_was_skipped(tmp_path):
    """A run with every unit trained is marked completed, and a skipped inference stage is named instead of counting zero inferences."""
    def stages(pipeline):
        """Finishes with every unit trained and no inference stage."""
        pipeline._finish([{"name": "a", "status": "DONE"}, {"name": "b", "status": "DONE"}], [], {})

    pipeline = StubPipeline(_config(tmp_path), stages)
    pipeline.run()

    text = _log(pipeline)

    assert FileIO.load_json(pipeline.state_path)["stages"]["pipeline"]["status"] == "completed"
    assert "Units planned   : 2" in text
    assert "Inference       : stage skipped" in text
