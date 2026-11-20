
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib     import Path
from unittest.mock import Mock

import pytest

from pipelines.shared.orchestration.staged_pipeline import PartialRun, StagedPipeline
from tools.data.io                                  import FileIO
from tools.runtime.git_state                        import GitState


@dataclass
class PathsSection:
    log_base_dir : Path = Path("runs")


@dataclass
class StagedConfig:
    paths   : PathsSection = field(default_factory=PathsSection)
    run_tag : str          = "unit_run"
    stage   : str          = "full"
    gpus    : list         = field(default_factory=lambda: [0, 1])
    epochs  : int          = 10


class StubPipeline(StagedPipeline):

    LOGGER_NAME = "stub_pipeline"

    def __init__(self, config, stages=None, resumed=None) -> None:
        self.stages = stages
        super().__init__(config, Path("main/stub.py"), resumed=resumed)

    def _execute(self) -> None:
        self.stages(self)


def _config(tmp_path, **kwargs) -> StagedConfig:
    config                    = StagedConfig(**kwargs)
    config.paths.log_base_dir = tmp_path

    return config


def _log(pipeline) -> str:
    return (pipeline.pipeline_dir / f"{pipeline.LOGGER_NAME}.log").read_text(encoding="utf-8")


def test_every_launch_appends_its_provenance_to_the_launch_record(tmp_path):
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
    StubPipeline(_config(tmp_path, stage="prepare")).logger.close()

    second = StubPipeline(_config(tmp_path, stage="compare", gpus=[0]))
    second._log_launch()
    second.logger.close()

    assert set(second.config_changes) == {"stage", "gpus"}
    assert "differ from the configuration this run recorded" not in _log(second)


def test_a_first_launch_records_no_config_change(tmp_path):
    pipeline = StubPipeline(_config(tmp_path))
    pipeline.logger.close()

    assert pipeline.config_changes  == {}
    assert pipeline.archived_config is None


def test_a_resumed_launch_logs_the_resumed_report_instead_of_the_relaunch_warning(tmp_path):
    StubPipeline(_config(tmp_path)).logger.close()

    class FakeResumed:

        def __init__(self) -> None:
            self.root   = tmp_path / "unit_run"
            self.logged = False

        def log(self, logger) -> None:
            self.logged = True

    changed = _config(tmp_path, epochs=13)
    resumed = FakeResumed()
    second  = StubPipeline(changed, resumed=resumed)

    second._log_launch()
    second.logger.close()

    assert resumed.logged is True
    assert "differ from the configuration this run recorded" not in _log(second)


def test_an_exception_in_a_stage_is_logged_with_its_stage_and_marked_failed(tmp_path):
    def stages(pipeline):
        pipeline._begin_stage("training")
        Mock(side_effect=RuntimeError("queue died"))()

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
    def stages(pipeline):
        Mock(side_effect=SystemExit(0))()

    pipeline = StubPipeline(_config(tmp_path), stages)

    with pytest.raises(SystemExit):
        pipeline.run()

    assert not pipeline.state_path.exists()
    assert ">>> END stub_pipeline" in _log(pipeline)
    assert "Aborted" not in _log(pipeline)


def test_failed_units_end_the_run_as_partial_with_a_non_zero_exit(tmp_path):
    def stages(pipeline):
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
    def stages(pipeline):
        pipeline._finish([{"name": "a", "status": "DONE"}, {"name": "b", "status": "DONE"}], [], {})

    pipeline = StubPipeline(_config(tmp_path), stages)
    pipeline.run()

    text = _log(pipeline)

    assert FileIO.load_json(pipeline.state_path)["stages"]["pipeline"]["status"] == "completed"
    assert "Units planned   : 2" in text
    assert "Inference       : stage skipped" in text
