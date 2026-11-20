
from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from pipelines.benchmark.pipeline import BenchmarkPipeline

from configuration.benchmark import BenchmarkConfig
from tools.data.io           import FileIO


ENTRY = Path("/entry/run_benchmark.py")


def _config(tmp_path, **kwargs):
    config                    = BenchmarkConfig()
    config.paths.log_base_dir = tmp_path
    config.run_tag            = "unit_run"
    for key, value in kwargs.items():
        setattr(config, key, value)
    return config


def test_pipeline_builds_model_list_from_registry(tmp_path):
    pipeline = BenchmarkPipeline(_config(tmp_path), entry_script=ENTRY)

    assert "unet" in pipeline.models
    assert len(pipeline.models) == len(set(pipeline.models))


def test_pipeline_excludes_skipped_models(tmp_path):
    pipeline = BenchmarkPipeline(_config(tmp_path, skip_models=["unet"]), entry_script=ENTRY)

    assert "unet" not in pipeline.models


def test_pipeline_writes_resolved_config(tmp_path):
    BenchmarkPipeline(_config(tmp_path), entry_script=ENTRY)

    assert (tmp_path / "unit_run" / "pipeline" / "resolved_config.json").exists()


def test_mark_stage_persists_state(tmp_path):
    pipeline = BenchmarkPipeline(_config(tmp_path), entry_script=ENTRY)

    pipeline._mark_stage("training", "completed")

    state = FileIO.load_json(pipeline.state_path)
    assert state["stages"]["training"]["status"] == "completed"
    assert state["run_tag"] == "unit_run"


def test_a_later_stage_process_keeps_the_earlier_stage_history(tmp_path):
    prepare = BenchmarkPipeline(_config(tmp_path), entry_script=ENTRY)
    prepare._mark_stage("prepare", "completed")

    compare = BenchmarkPipeline(_config(tmp_path), entry_script=ENTRY)
    compare._mark_stage("pipeline", "completed")

    state = FileIO.load_json(compare.state_path)
    assert set(state["stages"]) == {"prepare", "pipeline"}


def test_runs_inference_follows_infer_after(tmp_path):
    assert _config(tmp_path).runs_inference()                  is True
    assert _config(tmp_path, infer_after=False).runs_inference() is False


def _patch_stage_methods(pipeline, monkeypatch, order):
    monkeypatch.setattr(pipeline, "_run_training",    lambda: order.append("training") or [{"name": "unet/seed0", "status": "DONE"}])
    monkeypatch.setattr(pipeline, "_run_inference",   lambda: order.append("inference") or [{"name": "unet/seed0", "status": "DONE"}])
    monkeypatch.setattr(pipeline, "_run_comparison",  lambda: order.append("comparison") or (Path("/tmp/cmp")))


@pytest.mark.slow
def test_run_executes_stages_in_expected_order(tmp_path, monkeypatch):
    pipeline = BenchmarkPipeline(_config(tmp_path), entry_script=ENTRY)

    order = []
    _patch_stage_methods(pipeline, monkeypatch, order)

    pipeline.run()

    assert order == ["training", "inference", "comparison"]

    state = FileIO.load_json(pipeline.state_path)
    assert state["stages"]["pipeline"]["status"] == "completed"


def test_run_with_infer_after_off_logs_and_marks_the_skipped_inference(tmp_path, monkeypatch):
    pipeline = BenchmarkPipeline(_config(tmp_path, infer_after=False), entry_script=ENTRY)

    order = []
    _patch_stage_methods(pipeline, monkeypatch, order)
    monkeypatch.delattr(pipeline, "_run_inference")

    pipeline.run()

    state = FileIO.load_json(pipeline.state_path)
    text  = (pipeline.pipeline_dir / "benchmark_pipeline.log").read_text(encoding="utf-8")

    assert "inference" not in order
    assert state["stages"]["inference"]["status"] == "skipped"
    assert "Inference skipped: infer_after is off" in text
    assert "has no inference stage in the benchmark" not in text


def _recording_pipeline(tmp_path, monkeypatch, **kwargs):
    pipeline = BenchmarkPipeline(_config(tmp_path, **kwargs), entry_script=ENTRY)
    calls    = []
    returns  = {
        "_log_unit_status" : None,
        "_run_training"    : [],
        "_run_inference"   : [],
        "_run_comparison"  : tmp_path / "comparison",
    }

    for step, value in returns.items():
        monkeypatch.setattr(pipeline, step, lambda step=step, value=value: calls.append(step) or value)

    return pipeline, calls


def test_prepare_stage_stops_before_training(tmp_path, monkeypatch):
    pipeline, calls = _recording_pipeline(tmp_path, monkeypatch, stage="prepare")

    pipeline.run()

    assert calls == ["_log_unit_status"]
    state = FileIO.load_json(pipeline.state_path)
    assert state["stages"]["prepare"]["status"] == "completed"


def test_compare_stage_runs_the_comparison_only(tmp_path, monkeypatch):
    pipeline, calls = _recording_pipeline(tmp_path, monkeypatch, stage="compare")

    pipeline.run()

    assert calls == ["_run_comparison"]
    state = FileIO.load_json(pipeline.state_path)
    assert state["stages"]["pipeline"]["status"] == "completed"


def test_an_exception_in_a_stage_is_logged_with_its_stage_and_marked_failed(tmp_path, monkeypatch):
    pipeline = BenchmarkPipeline(_config(tmp_path), entry_script=ENTRY)

    def failing_training():
        pipeline._begin_stage("training")
        Mock(side_effect=RuntimeError("training queue died"))()

    monkeypatch.setattr(pipeline, "_run_training", failing_training)

    with pytest.raises(RuntimeError, match="training queue died"):
        pipeline.run()

    state = FileIO.load_json(pipeline.state_path)
    text  = (pipeline.pipeline_dir / "benchmark_pipeline.log").read_text(encoding="utf-8")

    assert state["stages"]["training"]["status"] == "failed"
    assert state["stages"]["pipeline"]["status"] == "failed"
    assert "Aborted in stage 'training' by RuntimeError: training queue died" in text
    assert ">>> ABORTED benchmark_pipeline" in text
    assert ">>> END benchmark_pipeline" not in text


def test_an_abort_overrides_the_completed_mark_of_an_earlier_launch(tmp_path, monkeypatch):
    earlier = BenchmarkPipeline(_config(tmp_path), entry_script=ENTRY)
    earlier._mark_stage("pipeline", "completed")
    earlier.logger.close()

    pipeline = BenchmarkPipeline(_config(tmp_path), entry_script=ENTRY)

    def failing_training():
        pipeline._begin_stage("training")
        Mock(side_effect=SystemExit("bogus"))()

    monkeypatch.setattr(pipeline, "_run_training", failing_training)

    with pytest.raises(SystemExit, match="bogus"):
        pipeline.run()

    state = FileIO.load_json(pipeline.state_path)

    assert state["stages"]["training"]["status"] == "failed"
    assert state["stages"]["pipeline"]["status"] == "failed"


def test_a_clean_system_exit_is_not_recorded_as_an_abort(tmp_path, monkeypatch):
    pipeline = BenchmarkPipeline(_config(tmp_path), entry_script=ENTRY)

    def clean_exit():
        Mock(side_effect=SystemExit(0))()

    monkeypatch.setattr(pipeline, "_execute", clean_exit)

    with pytest.raises(SystemExit):
        pipeline.run()

    text = (pipeline.pipeline_dir / "benchmark_pipeline.log").read_text(encoding="utf-8")

    assert not pipeline.state_path.exists()
    assert ">>> END benchmark_pipeline" in text
    assert "Aborted" not in text


def _unit_results(statuses: dict) -> list[dict]:
    return [{"name": name, "status": status} for name, status in statuses.items()]


def test_a_failed_training_unit_ends_the_run_as_partial_with_a_non_zero_exit(tmp_path, monkeypatch):
    pipeline = BenchmarkPipeline(_config(tmp_path), entry_script=ENTRY)

    order = []
    _patch_stage_methods(pipeline, monkeypatch, order)
    monkeypatch.setattr(pipeline, "_run_training",  lambda: order.append("training") or _unit_results({"unet/seed0": "DONE", "unet/seed1": "FAILED", "resunet/seed0": "DONE"}))
    monkeypatch.setattr(pipeline, "_run_inference", lambda: order.append("inference") or _unit_results({"unet/seed0": "DONE", "unet/seed1": "SKIPPED", "resunet/seed0": "DONE"}))

    with pytest.raises(SystemExit, match="Partial: 1 of 3 units failed"):
        pipeline.run()

    state = FileIO.load_json(pipeline.state_path)
    text  = (pipeline.pipeline_dir / "benchmark_pipeline.log").read_text(encoding="utf-8")

    assert order[-1] == "comparison"
    assert state["stages"]["pipeline"]["status"] == "partial"
    assert "training failed: unet/seed1" in text
    assert "Partial: 1 of 3 units failed" in text
    assert ">>> END benchmark_pipeline" in text
    assert ">>> ABORTED" not in text


def test_a_failed_inference_unit_counts_toward_the_partial_exit(tmp_path, monkeypatch):
    pipeline = BenchmarkPipeline(_config(tmp_path), entry_script=ENTRY)

    order = []
    _patch_stage_methods(pipeline, monkeypatch, order)
    monkeypatch.setattr(pipeline, "_run_training",  lambda: _unit_results({"unet/seed0": "DONE", "unet/seed1": "DONE"}))
    monkeypatch.setattr(pipeline, "_run_inference", lambda: _unit_results({"unet/seed0": "DONE", "unet/seed1": "FAILED"}))

    with pytest.raises(SystemExit, match="Partial: 1 of 2 units failed"):
        pipeline.run()

    text = (pipeline.pipeline_dir / "benchmark_pipeline.log").read_text(encoding="utf-8")

    assert FileIO.load_json(pipeline.state_path)["stages"]["pipeline"]["status"] == "partial"
    assert "inference failed: unet/seed1" in text


def test_a_run_whose_units_all_finished_ends_as_done(tmp_path, monkeypatch):
    pipeline = BenchmarkPipeline(_config(tmp_path), entry_script=ENTRY)

    order = []
    _patch_stage_methods(pipeline, monkeypatch, order)
    monkeypatch.setattr(pipeline, "_run_training",  lambda: _unit_results({"unet/seed0": "DONE", "unet/seed1": "DONE"}))
    monkeypatch.setattr(pipeline, "_run_inference", lambda: _unit_results({"unet/seed0": "DONE", "unet/seed1": "DONE"}))

    pipeline.run()

    text = (pipeline.pipeline_dir / "benchmark_pipeline.log").read_text(encoding="utf-8")

    assert FileIO.load_json(pipeline.state_path)["stages"]["pipeline"]["status"] == "completed"
    assert "Units planned     : 2" in text
    assert "Training failed   : 0" in text


def test_a_relaunch_appends_to_the_pipeline_log_instead_of_truncating_it(tmp_path):
    first = BenchmarkPipeline(_config(tmp_path, stage="prepare"), entry_script=ENTRY)
    first.logger.info("first launch marker")
    first.logger.close()

    second = BenchmarkPipeline(_config(tmp_path, stage="compare"), entry_script=ENTRY)
    second.logger.info("second launch marker")
    second.logger.close()

    text = (second.pipeline_dir / "benchmark_pipeline.log").read_text(encoding="utf-8")

    assert text.count(">>> RUN benchmark_pipeline") == 2
    assert "first launch marker"  in text
    assert "second launch marker" in text
    assert text.index("first launch marker") < text.index("second launch marker")


def test_a_relaunch_with_a_changed_config_warns_and_names_the_archived_recording(tmp_path):
    first = BenchmarkPipeline(_config(tmp_path), entry_script=ENTRY)
    first.logger.close()

    changed                 = _config(tmp_path)
    changed.training.epochs = first.config.training.epochs + 3
    second                  = BenchmarkPipeline(changed, entry_script=ENTRY)

    second._log_launch()
    second.logger.close()

    text = (second.pipeline_dir / "benchmark_pipeline.log").read_text(encoding="utf-8")

    assert second.config_changes == {"training.epochs": (first.config.training.epochs, first.config.training.epochs + 3)}
    assert second.archived_config.parent == second.history_dir
    assert second.archived_config.is_file()
    assert "1 leaf(s) differ from the configuration this run recorded" in text
    assert str(second.archived_config) in text
    assert "training.epochs" in text


def test_a_relaunch_changing_only_launch_owned_fields_logs_no_config_warning(tmp_path):
    first = BenchmarkPipeline(_config(tmp_path, stage="prepare"), entry_script=ENTRY)
    first.logger.close()

    second = BenchmarkPipeline(_config(tmp_path, stage="compare", gpus=[0, 1]), entry_script=ENTRY)
    second._log_launch()
    second.logger.close()

    text = (second.pipeline_dir / "benchmark_pipeline.log").read_text(encoding="utf-8")

    assert set(second.config_changes) == {"stage", "gpus"}
    assert "differ from the configuration this run recorded" not in text


def test_a_first_launch_records_no_config_change(tmp_path):
    pipeline = BenchmarkPipeline(_config(tmp_path), entry_script=ENTRY)
    pipeline.logger.close()

    assert pipeline.config_changes  == {}
    assert pipeline.archived_config is None


def test_a_resumed_launch_leaves_the_change_report_to_the_resumed_run(tmp_path):
    first = BenchmarkPipeline(_config(tmp_path), entry_script=ENTRY)
    first.logger.close()

    class FakeResumed:
        def __init__(self):
            self.root   = tmp_path / "unit_run"
            self.logged = False

        def log(self, logger):
            self.logged = True

    changed                 = _config(tmp_path)
    changed.training.epochs = first.config.training.epochs + 3
    resumed                 = FakeResumed()
    second                  = BenchmarkPipeline(changed, entry_script=ENTRY, resumed=resumed)

    second._log_launch()
    second.logger.close()

    text = (second.pipeline_dir / "benchmark_pipeline.log").read_text(encoding="utf-8")

    assert resumed.logged is True
    assert "differ from the configuration this run recorded" not in text


def test_every_launch_appends_its_provenance_to_the_launch_record(tmp_path, monkeypatch):
    import json

    from tools.runtime.git_state import GitState

    BenchmarkPipeline(_config(tmp_path), entry_script=ENTRY).logger.close()

    changed                 = _config(tmp_path, stage="compare")
    changed.training.epochs = changed.training.epochs + 1
    second                  = BenchmarkPipeline(changed, entry_script=ENTRY)

    second._log_launch()
    second.logger.close()

    lines = [json.loads(line) for line in second.launches_path.read_text(encoding="utf-8").splitlines()]
    text  = (second.pipeline_dir / "benchmark_pipeline.log").read_text(encoding="utf-8")

    assert [line["stage"] for line in lines]    == ["full", "compare"]
    assert lines[1]["archived_config"]          == str(second.archived_config)
    assert lines[1]["provenance"]["git_commit"] == GitState.commit()
    assert lines[0]["archived_config"]          is None
    assert lines[1]["resumed_from"]             is None
    assert f"Commit        : {GitState.commit()}" in text


def test_the_pipeline_logs_the_selected_and_the_skipped_models(tmp_path, monkeypatch):
    pipeline = BenchmarkPipeline(_config(tmp_path, skip_models=["resunet"]), entry_script=ENTRY)

    order = []
    _patch_stage_methods(pipeline, monkeypatch, order)

    pipeline.run()

    text = (pipeline.pipeline_dir / "benchmark_pipeline.log").read_text(encoding="utf-8")

    assert f"Model selection: {len(pipeline.models)} benchmarked, 1 skipped" in text
    assert "Skipped by skip_models : resunet" in text
    assert f"Selected               : {', '.join(pipeline.models)}" in text


def test_comparison_receives_the_unit_plan_of_the_launch(tmp_path, monkeypatch):
    import pipelines.benchmark.pipeline as pipeline_module

    class FakeTraining:
        items = ["unet/seed0", "unet/seed1"]

        def __init__(self, **kwargs):
            pass

        def unit_name(self, item):
            return "renamed/seed1" if item == "unet/seed1" else item

    seen = {}

    class FakeComparison:
        def __init__(self, **kwargs):
            seen["planned_units"] = kwargs["planned_units"]

        def run(self):
            return tmp_path / "comparison"

    monkeypatch.setattr(pipeline_module, "TrainingStage",   FakeTraining)
    monkeypatch.setattr(pipeline_module, "ComparisonStage", FakeComparison)

    pipeline = BenchmarkPipeline(_config(tmp_path, stage="compare"), entry_script=ENTRY)
    pipeline.run()

    assert seen["planned_units"] == {"unet/seed0": "unet/seed0", "unet/seed1": "renamed/seed1"}
