"""Tests for the benchmark pipeline's model selection, stage state and stage ordering.

Covers the model list derived from the backbone registry, the resolved-config artifact,
the persisted per-stage state across processes, the per-training-type stage gating, and
the prepare and compare partial-stage entry points.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipelines.benchmark.pipeline import BenchmarkPipeline

from configuration.benchmark import BenchmarkConfig
from tools.data.io           import FileIO


ENTRY = Path("/entry/run_benchmark.py")


def _config(tmp_path, **kwargs):
    """Returns a benchmark config rooted at tmp_path with a fixed run tag and the given field overrides."""
    config                    = BenchmarkConfig()
    config.paths.log_base_dir = tmp_path
    config.run_tag            = "unit_run"
    for key, value in kwargs.items():
        setattr(config, key, value)
    return config


def test_pipeline_builds_model_list_from_registry(tmp_path):
    """Verifies the model list comes from the registry without duplicates."""
    pipeline = BenchmarkPipeline(_config(tmp_path), entry_script=ENTRY)

    assert "unet" in pipeline.models
    assert len(pipeline.models) == len(set(pipeline.models))


def test_pipeline_excludes_skipped_models(tmp_path):
    """Verifies a skipped model is absent from the model list."""
    pipeline = BenchmarkPipeline(_config(tmp_path, skip_models=["unet"]), entry_script=ENTRY)

    assert "unet" not in pipeline.models


def test_pipeline_writes_resolved_config(tmp_path):
    """Verifies construction writes the resolved config into the run's pipeline directory."""
    BenchmarkPipeline(_config(tmp_path), entry_script=ENTRY)

    assert (tmp_path / "unit_run" / "pipeline" / "resolved_config.json").exists()


def test_mark_stage_persists_state(tmp_path):
    """Verifies marking a stage writes its status and the run tag to the state file."""
    pipeline = BenchmarkPipeline(_config(tmp_path), entry_script=ENTRY)

    pipeline._mark_stage("size_match", "completed")

    state = FileIO.load_json(pipeline.state_path)
    assert state["stages"]["size_match"]["status"] == "completed"
    assert state["run_tag"] == "unit_run"


def test_a_later_stage_process_keeps_the_earlier_stage_history(tmp_path):
    """Verifies a second pipeline process appends to rather than replaces the recorded stage history."""
    prepare = BenchmarkPipeline(_config(tmp_path), entry_script=ENTRY)
    prepare._mark_stage("size_match", "completed")
    prepare._mark_stage("prepare", "completed")

    compare = BenchmarkPipeline(_config(tmp_path), entry_script=ENTRY)
    compare._mark_stage("pipeline", "completed")

    state = FileIO.load_json(compare.state_path)
    assert set(state["stages"]) == {"size_match", "prepare", "pipeline"}


def test_runs_gating_for_backbone(tmp_path):
    """Verifies backbone sweeps run the size-match and inference stages."""
    config = _config(tmp_path, training_type="backbone")

    assert config.runs_size_match() is True
    assert config.runs_inference()  is True


def test_runs_gating_for_profile_autoencoder(tmp_path):
    """Verifies profile-autoencoder sweeps skip the size-match and inference stages."""
    config = _config(tmp_path, training_type="profile_autoencoder")

    assert config.runs_size_match() is False
    assert config.runs_inference()  is False


def _patch_stage_methods(pipeline, monkeypatch, order):
    """Replaces each pipeline stage method with a recorder appending its name to order."""
    monkeypatch.setattr(pipeline, "_run_size_match",  lambda: order.append("size_match") or {})
    monkeypatch.setattr(pipeline, "_run_max_batch",   lambda: order.append("max_batch") or {})
    monkeypatch.setattr(pipeline, "_run_training",    lambda: order.append("training") or [{"name": "unet/seed0", "status": "DONE"}])
    monkeypatch.setattr(pipeline, "_run_inference",   lambda: order.append("inference") or [{"name": "unet/seed0", "status": "DONE"}])
    monkeypatch.setattr(pipeline, "_run_comparison",  lambda: order.append("comparison") or (Path("/tmp/cmp")))


@pytest.mark.slow
def test_run_executes_stages_in_expected_order(tmp_path, monkeypatch):
    """Verifies a full run executes the five stages in order and marks the pipeline completed."""
    pipeline = BenchmarkPipeline(_config(tmp_path), entry_script=ENTRY)

    order = []
    _patch_stage_methods(pipeline, monkeypatch, order)

    pipeline.run()

    assert order == ["size_match", "max_batch", "training", "inference", "comparison"]

    state = FileIO.load_json(pipeline.state_path)
    assert state["stages"]["pipeline"]["status"] == "completed"


@pytest.mark.slow
def test_run_skips_size_match_and_inference_for_autoencoder(tmp_path, monkeypatch):
    """Verifies an autoencoder run probes, trains and compares without size match or inference, warns that infer_after is ignored, and logs and marks the skipped inference stage."""
    pipeline = BenchmarkPipeline(_config(tmp_path, training_type="profile_autoencoder"), entry_script=ENTRY)

    order = []
    _patch_stage_methods(pipeline, monkeypatch, order)
    monkeypatch.delattr(pipeline, "_run_inference")

    pipeline.run()

    state = FileIO.load_json(pipeline.state_path)
    text  = (pipeline.pipeline_dir / "benchmark_pipeline.log").read_text(encoding="utf-8")

    assert "size_match" not in order
    assert "inference"  not in order
    assert "max_batch"  in order
    assert "training"   in order
    assert "comparison" in order
    assert state["stages"]["inference"]["status"] == "skipped"
    assert "infer_after is on, but training_type 'profile_autoencoder' has no inference stage in the benchmark; it is ignored" in text
    assert "Inference skipped: training_type 'profile_autoencoder' has no inference stage" in text
    assert "Inference       : stage skipped" in text


def test_run_with_infer_after_off_logs_and_marks_the_skipped_inference(tmp_path, monkeypatch):
    """A backbone run with infer_after off skips the inference stage with a line naming the reason and a 'skipped' mark, without a warning."""
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


def test_run_rejects_unsupported_training_type(tmp_path):
    """Verifies an unsupported training type exits the pipeline."""
    pipeline = BenchmarkPipeline(_config(tmp_path, training_type="nonexistent_type"), entry_script=ENTRY)

    with pytest.raises(SystemExit, match="nonexistent_type"):
        pipeline.run()


def _recording_pipeline(tmp_path, monkeypatch, **kwargs):
    """Returns a pipeline whose stage methods record their names, together with the call log."""
    pipeline = BenchmarkPipeline(_config(tmp_path, **kwargs), entry_script=ENTRY)
    calls    = []
    returns  = {
        "_run_size_match"  : {},
        "_run_max_batch"   : {},
        "_log_unit_status" : None,
        "_run_training"    : [],
        "_run_inference"   : [],
        "_run_comparison"  : tmp_path / "comparison",
    }

    for step, value in returns.items():
        monkeypatch.setattr(pipeline, step, lambda step=step, value=value: calls.append(step) or value)

    return pipeline, calls


def test_prepare_stage_stops_before_training(tmp_path, monkeypatch):
    """Verifies the prepare stage runs size-match and max-batch only and marks itself completed."""
    pipeline, calls = _recording_pipeline(tmp_path, monkeypatch, stage="prepare")

    pipeline.run()

    assert calls == ["_run_size_match", "_run_max_batch", "_log_unit_status"]
    state = FileIO.load_json(pipeline.state_path)
    assert state["stages"]["prepare"]["status"] == "completed"


def test_compare_stage_runs_the_comparison_only(tmp_path, monkeypatch):
    """Verifies the compare stage runs only the comparison and marks the pipeline completed."""
    pipeline, calls = _recording_pipeline(tmp_path, monkeypatch, stage="compare")

    pipeline.run()

    assert calls == ["_run_comparison"]
    state = FileIO.load_json(pipeline.state_path)
    assert state["stages"]["pipeline"]["status"] == "completed"


def test_unknown_stage_is_rejected(tmp_path):
    """Verifies an unrecognised stage name exits the pipeline run."""
    pipeline = BenchmarkPipeline(_config(tmp_path, stage="warmup"), entry_script=ENTRY)

    with pytest.raises(SystemExit, match="unknown benchmark stage"):
        pipeline.run()


def test_an_exception_in_a_stage_is_logged_with_its_stage_and_marked_failed(tmp_path, monkeypatch):
    """An exception escaping a stage is re-raised after the pipeline log names the stage and state.json marks it and the pipeline failed."""
    pipeline = BenchmarkPipeline(_config(tmp_path), entry_script=ENTRY)

    def failing_max_batch():
        """Stands in for the probe stage and fails inside it."""
        pipeline._begin_stage("max_batch")
        raise RuntimeError("probe queue died")

    monkeypatch.setattr(pipeline, "_run_size_match", lambda: {})
    monkeypatch.setattr(pipeline, "_run_max_batch",  failing_max_batch)

    with pytest.raises(RuntimeError, match="probe queue died"):
        pipeline.run()

    state = FileIO.load_json(pipeline.state_path)
    text  = (pipeline.pipeline_dir / "benchmark_pipeline.log").read_text(encoding="utf-8")

    assert state["stages"]["max_batch"]["status"] == "failed"
    assert state["stages"]["pipeline"]["status"]  == "failed"
    assert "Aborted in stage 'max_batch' by RuntimeError: probe queue died" in text
    assert ">>> ABORTED benchmark_pipeline" in text
    assert ">>> END benchmark_pipeline" not in text


def test_an_abort_overrides_the_completed_mark_of_an_earlier_launch(tmp_path, monkeypatch):
    """A relaunch that fails no longer leaves the earlier launch's 'pipeline: completed' in the state file."""
    earlier = BenchmarkPipeline(_config(tmp_path), entry_script=ENTRY)
    earlier._mark_stage("pipeline", "completed")
    earlier.logger.close()

    pipeline = BenchmarkPipeline(_config(tmp_path, training_type="dual"), entry_script=ENTRY)

    with pytest.raises(SystemExit, match="dual"):
        pipeline.run()

    state = FileIO.load_json(pipeline.state_path)

    assert state["stages"]["validation"]["status"] == "failed"
    assert state["stages"]["pipeline"]["status"]   == "failed"


def test_a_clean_system_exit_is_not_recorded_as_an_abort(tmp_path, monkeypatch):
    """A SystemExit with code zero ends the run with the END banner and leaves the stage state untouched."""
    pipeline = BenchmarkPipeline(_config(tmp_path), entry_script=ENTRY)

    def clean_exit():
        """Ends the stage sequence deliberately."""
        raise SystemExit(0)

    monkeypatch.setattr(pipeline, "_execute", clean_exit)

    with pytest.raises(SystemExit):
        pipeline.run()

    text = (pipeline.pipeline_dir / "benchmark_pipeline.log").read_text(encoding="utf-8")

    assert not pipeline.state_path.exists()
    assert ">>> END benchmark_pipeline" in text
    assert "Aborted" not in text


def _unit_results(statuses: dict) -> list[dict]:
    """Returns stage results named by unit with the given statuses."""
    return [{"name": name, "status": status} for name, status in statuses.items()]


def test_a_failed_training_unit_ends_the_run_as_partial_with_a_non_zero_exit(tmp_path, monkeypatch):
    """A unit that failed training still gets the comparison written, then the run is marked partial, names the unit and exits non-zero with the END banner."""
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
    """A unit whose training finished but whose inference failed makes the run partial as well."""
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
    """A run with every unit trained and inferred is marked completed and its summary counts the units, not the models."""
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
    """A second launch into the same run tag keeps the first launch's log above its own RUN banner, so a compare job never wipes the prepare job's record."""
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
    """Relaunching a run tag with other settings archives the recording and the log lists the changed leaves with the archive path."""
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
    """A compare job following a prepare job changes only the stage, which is not reported as a configuration change."""
    first = BenchmarkPipeline(_config(tmp_path, stage="prepare"), entry_script=ENTRY)
    first.logger.close()

    second = BenchmarkPipeline(_config(tmp_path, stage="compare", gpus=[0, 1]), entry_script=ENTRY)
    second._log_launch()
    second.logger.close()

    text = (second.pipeline_dir / "benchmark_pipeline.log").read_text(encoding="utf-8")

    assert set(second.config_changes) == {"stage", "gpus"}
    assert "differ from the configuration this run recorded" not in text


def test_a_first_launch_records_no_config_change(tmp_path):
    """A fresh run tag has nothing to compare against."""
    pipeline = BenchmarkPipeline(_config(tmp_path), entry_script=ENTRY)
    pipeline.logger.close()

    assert pipeline.config_changes  == {}
    assert pipeline.archived_config is None


def test_a_resumed_launch_leaves_the_change_report_to_the_resumed_run(tmp_path):
    """With a resumed-run description the launch logs its report and no relaunch warning, since that report already lists the changes."""
    first = BenchmarkPipeline(_config(tmp_path), entry_script=ENTRY)
    first.logger.close()

    class FakeResumed:
        """Resumed-run stand-in recording whether its report was logged."""
        def __init__(self):
            """Starts with no report logged and no refresh."""
            self.root      = tmp_path / "unit_run"
            self.logged    = False
            self.refreshed = None

        def refresh(self, config):
            """Records the configuration the pipeline refreshed the change report with."""
            self.refreshed = config

        def log(self, logger):
            """Records the report call."""
            self.logged = True

    changed                 = _config(tmp_path)
    changed.training.epochs = first.config.training.epochs + 3
    resumed                 = FakeResumed()
    second                  = BenchmarkPipeline(changed, entry_script=ENTRY, resumed=resumed)

    second._log_launch()
    second.logger.close()

    text = (second.pipeline_dir / "benchmark_pipeline.log").read_text(encoding="utf-8")

    assert resumed.logged    is True
    assert resumed.refreshed is changed
    assert "differ from the configuration this run recorded" not in text


def test_every_launch_appends_its_provenance_to_the_launch_record(tmp_path, monkeypatch):
    """Each launch into a run tag adds one line naming its stage, the recording it archived and the commit and host it ran from, and the log names the commit."""
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


def test_a_skip_entry_matching_no_model_is_refused(tmp_path):
    """A skip_models entry that names no model key and no backbone name stops the sweep before anything runs, naming the entry."""
    config = _config(tmp_path, skip_models=["unet", "resnut"])

    with pytest.raises(SystemExit, match=r"skip_models lists \['resnut'\], which match no backbone model key"):
        BenchmarkPipeline.validate_sweep(config)


def test_a_skip_entry_may_name_a_full_key_or_a_backbone_name(tmp_path):
    """A skip entry naming a backbone-and-head key or a bare backbone name is accepted."""
    config = _config(tmp_path, heads=["conv", "set_pred"], skip_models=["unet-set_pred", "resunet"])

    BenchmarkPipeline.validate_sweep(config)

    assert BenchmarkPipeline.unknown_skips(config) == []


def test_the_pipeline_logs_the_selected_and_the_skipped_models(tmp_path, monkeypatch):
    """The pipeline head lists the benchmarked keys and the keys skip_models removed."""
    pipeline = BenchmarkPipeline(_config(tmp_path, skip_models=["resunet"]), entry_script=ENTRY)

    order = []
    _patch_stage_methods(pipeline, monkeypatch, order)

    pipeline.run()

    text = (pipeline.pipeline_dir / "benchmark_pipeline.log").read_text(encoding="utf-8")

    assert f"Model selection: {len(pipeline.models)} benchmarked, 1 skipped" in text
    assert "Skipped by skip_models : resunet" in text
    assert f"Selected               : {', '.join(pipeline.models)}" in text


def test_comparison_receives_the_unit_plan_of_the_launch(tmp_path, monkeypatch):
    """The comparison stage is handed every planned unit with the directory the training stage matched it to."""
    import pipelines.benchmark.pipeline as pipeline_module

    class FakeTraining:
        """Training stage stub planning two units, one matched to a renamed directory."""
        items = ["unet/seed0", "unet/seed1"]

        def __init__(self, **kwargs):
            """Accepts the stage arguments."""
            pass

        def unit_name(self, item):
            """Maps the second unit to its renamed directory."""
            return "renamed/seed1" if item == "unet/seed1" else item

    seen = {}

    class FakeComparison:
        """Comparison stage stub recording the plan it receives."""
        def __init__(self, **kwargs):
            """Records the planned units."""
            seen["planned_units"] = kwargs["planned_units"]

        def run(self):
            """Returns a fixed output directory."""
            return tmp_path / "comparison"

    monkeypatch.setattr(pipeline_module, "TrainingStage",   FakeTraining)
    monkeypatch.setattr(pipeline_module, "ComparisonStage", FakeComparison)

    pipeline = BenchmarkPipeline(_config(tmp_path, stage="compare"), entry_script=ENTRY)
    pipeline.run()

    assert seen["planned_units"] == {"unet/seed0": "unet/seed0", "unet/seed1": "renamed/seed1"}
