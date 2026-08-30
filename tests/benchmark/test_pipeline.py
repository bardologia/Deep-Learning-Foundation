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
    monkeypatch.setattr(pipeline, "_run_training",    lambda: order.append("training") or [{"status": "DONE"}])
    monkeypatch.setattr(pipeline, "_run_inference",   lambda: order.append("inference") or [{"status": "DONE"}])
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
    """Verifies an autoencoder run probes, trains and compares without size match or inference."""
    pipeline = BenchmarkPipeline(_config(tmp_path, training_type="profile_autoencoder"), entry_script=ENTRY)

    order = []
    _patch_stage_methods(pipeline, monkeypatch, order)

    pipeline.run()

    assert "size_match" not in order
    assert "inference"  not in order
    assert "max_batch"  in order
    assert "training"   in order
    assert "comparison" in order


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
        "_run_size_match" : {},
        "_run_max_batch"  : {},
        "_run_training"   : [],
        "_run_inference"  : [],
        "_run_comparison" : tmp_path / "comparison",
    }

    for step, value in returns.items():
        monkeypatch.setattr(pipeline, step, lambda step=step, value=value: calls.append(step) or value)

    return pipeline, calls


def test_prepare_stage_stops_before_training(tmp_path, monkeypatch):
    """Verifies the prepare stage runs size-match and max-batch only and marks itself completed."""
    pipeline, calls = _recording_pipeline(tmp_path, monkeypatch, stage="prepare")

    pipeline.run()

    assert calls == ["_run_size_match", "_run_max_batch"]
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
