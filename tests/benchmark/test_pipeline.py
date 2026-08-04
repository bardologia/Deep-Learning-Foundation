from __future__ import annotations

from pathlib import Path

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

    pipeline._mark_stage("size_match", "completed")

    state = FileIO.load_json(pipeline.state_path)
    assert state["stages"]["size_match"]["status"] == "completed"
    assert state["run_tag"] == "unit_run"


def test_runs_gating_for_backbone(tmp_path):
    config = _config(tmp_path, training_type="backbone")

    assert config.runs_size_match() is True
    assert config.runs_max_batch()  is True
    assert config.runs_inference()  is True


def test_runs_gating_for_profile_autoencoder(tmp_path):
    config = _config(tmp_path, training_type="profile_autoencoder")

    assert config.runs_size_match() is False
    assert config.runs_max_batch()  is False
    assert config.runs_inference()  is False


def _patch_stage_methods(pipeline, monkeypatch, order):
    monkeypatch.setattr(pipeline, "_run_size_match",  lambda: order.append("size_match") or {})
    monkeypatch.setattr(pipeline, "_run_max_batch",   lambda: order.append("max_batch") or {})
    monkeypatch.setattr(pipeline, "_run_training",    lambda: order.append("training") or [{"status": "DONE"}])
    monkeypatch.setattr(pipeline, "_run_inference",   lambda: order.append("inference") or [{"status": "DONE"}])
    monkeypatch.setattr(pipeline, "_run_comparison",  lambda: order.append("comparison") or (Path("/tmp/cmp")))


@pytest.mark.slow
def test_run_executes_stages_in_expected_order(tmp_path, monkeypatch):
    pipeline = BenchmarkPipeline(_config(tmp_path), entry_script=ENTRY)

    order = []
    _patch_stage_methods(pipeline, monkeypatch, order)

    pipeline.run()

    assert order == ["size_match", "max_batch", "training", "inference", "comparison"]

    state = FileIO.load_json(pipeline.state_path)
    assert state["stages"]["pipeline"]["status"] == "completed"


@pytest.mark.slow
def test_run_skips_size_match_and_maxbatch_for_autoencoder(tmp_path, monkeypatch):
    pipeline = BenchmarkPipeline(_config(tmp_path, training_type="profile_autoencoder"), entry_script=ENTRY)

    order = []
    _patch_stage_methods(pipeline, monkeypatch, order)

    pipeline.run()

    assert "size_match" not in order
    assert "max_batch"  not in order
    assert "inference"  not in order
    assert "training"   in order
    assert "comparison" in order


def test_run_rejects_unsupported_training_type(tmp_path):
    pipeline = BenchmarkPipeline(_config(tmp_path, training_type="unrolled"), entry_script=ENTRY)

    with pytest.raises(SystemExit, match="unrolled"):
        pipeline.run()
