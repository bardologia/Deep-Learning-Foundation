from __future__ import annotations

import json
from pathlib import Path

import pytest

import pipelines.cross_validation.pipeline as pipeline_module
from configuration.cross_validation      import CrossValidationConfig, FoldConfig
from pipelines.cross_validation.pipeline import CrossValidationPipeline


def pipeline_config(tmp_path: Path, test_data_dir: Path, training_type: str = "backbone") -> CrossValidationConfig:
    config                    = CrossValidationConfig(training_type=training_type)
    config.paths.log_base_dir = tmp_path
    config.paths.dataset_path = test_data_dir
    config.folds              = FoldConfig(n_folds=5, azimuth_start=1000, azimuth_end=2000)
    config.run_tag            = "rt"
    return config


def install_fake_stages(monkeypatch, order: list, training_results=None, inference_results=None):
    train_results = training_results if training_results is not None else [{"name": f"fold_{i}", "status": "DONE"} for i in range(5)]
    infer_results = inference_results if inference_results is not None else [{"name": "fold_0:test", "status": "DONE"}]

    class FakeTraining:
        def __init__(self, **kwargs):
            pass

        def run(self):
            order.append("training")
            return train_results

    class FakeInference:
        def __init__(self, **kwargs):
            order.append("inference_constructed")

        def run(self):
            order.append("inference")
            return infer_results

    class FakeReport:
        def __init__(self, **kwargs):
            self.out_dir = Path("/tmp/reports")

        def run(self):
            order.append("reports")
            return self.out_dir

    monkeypatch.setattr(pipeline_module, "FoldTrainingStage",          FakeTraining,  raising=True)
    monkeypatch.setattr(pipeline_module, "FoldInferenceStage",         FakeInference, raising=True)
    monkeypatch.setattr(pipeline_module, "CrossValidationReportStage", FakeReport,    raising=True)


@pytest.mark.real_data
def test_pipeline_init_writes_resolved_config(tmp_path, test_data_dir):
    pipeline = CrossValidationPipeline(pipeline_config(tmp_path, test_data_dir), entry_script=Path("e.py"))

    assert (pipeline.pipeline_dir / "resolved_config.json").exists()
    assert pipeline.run_tag == "rt"
    assert pipeline.run_dir == tmp_path / "rt"


@pytest.mark.real_data
def test_pipeline_runs_stages_in_order(tmp_path, test_data_dir, monkeypatch):
    order = []
    install_fake_stages(monkeypatch, order)

    pipeline = CrossValidationPipeline(pipeline_config(tmp_path, test_data_dir), entry_script=Path("e.py"))
    pipeline.run()

    assert [step for step in order if step in ("training", "inference", "reports")] == ["training", "inference", "reports"]


@pytest.mark.real_data
def test_pipeline_marks_all_stages_completed(tmp_path, test_data_dir, monkeypatch):
    install_fake_stages(monkeypatch, [])

    pipeline = CrossValidationPipeline(pipeline_config(tmp_path, test_data_dir), entry_script=Path("e.py"))
    pipeline.run()

    state    = json.loads(pipeline.state_path.read_text())
    statuses = {name: entry["status"] for name, entry in state["stages"].items()}

    assert statuses["training"]  == "completed"
    assert statuses["inference"] == "completed"
    assert statuses["reports"]   == "completed"
    assert statuses["pipeline"]  == "completed"


@pytest.mark.real_data
def test_pipeline_training_partial_when_a_fold_failed(tmp_path, test_data_dir, monkeypatch):
    failed = [{"name": "fold_0", "status": "DONE"}, {"name": "fold_1", "status": "FAILED"}]
    install_fake_stages(monkeypatch, [], training_results=failed)

    pipeline = CrossValidationPipeline(pipeline_config(tmp_path, test_data_dir), entry_script=Path("e.py"))
    pipeline.run()

    state = json.loads(pipeline.state_path.read_text())
    assert state["stages"]["training"]["status"] == "partial"


@pytest.mark.real_data
def test_pipeline_inference_partial_when_failed(tmp_path, test_data_dir, monkeypatch):
    failed = [{"name": "fold_0:test", "status": "FAILED"}]
    install_fake_stages(monkeypatch, [], inference_results=failed)

    pipeline = CrossValidationPipeline(pipeline_config(tmp_path, test_data_dir), entry_script=Path("e.py"))
    pipeline.run()

    state = json.loads(pipeline.state_path.read_text())
    assert state["stages"]["inference"]["status"] == "partial"


@pytest.mark.real_data
def test_pipeline_skips_inference_for_profile_autoencoder(tmp_path, test_data_dir, monkeypatch):
    order = []
    install_fake_stages(monkeypatch, order)

    config   = pipeline_config(tmp_path, test_data_dir, training_type="profile_autoencoder")
    pipeline = CrossValidationPipeline(config, entry_script=Path("e.py"))
    pipeline.run()

    assert "inference"             not in order
    assert "inference_constructed" not in order

    state = json.loads(pipeline.state_path.read_text())
    assert state["stages"]["inference"]["status"] == "skipped"


@pytest.mark.real_data
def test_pipeline_state_records_timestamps(tmp_path, test_data_dir, monkeypatch):
    install_fake_stages(monkeypatch, [])

    pipeline = CrossValidationPipeline(pipeline_config(tmp_path, test_data_dir), entry_script=Path("e.py"))
    pipeline.run()

    state = json.loads(pipeline.state_path.read_text())
    assert state["run_tag"] == "rt"

    for entry in state["stages"].values():
        assert "timestamp" in entry


@pytest.mark.real_data
@pytest.mark.slow
def test_pipeline_run_with_real_planner_and_fake_stages(tmp_path, test_data_dir, monkeypatch):
    order = []
    install_fake_stages(monkeypatch, order)

    config   = pipeline_config(tmp_path, test_data_dir)
    pipeline = CrossValidationPipeline(config, entry_script=Path("e.py"))

    planner = pipeline.factory.planner()
    assert len(planner.plans()) == 5

    pipeline.run()

    assert order.count("training") == 1
    assert order.count("reports")  == 1
