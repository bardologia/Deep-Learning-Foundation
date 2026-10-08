"""Tests for the cross-validation pipeline orchestration and its stage state file."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import pipelines.cross_validation.pipeline as pipeline_module
from configuration.cross_validation      import CrossValidationConfig, FoldConfig
from pipelines.cross_validation.pipeline import CrossValidationPipeline


def pipeline_config(tmp_path: Path, test_data_dir: Path, training_type: str = "backbone") -> CrossValidationConfig:
    """Returns a cross-validation config rooted at tmp_path over the real dataset."""
    config                    = CrossValidationConfig(training_type=training_type)
    config.paths.log_base_dir = tmp_path
    config.paths.dataset_path = test_data_dir
    config.folds              = FoldConfig(n_folds=5, row_start=1000, row_end=1512)
    config.run_tag            = "rt"
    return config


def install_fake_stages(monkeypatch, order: list, training_results=None, inference_results=None):
    """Replaces the pipeline stages with fakes that record their execution order.

    Args:
        monkeypatch       : Pytest patcher used to swap the stage classes.
        order             : List that receives a marker per executed stage.
        training_results  : Fold training results returned by the fake training stage.
        inference_results : Fold inference results returned by the fake inference stage.
    """
    train_results = training_results if training_results is not None else [{"name": f"fold_{i}", "status": "DONE"} for i in range(5)]
    infer_results = inference_results if inference_results is not None else [{"name": "fold_0", "status": "DONE"}]

    class FakeStatus:
        """Unit status stub that records nothing."""
        def log(self, logger, title):
            """Accepts the logger and title without writing anything."""
            pass

    class FakeTraining:
        """Training stage stub returning canned fold results."""
        unidentified = []

        def __init__(self, **kwargs):
            """Records the folds directory the stage would write under."""
            self.stage_dir = Path(kwargs["config"].paths.log_base_dir) / kwargs["run_tag"] / "folds"

        def status_report(self):
            """Returns the inert status stub."""
            return FakeStatus()

        def run(self):
            """Records the training step and returns the canned results."""
            order.append("training")
            return train_results

    class FakeInference:
        """Inference stage stub returning canned fold results."""
        def __init__(self, **kwargs):
            """Records that the stage was constructed."""
            order.append("inference_constructed")

        def run(self):
            """Records the inference step and returns the canned results."""
            order.append("inference")
            return infer_results

    class FakeReport:
        """Report stage stub writing nothing."""
        def __init__(self, **kwargs):
            """Sets a fixed output directory."""
            self.out_dir = Path("/tmp/reports")

        def run(self):
            """Records the report step and returns the output directory."""
            order.append("reports")
            return self.out_dir

    monkeypatch.setattr(pipeline_module, "FoldTrainingStage",          FakeTraining,  raising=True)
    monkeypatch.setattr(pipeline_module, "FoldInferenceStage",         FakeInference, raising=True)
    monkeypatch.setattr(pipeline_module, "CrossValidationReportStage", FakeReport,    raising=True)


@pytest.mark.dataset
def test_pipeline_init_writes_resolved_config(tmp_path, test_data_dir):
    """Verifies construction writes the resolved config and derives the run tag and directory."""
    pipeline = CrossValidationPipeline(pipeline_config(tmp_path, test_data_dir), entry_script=Path("e.py"))

    assert (pipeline.pipeline_dir / "resolved_config.json").exists()
    assert pipeline.run_tag == "rt"
    assert pipeline.run_dir == tmp_path / "rt"


@pytest.mark.dataset
def test_pipeline_runs_stages_in_order(tmp_path, test_data_dir, monkeypatch):
    """Verifies training, inference and reports run in that order."""
    order = []
    install_fake_stages(monkeypatch, order)

    pipeline = CrossValidationPipeline(pipeline_config(tmp_path, test_data_dir), entry_script=Path("e.py"))
    pipeline.run()

    assert [step for step in order if step in ("training", "inference", "reports")] == ["training", "inference", "reports"]


@pytest.mark.dataset
def test_pipeline_marks_all_stages_completed(tmp_path, test_data_dir, monkeypatch):
    """Verifies every stage and the pipeline itself are marked completed in the state file."""
    install_fake_stages(monkeypatch, [])

    pipeline = CrossValidationPipeline(pipeline_config(tmp_path, test_data_dir), entry_script=Path("e.py"))
    pipeline.run()

    state    = json.loads(pipeline.state_path.read_text())
    statuses = {name: entry["status"] for name, entry in state["stages"].items()}

    assert statuses["training"]  == "completed"
    assert statuses["inference"] == "completed"
    assert statuses["reports"]   == "completed"
    assert statuses["pipeline"]  == "completed"


@pytest.mark.dataset
def test_pipeline_training_partial_when_a_fold_failed(tmp_path, test_data_dir, monkeypatch):
    """Verifies a failed fold marks the training stage partial."""
    failed = [{"name": "fold_0", "status": "DONE"}, {"name": "fold_1", "status": "FAILED"}]
    install_fake_stages(monkeypatch, [], training_results=failed)

    pipeline = CrossValidationPipeline(pipeline_config(tmp_path, test_data_dir), entry_script=Path("e.py"))

    with pytest.raises(SystemExit, match="Partial: 1 of 2 units failed"):
        pipeline.run()

    state = json.loads(pipeline.state_path.read_text())
    text  = (pipeline.pipeline_dir / "cross_validation_pipeline.log").read_text(encoding="utf-8")

    assert state["stages"]["training"]["status"] == "partial"
    assert state["stages"]["reports"]["status"]  == "completed"
    assert state["stages"]["pipeline"]["status"] == "partial"
    assert "training failed: fold_1" in text


@pytest.mark.dataset
def test_pipeline_inference_partial_when_failed(tmp_path, test_data_dir, monkeypatch):
    """Verifies a failed fold inference marks the inference stage partial."""
    failed = [{"name": "fold_0", "status": "FAILED"}]
    install_fake_stages(monkeypatch, [], inference_results=failed)

    pipeline = CrossValidationPipeline(pipeline_config(tmp_path, test_data_dir), entry_script=Path("e.py"))

    with pytest.raises(SystemExit, match="Partial"):
        pipeline.run()

    state = json.loads(pipeline.state_path.read_text())
    assert state["stages"]["inference"]["status"] == "partial"
    assert state["stages"]["pipeline"]["status"]  == "partial"


@pytest.mark.dataset
def test_pipeline_runs_inference_for_every_training_type(tmp_path, test_data_dir, monkeypatch):
    """Verifies inference runs for each supported training type."""
    for training_type in ("backbone", "dual", "profile_autoencoder", "image_autoencoder"):
        order = []
        install_fake_stages(monkeypatch, order)

        config   = pipeline_config(tmp_path / training_type, test_data_dir, training_type=training_type)
        pipeline = CrossValidationPipeline(config, entry_script=Path("e.py"))
        pipeline.run()

        assert "inference" in order


@pytest.mark.dataset
def test_pipeline_skips_inference_without_splits(tmp_path, test_data_dir, monkeypatch):
    """Verifies an empty inference split list marks inference skipped and never constructs the stage."""
    order = []
    install_fake_stages(monkeypatch, order)

    config                  = pipeline_config(tmp_path, test_data_dir, training_type="profile_autoencoder")
    config.inference_splits = []
    pipeline                = CrossValidationPipeline(config, entry_script=Path("e.py"))
    pipeline.run()

    assert "inference"             not in order
    assert "inference_constructed" not in order

    state = json.loads(pipeline.state_path.read_text())
    assert state["stages"]["inference"]["status"] == "skipped"


@pytest.mark.dataset
def test_pipeline_state_records_timestamps(tmp_path, test_data_dir, monkeypatch):
    """Verifies the state file carries the run tag and a timestamp per stage."""
    install_fake_stages(monkeypatch, [])

    pipeline = CrossValidationPipeline(pipeline_config(tmp_path, test_data_dir), entry_script=Path("e.py"))
    pipeline.run()

    state = json.loads(pipeline.state_path.read_text())
    assert state["run_tag"] == "rt"

    for entry in state["stages"].values():
        assert "timestamp" in entry


@pytest.mark.dataset
@pytest.mark.slow
def test_pipeline_run_with_real_planner_and_fake_stages(tmp_path, test_data_dir, monkeypatch):
    """Verifies the real planner yields one plan per fold and the stages each run once."""
    order = []
    install_fake_stages(monkeypatch, order)

    config   = pipeline_config(tmp_path, test_data_dir)
    pipeline = CrossValidationPipeline(config, entry_script=Path("e.py"))

    planner = pipeline.factory.planner()
    assert len(planner.plans()) == 5

    pipeline.run()

    assert order.count("training") == 1
    assert order.count("reports")  == 1


@pytest.mark.dataset
def test_pipeline_prepare_stage_stops_before_training(tmp_path, test_data_dir, monkeypatch):
    """Verifies the prepare stage writes the resolved config without running any stage."""
    order = []
    install_fake_stages(monkeypatch, order)

    config       = pipeline_config(tmp_path, test_data_dir)
    config.stage = "prepare"
    pipeline     = CrossValidationPipeline(config, entry_script=Path("e.py"))
    pipeline.run()

    assert order == []
    assert (pipeline.pipeline_dir / "resolved_config.json").exists()

    state = json.loads(pipeline.state_path.read_text())
    assert state["stages"]["prepare"]["status"] == "completed"
    assert "training" not in state["stages"]


@pytest.mark.dataset
def test_pipeline_compare_stage_runs_reports_only(tmp_path, test_data_dir, monkeypatch):
    """Verifies the compare stage runs only the report stage."""
    order = []
    install_fake_stages(monkeypatch, order)

    config       = pipeline_config(tmp_path, test_data_dir)
    config.stage = "compare"
    pipeline     = CrossValidationPipeline(config, entry_script=Path("e.py"))
    pipeline.run()

    assert order == ["reports"]

    state = json.loads(pipeline.state_path.read_text())
    assert state["stages"]["reports"]["status"]  == "completed"
    assert state["stages"]["pipeline"]["status"] == "completed"
    assert "training" not in state["stages"]


@pytest.mark.dataset
def test_pipeline_rejects_an_unknown_stage(tmp_path, test_data_dir, monkeypatch):
    """Verifies an unknown stage name aborts the run."""
    install_fake_stages(monkeypatch, [])

    config       = pipeline_config(tmp_path, test_data_dir)
    config.stage = "warmup"
    pipeline     = CrossValidationPipeline(config, entry_script=Path("e.py"))

    with pytest.raises(SystemExit, match="unknown cross-validation stage"):
        pipeline.run()


@pytest.mark.dataset
def test_pipeline_rejects_an_unsupported_inference_split(tmp_path, test_data_dir, monkeypatch):
    """Verifies an inference split outside the supported set aborts sweep validation."""
    install_fake_stages(monkeypatch, [])

    config                  = pipeline_config(tmp_path, test_data_dir)
    config.inference_splits = ["val", "train"]

    with pytest.raises(SystemExit, match="unknown inference split"):
        CrossValidationPipeline.validate_sweep(config)


@pytest.mark.dataset
def test_pipeline_abort_in_training_is_logged_and_marked_failed(tmp_path, test_data_dir, monkeypatch):
    """A training stage that raises ends the run with the stage named in the log and marked failed in the state file."""
    install_fake_stages(monkeypatch, [])

    class FailingTraining:
        """Training stage stub that dies while queueing."""
        def __init__(self, **kwargs):
            """Accepts the stage arguments."""
            pass

        def run(self):
            """Raises as a GPU queue failure would."""
            raise RuntimeError("no device left in the pool")

    monkeypatch.setattr(pipeline_module, "FoldTrainingStage", FailingTraining, raising=True)

    pipeline = CrossValidationPipeline(pipeline_config(tmp_path, test_data_dir), entry_script=Path("e.py"))

    with pytest.raises(RuntimeError, match="no device left"):
        pipeline.run()

    state = json.loads(pipeline.state_path.read_text())
    text  = (pipeline.pipeline_dir / "cross_validation_pipeline.log").read_text(encoding="utf-8")

    assert state["stages"]["training"]["status"] == "failed"
    assert state["stages"]["pipeline"]["status"] == "failed"
    assert "Aborted in stage 'training' by RuntimeError: no device left in the pool" in text
    assert ">>> ABORTED cross_validation_pipeline" in text
