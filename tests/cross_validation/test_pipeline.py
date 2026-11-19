
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
    config.folds              = FoldConfig(n_folds=5, row_start=1000, row_end=1512)
    config.run_tag            = "rt"
    return config


def install_fake_stages(monkeypatch, order: list, training_results=None, inference_results=None):
    train_results = training_results if training_results is not None else [{"name": f"fold_{i}", "status": "DONE"} for i in range(5)]
    infer_results = inference_results if inference_results is not None else [{"name": "fold_0", "status": "DONE"}]

    class FakeStatus:
        def log(self, logger, title):
            pass

    class FakeTraining:
        unidentified = []

        def __init__(self, **kwargs):
            self.stage_dir = Path(kwargs["config"].paths.log_base_dir) / kwargs["run_tag"] / "folds"

        def status_report(self):
            return FakeStatus()

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


@pytest.mark.dataset
def test_pipeline_init_writes_resolved_config(tmp_path, test_data_dir):
    pipeline = CrossValidationPipeline(pipeline_config(tmp_path, test_data_dir), entry_script=Path("e.py"))

    assert (pipeline.pipeline_dir / "resolved_config.json").exists()
    assert pipeline.run_tag == "rt"
    assert pipeline.run_dir == tmp_path / "rt"


@pytest.mark.dataset
def test_pipeline_runs_stages_in_order(tmp_path, test_data_dir, monkeypatch):
    order = []
    install_fake_stages(monkeypatch, order)

    pipeline = CrossValidationPipeline(pipeline_config(tmp_path, test_data_dir), entry_script=Path("e.py"))
    pipeline.run()

    assert [step for step in order if step in ("training", "inference", "reports")] == ["training", "inference", "reports"]


@pytest.mark.dataset
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


@pytest.mark.dataset
def test_pipeline_training_partial_when_a_fold_failed(tmp_path, test_data_dir, monkeypatch):
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
    for training_type in ("backbone", "profile_autoencoder", "image_autoencoder"):
        order = []
        install_fake_stages(monkeypatch, order)

        config   = pipeline_config(tmp_path / training_type, test_data_dir, training_type=training_type)
        pipeline = CrossValidationPipeline(config, entry_script=Path("e.py"))
        pipeline.run()

        assert "inference" in order


@pytest.mark.dataset
def test_pipeline_skips_inference_without_splits(tmp_path, test_data_dir, monkeypatch):
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
    install_fake_stages(monkeypatch, [])

    config       = pipeline_config(tmp_path, test_data_dir)
    config.stage = "warmup"
    pipeline     = CrossValidationPipeline(config, entry_script=Path("e.py"))

    with pytest.raises(SystemExit, match="unknown cross-validation stage"):
        pipeline.run()


@pytest.mark.dataset
def test_pipeline_rejects_an_unsupported_inference_split(tmp_path, test_data_dir, monkeypatch):
    install_fake_stages(monkeypatch, [])

    config                  = pipeline_config(tmp_path, test_data_dir)
    config.inference_splits = ["val", "train"]

    with pytest.raises(SystemExit, match="unknown inference split"):
        CrossValidationPipeline.validate_sweep(config)


@pytest.mark.dataset
def test_pipeline_abort_in_training_is_logged_and_marked_failed(tmp_path, test_data_dir, monkeypatch):
    install_fake_stages(monkeypatch, [])

    class FailingTraining:
        def __init__(self, **kwargs):
            pass

        def run(self):
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


@pytest.mark.dataset
def test_pipeline_logs_the_inherited_warmup_fields_and_reports_them_as_resumed_changes(tmp_path, test_data_dir, monkeypatch):
    from configuration.training                     import CurriculumInheritance, default_curriculum
    from pipelines.shared.orchestration.resumed_run import ResumedRun
    from tools.runtime.config_cli                   import ConfigCli

    CrossValidationPipeline(pipeline_config(tmp_path, test_data_dir), entry_script=Path("e.py"))

    cli       = ConfigCli(CrossValidationConfig())
    launch    = cli.apply(["--resume_from", str(tmp_path / "rt"), "--curriculum.complete.weight_mse_curve", "0.25"])
    resumed   = ResumedRun(CrossValidationConfig, launch, cli.overrides)
    config    = resumed.load()
    inherited = CurriculumInheritance(config.curriculum, default_curriculum(), cli.overrides).apply()

    order = []
    install_fake_stages(monkeypatch, order)

    pipeline = CrossValidationPipeline(config, entry_script=Path("e.py"), resumed=resumed, inherited=inherited)
    pipeline.run()

    text = (pipeline.pipeline_dir / "cross_validation_pipeline.log").read_text(encoding="utf-8")

    assert "1 warmup field(s) inherited from the complete stage" in text
    assert "curriculum.warmup.weight_mse_curve" in text
    assert resumed.changed["curriculum.warmup.weight_mse_curve"] == (1.0, 0.25)
