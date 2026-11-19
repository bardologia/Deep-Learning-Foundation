
from __future__ import annotations


import pytest

from pipelines.benchmark.workers import (
    BenchmarkWorker,
    MaxBatchWorker,
)
from configuration.benchmark import BenchmarkConfig
from tools.data.io           import FileIO
from configuration.synthesis import SynthesisConfig
from tools.runtime.git_state import GitState


SCENE_K_TAG = f"K_{SynthesisConfig().k_max}"


@pytest.fixture
def config(tmp_path):
    config                    = BenchmarkConfig()
    config.paths.log_base_dir = tmp_path
    return config


@pytest.mark.dataset
def test_run_name_encodes_model_head_matching_gaussians_aug_presence_component_and_seed(config, test_data_dir, params_dir):
    from pipelines.shared.training.run_naming import RunNaming

    config.paths.dataset_path    = test_data_dir
    config.paths.parameters_path = params_dir / "parameters.npy"

    worker = BenchmarkWorker(config, "tag")
    tag    = RunNaming.benchmark_unit("unet", None, config.loss, SynthesisConfig().k_max, config.augmentation)

    assert worker._run_name("unet", None, None)       == tag
    assert worker._run_name("unet", None, 5)          == f"{tag}/seed5"
    assert worker._run_name("unet", "param_l1", None) == f"unet-conv-hungarian-{SCENE_K_TAG}-hv-AB__param_l1"
    assert worker._run_name("unet", "param_l1", 5)    == f"unet-conv-hungarian-{SCENE_K_TAG}-hv-AB__param_l1/seed5"
    assert tag.startswith(f"unet-conv-hungarian-{SCENE_K_TAG}-hv-AB-param_l1_1")


def test_size_overrides_empty_without_file(config):
    worker = BenchmarkWorker(config, "tag")

    assert worker._size_overrides("unet") == {}


def test_size_overrides_reads_overrides(config):
    worker = BenchmarkWorker(config, "tag")
    path   = worker.run_dir / "pipeline" / "size_match.json"

    FileIO.save_json({"unet": {"overrides": {"features": [32, 64]}}}, path)

    assert worker._size_overrides("unet") == {"features": [32, 64]}


def test_backbone_training_rejects_a_stale_size_override(config):
    from pipelines.benchmark.workers import TrainingWorker

    worker = TrainingWorker(config, "tag")

    FileIO.save_json({"unet": {"overrides": {"renamed_features": [32, 64]}}}, worker.run_dir / "pipeline" / "size_match.json")

    with pytest.raises(AttributeError, match="Unknown model override 'renamed_features'"):
        worker._run_backbone("unet", run_name="unet", seed=None, loss_component=None)


def test_size_overrides_missing_model_exits(config):
    worker = BenchmarkWorker(config, "tag")
    path   = worker.run_dir / "pipeline" / "size_match.json"

    FileIO.save_json({"other_model": {"overrides": {}}}, path)

    with pytest.raises(SystemExit, match="missing an entry for 'unet'"):
        worker._size_overrides("unet")


def test_max_batch_size_none_without_file(config):
    worker = BenchmarkWorker(config, "tag")

    assert worker._max_batch_size("unet") is None


def test_max_batch_size_returns_measured_value(config):
    worker = BenchmarkWorker(config, "tag")
    path   = worker.run_dir / "pipeline" / "max_batch.json"

    FileIO.save_json({"unet": {"status": "PASS", "batch_size": 128}}, path)

    assert worker._max_batch_size("unet") == 128


def test_max_batch_size_raises_when_model_missing(config):
    worker = BenchmarkWorker(config, "tag")
    path   = worker.run_dir / "pipeline" / "max_batch.json"

    FileIO.save_json({"resunet": {"status": "PASS", "batch_size": 128}}, path)

    with pytest.raises(SystemExit):
        worker._max_batch_size("unet")


def test_max_batch_size_raises_when_not_passed(config):
    worker = BenchmarkWorker(config, "tag")
    path   = worker.run_dir / "pipeline" / "max_batch.json"

    FileIO.save_json({"unet": {"status": "FAIL", "batch_size": None}}, path)

    with pytest.raises(SystemExit):
        worker._max_batch_size("unet")


def test_lr_multiplier_none_without_file(config):
    worker = BenchmarkWorker(config, "tag")

    assert worker._lr_multiplier("unet") is None


def test_lr_multiplier_returns_probed_value(config):
    worker = BenchmarkWorker(config, "tag")
    path   = worker.run_dir / "pipeline" / "max_batch.json"

    FileIO.save_json({"unet": {"status": "PASS", "batch_size": 128, "lr_multiplier": 0.25}}, path)

    assert worker._lr_multiplier("unet") == pytest.approx(0.25)


def test_lr_multiplier_none_when_the_test_selected_nothing(config):
    worker = BenchmarkWorker(config, "tag")
    path   = worker.run_dir / "pipeline" / "max_batch.json"

    FileIO.save_json({"unet": {"status": "PASS", "batch_size": 128, "lr_multiplier": None}}, path)

    assert worker._lr_multiplier("unet") is None


def test_lr_multiplier_raises_when_model_missing(config):
    worker = BenchmarkWorker(config, "tag")
    path   = worker.run_dir / "pipeline" / "max_batch.json"

    FileIO.save_json({"resunet": {"status": "PASS", "batch_size": 128, "lr_multiplier": 0.5}}, path)

    with pytest.raises(SystemExit):
        worker._lr_multiplier("unet")


def test_probe_config_is_disabled(config):
    probe = BenchmarkWorker(config, "tag").config.probe

    assert probe.enabled    is False
    assert probe.exit_after is True


def test_probe_config_is_the_configured_one(config):
    config.probe.enabled   = True
    config.probe.n_batches = 7

    probe = BenchmarkWorker(config, "tag").config.probe

    assert probe.enabled   is True
    assert probe.n_batches == 7


def test_max_batch_worker_hands_the_probe_its_model_and_work_dir(config, monkeypatch):
    worker = MaxBatchWorker(config, "tag")

    calls = {}

    class FakeProbe:
        def __init__(self, config, model_name, overrides, work_dir):
            calls["model_name"] = model_name
            calls["overrides"]  = overrides
            calls["work_dir"]   = work_dir
        def run(self):
            return {"model": "unet", "status": "PASS", "batch_size": 64, "peak_gb": 12.0}

    monkeypatch.setattr("pipelines.benchmark.batch_probe.MaxBatchProbe", FakeProbe)

    worker.run("unet")

    assert calls["model_name"] == "unet"
    assert calls["work_dir"]   == worker.run_dir / "max_batch" / "unet" / "probe"
    result_path = worker.run_dir / "max_batch" / "unet" / "max_batch_result.json"
    assert result_path.exists()
    assert FileIO.load_json(result_path)["batch_size"] == 64
    assert FileIO.load_json(result_path)["provenance"]["git_commit"] == GitState.commit()


def test_max_batch_worker_raises_on_probe_fail(config, monkeypatch):
    worker = MaxBatchWorker(config, "tag")

    class FakeProbe:
        def __init__(self, *args, **kwargs):
            pass
        def run(self):
            return {"model": "unet", "status": "FAIL", "batch_size": None, "peak_gb": None}

    monkeypatch.setattr("pipelines.benchmark.batch_probe.MaxBatchProbe", FakeProbe)

    with pytest.raises(SystemExit):
        worker.run("unet")


@pytest.mark.dataset
def test_training_worker_scales_the_lr_from_the_measured_batch(config, test_data_dir, params_dir, monkeypatch):
    from pipelines.benchmark.workers import TrainingWorker

    config.paths.dataset_path    = test_data_dir
    config.paths.parameters_path = params_dir / "parameters.npy"
    config.training.train_rows   = (1000, 1400)
    config.training.val_rows     = (1400, 1700)
    config.training.test_rows    = (1700, 2000)

    worker = TrainingWorker(config, "tag")
    FileIO.save_json({"resunet-conv": {"status": "PASS", "batch_size": 500}}, worker.run_dir / "pipeline" / "max_batch.json")

    captured = {}

    class FakePipeline:
        def __init__(self, **kwargs):
            captured.update(kwargs)
        def run(self, probe_config=None):
            return None

    monkeypatch.setattr("pipelines.backbone.training.pipeline.TrainingPipeline", FakePipeline)

    worker.run("resunet-conv")

    assert captured["trainer_config"].optimizer.lr_scale == pytest.approx(500 / config.training.lr_reference_batch_size)
    assert captured["dataset_config"].batch_size         == 500


@pytest.mark.dataset
def test_training_worker_applies_the_probed_lr_multiplier(config, test_data_dir, params_dir, monkeypatch):
    from pipelines.benchmark.workers import TrainingWorker

    config.paths.dataset_path     = test_data_dir
    config.paths.parameters_path  = params_dir / "parameters.npy"
    config.training.train_rows = (1000, 1400)
    config.training.val_rows   = (1400, 1700)
    config.training.test_rows  = (1700, 2000)
    config.find_lr.enabled        = True

    worker = TrainingWorker(config, "tag")
    FileIO.save_json({"resunet-conv": {"status": "PASS", "batch_size": 500, "lr_multiplier": 0.25}}, worker.run_dir / "pipeline" / "max_batch.json")

    captured = {}

    class FakePipeline:
        def __init__(self, **kwargs):
            captured.update(kwargs)
        def run(self, probe_config=None):
            return None

    monkeypatch.setattr("pipelines.backbone.training.pipeline.TrainingPipeline", FakePipeline)

    worker.run("resunet-conv")

    assert captured["trainer_config"].optimizer.lr_scale == pytest.approx(500 / config.training.lr_reference_batch_size * 0.25)


@pytest.mark.dataset
def test_training_worker_ignores_the_multiplier_when_the_test_is_disabled(config, test_data_dir, params_dir, monkeypatch):
    from pipelines.benchmark.workers import TrainingWorker

    config.paths.dataset_path     = test_data_dir
    config.paths.parameters_path  = params_dir / "parameters.npy"
    config.training.train_rows = (1000, 1400)
    config.training.val_rows   = (1400, 1700)
    config.training.test_rows  = (1700, 2000)

    worker = TrainingWorker(config, "tag")
    FileIO.save_json({"resunet-conv": {"status": "PASS", "batch_size": 500, "lr_multiplier": 0.25}}, worker.run_dir / "pipeline" / "max_batch.json")

    captured = {}

    class FakePipeline:
        def __init__(self, **kwargs):
            captured.update(kwargs)
        def run(self, probe_config=None):
            return None

    monkeypatch.setattr("pipelines.backbone.training.pipeline.TrainingPipeline", FakePipeline)

    worker.run("resunet-conv")

    assert captured["trainer_config"].optimizer.lr_scale == pytest.approx(500 / config.training.lr_reference_batch_size)


@pytest.mark.dataset
def test_training_worker_leaves_a_finished_unit_alone(config, test_data_dir, params_dir, monkeypatch):
    from pipelines.benchmark.workers import TrainingWorker
    from tools.runtime.completion    import CompletionMarker

    config.paths.dataset_path    = test_data_dir
    config.paths.parameters_path = params_dir / "parameters.npy"

    worker = TrainingWorker(config, "tag")
    unit   = worker.run_dir / "training" / "renamed/seed3"
    unit.mkdir(parents=True)
    CompletionMarker.stamp(unit, {"stage": "training"})

    trained = []
    monkeypatch.setattr(TrainingWorker, "_run_backbone", lambda self, model_name, run_name, seed, loss_component: trained.append(run_name), raising=True)

    worker.run("unet", seed=3, loss_component="param_l1", run_name="renamed/seed3")

    assert trained == []


@pytest.mark.dataset
def test_training_worker_writes_the_unit_manifest_into_the_handed_over_directory(config, test_data_dir, params_dir, monkeypatch):
    from pipelines.benchmark.stages                   import BenchmarkUnitManifest
    from pipelines.benchmark.workers                  import TrainingWorker
    from pipelines.shared.orchestration.unit_manifest import UnitManifest

    config.paths.dataset_path    = test_data_dir
    config.paths.parameters_path = params_dir / "parameters.npy"

    seen = {}
    monkeypatch.setattr(TrainingWorker, "_run_backbone", lambda self, model_name, run_name, seed, loss_component: seen.update(run_name=run_name), raising=True)

    worker = TrainingWorker(config, "tag")
    worker.run("unet", seed=3, loss_component="param_l1", run_name="renamed/seed3")

    manifest = UnitManifest.read(worker.run_dir / "training" / "renamed/seed3")
    assert seen["run_name"] == "renamed/seed3"
    assert manifest["kind"] == BenchmarkUnitManifest.KIND
    assert BenchmarkUnitManifest.identity(manifest) == ("unet", "param_l1", 3)
    assert manifest["unit"] == "renamed/seed3"


def test_training_worker_refuses_a_finished_unit_when_resume_is_off(config, monkeypatch):
    from pipelines.benchmark.workers import TrainingWorker
    from tools.runtime.completion    import CompletionMarker

    config.resume = False
    worker        = TrainingWorker(config, "tag")
    unit          = worker.run_dir / "training" / "renamed/seed3"
    unit.mkdir(parents=True)
    CompletionMarker.stamp(unit, {"stage": "training"})

    trained = []
    monkeypatch.setattr(TrainingWorker, "_run_backbone", lambda self, model_name, run_name, seed, loss_component: trained.append(run_name), raising=True)

    with pytest.raises(SystemExit, match="already holds a completed run and resume is off"):
        worker.run("unet", seed=3, loss_component="param_l1", run_name="renamed/seed3")

    assert trained == []
    assert CompletionMarker.is_complete(unit)


def test_training_worker_trains_an_unfinished_unit_when_resume_is_off(config, monkeypatch):
    from pipelines.benchmark.workers import TrainingWorker

    config.resume = False
    worker        = TrainingWorker(config, "tag")

    trained = []
    monkeypatch.setattr(TrainingWorker, "_run_backbone", lambda self, model_name, run_name, seed, loss_component: trained.append(run_name), raising=True)

    worker.run("unet", seed=3, loss_component="param_l1", run_name="renamed/seed3")

    assert trained == ["renamed/seed3"]


def _stub_inference(monkeypatch) -> list:
    import pipelines.backbone.inference.pipeline as pipeline_module

    ran = []

    class FakePipeline:

        def __init__(self, inference_config):
            self.run_directory = inference_config.run_directory

        def run(self):
            ran.append(self.run_directory)

    monkeypatch.setattr(pipeline_module, "InferencePipeline", FakePipeline, raising=True)

    return ran


def test_inference_worker_reuses_a_completed_inference_on_resume(config, monkeypatch):
    from pipelines.benchmark.workers import InferenceWorker
    from tools.runtime.completion    import CompletionMarker

    ran    = _stub_inference(monkeypatch)
    worker = InferenceWorker(config, "tag")
    unit   = worker.run_dir / "training" / "renamed/seed3"
    done   = unit / "inference" / "stamp_done"
    torn   = unit / "inference" / "stamp_torn"
    done.mkdir(parents=True)
    torn.mkdir(parents=True)
    CompletionMarker.stamp(done, {"stage": "inference", "split": config.inference.split})

    worker.run("unet", seed=3, loss_component="param_l1", run_name="renamed/seed3")

    log = (unit / "logs" / "unit_resume.log").read_text()
    assert ran == []
    assert done.is_dir()
    assert not torn.exists()
    assert f"existing '{config.inference.split}' inference reused" in log
    assert "unfinished inference 'stamp_torn' deleted" in log


def test_inference_worker_scores_a_unit_without_a_completed_inference_for_the_split(config, monkeypatch):
    from pipelines.benchmark.workers import InferenceWorker
    from tools.runtime.completion    import CompletionMarker

    ran    = _stub_inference(monkeypatch)
    worker = InferenceWorker(config, "tag")
    unit   = worker.run_dir / "training" / "renamed/seed3"
    other  = unit / "inference" / "stamp_other"
    other.mkdir(parents=True)
    CompletionMarker.stamp(other, {"stage": "inference", "split": "train"})

    worker.run("unet", seed=3, loss_component="param_l1", run_name="renamed/seed3")

    assert ran == [unit]


def test_inference_worker_rescores_a_completed_unit_when_resume_is_off(config, monkeypatch):
    from pipelines.benchmark.workers import InferenceWorker
    from tools.runtime.completion    import CompletionMarker

    config.resume = False
    ran           = _stub_inference(monkeypatch)
    worker        = InferenceWorker(config, "tag")
    unit          = worker.run_dir / "training" / "renamed/seed3"
    done          = unit / "inference" / "stamp_done"
    done.mkdir(parents=True)
    CompletionMarker.stamp(done, {"stage": "inference", "split": config.inference.split})

    worker.run("unet", seed=3, loss_component="param_l1", run_name="renamed/seed3")

    assert ran == [unit]
