
from __future__ import annotations


import pytest

from pipelines.benchmark.workers import BenchmarkWorker
from configuration.benchmark import BenchmarkConfig
from tools.data.io           import FileIO
from configuration.synthesis import SynthesisConfig


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


@pytest.mark.dataset
def test_training_worker_trains_with_the_configured_batch_size(config, test_data_dir, params_dir, monkeypatch):
    from pipelines.benchmark.workers import TrainingWorker

    config.paths.dataset_path    = test_data_dir
    config.paths.parameters_path = params_dir / "parameters.npy"
    config.training.train_rows   = (1000, 1400)
    config.training.val_rows     = (1400, 1700)
    config.training.test_rows    = (1700, 2000)
    config.training.batch_size   = 500

    worker = TrainingWorker(config, "tag")

    captured = {}

    class FakePipeline:
        def __init__(self, **kwargs):
            captured.update(kwargs)
        def run(self):
            return None

    monkeypatch.setattr("pipelines.backbone.training.pipeline.TrainingPipeline", FakePipeline)

    worker.run("resunet-conv")

    assert captured["trainer_config"].optimizer.lr_scale == pytest.approx(500 / config.training.lr_reference_batch_size)
    assert captured["dataset_config"].batch_size         == 500


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
