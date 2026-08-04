from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import pipelines.benchmark.batch_probe as module
from configuration.benchmark                 import BenchmarkConfig
from pipelines.backbone.dataset.pipeline     import DatasetPipeline
from pipelines.backbone.training.loss_terms  import LossComponentCatalog
from pipelines.backbone.training.pipeline    import TrainingPipeline
from pipelines.backbone.training.trainer     import Trainer
from pipelines.benchmark.batch_probe         import MaxBatchProbe
from tools.monitoring.logger                 import Logger
from tools.training.pretraining.batch_finder import TrainStepMemoryProbe

from tests.backbone_training._helpers import identity_normalizer, tiny_model, tiny_trainer_config, x_axis_numpy


GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="batch probe memory measurement requires CUDA")


class _RecordingOptimizer:
    def __init__(self, parameter) -> None:
        self.parameter    = parameter
        self.zero_grads   = 0
        self.steps        = 0
        self.grad_at_step = []

    def zero_grad(self, set_to_none: bool = False) -> None:
        self.zero_grads     += 1
        self.parameter.grad  = None

    def step(self) -> None:
        self.steps += 1
        self.grad_at_step.append(self.parameter.grad is not None)


class _RecordingTrainer:
    def __init__(self) -> None:
        self.parameter = torch.nn.Parameter(torch.ones(1))
        self.optimizer = _RecordingOptimizer(self.parameter)
        self.use_amp   = False
        self.batches   = []

    def _compute_loss(self, batch) -> dict:
        self.batches.append(batch)
        total = (self.parameter * batch.sum()).squeeze()

        return {"total_loss": total, "param_l1": total.detach()}


class _StubDataset:
    def __init__(self, size: int, input_channels: int) -> None:
        self.size           = size
        self.input_channels = input_channels

    def __len__(self) -> int:
        return self.size


def _bare_probe(ceiling):
    probe         = MaxBatchProbe.__new__(MaxBatchProbe)
    probe.ceiling = ceiling
    return probe


def _curve_dataset(n_samples: int = 8):
    return [torch.full((3,), float(index)) for index in range(n_samples)]


def _stub_cuda(monkeypatch, trainer, reserved_bytes: int) -> dict:
    calls = {"reset_devices": [], "loss_calls_at_reset": None}

    def reset_peak_memory_stats(device):
        calls["reset_devices"].append(device)
        calls["loss_calls_at_reset"] = len(trainer.batches)

    monkeypatch.setattr(torch.cuda, "empty_cache",             lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset_peak_memory_stats)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved",     lambda device: reserved_bytes)

    return calls


def _run_with_stubbed_context(monkeypatch, tmp_path, dataset, ceiling: int = 16, context_gb: float = 0.0):
    config                    = BenchmarkConfig()
    config.paths.log_base_dir = tmp_path

    probe               = _bare_probe(ceiling=ceiling)
    probe.config        = config
    probe.model_name    = "unet"
    probe.overrides     = {}
    probe.budget_gb     = config.max_batch.vram_budget_gb
    probe.measure_steps = config.max_batch.measure_steps
    probe.seed          = config.max_batch.seed
    probe.device        = torch.device("cpu")
    probe.context_gb    = 0.0
    probe.work_dir      = tmp_path / "work"
    probe.work_dir.mkdir(parents=True, exist_ok=True)
    probe.logger        = Logger(log_dir=str(tmp_path / "logs"), name="probe", level="ERROR")

    captured = {}

    class _RecordingFinder:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        def run(self) -> dict:
            return {"status": "PASS", "batch_size": 4, "peak_gb": 1.0}

    monkeypatch.setattr(module, "BatchSizeFinder", _RecordingFinder)
    monkeypatch.setattr(probe, "_measure_context", lambda: context_gb)
    monkeypatch.setattr(probe, "_build_context",   lambda: (SimpleNamespace(), SimpleNamespace(), dataset, SimpleNamespace()))
    monkeypatch.setattr(probe, "_build_model",     lambda dataset_config, dataset_split, gaussian_cfg: (None, None))
    monkeypatch.setattr(probe, "_build_trainer",   lambda *args: SimpleNamespace())

    return probe.run(), captured


def test_candidates_are_powers_of_two_up_to_ceiling():
    probe = _bare_probe(ceiling=16)

    assert probe._candidates() == [1, 2, 4, 8, 16]


def test_candidates_stop_below_non_power_of_two_ceiling():
    probe = _bare_probe(ceiling=10)

    assert probe._candidates() == [1, 2, 4, 8]


def test_trial_drives_the_real_trainer_loss_and_optimizer(monkeypatch):
    trainer = _RecordingTrainer()
    _stub_cuda(monkeypatch, trainer, reserved_bytes=0)

    TrainStepMemoryProbe(trainer, _curve_dataset(), measure_steps=3, device=torch.device("cpu"))(batch_size=2)

    assert [tuple(batch.shape) for batch in trainer.batches] == [(2, 3)] * 3
    assert trainer.optimizer.zero_grads   == 3
    assert trainer.optimizer.steps        == 3
    assert trainer.optimizer.grad_at_step == [True, True, True]


def test_trial_reports_context_plus_peak_reserved_memory(monkeypatch):
    trainer = _RecordingTrainer()
    calls   = _stub_cuda(monkeypatch, trainer, reserved_bytes=2 * 1024 ** 3)

    peak_gb = TrainStepMemoryProbe(trainer, _curve_dataset(), measure_steps=2, device=torch.device("cpu"), context_gb=0.5)(batch_size=2)

    assert peak_gb == pytest.approx(2.5)
    assert calls["reset_devices"]       == [torch.device("cpu")]
    assert calls["loss_calls_at_reset"] == 0


def test_build_trainer_returns_a_real_trainer_on_the_complete_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    probe          = _bare_probe(ceiling=1)
    probe.work_dir = tmp_path / "work"
    probe.work_dir.mkdir(parents=True, exist_ok=True)
    probe.logger   = Logger(log_dir=str(tmp_path / "logs"), name="probe", level="ERROR")

    trainer_config   = tiny_trainer_config(n_gaussians=2)
    dataset_config   = SimpleNamespace(x_axis=x_axis_numpy())
    model, model_cfg = tiny_model(in_channels=2, n_gaussians=2)
    dataset          = SimpleNamespace(normalizer=identity_normalizer(6))

    trainer = probe._build_trainer(trainer_config, dataset_config, model, model_cfg, dataset)

    assert isinstance(trainer, Trainer)
    assert trainer.criterion.loss_cfg        is trainer_config.curriculum.complete
    assert trainer.criterion.loss_generation == 1
    assert trainer.model.training            is True

    probe.logger.close()


def test_probe_binds_the_production_dataset_and_trainer_classes():
    assert module.DatasetPipeline is DatasetPipeline
    assert module.Trainer         is Trainer


def test_run_catches_exceptions_and_reports_failure_dict(monkeypatch, tmp_path):
    config                    = BenchmarkConfig()
    config.paths.log_base_dir = tmp_path

    probe = _bare_probe(ceiling=config.max_batch.max_batch)
    probe.config        = config
    probe.model_name    = "unet"
    probe.overrides     = {}
    probe.budget_gb     = config.max_batch.vram_budget_gb
    probe.measure_steps = config.max_batch.measure_steps
    probe.seed          = config.max_batch.seed
    probe.device        = torch.device("cpu")
    probe.context_gb    = 0.0
    probe.work_dir      = tmp_path / "work"
    probe.work_dir.mkdir(parents=True, exist_ok=True)

    probe.logger = Logger(log_dir=str(tmp_path / "logs"), name="probe", level="INFO")

    monkeypatch.setattr(probe, "_measure_context", lambda: (_ for _ in ()).throw(RuntimeError("no cuda")))

    result = probe.run()

    assert result["status"] == "FAIL"
    assert result["model"]  == "unet"
    assert result["error"] is not None


@GPU
def test_probe_init_requires_cuda_device(tmp_path):
    config                    = BenchmarkConfig()
    config.paths.log_base_dir = tmp_path

    probe = MaxBatchProbe(config=config, model_name="unet", overrides={}, work_dir=tmp_path / "run" / "max_batch" / "unet" / "probe")

    assert probe.device.type == "cuda"
    assert probe.budget_gb   == config.max_batch.vram_budget_gb
    assert probe.ceiling     == config.max_batch.max_batch
    assert probe.work_dir    == tmp_path / "run" / "max_batch" / "unet" / "probe"


def test_run_caps_the_ceiling_at_the_dataset_size(monkeypatch, tmp_path):
    result, captured = _run_with_stubbed_context(monkeypatch, tmp_path, _StubDataset(size=5, input_channels=9), ceiling=16)

    assert result["error"]     is None
    assert result["status"]    == "PASS"
    assert captured["ceiling"] == 5


def test_run_keeps_the_ceiling_when_the_dataset_is_larger(monkeypatch, tmp_path):
    result, captured = _run_with_stubbed_context(monkeypatch, tmp_path, _StubDataset(size=64, input_channels=9), ceiling=16)

    assert result["status"]    == "PASS"
    assert captured["ceiling"] == 16


def test_run_validates_size_match_channels_against_the_dataset(monkeypatch, tmp_path):
    result, captured = _run_with_stubbed_context(monkeypatch, tmp_path, _StubDataset(size=64, input_channels=7))

    assert result["status"] == "FAIL"
    assert "size_match.in_channels" in result["error"]
    assert captured == {}


def test_run_records_the_measured_context(monkeypatch, tmp_path):
    result, captured = _run_with_stubbed_context(monkeypatch, tmp_path, _StubDataset(size=64, input_channels=9), context_gb=0.75)

    assert result["context_gb"]   == 0.75
    assert captured["context_gb"] == 0.75


@pytest.mark.real_data
def test_build_context_probes_with_the_swept_loss_union(monkeypatch, tmp_path, test_data_dir):
    config                       = BenchmarkConfig()
    config.paths.log_base_dir    = tmp_path
    config.paths.dataset_path    = str(test_data_dir)
    config.paths.parameters_path = test_data_dir / "params" / "params_k5_lam0.01_sig4_sigma" / "parameters.npy"
    config.sweep_loss_components = ["param_l1", "covariance_match"]

    probe          = _bare_probe(ceiling=1)
    probe.config   = config
    probe.seed     = config.max_batch.seed
    probe.work_dir = tmp_path / "work"
    probe.work_dir.mkdir(parents=True, exist_ok=True)
    probe.logger   = Logger(log_dir=str(tmp_path / "logs"), name="probe", level="ERROR")

    captured = {}

    class _RecordingPipeline:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)
            self.layout = SimpleNamespace(profile_length=64)

        def run(self):
            return None, None, None, {"train": "train-split"}

    monkeypatch.setattr(module, "DatasetPipeline", _RecordingPipeline)

    trainer_config, dataset_config, dataset, gaussian_cfg = probe._build_context()

    assert dataset == "train-split"
    assert trainer_config.curriculum.complete.use_param_l1         is True
    assert trainer_config.curriculum.complete.use_covariance_match is True
    assert trainer_config.curriculum.complete.use_mse_curve        is False
    assert captured["build_geometry_field"] is True
    assert len(dataset_config.x_axis)       == 64
    assert dataset_config.n_gaussians       == gaussian_cfg.n_default_gaussians

    probe.logger.close()


def test_probe_curriculum_activates_geometry_field_for_physics_sweeps():
    config = BenchmarkConfig()

    config.sweep_loss_components = ["param_l1"]
    plain = LossComponentCatalog.combined_curriculum(config.sweep_loss_components, base=config.loss)
    stub  = type("Cfg", (), {"curriculum": plain})()
    assert TrainingPipeline.physics_geometry_active(stub) is False

    config.sweep_loss_components = ["param_l1", "covariance_match"]
    physics = LossComponentCatalog.combined_curriculum(config.sweep_loss_components, base=config.loss)
    stub    = type("Cfg", (), {"curriculum": physics})()
    assert TrainingPipeline.physics_geometry_active(stub) is True
