"""Tests for the benchmark max-batch probe and its training-step memory measurement.

Covers the power-of-two batch candidate ladder, the memory probe driving the real trainer
loss and optimizer, the trainer built on the complete curriculum stage, dataset-size
capping of the search ceiling, channel validation against the dataset, and the loss union
used when building the probe context.
"""

from __future__ import annotations

from pathlib import Path
from types   import SimpleNamespace

import pytest
import torch

import pipelines.benchmark.batch_probe as module
from configuration.benchmark                 import BenchmarkConfig
from pipelines.backbone.dataset.pipeline     import DatasetPipeline
from pipelines.backbone.training.trainer     import Trainer
from pipelines.benchmark.batch_probe         import EntryMaxBatchProbe, MaxBatchProbe
from tools.monitoring.logger                 import Logger
from tools.training.pretraining.batch_finder import TrainStepMemoryProbe

from tests.backbone_training._helpers   import identity_normalizer, tiny_model, tiny_trainer_config, x_axis_numpy
from configuration.training.general.run import DEFAULT_PARAMS_RUN


GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="batch probe memory measurement requires CUDA")


class _RecordingOptimizer:
    """Optimizer stub that counts zero_grad and step calls and records gradient presence.

    Attributes:
        parameter    : The single parameter whose gradient is cleared and inspected.
        zero_grads   : Number of zero_grad calls received.
        steps        : Number of step calls received.
        grad_at_step : Whether the parameter had a gradient at each step call.
    """
    def __init__(self, parameter) -> None:
        """Stores the tracked parameter and initialises the call counters."""
        self.parameter    = parameter
        self.zero_grads   = 0
        self.steps        = 0
        self.grad_at_step = []

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Counts the call and clears the tracked parameter's gradient."""
        self.zero_grads     += 1
        self.parameter.grad  = None

    def step(self) -> None:
        """Counts the call and records whether the parameter carries a gradient."""
        self.steps += 1
        self.grad_at_step.append(self.parameter.grad is not None)


class _RecordingTrainer:
    """Trainer stub exposing the loss and optimizer surface the memory probe drives.

    Attributes:
        parameter : The single trainable parameter the stub loss depends on.
        optimizer : Recording optimizer over that parameter.
        use_amp   : Always False, so the probe takes the non-AMP path.
        batches   : Every batch tensor passed to the loss.
    """
    def __init__(self) -> None:
        """Builds the tracked parameter, the recording optimizer and the batch log."""
        self.parameter = torch.nn.Parameter(torch.ones(1))
        self.optimizer = _RecordingOptimizer(self.parameter)
        self.use_amp   = False
        self.batches   = []

    def _compute_loss(self, batch) -> dict:
        """Returns a differentiable scalar loss over the batch sum and records the batch.

        Args:
            batch : Batch tensor of shape (batch, features).

        Returns:
            Mapping with the scalar 'total_loss' and its detached copy under 'param_l1'.
        """
        self.batches.append(batch)
        total = (self.parameter * batch.sum()).squeeze()

        return {"total_loss": total, "param_l1": total.detach()}


class _StubDataset:
    """Dataset stub exposing only the length and input channel count the probe reads.

    Attributes:
        size           : Number of samples reported by len().
        input_channels : Channel count the probe validates against size_match.in_channels.
    """
    def __init__(self, size: int, input_channels: int) -> None:
        """Stores the reported sample count and input channel count."""
        self.size           = size
        self.input_channels = input_channels

    def __len__(self) -> int:
        """Returns the reported sample count."""
        return self.size


def _bare_probe(ceiling, model_name: str = "unet"):
    """Returns an uninitialised MaxBatchProbe carrying only a ceiling and a model name."""
    probe            = MaxBatchProbe.__new__(MaxBatchProbe)
    probe.ceiling    = ceiling
    probe.model_name = model_name
    return probe


def _curve_dataset(n_samples: int = 8):
    """Returns a list of n_samples constant tensors of shape (3,)."""
    return [torch.full((3,), float(index)) for index in range(n_samples)]


def _stub_cuda(monkeypatch, trainer, reserved_bytes: int) -> dict:
    """Replaces the CUDA memory calls with recorders and a fixed reserved-byte reading.

    Args:
        monkeypatch    : Pytest monkeypatch fixture.
        trainer        : Recording trainer whose batch count is sampled at the reset call.
        reserved_bytes : Value returned by torch.cuda.max_memory_reserved, in bytes.

    Returns:
        Mapping with the devices passed to the reset call and the loss-call count at reset.
    """
    calls = {"reset_devices": [], "loss_calls_at_reset": None}

    def reset_peak_memory_stats(device):
        calls["reset_devices"].append(device)
        calls["loss_calls_at_reset"] = len(trainer.batches)

    monkeypatch.setattr(torch.cuda, "empty_cache",             lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset_peak_memory_stats)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved",     lambda device: reserved_bytes)

    return calls


def _run_with_stubbed_context(monkeypatch, tmp_path, dataset, ceiling: int = 16, context_gb: float = 0.0, configure=None, trainer=None):
    """Runs a max-batch probe whose context, model, trainer and batch finder are stubbed.

    Args:
        monkeypatch : Pytest monkeypatch fixture.
        tmp_path    : Directory used for the probe work directory and logs.
        dataset     : Stub dataset returned as the training split.
        ceiling     : Batch-size search ceiling before dataset capping.
        context_gb  : Measured context memory in gigabytes.
        configure   : Optional callable mutating the benchmark config before the run.
        trainer     : Optional trainer stub handed out by the trainer builder.

    Returns:
        Tuple of the probe result dict and the keyword arguments seen by the batch finder.
    """
    config                    = BenchmarkConfig()
    config.paths.log_base_dir = tmp_path

    if configure is not None:
        configure(config)

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
        """Batch-size finder stub that captures its keyword arguments and reports a passing search."""
        def __init__(self, **kwargs) -> None:
            """Captures the finder's keyword arguments."""
            captured.update(kwargs)

        def run(self) -> dict:
            """Returns a fixed passing search result."""
            return {"status": "PASS", "batch_size": 4, "peak_gb": 1.0}

    monkeypatch.setattr(module, "BatchSizeFinder", _RecordingFinder)
    monkeypatch.setattr(probe, "_measure_context", lambda: context_gb)
    monkeypatch.setattr(probe, "_build_context",   lambda: (SimpleNamespace(), SimpleNamespace(), dataset, SimpleNamespace()))
    monkeypatch.setattr(probe, "_build_model",     lambda dataset_config, dataset_split, gaussian_cfg: (None, None))
    monkeypatch.setattr(probe, "_build_trainer",   lambda *args: trainer if trainer is not None else SimpleNamespace())

    return probe.run(), captured


def test_candidates_are_powers_of_two_up_to_ceiling():
    """Verifies the candidate ladder runs from one to a power-of-two ceiling."""
    probe = _bare_probe(ceiling=16)

    assert probe._candidates() == [1, 2, 4, 8, 16]


def test_candidates_stop_below_non_power_of_two_ceiling():
    """Verifies the candidate ladder stops at the last power of two below the ceiling."""
    probe = _bare_probe(ceiling=10)

    assert probe._candidates() == [1, 2, 4, 8]


def test_trial_drives_the_real_trainer_loss_and_optimizer(monkeypatch):
    """Verifies each measured step forms a full batch, zeroes gradients, backpropagates and steps."""
    trainer = _RecordingTrainer()
    _stub_cuda(monkeypatch, trainer, reserved_bytes=0)

    TrainStepMemoryProbe(trainer, _curve_dataset(), measure_steps=3, device=torch.device("cpu"))(batch_size=2)

    assert [tuple(batch.shape) for batch in trainer.batches] == [(2, 3)] * 3
    assert trainer.optimizer.zero_grads   == 3
    assert trainer.optimizer.steps        == 3
    assert trainer.optimizer.grad_at_step == [True, True, True]


def test_trial_reports_context_plus_peak_reserved_memory(monkeypatch):
    """Verifies the reported peak adds the context memory to the peak reserved bytes and resets before the first loss."""
    trainer = _RecordingTrainer()
    calls   = _stub_cuda(monkeypatch, trainer, reserved_bytes=2 * 1024 ** 3)

    peak_gb = TrainStepMemoryProbe(trainer, _curve_dataset(), measure_steps=2, device=torch.device("cpu"), context_gb=0.5)(batch_size=2)

    assert peak_gb == pytest.approx(2.5)
    assert calls["reset_devices"]       == [torch.device("cpu")]
    assert calls["loss_calls_at_reset"] == 0


def test_build_trainer_returns_a_real_trainer_on_the_complete_stage(tmp_path, monkeypatch):
    """Verifies the probe builds a real Trainer pinned to the complete curriculum stage in training mode."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    probe          = _bare_probe(ceiling=1)
    probe.work_dir = tmp_path / "work"
    probe.work_dir.mkdir(parents=True, exist_ok=True)
    probe.logger   = Logger(log_dir=str(tmp_path / "logs"), name="probe", level="ERROR")

    trainer_config   = tiny_trainer_config(n_gaussians=2)
    dataset_config   = SimpleNamespace(z_axis=x_axis_numpy())
    model, model_cfg = tiny_model(in_channels=2, n_gaussians=2)
    dataset          = SimpleNamespace(normalizer=identity_normalizer(6))

    trainer = probe._build_trainer(trainer_config, dataset_config, model, model_cfg, dataset)

    assert isinstance(trainer, Trainer)
    assert trainer.criterion.loss_cfg        is trainer_config.curriculum.complete
    assert trainer.criterion.loss_generation == 1
    assert trainer.model.training            is True

    probe.logger.close()


def test_probe_binds_the_production_dataset_and_trainer_classes():
    """Verifies the probe module imports the production dataset pipeline and trainer classes."""
    assert module.DatasetPipeline is DatasetPipeline
    assert module.Trainer         is Trainer


def test_run_catches_exceptions_and_reports_failure_dict(monkeypatch, tmp_path):
    """Verifies a failure during context measurement is reported as a FAIL result rather than raised."""
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
    """Verifies the probe initialises on CUDA with the configured budget, ceiling and work directory."""
    config                    = BenchmarkConfig()
    config.paths.log_base_dir = tmp_path

    probe = MaxBatchProbe(config=config, model_name="unet", overrides={}, work_dir=tmp_path / "run" / "max_batch" / "unet" / "probe")

    assert probe.device.type == "cuda"
    assert probe.budget_gb   == config.max_batch.vram_budget_gb
    assert probe.ceiling     == config.max_batch.max_batch
    assert probe.work_dir    == tmp_path / "run" / "max_batch" / "unet" / "probe"


def test_run_caps_the_ceiling_at_the_dataset_size(monkeypatch, tmp_path):
    """Verifies a dataset smaller than the ceiling lowers the batch-size search ceiling."""
    result, captured = _run_with_stubbed_context(monkeypatch, tmp_path, _StubDataset(size=5, input_channels=9), ceiling=16)

    assert result["error"]     is None
    assert result["status"]    == "PASS"
    assert captured["ceiling"] == 5


def test_run_keeps_the_ceiling_when_the_dataset_is_larger(monkeypatch, tmp_path):
    """Verifies the configured ceiling survives when the dataset is larger than it."""
    result, captured = _run_with_stubbed_context(monkeypatch, tmp_path, _StubDataset(size=64, input_channels=9), ceiling=16)

    assert result["status"]    == "PASS"
    assert captured["ceiling"] == 16


def test_run_runs_the_lr_range_test_at_the_probed_batch(monkeypatch, tmp_path):
    """Verifies the LR test sees batch-rescaled warmup-free base rates and its outcome lands in the record."""
    lr_captured = {}

    class _FakeRangeTest:
        """Range-test stub capturing its inputs and returning a fixed selection."""
        def __init__(self, **kwargs) -> None:
            """Captures the range-test keyword arguments."""
            lr_captured.update(kwargs)

        def run(self, batch_size):
            """Records the probed batch and the warmup-free base rates at run time."""
            lr_captured["batch_size"] = batch_size
            lr_captured["lrs"]        = list(lr_captured["trainer"].base_lrs)
            return {"multiplier": 0.5, "min_loss_multiplier": 5.0, "divergence_multiplier": 20.0, "rungs_run": 30, "diverged": True, "multipliers": [], "losses": [], "spreads": []}

    monkeypatch.setattr(module, "LrRangeTest", _FakeRangeTest)

    trainer = SimpleNamespace(optimizer=SimpleNamespace(param_groups=[{"lr": 0.064}]), base_lrs=[0.64])

    def configure(config):
        """Switches the LR range test on."""
        config.find_lr.enabled = True

    result, _captured = _run_with_stubbed_context(monkeypatch, tmp_path, _StubDataset(size=64, input_channels=9), ceiling=16, configure=configure, trainer=trainer)

    assert result["status"]                 == "PASS"
    assert result["lr_multiplier"]            == pytest.approx(0.5)
    assert result["lr_min_loss_multiplier"]   == pytest.approx(5.0)
    assert result["lr_divergence_multiplier"] == pytest.approx(20.0)
    assert result["lr_rungs_run"]             == 30
    assert lr_captured["batch_size"]          == 4
    assert lr_captured["lrs"]                 == [pytest.approx(0.64 * 4 / 64)]


def test_run_skips_the_lr_range_test_when_disabled(monkeypatch, tmp_path):
    """Verifies the default configuration never constructs the range test."""
    class _ForbiddenRangeTest:
        """Range-test stub that fails the test when instantiated."""
        def __init__(self, **kwargs) -> None:
            """Raises because the disabled probe must not build a range test."""
            raise AssertionError("LrRangeTest was constructed although find_lr is disabled")

    monkeypatch.setattr(module, "LrRangeTest", _ForbiddenRangeTest)

    result, _captured = _run_with_stubbed_context(monkeypatch, tmp_path, _StubDataset(size=64, input_channels=9), ceiling=16)

    assert result["status"]        == "PASS"
    assert result["lr_multiplier"] is None


def test_entry_probe_builds_through_the_pipeline_trainer_builder(tmp_path):
    """Verifies the entry probe drives the family builder and puts the model in train mode."""
    config                    = BenchmarkConfig()
    config.paths.log_base_dir = tmp_path

    calls = {}

    class _TrainableModel:
        """Model stub recording whether train mode was requested."""
        def __init__(self) -> None:
            """Starts outside train mode."""
            self.training = False

        def train(self) -> None:
            """Records the switch to train mode."""
            self.training = True

    model   = _TrainableModel()
    trainer = SimpleNamespace(model=model)
    dataset = _StubDataset(size=8, input_channels=9)

    def build(work_dir, logger):
        """Records the builder call and returns the stub triple."""
        calls["work_dir"] = Path(work_dir)
        calls["logger"]   = logger
        return trainer, dataset, model

    probe = EntryMaxBatchProbe(config=config, model_name="pae_mlp", work_dir=tmp_path / "probe", build_trainer=build)

    built_trainer, built_dataset = probe._build_probe_setup()

    assert built_trainer is trainer
    assert built_dataset is dataset
    assert model.training is True
    assert calls["work_dir"] == tmp_path / "probe" / "context"
    assert calls["logger"] is probe.logger


def test_run_validates_size_match_channels_against_the_dataset(monkeypatch, tmp_path):
    """Verifies a dataset channel count differing from size_match.in_channels fails before the search."""
    result, captured = _run_with_stubbed_context(monkeypatch, tmp_path, _StubDataset(size=64, input_channels=7))

    assert result["status"] == "FAIL"
    assert "size_match.in_channels" in result["error"]
    assert captured == {}


def test_run_records_the_measured_context(monkeypatch, tmp_path):
    """Verifies the measured context memory reaches both the result and the batch finder."""
    result, captured = _run_with_stubbed_context(monkeypatch, tmp_path, _StubDataset(size=64, input_channels=9), context_gb=0.75)

    assert result["context_gb"]   == 0.75
    assert captured["context_gb"] == 0.75


@pytest.mark.dataset
def test_build_context_probes_with_the_swept_loss_union(monkeypatch, tmp_path, test_data_dir):
    """Verifies the probe context enables exactly the swept loss components and requests the geometry field."""
    config                       = BenchmarkConfig()
    config.paths.log_base_dir    = tmp_path
    config.paths.dataset_path    = str(test_data_dir)
    config.paths.parameters_path = test_data_dir / "params" / DEFAULT_PARAMS_RUN / "parameters.npy"
    config.sweep_loss_components = ["param_l1", "moments"]

    probe            = _bare_probe(ceiling=1)
    probe.config     = config
    probe.model_name = "unet"
    probe.seed       = config.max_batch.seed
    probe.work_dir   = tmp_path / "work"
    probe.work_dir.mkdir(parents=True, exist_ok=True)
    probe.logger     = Logger(log_dir=str(tmp_path / "logs"), name="probe", level="ERROR")

    captured = {}

    class _RecordingPipeline:
        """Dataset pipeline stub that captures its keyword arguments and yields a named train split."""
        def __init__(self, **kwargs) -> None:
            """Captures the pipeline keyword arguments and exposes a 64-bin profile layout."""
            captured.update(kwargs)
            self.layout = SimpleNamespace(profile_length=64)

        def run(self):
            """Returns empty loaders and a split mapping whose train entry is a sentinel string."""
            return None, None, None, {"train": "train-split"}

    monkeypatch.setattr(module, "DatasetPipeline", _RecordingPipeline)

    trainer_config, dataset_config, dataset, gaussian_cfg = probe._build_context()

    assert dataset == "train-split"
    assert trainer_config.curriculum.complete.use_param_l1  is True
    assert trainer_config.curriculum.complete.use_moments   is True
    assert trainer_config.curriculum.complete.use_mse_curve is False
    assert len(dataset_config.z_axis) == 64
    assert dataset_config.n_gaussians == gaussian_cfg.n_default_gaussians

    probe.logger.close()


