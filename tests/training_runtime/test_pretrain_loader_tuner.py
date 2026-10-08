"""Tests for the data loader throughput tuner used before training.

Covers the prefetch grid it explores, the worker/prefetch choice it returns
on a CPU device, and the per-configuration logging and records of both
sweeps, out-of-memory configurations included.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import tools.training.pretraining.loader_tuner as tuner_module
from tools.training.pretraining.loader_tuner import LoaderTuner


class _Vectors(Dataset):
    """Dataset of zero vectors used as a cheap loader payload."""
    def __init__(self, n_samples: int, dim: int) -> None:
        """Stores the sample count and vector dimension."""
        self.n_samples = n_samples
        self.dim       = dim

    def __len__(self) -> int:
        """Returns the number of samples."""
        return self.n_samples

    def __getitem__(self, index: int):
        """Returns a zero vector of shape (dim,)."""
        return torch.zeros(self.dim)


class _Model(nn.Module):
    """Single linear layer standing in for a trainable model."""
    def __init__(self, dim: int) -> None:
        """Builds a dim-to-dim linear layer."""
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        """Returns the linear projection of `x` of shape (batch, dim)."""
        return self.linear(x)


class _Logger:
    """Logger stub recording the subsection and warning lines of the tuner.

    Attributes:
        lines    : Subsection messages in emission order.
        warnings : Warning messages in emission order, NVML-unavailable notices included.
    """
    def __init__(self):
        """Starts with no recorded output."""
        self.lines    = []
        self.warnings = []

    def section(self, *args, **kwargs):
        """Ignores a section heading."""
        pass

    def subsection(self, message, *args, **kwargs):
        """Records a subsection line."""
        self.lines.append(message)

    def warning(self, message, *args, **kwargs):
        """Records a warning line."""
        self.warnings.append(message)


def _forward_loss(model, batch):
    """Returns the mean squared activation of the model output as a scalar loss."""
    prediction = model(batch)
    return (prediction ** 2).mean()


def test_refine_grid_pins_memory_and_covers_each_prefetch_once():
    """Verifies that each prefetch factor is probed once with pinned memory."""
    dim = 8

    tuner = LoaderTuner(
        dataset          = _Vectors(64, dim),
        model            = _Model(dim),
        to_model_input   = lambda batch, device: batch.to(device),
        forward_loss     = _forward_loss,
        device           = torch.device("cpu"),
        logger           = _Logger(),
        prefetch_factors = (2, 4, 8),
    )

    specs = tuner._prefetch_specs(batch_size=8, num_workers=2)

    assert [spec.prefetch_factor for spec in specs] == [2, 4, 8]
    assert all(spec.pin_memory for spec in specs)


def test_loader_tuner_returns_two_field_config_on_cpu():
    """Verifies that the tuner returns only num_workers and prefetch_factor."""
    dim = 8

    tuner = LoaderTuner(
        dataset          = _Vectors(2000, dim),
        model            = _Model(dim),
        to_model_input   = lambda batch, device: batch.to(device),
        forward_loss     = _forward_loss,
        device           = torch.device("cpu"),
        logger           = _Logger(),
        worker_counts    = (0,),
        prefetch_factors = (2,),
        warmup_batches   = 2,
        timed_batches    = 4,
    )

    choice = tuner.run(batch_size=32)

    assert choice is not None
    assert set(choice)           == {"num_workers", "prefetch_factor"}
    assert choice["num_workers"] == 0
    assert choice["prefetch_factor"] > 0


def _tuner(logger, **overrides) -> LoaderTuner:
    """Returns a CPU loader tuner over cheap vectors with a one-point grid unless overridden."""
    dim      = 8
    settings = dict(
        dataset          = _Vectors(2000, dim),
        model            = _Model(dim),
        to_model_input   = lambda batch, device: batch.to(device),
        forward_loss     = _forward_loss,
        device           = torch.device("cpu"),
        logger           = logger,
        worker_counts    = (0,),
        prefetch_factors = (2,),
        warmup_batches   = 2,
        timed_batches    = 4,
    )
    settings.update(overrides)
    return LoaderTuner(**settings)


def test_every_measured_configuration_is_logged_and_kept():
    """Verifies each configuration of both sweeps is logged with its throughput and kept, and the final line carries the chosen one's measurements."""
    logger = _Logger()
    tuner  = _tuner(logger)

    tuner.run(batch_size=32)

    assert [record["phase"] for record in tuner.records] == ["workers", "prefetch"]
    assert all(record["status"] == "ok" for record in tuner.records)
    assert sum("samples/s end to end" in line for line in logger.lines) == 3
    assert "feed ratio" in logger.lines[-1] and logger.lines[-1].startswith("Loader tuner -> workers=0")


def test_out_of_memory_configurations_and_an_empty_sweep_are_warnings(monkeypatch):
    """Verifies an out-of-memory configuration is a warning naming it, and a sweep without any success warns that the settings are kept."""
    class _OomSweep:
        """Sweep stub reporting every configuration out of memory through the callback."""
        def __init__(self, benchmark, specs, on_result=None):
            """Stores the specs and the callback."""
            self.specs     = specs
            self.on_result = on_result

        def run(self):
            """Reports one 'oom' record per spec."""
            records = [{**spec.as_record(), "status": "oom"} for spec in self.specs]
            for record in records:
                self.on_result(record)
            return records

    monkeypatch.setattr(tuner_module, "DataLoaderSweep", _OomSweep)

    logger = _Logger()
    tuner  = _tuner(logger, worker_counts=(0, 1))

    assert tuner.run(batch_size=32) is None
    assert [record["status"] for record in tuner.records] == ["oom", "oom"]
    assert any("workers=1" in line and "ran out of memory" in line for line in logger.warnings)
    assert "found no successful configuration among 2 measured" in logger.warnings[-1]
