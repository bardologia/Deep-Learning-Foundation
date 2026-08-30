"""Tests for the data loader throughput tuner used before training.

Covers the prefetch grid it explores and the worker/prefetch choice it
returns on a CPU device.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import Dataset

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
    """Logger stub swallowing the section calls made by the tuner."""
    def section(self, *args, **kwargs):
        """Ignores a section heading."""
        pass

    def subsection(self, *args, **kwargs):
        """Ignores a subsection heading."""
        pass

    def warning(self, *args, **kwargs):
        """Ignores a warning, emitted when NVML is unavailable on the test machine."""
        pass


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
