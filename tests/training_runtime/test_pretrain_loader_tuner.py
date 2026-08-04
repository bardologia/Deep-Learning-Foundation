from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from tools.training.pretraining.loader_tuner import LoaderTuner


class _Vectors(Dataset):
    def __init__(self, n_samples: int, dim: int) -> None:
        self.n_samples = n_samples
        self.dim       = dim

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int):
        return torch.zeros(self.dim)


class _Model(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x)


class _Logger:
    def section(self, *args, **kwargs):
        pass

    def subsection(self, *args, **kwargs):
        pass


def _forward_loss(model, batch):
    prediction = model(batch)
    return (prediction ** 2).mean()


def test_loader_tuner_returns_three_field_config_on_cpu():
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

    if choice is not None:
        assert set(choice)        == {"num_workers", "prefetch_factor", "pin_memory"}
        assert choice["num_workers"] == 0
