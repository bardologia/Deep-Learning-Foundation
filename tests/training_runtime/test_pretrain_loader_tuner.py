
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import tools.training.pretraining.loader_tuner as tuner_module
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
    def __init__(self):
        self.lines    = []
        self.warnings = []

    def section(self, *args, **kwargs):
        pass

    def subsection(self, message, *args, **kwargs):
        self.lines.append(message)

    def warning(self, message, *args, **kwargs):
        self.warnings.append(message)


def _forward_loss(model, batch):
    prediction = model(batch)
    return (prediction ** 2).mean()


def test_refine_grid_pins_memory_and_covers_each_prefetch_once():
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
    logger = _Logger()
    tuner  = _tuner(logger)

    tuner.run(batch_size=32)

    assert [record["phase"] for record in tuner.records] == ["workers", "prefetch"]
    assert all(record["status"] == "ok" for record in tuner.records)
    assert sum("samples/s end to end" in line for line in logger.lines) == 3
    assert "feed ratio" in logger.lines[-1] and logger.lines[-1].startswith("Loader tuner -> workers=0")


def test_out_of_memory_configurations_and_an_empty_sweep_are_warnings(monkeypatch):
    class _OomSweep:
        def __init__(self, benchmark, specs, on_result=None):
            self.specs     = specs
            self.on_result = on_result

        def run(self):
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
