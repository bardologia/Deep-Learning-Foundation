from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from configuration.training                  import OverfitCheckConfig, ProfileAeTrainerConfig
from pipelines.shared.training.overfit_check import OverfitCheck

from tests.conftest import SilentLogger


class _StubTrainer:
    def __init__(self, losses):
        self.losses  = losses
        self.loaders = None

    def train(self, train_loader, val_loader, test_loader):
        self.loaders = (train_loader, val_loader, test_loader)
        torch.rand(3)
        return self.losses, [], min(self.losses)


class _TupleDataset(torch.utils.data.Dataset):
    def __init__(self, n=10):
        self.n                      = n
        self.augmenter              = object()
        self.fetched_with_augmenter = []

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        self.fetched_with_augmenter.append(self.augmenter is not None)
        return torch.full((2, 4, 4), float(idx)), torch.full((3,), float(idx))


class _TensorDataset(torch.utils.data.Dataset):
    def __init__(self, n=10):
        self.n         = n
        self.augmenter = object()

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return torch.full((16,), float(idx))


@dataclass
class _ArchConfig:
    embedding_dim     : int   = 8
    dropout           : float = 0.3
    attention_dropout : float = 0.1
    encoder_wd        : float = 0.05
    decoder_wd        : float = 0.05


def _check(tmp_path, **overrides):
    config = OverfitCheckConfig(enabled=True, **overrides)
    return OverfitCheck(config, tmp_path, SilentLogger())


def test_disabled_check_is_inert(tmp_path):
    check = OverfitCheck(OverfitCheckConfig(), tmp_path, SilentLogger())

    assert check.enabled is False
    assert check.rng     is None


def test_epoch_steps_and_planned_epochs(tmp_path):
    check = _check(tmp_path, max_steps=300, steps_per_epoch=25)
    assert check.epoch_steps    == 25
    assert check.planned_epochs == 12

    check = _check(tmp_path, max_steps=10, steps_per_epoch=25)
    assert check.epoch_steps    == 10
    assert check.planned_epochs == 1


def test_sanitized_trainer_config_disables_regularization(tmp_path):
    check    = _check(tmp_path, n_examples=4, max_steps=100, steps_per_epoch=20, stop_threshold=1e-5)
    original = ProfileAeTrainerConfig(gaussian=None)

    original.optimizer.weight_decay = 0.1
    original.training.use_ema       = True

    cfg = check.sanitized_trainer_config(original)

    assert cfg.training.epochs               == 5
    assert cfg.training.validation_frequency == 5
    assert cfg.training.use_ema              is False
    assert cfg.training.resume               is False

    assert cfg.optimizer.weight_decay      == 0.0
    assert cfg.warmup.warmup_enabled       is False
    assert cfg.scheduler.type              == "constant"
    assert cfg.early_stopping.restore_best is False
    assert cfg.resources.enabled           is False
    assert cfg.io.logdir                   == str(check.work_directory)

    assert original.optimizer.weight_decay == 0.1
    assert original.training.use_ema       is True

    assert check.overrides["optimizer.weight_decay"] == 0.0
    assert check.overrides["training.use_ema"]       is False


def test_sanitized_model_config_zeroes_dropout_and_weight_decay(tmp_path):
    check    = _check(tmp_path)
    original = _ArchConfig()

    cfg = check.sanitized_model_config(original)

    assert cfg.dropout           == 0.0
    assert cfg.attention_dropout == 0.0
    assert cfg.encoder_wd        == 0.0
    assert cfg.decoder_wd        == 0.0
    assert cfg.embedding_dim     == 8

    assert original.dropout    == 0.3
    assert original.encoder_wd == 0.05

    assert check.overrides["model.dropout"]    == 0.0
    assert check.overrides["model.encoder_wd"] == 0.0


def test_gate_batch_disables_augmentation_and_restores_it(tmp_path):
    check   = _check(tmp_path, n_examples=2)
    dataset = _TupleDataset(n=10)

    batch, indices = check._gate_batch(dataset)

    assert dataset.fetched_with_augmenter == [False, False]
    assert dataset.augmenter is not None
    assert indices == [0, 5]

    assert batch[0].shape == (2, 2, 4, 4)
    assert batch[1].shape == (2, 3)
    assert check.overrides["augmentation"] == "disabled"


def test_gate_batch_collates_tensor_items(tmp_path):
    check   = _check(tmp_path, n_examples=3)
    dataset = _TensorDataset(n=9)

    batch, indices = check._gate_batch(dataset)

    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (3, 16)
    assert indices     == [0, 3, 6]


def test_gate_batch_caps_examples_at_dataset_size(tmp_path):
    check      = _check(tmp_path, n_examples=8)
    batch, idx = check._gate_batch(_TensorDataset(n=3))

    assert batch.shape == (3, 16)
    assert idx         == [0, 1, 2]


def test_run_passing_writes_report_and_cleans_work_directory(tmp_path):
    check   = _check(tmp_path, n_examples=2, max_steps=50, steps_per_epoch=10, pass_loss_ratio=0.05)
    trainer = _StubTrainer(losses=[1.0, 0.1, 0.001])

    check.work_directory.mkdir(parents=True)
    (check.work_directory / "best_model.pt").write_bytes(b"x")
    (check.work_directory / "last.pt").write_bytes(b"x")
    (check.work_directory / "complete.json").write_text("{}")

    verdict = check.run(trainer, _TensorDataset(n=6))

    assert verdict["passed"]     is True
    assert verdict["best_loss"]  == pytest.approx(0.001)
    assert verdict["loss_ratio"] == pytest.approx(0.001)

    assert len(trainer.loaders[0]) == 10
    assert trainer.loaders[0][0].shape == (2, 16)

    report = json.loads(check.report_path.read_text())
    assert report["passed"]              is True
    assert report["epochs_run"]          == 3
    assert report["steps_run"]           == 30
    assert report["epoch_losses"]        == [1.0, 0.1, 0.001]
    assert report["sanitized_overrides"] == check.overrides

    assert not check.work_directory.exists()


def test_run_failing_raises_and_still_writes_report(tmp_path):
    check   = _check(tmp_path, max_steps=50, steps_per_epoch=10, pass_loss_ratio=0.05)
    trainer = _StubTrainer(losses=[1.0, 0.9, 0.8])

    with pytest.raises(RuntimeError, match="Overfit check failed"):
        check.run(trainer, _TensorDataset(n=6))

    report = json.loads(check.report_path.read_text())
    assert report["passed"]     is False
    assert report["loss_ratio"] == pytest.approx(0.8)


def test_run_passes_on_absolute_threshold_even_with_high_ratio(tmp_path):
    check   = _check(tmp_path, pass_loss_ratio=0.0001, stop_threshold=1e-3)
    trainer = _StubTrainer(losses=[1e-4, 5e-5])

    verdict = check.run(trainer, _TensorDataset(n=6))

    assert verdict["passed"] is True


def test_run_restores_global_rng_state(tmp_path):
    torch.manual_seed(1234)
    np.random.seed(1234)

    check          = _check(tmp_path)
    expected_torch = torch.get_rng_state().clone()
    expected_numpy = np.random.get_state()[1].copy()

    check.run(_StubTrainer(losses=[1.0, 0.001]), _TensorDataset(n=6))

    assert torch.equal(torch.get_rng_state(), expected_torch)
    assert np.array_equal(np.random.get_state()[1], expected_numpy)
