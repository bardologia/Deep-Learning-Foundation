from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from configuration.training.general.runtime       import ResourceConfig, TrainingLoopConfig
from configuration.training.image_autoencoder     import ImageAeLossConfig, ImageAeTrainerConfig
from models.image_autoencoder                     import IMAGE_AE_CONFIG_REGISTRY, get_image_autoencoder
from pipelines.image_autoencoder.training.trainer import Trainer
from tools.monitoring.logger                      import Logger


pytestmark = pytest.mark.usefixtures("force_cpu")


IN_CHANNELS   = 2
EMBEDDING_DIM = 8
PATCH         = 16


def _tiny_model():
    cfg               = IMAGE_AE_CONFIG_REGISTRY["conv2d_ae"](in_channels=IN_CHANNELS, embedding_dim=EMBEDDING_DIM)
    cfg.base_channels = 8
    cfg.depth         = 1
    model, cfg        = get_image_autoencoder("conv2d_ae", cfg)
    return model, cfg


def _trainer_config(cfg, epochs=2):
    tc                         = ImageAeTrainerConfig(gaussian=None, image_autoencoder=cfg, ae_loss=ImageAeLossConfig(recon_kind="mse"))
    tc.training                = TrainingLoopConfig(epochs=epochs, validation_frequency=1, use_amp=False)
    tc.resources               = ResourceConfig(enabled=False)
    tc.early_stopping.patience = 1000
    tc.warmup.warmup_steps     = 0
    return tc


def _loader(n=8, batch_size=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, IN_CHANNELS, PATCH, PATCH, generator=g)
    return DataLoader(TensorDataset(x), batch_size=batch_size)


def _build_trainer(tmp_path, epochs=2):
    model, cfg = _tiny_model()
    tc         = _trainer_config(cfg, epochs=epochs)
    logger     = Logger(log_dir=str(tmp_path), name="image_ae_trainer_test")
    x_axis     = np.arange(PATCH, dtype=np.float32)
    return Trainer(model, cfg, x_axis, tc, str(tmp_path), logger), cfg


def test_trainer_builds_on_cpu(tmp_path):
    trainer, _ = _build_trainer(tmp_path)

    assert trainer.device.type == "cpu"
    assert trainer.criterion is not None
    assert len(trainer.optimizer.param_groups) == 2


def test_compute_loss_finite_and_structured(tmp_path):
    trainer, _ = _build_trainer(tmp_path)
    batch      = next(iter(_loader()))

    out = trainer._compute_loss(batch)

    assert torch.isfinite(out["total_loss"])
    assert "image_recon" in out["components"]


def test_single_train_epoch_updates_params(tmp_path):
    trainer, _ = _build_trainer(tmp_path, epochs=1)
    loader     = _loader()

    before = copy.deepcopy(trainer.model.state_dict())
    avg    = trainer.train_epoch(loader, epoch=0)
    after  = trainer.model.state_dict()

    assert np.isfinite(avg)

    changed = any(not torch.equal(before[k], after[k]) for k in before)
    assert changed


def test_fit_returns_finite_losses(tmp_path):
    trainer, _ = _build_trainer(tmp_path, epochs=2)
    loader     = _loader()

    train_losses, val_losses, best = trainer.train(loader, loader, loader)

    assert len(train_losses) == 2
    assert all(np.isfinite(l) for l in train_losses)
    assert np.isfinite(best)


def test_fit_reduces_or_stable_loss(tmp_path):
    trainer, _ = _build_trainer(tmp_path, epochs=3)
    loader     = _loader()

    train_losses, _, _ = trainer.train(loader, loader, loader)

    assert train_losses[-1] <= train_losses[0] + 1e-3


def test_checkpoint_saved_and_roundtrips(tmp_path):
    trainer, _ = _build_trainer(tmp_path, epochs=2)
    loader     = _loader()

    trainer.train(loader, loader, loader)

    assert trainer.checkpoint_path.is_file()

    ckpt = torch.load(trainer.checkpoint_path, map_location="cpu", weights_only=False)

    assert "params" in ckpt
    assert "x_axis" in ckpt
    assert isinstance(ckpt["x_axis"], np.ndarray)
    assert "best_val_loss" in ckpt

    fresh_model, fresh_cfg = _tiny_model()
    fresh_model.load_state_dict(ckpt["params"])

    for k, v in trainer.model.state_dict().items():
        assert torch.equal(v.cpu(), fresh_model.state_dict()[k])


def test_capture_state_contents(tmp_path):
    trainer, _ = _build_trainer(tmp_path)

    state = trainer.capture_state(epoch=3)

    assert state["epoch"] == 3
    assert set(state.keys()) >= {"epoch", "params", "x_axis"}
    np.testing.assert_allclose(state["x_axis"], np.arange(PATCH, dtype=np.float32))


def test_non_finite_loss_raises(tmp_path, monkeypatch):
    trainer, _ = _build_trainer(tmp_path, epochs=1)
    loader     = _loader()

    def _nan_loss(batch):
        return {"total_loss": torch.tensor(float("nan")), "components": {}, "monitor": {}, "occupancy": {}, "physical": {}}

    monkeypatch.setattr(trainer, "_compute_loss", _nan_loss)

    with pytest.raises(FloatingPointError):
        trainer.train_epoch(loader, epoch=0)
