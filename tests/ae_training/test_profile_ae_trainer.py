from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from configuration.training.general.runtime         import ResourceConfig, TrainingLoopConfig
from configuration.training.profile_autoencoder     import ProfileAeLossConfig, ProfileAeTrainerConfig
from models.profile_autoencoder                     import PROFILE_AE_CONFIG_REGISTRY, get_profile_autoencoder
from pipelines.profile_autoencoder.training.trainer import Trainer
from tools.monitoring.logger                        import Logger


pytestmark = pytest.mark.usefixtures("force_cpu")


PROFILE_LENGTH = 16
EMBEDDING_DIM  = 8


def _tiny_model():
    cfg            = PROFILE_AE_CONFIG_REGISTRY["mlp_ae"](profile_length=PROFILE_LENGTH, embedding_dim=EMBEDDING_DIM)
    cfg.hidden_dim = 16
    cfg.depth      = 2
    model, cfg     = get_profile_autoencoder("mlp_ae", cfg)
    return model, cfg


def _trainer_config(cfg, epochs=2):
    tc                         = ProfileAeTrainerConfig(gaussian=None, autoencoder=cfg, ae_loss=ProfileAeLossConfig(curve_kind="mse"))
    tc.training                = TrainingLoopConfig(epochs=epochs, validation_frequency=1, use_amp=False)
    tc.resources               = ResourceConfig(enabled=False)
    tc.early_stopping.patience = 1000
    tc.warmup.warmup_steps     = 0
    return tc


def _loader(n=8, batch_size=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, PROFILE_LENGTH, generator=g)
    return DataLoader(x, batch_size=batch_size)


def _build_trainer(tmp_path, epochs=2):
    model, cfg = _tiny_model()
    tc         = _trainer_config(cfg, epochs=epochs)
    logger     = Logger(log_dir=str(tmp_path), name="profile_ae_trainer_test")
    x_axis     = np.arange(PROFILE_LENGTH, dtype=np.float32)
    return Trainer(model, cfg, x_axis, tc, str(tmp_path), logger), cfg


def test_trainer_builds_on_cpu(tmp_path):
    trainer, _ = _build_trainer(tmp_path)

    assert trainer.device.type == "cpu"
    assert trainer.criterion is not None
    assert len(trainer.optimizer.param_groups) == 2


def test_compute_loss_reshapes_and_finite(tmp_path):
    trainer, _ = _build_trainer(tmp_path)
    batch      = next(iter(_loader()))

    out = trainer._compute_loss(batch)

    assert torch.isfinite(out["total_loss"])
    assert "curve_recon" in out["components"]


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
    assert isinstance(ckpt["x_axis"], np.ndarray)
    assert "best_val_loss" in ckpt

    fresh_model, _ = _tiny_model()
    fresh_model.load_state_dict(ckpt["params"])

    for k, v in trainer.model.state_dict().items():
        assert torch.equal(v.cpu(), fresh_model.state_dict()[k])


def test_capture_state_contents(tmp_path):
    trainer, _ = _build_trainer(tmp_path)

    state = trainer.capture_state(epoch=2)

    assert state["epoch"] == 2
    np.testing.assert_allclose(state["x_axis"], np.arange(PROFILE_LENGTH, dtype=np.float32))


def test_non_finite_loss_raises(tmp_path, monkeypatch):
    trainer, _ = _build_trainer(tmp_path, epochs=1)
    loader     = _loader()

    def _nan_loss(batch):
        return {"total_loss": torch.tensor(float("inf")), "components": {}, "monitor": {}, "occupancy": {}, "physical": {}}

    monkeypatch.setattr(trainer, "_compute_loss", _nan_loss)

    with pytest.raises(FloatingPointError):
        trainer.train_epoch(loader, epoch=0)
