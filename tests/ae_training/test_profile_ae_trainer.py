"""Tests for the profile autoencoder trainer.

Covers CPU construction and the three optimizer parameter groups, that the reconstruction
target is the clean curve rather than the noisy input, the extra components contributed by
the latent-noise and Sobolev terms, the checkpoint round trip, and the non-finite loss guard.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from configuration.training.general.runtime         import ResourceConfig, TrainingLoopConfig
from configuration.training.profile_autoencoder     import ProfileAeLossConfig, ProfileAeTrainerConfig
from models.profile_autoencoder                     import PROFILE_AE_CONFIG_REGISTRY, get_profile_autoencoder
from pipelines.profile_autoencoder.training.trainer import Trainer
from tools.monitoring.logger                        import Logger


pytestmark = pytest.mark.usefixtures("force_cpu")


PROFILE_LENGTH = 16
EMBEDDING_DIM  = 8


def _tiny_model():
    """Returns a tiny MLP profile autoencoder and its config: 16 hidden units at depth two."""
    cfg            = PROFILE_AE_CONFIG_REGISTRY["mlp_ae"](profile_length=PROFILE_LENGTH, embedding_dim=EMBEDDING_DIM)
    cfg.hidden_dim = 16
    cfg.depth      = 2
    model, cfg     = get_profile_autoencoder("mlp_ae", cfg)
    return model, cfg


def _trainer_config(cfg, epochs=2, ae_loss=None):
    """Returns a CPU trainer config over the given epoch budget, with early stopping and warmup disabled."""
    tc                         = ProfileAeTrainerConfig(gaussian=None, autoencoder=cfg, ae_loss=ae_loss or ProfileAeLossConfig())
    tc.training                = TrainingLoopConfig(epochs=epochs, validation_frequency=1, use_amp=False)
    tc.resources               = ResourceConfig(enabled=False)
    tc.early_stopping.patience = 1000
    tc.warmup.warmup_fraction  = 0.0
    return tc


def _loader(n=8, batch_size=4, seed=0):
    """Returns a loader over random curves of shape (n, 16), yielded as a (noisy, clean) pair."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, PROFILE_LENGTH, generator=g)
    return DataLoader(TensorDataset(x, x), batch_size=batch_size)


def _build_trainer(tmp_path, epochs=2, ae_loss=None):
    """Returns a trainer over the tiny model writing its run into tmp_path, and the architecture config."""
    model, cfg = _tiny_model()
    tc         = _trainer_config(cfg, epochs=epochs, ae_loss=ae_loss)
    logger     = Logger(log_dir=str(tmp_path), name="profile_ae_trainer_test")
    z_axis     = np.arange(PROFILE_LENGTH, dtype=np.float32)
    return Trainer(model, cfg, z_axis, tc, str(tmp_path), logger), cfg


def test_trainer_builds_on_cpu(tmp_path):
    """The trainer lands on CPU with encoder, embedding-norm and decoder parameter groups in that order."""
    trainer, _ = _build_trainer(tmp_path)

    assert trainer.device.type == "cpu"
    assert trainer.criterion is not None
    assert [group["name"] for group in trainer.optimizer.param_groups] == ["ae_encoder", "ae_embedding_norm", "ae_decoder"]


def test_compute_loss_reshapes_and_finite(tmp_path):
    """A training step reshapes the curve batch and returns a finite loss with the curve_mse component."""
    trainer, _ = _build_trainer(tmp_path)
    batch      = next(iter(_loader()))

    out = trainer._compute_loss(batch)

    assert torch.isfinite(out["total_loss"])
    assert "curve_mse" in out["components"]


def test_compute_loss_reconstructs_against_the_clean_target(tmp_path, monkeypatch):
    """The criterion target is the clean curve of the batch, not the noisy encoder input."""
    trainer, _   = _build_trainer(tmp_path)
    noisy, clean = next(iter(_loader()))
    noisy        = noisy + 100.0

    captured  = {}
    criterion = trainer.criterion

    def spy(prediction, target):
        captured["target"] = target
        return criterion(prediction, target)

    monkeypatch.setattr(trainer, "criterion", spy)
    trainer._compute_loss((noisy, clean))

    assert torch.equal(captured["target"], clean.unsqueeze(-1).unsqueeze(-1))


def test_compute_loss_latent_noise_adds_weighted_term(tmp_path):
    """Latent noise adds a second reconstruction component folded into the total at its weight."""
    ae_loss    = ProfileAeLossConfig(use_latent_noise=True, latent_noise_std=0.1, weight_latent_noise=2.0)
    trainer, _ = _build_trainer(tmp_path, ae_loss=ae_loss)
    batch      = next(iter(_loader()))

    out = trainer._compute_loss(batch)

    assert set(out["components"]) == {"curve_mse", "latent_curve_mse"}
    assert torch.isfinite(out["total_loss"])

    expected = out["components"]["curve_mse"] + 2.0 * out["components"]["latent_curve_mse"]
    assert out["total_loss"].item() == pytest.approx(expected.item(), rel=1e-6)


def test_compute_loss_latent_noise_carries_the_sobolev_term(tmp_path):
    """With both terms on, the latent-noise pass reports its own Sobolev component alongside the clean one."""
    ae_loss    = ProfileAeLossConfig(use_sobolev=True, weight_sobolev=1.0, use_latent_noise=True, latent_noise_std=0.1, weight_latent_noise=1.0)
    trainer, _ = _build_trainer(tmp_path, ae_loss=ae_loss)
    batch      = next(iter(_loader()))

    out = trainer._compute_loss(batch)

    assert set(out["components"]) == {"curve_mse", "curve_sobolev", "latent_curve_mse", "latent_curve_sobolev"}


def test_single_train_epoch_updates_params(tmp_path):
    """One epoch returns a finite average loss and changes at least one model parameter."""
    trainer, _ = _build_trainer(tmp_path, epochs=1)
    loader     = _loader()

    before = copy.deepcopy(trainer.model.state_dict())
    avg    = trainer.train_epoch(loader, epoch=0)
    after  = trainer.model.state_dict()

    assert np.isfinite(avg)

    changed = any(not torch.equal(before[k], after[k]) for k in before)
    assert changed


def test_fit_returns_finite_losses(tmp_path):
    """Training over two epochs returns one finite loss per epoch and a finite best value."""
    trainer, _ = _build_trainer(tmp_path, epochs=2)
    loader     = _loader()

    train_losses, val_losses, best = trainer.train(loader, loader, loader)

    assert len(train_losses) == 2
    assert all(np.isfinite(l) for l in train_losses)
    assert np.isfinite(best)


def test_fit_reduces_or_stable_loss(tmp_path):
    """Over three epochs the training loss does not increase beyond a small tolerance."""
    trainer, _ = _build_trainer(tmp_path, epochs=3)
    loader     = _loader()

    train_losses, _, _ = trainer.train(loader, loader, loader)

    assert train_losses[-1] <= train_losses[0] + 1e-3


def test_checkpoint_saved_and_roundtrips(tmp_path):
    """The checkpoint stores parameters, the z axis and the best validation loss, and reloads bit-identically."""
    trainer, _ = _build_trainer(tmp_path, epochs=2)
    loader     = _loader()

    trainer.train(loader, loader, loader)

    assert trainer.checkpoint_path.is_file()

    ckpt = torch.load(trainer.checkpoint_path, map_location="cpu", weights_only=False)

    assert "params" in ckpt
    assert isinstance(ckpt["z_axis"], np.ndarray)
    assert "best_val_loss" in ckpt

    fresh_model, _ = _tiny_model()
    fresh_model.load_state_dict(ckpt["params"])

    for k, v in trainer.model.state_dict().items():
        assert torch.equal(v.cpu(), fresh_model.state_dict()[k])


def test_capture_state_contents(tmp_path):
    """The captured state carries the epoch and the z axis."""
    trainer, _ = _build_trainer(tmp_path)

    state = trainer.capture_state(epoch=2)

    assert state["epoch"] == 2
    np.testing.assert_allclose(state["z_axis"], np.arange(PROFILE_LENGTH, dtype=np.float32))


def test_non_finite_loss_raises(tmp_path, monkeypatch):
    """An infinite total loss aborts the epoch with a FloatingPointError."""
    trainer, _ = _build_trainer(tmp_path, epochs=1)
    loader     = _loader()

    def _nan_loss(batch):
        return {"total_loss": torch.tensor(float("inf")), "components": {}, "monitor": {}, "occupancy": {}, "physical": {}}

    monkeypatch.setattr(trainer, "_compute_loss", _nan_loss)

    with pytest.raises(FloatingPointError):
        trainer.train_epoch(loader, epoch=0)
