from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from configuration.training                 import UnrolledEntryConfig, UnrolledTrainerConfig
from models.unrolled                        import get_unrolled
from pipelines.shared.config.config_factory import ConfigFactory
from pipelines.unrolled.training.trainer    import UnrolledTrainer

from tests.backbone_training._helpers import identity_normalizer, x_axis_numpy

from tools.monitoring.logger import Logger


HW      = 8
TRACKS  = 3
N_BATCH = 3


def _loader(n: int = 2 * N_BATCH) -> DataLoader:
    gen  = torch.Generator().manual_seed(0)
    imgs = torch.randn(n, 2, HW, HW, generator=gen)
    gt   = torch.randn(n, 6, HW, HW, generator=gen)
    kz   = torch.linspace(-0.15, 0.15, TRACKS).reshape(1, TRACKS, 1, 1).expand(n, TRACKS, HW, HW).contiguous()

    gt[:, 0::3] = gt[:, 0::3].abs() + 1.0
    gt[:, 1::3] = 20.0 + 10.0 * gt[:, 1::3].clamp(-1.5, 1.5)
    gt[:, 2::3] = gt[:, 2::3].abs() + 2.0

    return DataLoader(TensorDataset(imgs, gt, kz), batch_size=2)


def _entry_config(epochs: int = 2, warmup_steps: int = 4, use_ema: bool = True, reserve_vram: bool = False) -> UnrolledEntryConfig:
    config = UnrolledEntryConfig()

    config.training.epochs              = epochs
    config.training.warmup_enabled      = True
    config.training.warmup_steps        = warmup_steps
    config.training.use_ema             = use_ema
    config.training.ema_decay           = 0.5
    config.training.reserve_vram        = reserve_vram
    config.training.early_stop_patience = 100

    return config


def _trainer_config(tmp_path, entry_config: UnrolledEntryConfig) -> UnrolledTrainerConfig:
    config                   = ConfigFactory(entry_config).unrolled_trainer_config(tmp_path, params_per_gaussian=3)
    config.resources.enabled = False

    return config


def _build_trainer(tmp_path, entry_config: UnrolledEntryConfig) -> UnrolledTrainer:
    model, model_cfg = get_unrolled("gamma_net", n_iterations=2, prox_hidden=4)
    logger           = Logger(log_dir=str(tmp_path / "logs"), name="unrolled_trainer", level="ERROR")

    return UnrolledTrainer(
        model      = model,
        model_cfg  = model_cfg,
        x_axis     = x_axis_numpy(),
        config     = _trainer_config(tmp_path, entry_config),
        run_dir    = tmp_path,
        logger     = logger,
        norm_stats = identity_normalizer(6),
    )


def test_fit_engages_warmup_and_finishes(tmp_path, force_cpu):
    trainer = _build_trainer(tmp_path, _entry_config(epochs=2, warmup_steps=4))
    loader  = _loader()

    train_losses, val_losses, best_val_loss = trainer.train(loader, loader, loader)

    assert len(train_losses) == 2
    assert len(val_losses)   == 2
    assert trainer.warmup.current_step == trainer.warmup.warmup_steps
    assert trainer.warmup.is_finished()
    assert torch.isfinite(torch.tensor(best_val_loss)).item()
    assert torch.isfinite(torch.tensor(trainer.test_metrics["avg_loss"])).item()


def test_active_warmup_scales_learning_rates_below_base(tmp_path, force_cpu):
    trainer = _build_trainer(tmp_path, _entry_config(epochs=1, warmup_steps=100))
    loader  = _loader()

    trainer.train(loader, loader, loader)

    assert not trainer.warmup.is_finished()
    assert trainer.warmup.factor() < 1.0
    assert all(group["lr"] < base for group, base in zip(trainer.optimizer.param_groups, trainer.base_lrs))


def test_fit_updates_ema_shadow_and_checkpoints(tmp_path, force_cpu):
    trainer = _build_trainer(tmp_path, _entry_config(epochs=2))
    initial = {key: tensor.clone() for key, tensor in trainer.ema.shadow.items()}
    loader  = _loader()

    trainer.train(loader, loader, loader)

    assert trainer.ema.enabled
    assert any(not torch.equal(trainer.ema.shadow[key], initial[key]) for key in initial)
    assert (tmp_path / "best_model.pt").exists()
    assert (tmp_path / "last.pt").exists()


def test_best_checkpoint_holds_ema_weights(tmp_path, force_cpu):
    trainer = _build_trainer(tmp_path, _entry_config(epochs=1))
    loader  = _loader()

    trainer.train(loader, loader, loader)

    saved = torch.load(tmp_path / "best_model.pt", map_location="cpu", weights_only=False)["params"]

    assert all(torch.equal(saved[key], trainer.ema.shadow[key]) for key in trainer.ema.shadow)


def test_peak_error_is_tracked_for_every_stage(tmp_path, force_cpu):
    trainer = _build_trainer(tmp_path, _entry_config(epochs=1))
    loader  = _loader()

    trainer.train(loader, loader, loader)

    assert set(trainer.peak_error_m) == {"train", "val", "test"}
    assert all(value >= 0.0 for value in trainer.peak_error_m.values())


def test_resume_restores_the_saved_trainer_state(tmp_path, force_cpu):
    loader = _loader()
    _build_trainer(tmp_path, _entry_config(epochs=1)).train(loader, loader, loader)

    resumed_entry                 = _entry_config(epochs=2)
    resumed_entry.training.resume = True

    trainer = _build_trainer(tmp_path, resumed_entry)
    trainer.train(loader, loader, loader)

    assert len(trainer.train_losses) == 2


def test_batch_without_kz_field_fails_loudly(tmp_path, force_cpu):
    trainer = _build_trainer(tmp_path, _entry_config(epochs=1))
    imgs    = torch.randn(2, 2, HW, HW)
    gt      = torch.randn(2, 6, HW, HW)
    loader  = DataLoader(TensorDataset(imgs, gt), batch_size=2)

    with pytest.raises(RuntimeError):
        trainer.train(loader, loader, loader)


def test_lr_scales_linearly_with_batch_size(tmp_path, force_cpu):
    reference = _entry_config(epochs=1)
    reference.training.scale_lr_with_batch = False

    scaled = _entry_config(epochs=1)
    scaled.training.batch_size              = 512
    scaled.training.lr_reference_batch_size = 256
    scaled.training.scale_lr_with_batch     = True

    base_lrs   = _build_trainer(tmp_path / "reference", reference).base_lrs
    scaled_lrs = _build_trainer(tmp_path / "scaled", scaled).base_lrs

    assert scaled_lrs == [lr * 2.0 for lr in base_lrs]


def test_compute_loss_satisfies_the_probe_contract(tmp_path, force_cpu):
    trainer = _build_trainer(tmp_path, _entry_config(epochs=1))
    batch   = next(iter(_loader()))

    trainer.model.train()
    losses = trainer._compute_loss(batch)
    loss   = losses["total_loss"]

    assert trainer.use_amp is False
    assert loss.requires_grad
    assert torch.isfinite(loss).item()
    assert set(losses) == {"total_loss", "components", "monitor", "occupancy", "physical"}

    loss.backward()
    trainer.optimizer.step()


def test_unknown_curve_loss_fails_loudly(tmp_path, force_cpu):
    entry            = _entry_config(epochs=1)
    entry.curve_loss = "huber"

    with pytest.raises(ValueError):
        _build_trainer(tmp_path, entry)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="VRAM reservation requires CUDA")
def test_fit_with_vram_reservation_on_gpu(tmp_path):
    trainer = _build_trainer(tmp_path, _entry_config(epochs=1, reserve_vram=True))
    loader  = _loader()

    trainer.train(loader, loader, loader)

    assert trainer.device.type == "cuda"
    assert trainer.vram_reservation.enabled
    assert trainer.vram_reservation.filled
    assert torch.isfinite(torch.tensor(trainer.test_metrics["avg_loss"])).item()
