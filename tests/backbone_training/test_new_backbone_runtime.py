from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from models                              import get_backbone
from pipelines.backbone.training.trainer import Trainer

from tests.backbone_training._helpers import identity_normalizer, tiny_trainer_config, x_axis_numpy

from tools.monitoring.logger import Logger


NEW_BACKBONES = {
    "pixel_mlp"        : ("pixel_mlp", {"features": [16, 16]}),
    "local_cnn"        : ("local_cnn", {"features": [8, 8], "dropout": 0.0}),
    "nafnet"           : ("nafnet",    {"width": 8, "enc_blocks": [1, 1], "middle_blocks": 1, "dec_blocks": [1, 1], "dropout": 0.0}),
    "unet-set_pred"    : ("unet",      {"head": "set_pred", "features": [8, 16], "bottleneck_factor": 1, "dropout": 0.0, "normalization": "instance"}),
    "resunet-set_pred" : ("resunet",   {"head": "set_pred", "features": [8, 16], "bottleneck_factor": 1, "dropout": 0.0, "normalization": "instance"}),
}


def _loader(n: int = 6, hw: int = 16) -> DataLoader:
    gen  = torch.Generator().manual_seed(0)
    imgs = torch.randn(n, 2, hw, hw, generator=gen)
    tgt  = torch.randn(n, 6, hw, hw, generator=gen)

    tgt[:, 0::3] = tgt[:, 0::3].abs() + 1.0
    tgt[:, 1::3] = 20.0 + 10.0 * tgt[:, 1::3].clamp(-1.5, 1.5)
    tgt[:, 2::3] = tgt[:, 2::3].abs() + 2.0

    return DataLoader(TensorDataset(imgs, tgt), batch_size=2)


def _build_trainer(tmp_path, name: str, reserve_vram: bool = False) -> Trainer:
    backbone_name, overrides = NEW_BACKBONES[name]
    model, model_cfg         = get_backbone(backbone_name, in_channels=2, out_channels=6, **overrides)

    config = tiny_trainer_config(n_gaussians=2, epochs=1)
    config.training.use_ema      = True
    config.training.ema_decay    = 0.5
    config.warmup.warmup_enabled = True
    config.warmup.warmup_steps   = 16
    config.memory.reserve_vram   = reserve_vram

    logger     = Logger(log_dir=str(tmp_path / "logs"), name="trainer", level="ERROR")
    norm_stats = identity_normalizer(6)

    return Trainer(model, model_cfg, x_axis_numpy(), config, tmp_path, logger, norm_stats=norm_stats, emit_docs=False)


@pytest.mark.parametrize("name", sorted(NEW_BACKBONES))
def test_train_epoch_engages_warmup(name, tmp_path, force_cpu):
    trainer = _build_trainer(tmp_path, name)
    loader  = _loader()

    avg = trainer.train_epoch(loader, epoch=0)

    assert torch.isfinite(torch.tensor(avg)).item()
    assert trainer.warmup.current_step == len(loader)
    assert trainer.warmup.factor() < 1.0
    assert all(group["lr"] < base for group, base in zip(trainer.optimizer.param_groups, trainer.base_lrs))


@pytest.mark.parametrize("name", sorted(NEW_BACKBONES))
def test_train_epoch_updates_ema_shadow(name, tmp_path, force_cpu):
    trainer = _build_trainer(tmp_path, name)
    initial = {key: tensor.clone() for key, tensor in trainer.ema.shadow.items()}

    trainer.train_epoch(_loader(), epoch=0)

    assert trainer.ema.enabled
    assert any(not torch.equal(trainer.ema.shadow[key], initial[key]) for key in initial)
    assert all(torch.isfinite(tensor).all() for tensor in trainer.ema.shadow.values())


@pytest.mark.parametrize("name", sorted(NEW_BACKBONES))
def test_ema_applied_swaps_and_restores_weights(name, tmp_path, force_cpu):
    trainer = _build_trainer(tmp_path, name)
    trainer.train_epoch(_loader(), epoch=0)

    raw = {key: p.detach().clone() for key, p in trainer.model.named_parameters()}

    with trainer.ema.applied(trainer.model):
        applied = {key: p.detach().clone() for key, p in trainer.model.named_parameters()}
        with torch.no_grad():
            out = trainer.model(torch.randn(1, 2, 16, 16))

    restored = {key: p.detach().clone() for key, p in trainer.model.named_parameters()}

    assert torch.isfinite(out).all()
    assert all(torch.equal(applied[key], trainer.ema.shadow[key]) for key in applied)
    assert all(torch.equal(restored[key], raw[key]) for key in raw)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="VRAM reservation requires CUDA")
@pytest.mark.parametrize("name", sorted(NEW_BACKBONES))
def test_full_fit_with_vram_reservation_on_gpu(name, tmp_path):
    trainer = _build_trainer(tmp_path, name, reserve_vram=True)
    loader  = _loader()

    train_losses, _val_losses, best_val = trainer.train(loader, loader, loader)

    assert trainer.device.type == "cuda"
    assert trainer.vram_reservation.enabled
    assert trainer.vram_reservation.filled
    assert torch.isfinite(torch.tensor(best_val)).item()
    assert len(train_losses) == 1
