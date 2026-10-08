"""Tests for saving and restoring the full trainer state.

Covers the round trip of model weights, EMA shadow, optimizer, warmup,
scheduler, early stopping, checkpoint bookkeeping, gradient clipper history
and RNG state, plus the post-restore hook.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tools.training.checkpoint import Checkpoint, TrainerState, WeightEma, WeightSwa
from tools.training.gradients  import GradientClipper
from tools.training.scheduling import Scheduler, Warmup
from tools.training.stopping   import EarlyStopping

from tests.conftest                  import SilentLogger
from tests.training_runtime.conftest import RecordingTracker, TinyModel


def _warmup_config(fraction: float = 0.05) -> SimpleNamespace:
    """Returns a namespace holding a linear warmup section spanning `fraction` of the total steps."""
    return SimpleNamespace(warmup=SimpleNamespace(warmup_fraction=fraction, warmup_start_factor=0.1, warmup_enabled=True, warmup_mode="linear", warmup_poly_power=2.0))


def _scheduler_config(epochs: int = 10) -> SimpleNamespace:
    """Returns a namespace holding a cosine scheduler section and a warmup section."""
    return SimpleNamespace(scheduler=SimpleNamespace(type="cosine_annealing", epochs=epochs, eta_min=1e-6, step_size=30, gamma=0.1, power=1.0), optimizer=SimpleNamespace(lr_scale=1.0), warmup=_warmup_config().warmup)


def _stopping_config(patience: int = 3) -> SimpleNamespace:
    """Returns a namespace holding an early_stopping section with the given patience."""
    return SimpleNamespace(early_stopping=SimpleNamespace(patience=patience, min_delta=0.0, restore_best=True))


def _clipper_config(window: int = 4) -> SimpleNamespace:
    """Returns a namespace holding an adaptive-percentile gradient clipper section."""
    return SimpleNamespace(gradient_clipper=SimpleNamespace(clip_mode="adaptive_percentile", max_grad_norm=1.0, adaptive_window=window, adaptive_percentile=95.0, adaptive_mean_std_k=2.0, clip_epsilon=1e-6, log_histogram_freq=100))


def _trainer(tmp_path, seed: int = 0, batch_size: int = 8, lr_scale: float = 1.0) -> SimpleNamespace:
    """Returns a trainer shim with every stateful component wired up, seeded by `seed`, under the given batch geometry."""
    torch.manual_seed(seed)

    logger  = SilentLogger()
    tracker = RecordingTracker()

    model     = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    warmup    = Warmup(_warmup_config(), logger)
    warmup.resolve(100)
    scheduler = Scheduler([1e-3], warmup, _scheduler_config(), logger)

    return SimpleNamespace(
        model              = model,
        optimizer          = optimizer,
        ema                = WeightEma(model, decay=0.5, enabled=True, warmup=False),
        swa                = WeightSwa(model, enabled=True),
        warmup             = warmup,
        lr_scheduler       = scheduler,
        early_stopping     = EarlyStopping(_stopping_config(), logger, tracker),
        checkpoint         = Checkpoint(logger, tracker, str(tmp_path / "best_model.pt")),
        grad_clipper       = GradientClipper(_clipper_config(), logger, tracker),
        train_losses       = [],
        val_losses         = [],
        global_step        = 0,
        device             = torch.device("cpu"),
        config             = SimpleNamespace(optimizer=SimpleNamespace(lr_scale=lr_scale)),
        micro_batch        = batch_size,
        accumulation_steps = 1,
        distributed        = SimpleNamespace(world_size=1),
        restored_states    = [],
        _on_state_restored = lambda state: None,
    )


def _advance(trainer) -> None:
    """Advances every stateful component of `trainer` so a restore has something to compare."""
    trainer.model.linear.weight.grad = torch.ones_like(trainer.model.linear.weight)
    trainer.optimizer.step()
    trainer.ema.update(trainer.model)
    trainer.swa.update(trainer.model)

    for _ in range(3):
        trainer.warmup.step()

    trainer.lr_scheduler.step(4)
    trainer.early_stopping(0.5, 3)
    trainer.early_stopping(0.6, 4)
    trainer.checkpoint.best_val_loss = 0.5
    trainer.checkpoint.best_epoch    = 3

    trainer.train_losses = [1.0, 0.8]
    trainer.val_losses   = [1.1, 0.9]
    trainer.global_step  = 42

    for step, norm in enumerate((0.5, 0.7, 0.6)):
        trainer.grad_clipper.record(norm, step)


def test_roundtrip_restores_all_components(tmp_path):
    """Verifies that every trainer component is restored from a saved state."""
    source = _trainer(tmp_path, seed=0)
    _advance(source)

    path = tmp_path / "last.pt"
    TrainerState.save(source, epoch=7, path=path)

    target     = _trainer(tmp_path, seed=1)
    next_epoch = TrainerState.restore(target, path)

    assert next_epoch          == 8
    assert target.global_step  == 42
    assert target.train_losses == [1.0, 0.8]
    assert target.val_losses   == [1.1, 0.9]

    assert target.warmup.current_step           == 3
    assert target.lr_scheduler.current_lrs      == source.lr_scheduler.current_lrs
    assert target.early_stopping.best_loss      == pytest.approx(0.5)
    assert target.early_stopping.counter        == 1
    assert target.checkpoint.best_val_loss      == pytest.approx(0.5)
    assert target.checkpoint.best_epoch         == 3
    assert target.grad_clipper.history          == pytest.approx([0.5, 0.7, 0.6])

    for key in source.model.state_dict():
        assert torch.allclose(target.model.state_dict()[key], source.model.state_dict()[key])

    assert torch.allclose(target.ema.shadow["linear.weight"], source.ema.shadow["linear.weight"])


def test_restore_invokes_hook_with_state(tmp_path):
    """Verifies that the post-restore hook receives the loaded state payload."""
    source = _trainer(tmp_path)
    path   = tmp_path / "last.pt"
    TrainerState.save(source, epoch=2, path=path)

    seen                      = []
    target                    = _trainer(tmp_path, seed=1)
    target._on_state_restored = lambda state: seen.append(state["epoch"])

    TrainerState.restore(target, path)

    assert seen == [2]


def test_restore_missing_file_raises(tmp_path):
    """Verifies that restoring from a missing checkpoint raises FileNotFoundError."""
    trainer = _trainer(tmp_path)

    with pytest.raises(FileNotFoundError):
        TrainerState.restore(trainer, tmp_path / "absent.pt")


def test_capture_records_the_batch_geometry(tmp_path):
    """Verifies the saved state carries the micro-batch, accumulation window, rank count and LR scale it was trained under."""
    state = TrainerState.capture(_trainer(tmp_path, batch_size=16, lr_scale=2.0), epoch=0)

    assert state["batch_geometry"] == {"batch_size": 16, "accumulation_steps": 1, "world_size": 1, "lr_scale": 2.0}


def test_restore_refuses_a_different_batch_geometry(tmp_path):
    """Verifies a state saved under another micro-batch or LR scale is refused, naming each changed setting, before touching the trainer."""
    path = tmp_path / "last.pt"
    TrainerState.save(_trainer(tmp_path, batch_size=16, lr_scale=2.0), epoch=3, path=path)

    target = _trainer(tmp_path, seed=1, batch_size=32, lr_scale=4.0)
    before = {key: value.clone() for key, value in target.model.state_dict().items()}

    with pytest.raises(ValueError, match=r"batch_size 16 -> 32, lr_scale 2.0 -> 4.0"):
        TrainerState.restore(target, path)

    for key, value in target.model.state_dict().items():
        assert torch.equal(value, before[key])


def test_restore_refuses_a_state_without_batch_geometry(tmp_path):
    """Verifies a state saved before the geometry was recorded is refused rather than resumed unchecked."""
    path  = tmp_path / "last.pt"
    state = TrainerState.capture(_trainer(tmp_path), epoch=0)
    del state["batch_geometry"]
    torch.save(state, path)

    with pytest.raises(KeyError, match="records no batch geometry"):
        TrainerState.restore(_trainer(tmp_path), path)


def test_rng_roundtrip_reproduces_draws(tmp_path):
    """Verifies that torch RNG state is restored so subsequent draws repeat."""
    trainer = _trainer(tmp_path)
    path    = tmp_path / "last.pt"

    torch.manual_seed(123)
    TrainerState.save(trainer, epoch=0, path=path)
    expected = torch.rand(4)

    torch.manual_seed(999)
    TrainerState.restore(trainer, path)
    resumed = torch.rand(4)

    assert torch.allclose(resumed, expected)
