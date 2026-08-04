from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from tools.training.checkpoint import Checkpoint


class StubTrainer:
    def __init__(self, model, loss_generation=0):
        self.model     = model
        self.criterion = SimpleNamespace(loss_generation=loss_generation)
        self.ema       = SimpleNamespace(enabled=False)

    def capture_state(self, epoch: int) -> dict:
        return {
            "epoch"  : epoch,
            "params" : self.model.state_dict(),
            "x_axis" : torch.linspace(0, 1, 5),
        }


def _model():
    torch.manual_seed(0)
    return torch.nn.Linear(4, 3)


def test_step_records_first_best(tmp_path, logger, tracker):
    path    = str(tmp_path / "ckpt" / "best.pt")
    ckpt    = Checkpoint(logger, tracker, path)
    trainer = StubTrainer(_model())

    improved = ckpt.step(2.0, epoch=1, trainer=trainer)

    assert improved              is True
    assert ckpt.best_val_loss    == 2.0
    assert ckpt.best_epoch       == 1


def test_step_no_improvement_returns_false(tmp_path, logger, tracker):
    ckpt    = Checkpoint(logger, tracker, str(tmp_path / "best.pt"))
    trainer = StubTrainer(_model())

    ckpt.step(2.0, 1, trainer)
    improved = ckpt.step(3.0, 2, trainer)

    assert improved           is False
    assert ckpt.best_val_loss == 2.0
    assert ckpt.best_epoch    == 1


def test_step_improvement_updates_best(tmp_path, logger, tracker):
    ckpt    = Checkpoint(logger, tracker, str(tmp_path / "best.pt"))
    trainer = StubTrainer(_model())

    ckpt.step(2.0, 1, trainer)
    improved = ckpt.step(1.0, 2, trainer)

    assert improved           is True
    assert ckpt.best_val_loss == 1.0
    assert ckpt.best_epoch    == 2


def test_save_writes_file_with_metadata(tmp_path, logger, tracker):
    path    = tmp_path / "sub" / "best.pt"
    ckpt    = Checkpoint(logger, tracker, str(path))
    trainer = StubTrainer(_model())

    ckpt.step(0.5, epoch=3, trainer=trainer)

    assert path.is_file()
    loaded = torch.load(path, map_location="cpu")
    assert loaded["best_val_loss"] == 0.5
    assert loaded["best_epoch"]    == 3
    assert "params"                in loaded


def test_restore_best_round_trip_exact(tmp_path, logger, tracker):
    model   = _model()
    ckpt    = Checkpoint(logger, tracker, str(tmp_path / "best.pt"))
    trainer = StubTrainer(model)

    ckpt.step(0.5, epoch=2, trainer=trainer)
    saved_state = {k: v.clone() for k, v in model.state_dict().items()}

    with torch.no_grad():
        for p in model.parameters():
            p.add_(10.0)

    ckpt.restore_best(model, device="cpu")

    for k, v in model.state_dict().items():
        assert torch.equal(v, saved_state[k])


def test_restore_best_noop_when_no_checkpoint(tmp_path, logger, tracker):
    model = _model()
    ckpt  = Checkpoint(logger, tracker, str(tmp_path / "best.pt"))

    before = {k: v.clone() for k, v in model.state_dict().items()}
    ckpt.restore_best(model, device="cpu")

    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k])


def test_loss_generation_change_resets_baseline(tmp_path, logger, tracker):
    ckpt    = Checkpoint(logger, tracker, str(tmp_path / "best.pt"))
    trainer = StubTrainer(_model(), loss_generation=0)

    ckpt.step(1.0, 1, trainer)
    assert ckpt.best_val_loss == 1.0

    trainer.criterion.loss_generation = 1
    improved = ckpt.step(5.0, 2, trainer)

    assert improved              is True
    assert ckpt.loss_generation  == 1
    assert ckpt.best_val_loss    == 5.0
    assert ckpt.best_epoch       == 2


class NumpyAxisTrainer(StubTrainer):
    def capture_state(self, epoch: int) -> dict:
        return {
            "epoch"  : epoch,
            "params" : self.model.state_dict(),
            "x_axis" : torch.linspace(0, 1, 5).numpy(),
        }


def test_restore_best_with_numpy_axis_like_real_trainer(tmp_path, logger, tracker):
    model   = _model()
    ckpt    = Checkpoint(logger, tracker, str(tmp_path / "best.pt"))
    trainer = NumpyAxisTrainer(model)

    ckpt.step(0.5, epoch=2, trainer=trainer)
    ckpt.restore_best(model, device="cpu")

    restored = torch.load(str(tmp_path / "best.pt"), map_location="cpu", weights_only=False)
    assert isinstance(restored["x_axis"], np.ndarray)


def test_reset_baseline_clears_state(tmp_path, logger, tracker):
    ckpt = Checkpoint(logger, tracker, str(tmp_path / "best.pt"))
    ckpt.best_val_loss = 0.1
    ckpt.best_epoch    = 4

    ckpt.reset_baseline(loss_generation=2, epoch=10)

    assert ckpt.best_val_loss   == float("inf")
    assert ckpt.best_epoch      == -1
    assert ckpt.loss_generation == 2
