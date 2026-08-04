from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.optim as optim

from tools.training.trainer          import BaseTrainer
from tools.training.vram_reservation import VramReservation


class TrainerShim:
    pass


def _shim(model, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
    shim        = TrainerShim()
    shim.model  = model
    shim.config = SimpleNamespace(optimizer=SimpleNamespace(betas=betas, eps=eps, weight_decay=weight_decay))
    return shim


def test_build_optimizer_applies_defaults_and_is_adamw():
    model = torch.nn.Linear(4, 2)
    shim  = _shim(model, betas=(0.85, 0.95), eps=1e-7, weight_decay=0.05)

    param_groups = [{"params": list(model.parameters()), "lr": 0.01}]
    optimizer    = BaseTrainer._build_optimizer(shim, param_groups)

    assert isinstance(optimizer, optim.AdamW)
    pg = optimizer.param_groups[0]
    assert pg["betas"]        == (0.85, 0.95)
    assert pg["eps"]          == 1e-7
    assert pg["weight_decay"] == 0.05
    assert pg["lr"]           == 0.01


def test_build_optimizer_does_not_override_explicit_group_values():
    model = torch.nn.Linear(4, 2)
    shim  = _shim(model, weight_decay=0.99)

    param_groups = [{"params": list(model.parameters()), "lr": 0.01, "weight_decay": 0.0}]
    optimizer    = BaseTrainer._build_optimizer(shim, param_groups)

    assert optimizer.param_groups[0]["weight_decay"] == 0.0


def test_update_optimizer_sets_lrs_and_logs(tracker):
    model = torch.nn.Linear(4, 2)
    shim  = _shim(model)

    param_groups     = [{"params": list(model.parameters()), "lr": 0.01, "name": "main"}]
    shim.optimizer   = BaseTrainer._build_optimizer(shim, param_groups)
    shim.tracker     = tracker
    shim.global_step = 5

    BaseTrainer._update_optimizer(shim, [0.123])

    assert shim.optimizer.param_groups[0]["lr"] == 0.123
    assert any(tag == "optim/lr/main" and val == 0.123 for tag, val, _ in tracker.scalars)


def test_capture_state_contains_model_and_axis():
    model = torch.nn.Linear(3, 1)
    shim  = TrainerShim()
    shim.model  = model
    shim.x_axis = torch.linspace(0, 1, 4)

    state = BaseTrainer.capture_state(shim, epoch=7)

    assert state["epoch"] == 7
    assert set(state["params"].keys()) == set(model.state_dict().keys())
    assert state["x_axis"].shape == (4,)


def test_clear_cuda_cache_runs_on_cpu():
    shim = TrainerShim()
    shim.vram_reservation = VramReservation(enabled=False, keep_free_gb=1.0, device=torch.device("cpu"), logger=None)
    BaseTrainer._clear_cuda_cache(shim)


def test_clear_cuda_cache_refills_reservation():
    calls = []
    shim  = TrainerShim()
    shim.vram_reservation = SimpleNamespace(refill=lambda: calls.append("refill"))

    BaseTrainer._clear_cuda_cache(shim)

    assert calls == ["refill"]


def test_eval_step_delegates_to_compute_loss():
    sentinel = {"total_loss": torch.tensor(1.0)}

    class Shim(TrainerShim):
        def _compute_loss(self, batch):
            return sentinel

    shim   = Shim()
    result = BaseTrainer._eval_step(shim, batch=None, aggregator=None)
    assert result is sentinel


def test_abstract_methods_raise_not_implemented():
    shim = TrainerShim()
    with pytest.raises(NotImplementedError):
        BaseTrainer._build_param_groups(shim)
    with pytest.raises(NotImplementedError):
        BaseTrainer._build_criterion(shim)
    with pytest.raises(NotImplementedError):
        BaseTrainer._compute_loss(shim, batch=None)


def test_no_op_hooks_return_none():
    shim = TrainerShim()
    assert BaseTrainer._log_init_banner(shim)             is None
    assert BaseTrainer._on_optimizer_step(shim)           is None
    assert BaseTrainer._before_epoch(shim, 0)             is None
    assert BaseTrainer._after_eval(shim, 1.0, 0)          is None
