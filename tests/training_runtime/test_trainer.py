"""Tests for the shared BaseTrainer optimizer and state helpers.

Covers AdamW construction from parameter groups, learning rate updates,
state capture, cache clearing, the CPU-fallback warning, the param-group
table, the DDP scope of the logged metrics, the warning of an evaluation
without a finite batch and the abstract and no-op hooks.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.optim as optim

from tools.training.trainer          import BaseTrainer
from tools.training.vram_reservation import VramReservation


class TrainerShim:
    """Bare object standing in for a trainer instance in unbound method calls."""
    pass


def _shim(model, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
    """Returns a trainer shim carrying a model and an optimizer config section."""
    shim        = TrainerShim()
    shim.model  = model
    shim.config = SimpleNamespace(optimizer=SimpleNamespace(betas=betas, eps=eps, weight_decay=weight_decay))
    return shim


def test_build_optimizer_applies_defaults_and_is_adamw():
    """Verifies that the optimizer is AdamW and inherits the config defaults."""
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
    """Verifies that values set explicitly on a parameter group are not overwritten."""
    model = torch.nn.Linear(4, 2)
    shim  = _shim(model, weight_decay=0.99)

    param_groups = [{"params": list(model.parameters()), "lr": 0.01, "weight_decay": 0.0}]
    optimizer    = BaseTrainer._build_optimizer(shim, param_groups)

    assert optimizer.param_groups[0]["weight_decay"] == 0.0


def test_update_optimizer_sets_lrs_and_logs(tracker):
    """Verifies that new rates are written to the groups and logged per group name."""
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
    """Verifies that captured state holds the epoch, model parameters and x axis."""
    model = torch.nn.Linear(3, 1)
    shim  = TrainerShim()
    shim.model  = model
    shim.z_axis = torch.linspace(0, 1, 4)

    state = BaseTrainer.capture_state(shim, epoch=7)

    assert state["epoch"] == 7
    assert set(state["params"].keys()) == set(model.state_dict().keys())
    assert state["z_axis"].shape == (4,)


def test_clear_cuda_cache_runs_on_cpu():
    """Verifies that cache clearing is safe on a CPU device."""
    shim = TrainerShim()
    shim.vram_reservation = VramReservation(enabled=False, keep_free_gb=1.0, device=torch.device("cpu"), logger=None)
    BaseTrainer._clear_cuda_cache(shim)


def test_clear_cuda_cache_does_not_touch_the_reservation():
    """Verifies that cache clearing does not call into the VRAM reservation."""
    shim = TrainerShim()
    shim.vram_reservation = SimpleNamespace()

    BaseTrainer._clear_cuda_cache(shim)


def test_eval_step_delegates_to_compute_loss():
    """Verifies that the evaluation step returns whatever _compute_loss produced."""
    sentinel = {"total_loss": torch.tensor(1.0)}

    class Shim(TrainerShim):
        """Trainer shim whose loss computation returns a fixed sentinel dict."""
        def _compute_loss(self, batch):
            """Returns the sentinel loss dictionary."""
            return sentinel

    shim   = Shim()
    result = BaseTrainer._eval_step(shim, batch=None)
    assert result is sentinel


def test_abstract_methods_raise_not_implemented():
    """Verifies that the abstract trainer hooks raise NotImplementedError."""
    shim = TrainerShim()
    with pytest.raises(NotImplementedError):
        BaseTrainer._build_param_groups(shim)
    with pytest.raises(NotImplementedError):
        BaseTrainer._build_criterion(shim)
    with pytest.raises(NotImplementedError):
        BaseTrainer._compute_loss(shim, batch=None)


def test_no_op_hooks_return_none():
    """Verifies that the optional lifecycle hooks default to returning None."""
    shim = TrainerShim()
    assert BaseTrainer._log_init_banner(shim)             is None
    assert BaseTrainer._on_optimizer_step(shim)           is None
    assert BaseTrainer._before_epoch(shim, 0)             is None
    assert BaseTrainer._after_eval(shim, 1.0, 0)          is None


def test_cpu_device_is_warned_with_the_mask_and_the_amp_consequence(recording_logger, monkeypatch):
    """Verifies a CPU device raises a warning naming the CUDA mask and that AMP becomes CPU bfloat16 autocast."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")

    shim        = TrainerShim()
    shim.device = torch.device("cpu")
    shim.logger = recording_logger
    shim.config = SimpleNamespace(training=SimpleNamespace(use_amp=True))

    BaseTrainer._warn_cpu_device(shim)

    [(level, message)] = recording_logger.messages

    assert level == "warning"
    assert "CUDA_VISIBLE_DEVICES=3" in message
    assert "bfloat16 on the CPU" in message


def test_cuda_device_is_not_warned(recording_logger):
    """Verifies a CUDA device produces no fallback warning."""
    shim        = TrainerShim()
    shim.device = torch.device("cuda")
    shim.logger = recording_logger

    BaseTrainer._warn_cpu_device(shim)

    assert recording_logger.messages == []


def test_param_group_table_separates_the_base_rate_from_the_warmup_start_rate():
    """Verifies the param-group table prints the base LR after lr_scale and the warmup-scaled step-0 LR in separate columns."""
    tables = []
    model  = torch.nn.Linear(4, 2)
    shim   = _shim(model)

    shim.base_lrs  = [0.01]
    shim.optimizer = BaseTrainer._build_optimizer(shim, [{"params": list(model.parameters()), "lr": 0.001, "name": "main"}])
    shim.logger    = SimpleNamespace(metrics_table=lambda rows, columns, title: tables.append((rows, columns, title)))

    BaseTrainer._log_param_groups(shim)

    [(rows, columns, title)] = tables

    assert columns == ["Group", "Base LR", "Step-0 LR", "Weight decay"]
    assert rows[0]["Base LR"]   == "0.01"
    assert rows[0]["Step-0 LR"] == "0.001"
    assert "warmup start factor" in title


def test_evaluate_without_a_finite_batch_warns_that_the_loss_is_nan(recording_logger):
    """Verifies an evaluation in which every batch is non-finite warns that its loss is NaN instead of returning NaN silently."""
    shim = TrainerShim()

    shim.model                   = torch.nn.Linear(2, 1)
    shim.logger                  = recording_logger
    shim.stage_name              = "Backbone"
    shim.epochs                  = 4
    shim.device                  = torch.device("cpu")
    shim.use_amp                 = False
    shim.abort_on_nonfinite_loss = False
    shim.distributed             = SimpleNamespace(active=False, sum_scalars=lambda loss_sum, batches, samples: (loss_sum, batches, samples))
    shim._apply_memory_format    = lambda batch: batch
    shim._eval_step              = lambda batch: {"total_loss": torch.tensor(float("nan")), "components": {"param": torch.tensor(float("nan"))}}
    shim._log_epoch_metrics      = lambda stage, aggregator, epoch: None
    shim._nonfinite_detail       = lambda loss_dict, batch: BaseTrainer._nonfinite_detail(shim, loss_dict, batch)

    result = BaseTrainer.evaluate(shim, [(torch.zeros(1, 2),), (torch.zeros(1, 2),)], epoch=2, stage="val")

    assert result["num_batches"] == 0
    assert result["avg_loss"] != result["avg_loss"]
    assert ("warning", "Backbone val epoch 3: no finite batch out of 2; the val loss is NaN") in recording_logger.messages
    assert any(level == "warning" and "non-finite terms: ['param']" in message for level, message in recording_logger.messages)


def test_nonfinite_detail_says_when_every_weighted_term_is_finite():
    """Verifies the non-finite clause states that no weighted term is at fault when every component is finite."""
    detail = BaseTrainer._nonfinite_detail(TrainerShim(), {"components": {"param": torch.tensor(1.0)}}, batch=None)

    assert detail == "; non-finite terms: none (every weighted term is finite)"


def test_ddp_scope_names_the_rank_local_diagnostics(recording_logger):
    """Verifies a DDP run's loop row and one info line say the per-term diagnostics cover this rank's shard only."""
    shim             = TrainerShim()
    shim.logger      = recording_logger
    shim.distributed = SimpleNamespace(active=True, rank=0, world_size=2)

    row = BaseTrainer._ddp_row(shim)
    BaseTrainer._log_ddp_scope(shim)

    [(level, message)] = recording_logger.messages

    assert row.startswith("rank 1/2") and "per-term diagnostics from this rank's shard only" in row
    assert level == "info"
    assert "1/2 of the patches" in message


def test_ddp_scope_is_silent_without_ddp(recording_logger):
    """Verifies a single-process run prints 'off' and no DDP scope line."""
    shim             = TrainerShim()
    shim.logger      = recording_logger
    shim.distributed = SimpleNamespace(active=False)

    assert BaseTrainer._ddp_row(shim) == "off"

    BaseTrainer._log_ddp_scope(shim)

    assert recording_logger.messages == []
