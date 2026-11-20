
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from configuration.training              import OptimizerConfig
from pipelines.backbone.training.loss    import Loss
from pipelines.backbone.training.trainer import Trainer
from tools.training.aggregation           import WeightedMean
from tools.training.checkpoint            import WeightSwa

from tests.backbone_training._helpers import TARGET_CHANNELS, batch_tensors, tiny_model, tiny_trainer_config
from tests.conftest                   import RecordingWriter

from tools.monitoring.logger import Logger


pytestmark = pytest.mark.usefixtures("force_cpu")


def _loader(n: int = 6, batch_size: int = 2) -> DataLoader:
    return DataLoader(TensorDataset(*batch_tensors(n=n)), batch_size=batch_size)


def _build_trainer(tmp_path, epochs: int = 1, resume: bool = False, level: str = "ERROR") -> Trainer:
    model, model_cfg = tiny_model()
    config           = tiny_trainer_config(epochs=epochs)
    logger           = Logger(log_dir=str(tmp_path / "logs"), name="trainer", level=level)

    config.training.resume = resume

    trainer = Trainer(model, model_cfg, TARGET_CHANNELS, config, tmp_path, logger, emit_docs=False)
    trainer.warmup.resolve(200)
    return trainer


def test_trainer_builds_on_cpu_with_wired_components(tmp_path):
    trainer = _build_trainer(tmp_path)

    assert trainer.device.type == "cpu"
    assert isinstance(trainer.criterion, Loss)
    assert trainer.optimizer      is not None
    assert trainer.lr_scheduler   is not None
    assert trainer.early_stopping is not None
    assert trainer.warmup         is not None


def test_trainer_optimizer_has_named_param_groups(tmp_path):
    trainer = _build_trainer(tmp_path)
    names   = {group.get("name") for group in trainer.optimizer.param_groups}

    assert "encoder"     in names
    assert "output_head" in names
    assert len(trainer.base_lrs) == len(trainer.optimizer.param_groups)


def test_single_train_step_updates_parameters(tmp_path):
    trainer = _build_trainer(tmp_path)
    loader  = _loader()

    before  = [p.detach().clone() for p in trainer.model.parameters()]
    avg     = trainer.train_epoch(loader, epoch=0)
    after   = list(trainer.model.parameters())

    assert isinstance(avg, float)
    assert torch.isfinite(torch.tensor(avg)).item()
    assert any(not torch.equal(a, b) for a, b in zip(before, after))


def test_compute_loss_returns_finite_total(tmp_path):
    trainer = _build_trainer(tmp_path)
    batch   = next(iter(_loader()))

    out     = trainer._compute_loss(batch)

    assert set(out["channels"]) == set(TARGET_CHANNELS)
    assert torch.isfinite(out["total_loss"].value).item()


def test_one_epoch_fit_writes_checkpoint(tmp_path):
    trainer = _build_trainer(tmp_path, epochs=1)
    loader  = _loader()

    train_losses, val_losses, best_val = trainer.train(loader, loader, loader)

    assert len(train_losses) == 1
    assert (tmp_path / "best_model.pt").exists()
    assert torch.isfinite(torch.tensor(best_val)).item()


def test_trainer_builds_the_configured_optimizer_family(tmp_path):
    model, model_cfg = tiny_model()
    config           = tiny_trainer_config(epochs=1)

    config.optimizer.type     = "sgd"
    config.optimizer.momentum = 0.8

    logger  = Logger(log_dir=str(tmp_path / "logs"), name="trainer_sgd", level="ERROR")
    trainer = Trainer(model, model_cfg, TARGET_CHANNELS, config, tmp_path, logger, emit_docs=False)

    assert isinstance(trainer.optimizer, torch.optim.SGD)
    assert all(pg["momentum"] == 0.8 for pg in trainer.optimizer.param_groups)


def test_swa_writes_averaged_checkpoint_and_marker(tmp_path):
    trainer = _build_trainer(tmp_path, epochs=2)

    trainer.config.training.use_swa = True
    trainer.swa                     = WeightSwa(trainer.model, True)
    trainer.swa_start_epoch         = 0

    loader = _loader()
    trainer.train(loader, loader, loader)

    payload = torch.load(tmp_path / "swa_model.pt", map_location="cpu", weights_only=False)
    marker  = json.loads((tmp_path / "complete.json").read_text())

    assert payload["swa_epochs"] == 2
    assert torch.isfinite(torch.tensor(payload["swa_val_loss"])).item()
    assert marker["swa_val_loss"] == pytest.approx(payload["swa_val_loss"])


def test_train_writes_final_model_checkpoint(tmp_path):
    trainer = _build_trainer(tmp_path, epochs=2)
    loader  = _loader()

    trainer.train(loader, loader, loader)

    saved = torch.load(tmp_path / "final_model.pt", map_location="cpu", weights_only=False)
    best  = torch.load(tmp_path / "best_model.pt",  map_location="cpu", weights_only=False)

    assert saved["epoch"] == 1
    assert set(saved) == set(best)
    assert saved["params"].keys() == trainer.model.state_dict().keys()
    assert saved["best_val_loss"] == best["best_val_loss"]
    assert saved["best_epoch"]    == best["best_epoch"]


def test_multi_epoch_fit_records_losses(tmp_path):
    trainer = _build_trainer(tmp_path, epochs=3)
    loader  = _loader()

    train_losses, val_losses, _ = trainer.train(loader, loader, loader)

    assert len(train_losses) == 3
    assert all(torch.isfinite(torch.tensor(loss)).item() for loss in train_losses)
    assert len(val_losses) >= 1


def test_train_logs_throughput_and_diagnostics(tmp_path):
    writer           = RecordingWriter()
    model, model_cfg = tiny_model()
    config           = tiny_trainer_config(epochs=1)
    logger           = Logger(log_dir=str(tmp_path / "logs"), name="trainer", level="ERROR")
    trainer          = Trainer(model, model_cfg, TARGET_CHANNELS, config, tmp_path, logger, emit_docs=False, writer=writer)

    loader = _loader()
    trainer.train(loader, loader, loader)

    scalar_tags = {tag for tag, _, _ in writer.scalars}

    assert "throughput/samples_per_s"   in scalar_tags
    assert "throughput/epoch_time_s"    in scalar_tags
    assert "throughput/data_wait_frac"  in scalar_tags
    assert "controls/nonfinite_batches" in scalar_tags
    assert "optim/grad_norm"            in scalar_tags

    nonfinite = [value for tag, value, _ in writer.scalars if tag == "controls/nonfinite_batches"]
    assert nonfinite == [0.0]

    assert {"loss_channels/a/train", "loss_channels/b/val"} <= scalar_tags

    assert len(writer.layouts) == 1
    assert list(writer.layouts[0]) == ["Loss", "Optimiser"]


def _mean(value: float, weight: float = 1.0) -> WeightedMean:
    return WeightedMean(torch.tensor(value * weight), torch.tensor(weight))


def _loss_dict(total: float, weight: float = 1.0) -> dict:
    return {"total_loss": _mean(total, weight), "channels": {"a": _mean(total, weight)}}


def _diverged_loss_dict(bad_channel: str) -> dict:
    return {"total_loss": _mean(float("nan")), "channels": {"a": _mean(0.5), bad_channel: _mean(float("nan"))}}


def test_evaluate_raises_on_nonfinite_loss(tmp_path):
    trainer = _build_trainer(tmp_path)
    loader  = _loader()

    trainer._compute_loss = lambda batch: _loss_dict(float("nan"))

    with pytest.raises(FloatingPointError):
        trainer.evaluate(loader, epoch=0)


def test_evaluate_skips_nonfinite_batches_when_abort_disabled(tmp_path):
    trainer = _build_trainer(tmp_path)
    trainer.abort_on_nonfinite_loss = False

    loader  = _loader()
    batches = iter(range(len(loader)))

    trainer._compute_loss = lambda batch: _loss_dict(float("inf") if next(batches) == 0 else 2.0)

    result = trainer.evaluate(loader, epoch=0)

    assert result["num_batches"] == len(loader) - 1
    assert result["avg_loss"]    == pytest.approx(2.0)


def test_trainer_channels_last_converts_model_and_batches(tmp_path):
    model, model_cfg = tiny_model()
    config           = tiny_trainer_config(epochs=1)

    config.training.channels_last = True

    logger  = Logger(log_dir=str(tmp_path / "logs"), name="trainer_cl", level="ERROR")
    trainer = Trainer(model, model_cfg, TARGET_CHANNELS, config, tmp_path, logger, emit_docs=False)
    trainer.warmup.resolve(200)

    conv = next(module for module in trainer.model.modules() if isinstance(module, torch.nn.Conv2d))
    assert conv.weight.is_contiguous(memory_format=torch.channels_last)

    result = trainer.evaluate(_loader(), epoch=0)
    assert result["num_batches"] > 0


def test_trainer_compile_flag_compiles_the_forward(tmp_path, monkeypatch):
    recorded = {}

    def _fake_compile(fn, mode=None):
        recorded["mode"] = mode
        return fn

    monkeypatch.setattr(torch, "compile", _fake_compile)

    model, model_cfg = tiny_model()
    config           = tiny_trainer_config(epochs=1)

    config.training.compile_model = True
    config.training.compile_mode  = "reduce-overhead"

    logger  = Logger(log_dir=str(tmp_path / "logs"), name="trainer_compile", level="ERROR")
    trainer = Trainer(model, model_cfg, TARGET_CHANNELS, config, tmp_path, logger, emit_docs=False)

    assert recorded["mode"] == "reduce-overhead"
    assert trainer.model is not None


def test_trainer_activation_checkpointing_wraps_and_trains(tmp_path):
    model, model_cfg = tiny_model()
    reference_keys   = sorted(model.state_dict())
    config           = tiny_trainer_config(epochs=1)

    config.training.activation_checkpointing = ("encoder_blocks", "decoder_blocks")

    logger  = Logger(log_dir=str(tmp_path / "logs"), name="trainer_ckpt", level="ERROR")
    trainer = Trainer(model, model_cfg, TARGET_CHANNELS, config, tmp_path, logger, emit_docs=False)
    trainer.warmup.resolve(200)

    trainer.train_epoch(_loader(), epoch=0)

    assert sorted(trainer.model.state_dict()) == reference_keys


def test_evaluate_pools_batches_by_their_pixel_weight(tmp_path):
    trainer = _build_trainer(tmp_path)
    loader  = _loader(batch_size=4)

    trainer._compute_loss = lambda batch: _loss_dict(1.0, 30.0) if batch[0].shape[0] == 4 else _loss_dict(4.0, 10.0)

    result = trainer.evaluate(loader, epoch=0)

    assert result["num_batches"] == 2
    assert result["num_samples"] == 6
    assert result["avg_loss"]    == pytest.approx((1.0 * 30.0 + 4.0 * 10.0) / 40.0)


def test_evaluate_returns_nan_when_all_batches_nonfinite(tmp_path):
    trainer = _build_trainer(tmp_path)
    trainer.abort_on_nonfinite_loss = False

    trainer._compute_loss = lambda batch: _loss_dict(float("nan"))

    result = trainer.evaluate(_loader(), epoch=0)

    assert result["num_batches"] == 0
    assert result["avg_loss"]    != result["avg_loss"]


def test_nonfinite_train_loss_abort_names_the_term_and_clears_the_data(tmp_path):
    trainer = _build_trainer(tmp_path)

    trainer._compute_loss = lambda batch: _diverged_loss_dict("b")

    with pytest.raises(FloatingPointError, match=r"non-finite target channels: \['b'\]; inputs and targets finite"):
        trainer.train_epoch(_loader(), epoch=0)


def test_nonfinite_train_loss_abort_names_corrupt_inputs(tmp_path):
    trainer = _build_trainer(tmp_path)
    loader  = _loader()

    loader.dataset.tensors[0][0, 0, 0, 0] = float("nan")
    trainer._compute_loss                 = lambda batch: {"total_loss": _mean(float("nan")), "channels": {"a": _mean(0.5)}}

    with pytest.raises(FloatingPointError, match=r"non-finite target channels: none \(every channel loss is finite\); non-finite inputs in the batch"):
        trainer.train_epoch(loader, epoch=0)


def test_evaluate_skip_warning_names_the_nonfinite_term(tmp_path):
    model, model_cfg = tiny_model()
    config           = tiny_trainer_config(epochs=1)
    logger           = Logger(log_dir=str(tmp_path / "logs"), name="trainer_warn", level="WARNING")
    trainer          = Trainer(model, model_cfg, TARGET_CHANNELS, config, tmp_path, logger, emit_docs=False)

    trainer.abort_on_nonfinite_loss = False
    trainer._compute_loss           = lambda batch: _diverged_loss_dict("b")

    result = trainer.evaluate(_loader(), epoch=0)
    logger.close()

    text = (tmp_path / "logs" / "trainer_warn.log").read_text(encoding="utf-8")

    assert result["num_batches"] == 0
    assert "val loss is non-finite at epoch 1; non-finite target channels: ['b']; inputs and targets finite" in text


def test_checkpoint_save_restore_round_trip(tmp_path):
    trainer = _build_trainer(tmp_path, epochs=1)
    loader  = _loader()

    trainer.train(loader, loader, loader)

    saved = torch.load(tmp_path / "best_model.pt", map_location="cpu", weights_only=False)

    assert "params"        in saved
    assert "best_val_loss" in saved

    fresh_model, model_cfg = tiny_model()
    fresh_model.load_state_dict(saved["params"])

    for name, param in fresh_model.state_dict().items():
        assert torch.equal(param, saved["params"][name])


def test_best_checkpoint_epoch_matches_trainer_state_convention(tmp_path):
    trainer = _build_trainer(tmp_path, epochs=2)
    loader  = _loader()

    trainer.train(loader, loader, loader)

    best = torch.load(tmp_path / "best_model.pt", map_location="cpu", weights_only=False)
    last = torch.load(tmp_path / "last.pt",       map_location="cpu", weights_only=False)

    assert best["epoch"] == best["best_epoch"]
    assert best["epoch"] in (0, 1)
    assert last["epoch"] == 1


def test_resume_of_an_early_stopped_state_trains_no_further_epoch(tmp_path):
    loader = _loader()
    _build_trainer(tmp_path, epochs=1).train(loader, loader, loader)

    state_path = tmp_path / "last.pt"
    state      = torch.load(state_path, map_location="cpu", weights_only=False)

    state["early_stopping"]["triggered"] = True
    torch.save(state, state_path)

    trainer = _build_trainer(tmp_path, epochs=3, resume=True, level="INFO")
    trainer.train(loader, loader, loader)
    trainer.logger.close()

    marker = json.loads((tmp_path / "complete.json").read_text())
    log    = (tmp_path / "logs" / "trainer.log").read_text()

    assert len(trainer.train_losses) == 1
    assert marker["early_stopped"]    is True
    assert marker["epochs_completed"] == 1
    assert "the restored state had already early-stopped at epoch 1" in log
    assert "triggered=True" in log


def test_restore_best_loads_best_parameters(tmp_path):
    trainer = _build_trainer(tmp_path, epochs=2)
    loader  = _loader()

    trainer.train(loader, loader, loader)

    saved = torch.load(tmp_path / "best_model.pt", map_location="cpu", weights_only=False)

    for name, param in trainer.model.state_dict().items():
        assert torch.equal(param, saved["params"][name])



class TrainerShim:
    pass


def _shim(model, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
    shim        = TrainerShim()
    shim.model  = model
    shim.config = SimpleNamespace(optimizer=OptimizerConfig(type="adamw", betas=betas, eps=eps, weight_decay=weight_decay))
    return shim


def test_build_optimizer_applies_defaults_and_is_adamw():
    model = torch.nn.Linear(4, 2)
    shim  = _shim(model, betas=(0.85, 0.95), eps=1e-7, weight_decay=0.05)

    param_groups = [{"params": list(model.parameters()), "lr": 0.01}]
    optimizer    = Trainer._build_optimizer(shim, param_groups)

    assert isinstance(optimizer, torch.optim.AdamW)
    pg = optimizer.param_groups[0]
    assert pg["betas"]        == (0.85, 0.95)
    assert pg["eps"]          == 1e-7
    assert pg["weight_decay"] == 0.05
    assert pg["lr"]           == 0.01


def test_build_optimizer_does_not_override_explicit_group_values():
    model = torch.nn.Linear(4, 2)
    shim  = _shim(model, weight_decay=0.99)

    param_groups = [{"params": list(model.parameters()), "lr": 0.01, "weight_decay": 0.0}]
    optimizer    = Trainer._build_optimizer(shim, param_groups)

    assert optimizer.param_groups[0]["weight_decay"] == 0.0


@pytest.mark.parametrize("kind, expected", [("adamw", torch.optim.AdamW), ("adam", torch.optim.Adam), ("sgd", torch.optim.SGD), ("rmsprop", torch.optim.RMSprop)])
def test_build_optimizer_dispatches_on_the_configured_type(kind, expected):
    model                      = torch.nn.Linear(4, 2)
    shim                       = _shim(model)
    shim.config.optimizer.type = kind

    optimizer = Trainer._build_optimizer(shim, [{"params": list(model.parameters()), "lr": 0.01}])

    assert type(optimizer) is expected


def test_build_optimizer_fails_on_an_unknown_type():
    model                      = torch.nn.Linear(4, 2)
    shim                       = _shim(model)
    shim.config.optimizer.type = "lamb"

    with pytest.raises(KeyError):
        Trainer._build_optimizer(shim, [{"params": list(model.parameters()), "lr": 0.01}])


def test_update_optimizer_sets_lrs_and_logs():
    model  = torch.nn.Linear(4, 2)
    shim   = _shim(model)
    logged = []

    param_groups     = [{"params": list(model.parameters()), "lr": 0.01, "name": "main"}]
    shim.optimizer   = Trainer._build_optimizer(shim, param_groups)
    shim.tracker     = SimpleNamespace(log_scalar=lambda tag, value, step: logged.append((tag, value)))
    shim.global_step = 5

    Trainer._update_optimizer(shim, [0.123])

    assert shim.optimizer.param_groups[0]["lr"] == 0.123
    assert ("optim/lr/main", 0.123) in logged


def test_capture_state_contains_the_epoch_and_the_model():
    model = torch.nn.Linear(3, 1)
    shim  = TrainerShim()
    shim.model = model

    state = Trainer.capture_state(shim, epoch=7)

    assert set(state) == {"epoch", "params"}
    assert state["epoch"] == 7
    assert set(state["params"].keys()) == set(model.state_dict().keys())


def test_clear_cuda_cache_runs_on_cpu():
    Trainer._clear_cuda_cache(TrainerShim())


def test_cpu_device_is_warned_with_the_mask_and_the_amp_consequence(recording_logger, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")

    shim        = TrainerShim()
    shim.device = torch.device("cpu")
    shim.logger = recording_logger
    shim.config = SimpleNamespace(training=SimpleNamespace(use_amp=True))

    Trainer._warn_cpu_device(shim)

    [(level, message)] = recording_logger.messages

    assert level == "warning"
    assert "CUDA_VISIBLE_DEVICES=3" in message
    assert "bfloat16 on the CPU" in message


def test_cuda_device_is_not_warned(recording_logger):
    shim        = TrainerShim()
    shim.device = torch.device("cuda")
    shim.logger = recording_logger

    Trainer._warn_cpu_device(shim)

    assert recording_logger.messages == []


def test_param_group_table_separates_the_base_rate_from_the_warmup_start_rate():
    tables = []
    model  = torch.nn.Linear(4, 2)
    shim   = _shim(model)

    shim.base_lrs  = [0.01]
    shim.optimizer = Trainer._build_optimizer(shim, [{"params": list(model.parameters()), "lr": 0.001, "name": "main"}])
    shim.logger    = SimpleNamespace(metrics_table=lambda rows, columns, title: tables.append((rows, columns, title)))

    Trainer._log_param_groups(shim)

    [(rows, columns, title)] = tables

    assert columns == ["Group", "Base LR", "Step-0 LR", "Weight decay"]
    assert rows[0]["Base LR"]   == "0.01"
    assert rows[0]["Step-0 LR"] == "0.001"
    assert "warmup start factor" in title


def test_evaluate_without_a_finite_batch_warns_that_the_loss_is_nan(recording_logger):
    shim = TrainerShim()

    shim.model                   = torch.nn.Linear(2, 1)
    shim.logger                  = recording_logger
    shim.epochs                  = 4
    shim.device                  = torch.device("cpu")
    shim.use_amp                 = False
    shim.abort_on_nonfinite_loss = False
    shim._apply_memory_format    = lambda batch: batch
    shim._compute_loss           = lambda batch: {"total_loss": WeightedMean(torch.tensor(float("nan")), torch.tensor(1.0)), "channels": {"y": WeightedMean(torch.tensor(float("nan")), torch.tensor(1.0))}}
    shim._log_epoch_metrics      = lambda stage, aggregator, epoch: None
    shim._nonfinite_detail       = lambda loss_dict, batch: Trainer._nonfinite_detail(shim, loss_dict, batch)

    batch  = (torch.zeros(1, 2), torch.zeros(1, 2))
    result = Trainer.evaluate(shim, [batch, batch], epoch=2, stage="val")

    assert result["num_batches"] == 0
    assert result["avg_loss"] != result["avg_loss"]
    assert ("warning", "Training val epoch 3: no finite batch out of 2; the val loss is NaN") in recording_logger.messages
    assert any(level == "warning" and "non-finite target channels: ['y']" in message for level, message in recording_logger.messages)


def test_nonfinite_detail_says_when_every_channel_loss_is_finite():
    batch  = (torch.zeros(1, 2), torch.zeros(1, 2))
    detail = Trainer._nonfinite_detail(TrainerShim(), {"channels": {"y": WeightedMean(torch.tensor(1.0), torch.tensor(1.0))}}, batch)

    assert detail.startswith("; non-finite target channels: none (every channel loss is finite)")
