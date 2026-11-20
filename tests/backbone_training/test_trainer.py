
from __future__ import annotations

import json

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from configuration.training.general.loss import LossConfig
from pipelines.backbone.training.loss    import Loss
from pipelines.backbone.training.trainer import Trainer
from tools.training.checkpoint            import WeightSwa

from tests.backbone_training._helpers import identity_normalizer, tiny_model, tiny_trainer_config, x_axis_numpy
from tests.conftest                   import RecordingWriter

from tools.monitoring.logger import Logger


pytestmark = pytest.mark.usefixtures("force_cpu")


def _loader(in_channels: int = 2, n_gaussians: int = 2, n: int = 6, hw: int = 16) -> DataLoader:
    gen  = torch.Generator().manual_seed(0)
    imgs = torch.randn(n, in_channels, hw, hw, generator=gen)
    tgt  = torch.randn(n, n_gaussians * 3, hw, hw, generator=gen)

    for i, amp in enumerate((1.0, 5.0, 10.0)):
        tgt[0, :, i + 1, i + 1]   = 0.0
        tgt[0, 0:3, i + 1, i + 1] = torch.tensor([amp, 30.0, 5.0])

    for i, mu2 in enumerate((26.0, 28.0, 29.0)):
        tgt[1, 0:3, i + 1, i + 1] = torch.tensor([3.0, 20.0, 4.0])
        tgt[1, 3:6, i + 1, i + 1] = torch.tensor([3.0, mu2, 4.0])

    for i, amp2 in enumerate((5.0, 8.0, 9.0)):
        tgt[2, 0:3, i + 1, i + 1] = torch.tensor([2.0, 0.0, 3.0])
        tgt[2, 3:6, i + 1, i + 1] = torch.tensor([amp2, 60.0, 3.0])

    for i, mu2 in enumerate((22.0, 34.0, 53.0)):
        tgt[3, 0:3, i + 1, i + 1] = torch.tensor([3.0, 5.0, 3.0])
        tgt[3, 3:6, i + 1, i + 1] = torch.tensor([3.0, mu2, 3.0])

    return DataLoader(TensorDataset(imgs, tgt), batch_size=2)


def _build_trainer(tmp_path, epochs: int = 1, n_gaussians: int = 2, resume: bool = False, level: str = "ERROR") -> Trainer:
    model, model_cfg = tiny_model(in_channels=2, n_gaussians=n_gaussians)
    config           = tiny_trainer_config(n_gaussians=n_gaussians, epochs=epochs)
    logger           = Logger(log_dir=str(tmp_path / "logs"), name="trainer", level=level)
    norm_stats       = identity_normalizer(n_gaussians * 3)

    config.training.resume = resume

    trainer = Trainer(model, model_cfg, x_axis_numpy(), config, tmp_path, logger, norm_stats=norm_stats, emit_docs=False)
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

    assert "total_loss" in out
    assert torch.isfinite(out["total_loss"]).item()


def test_one_epoch_fit_writes_checkpoint(tmp_path):
    trainer = _build_trainer(tmp_path, epochs=1)
    loader  = _loader()

    train_losses, val_losses, best_val = trainer.train(loader, loader, loader)

    assert len(train_losses) == 1
    assert (tmp_path / "best_model.pt").exists()
    assert torch.isfinite(torch.tensor(best_val)).item()


def test_trainer_builds_the_configured_optimizer_family(tmp_path):
    model, model_cfg = tiny_model(in_channels=2, n_gaussians=2)
    config           = tiny_trainer_config(n_gaussians=2, epochs=1)

    config.optimizer.type     = "sgd"
    config.optimizer.momentum = 0.8

    logger  = Logger(log_dir=str(tmp_path / "logs"), name="trainer_sgd", level="ERROR")
    trainer = Trainer(model, model_cfg, x_axis_numpy(), config, tmp_path, logger, norm_stats=identity_normalizer(6), emit_docs=False)

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
    model, model_cfg = tiny_model(in_channels=2, n_gaussians=2)
    config           = tiny_trainer_config(n_gaussians=2, epochs=1)
    logger           = Logger(log_dir=str(tmp_path / "logs"), name="trainer", level="ERROR")
    norm_stats       = identity_normalizer(6)
    trainer          = Trainer(model, model_cfg, x_axis_numpy(), config, tmp_path, logger, norm_stats=norm_stats, emit_docs=False, writer=writer)

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

    assert len(writer.figures) == 12
    assert all(tag.startswith("reconstruction/") and tag.endswith("/val") for tag, *_ in writer.figures)

    assert len(writer.layouts) == 1
    assert list(writer.layouts[0]) == ["Loss", "Optimiser", "Diagnostics", "Slots"]
    assert "Active fraction per slot" in writer.layouts[0]["Slots"]


def _loss_dict(total: float) -> dict:
    return {"total_loss": torch.tensor(total), "components": {}, "monitor": {}, "occupancy": {}, "physical": {}}


def _diverged_loss_dict(bad_term: str) -> dict:
    return {"total_loss": torch.tensor(float("nan")), "components": {"param_l1": torch.tensor(0.5), bad_term: torch.tensor(float("nan"))}, "monitor": {}, "occupancy": {}, "physical": {}}


def test_evaluate_raises_on_nonfinite_loss(tmp_path):
    trainer = _build_trainer(tmp_path)
    loader  = _loader()

    trainer._eval_step = lambda batch: _loss_dict(float("nan"))

    with pytest.raises(FloatingPointError):
        trainer.evaluate(loader, epoch=0)


def test_evaluate_skips_nonfinite_batches_when_abort_disabled(tmp_path):
    trainer = _build_trainer(tmp_path)
    trainer.abort_on_nonfinite_loss = False

    loader  = _loader()
    batches = iter(range(len(loader)))

    trainer._eval_step = lambda batch: _loss_dict(float("inf") if next(batches) == 0 else 2.0)

    result = trainer.evaluate(loader, epoch=0)

    assert result["num_batches"] == len(loader) - 1
    assert result["avg_loss"]    == pytest.approx(2.0)


def test_trainer_channels_last_converts_model_and_batches(tmp_path):
    model, model_cfg = tiny_model(in_channels=2, n_gaussians=2)
    config           = tiny_trainer_config(n_gaussians=2, epochs=1)

    config.training.channels_last = True

    logger  = Logger(log_dir=str(tmp_path / "logs"), name="trainer_cl", level="ERROR")
    trainer = Trainer(model, model_cfg, x_axis_numpy(), config, tmp_path, logger, norm_stats=identity_normalizer(6), emit_docs=False)
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

    model, model_cfg = tiny_model(in_channels=2, n_gaussians=2)
    config           = tiny_trainer_config(n_gaussians=2, epochs=1)

    config.training.compile_model = True
    config.training.compile_mode  = "reduce-overhead"

    logger  = Logger(log_dir=str(tmp_path / "logs"), name="trainer_compile", level="ERROR")
    trainer = Trainer(model, model_cfg, x_axis_numpy(), config, tmp_path, logger, norm_stats=identity_normalizer(6), emit_docs=False)

    assert recorded["mode"] == "reduce-overhead"
    assert trainer.model is not None


def test_trainer_activation_checkpointing_wraps_and_trains(tmp_path):
    model, model_cfg = tiny_model(in_channels=2, n_gaussians=2)
    reference_keys   = sorted(model.state_dict())
    config           = tiny_trainer_config(n_gaussians=2, epochs=1)

    config.training.activation_checkpointing = ("encoder_blocks", "decoder_blocks")

    logger  = Logger(log_dir=str(tmp_path / "logs"), name="trainer_ckpt", level="ERROR")
    trainer = Trainer(model, model_cfg, x_axis_numpy(), config, tmp_path, logger, norm_stats=identity_normalizer(6), emit_docs=False)
    trainer.warmup.resolve(200)

    trainer.train_epoch(_loader(), epoch=0)

    assert sorted(trainer.model.state_dict()) == reference_keys


def test_evaluate_weights_batches_by_sample_count(tmp_path):
    trainer = _build_trainer(tmp_path)

    gen    = torch.Generator().manual_seed(0)
    imgs   = torch.randn(6, 2, 16, 16, generator=gen)
    tgt    = torch.randn(6, 6, 16, 16, generator=gen)
    loader = DataLoader(TensorDataset(imgs, tgt), batch_size=4)

    trainer._eval_step = lambda batch: _loss_dict(1.0 if batch[0].shape[0] == 4 else 4.0)

    result = trainer.evaluate(loader, epoch=0)

    assert result["num_batches"] == 2
    assert result["num_samples"] == 6
    assert result["avg_loss"]    == pytest.approx((1.0 * 4 + 4.0 * 2) / 6)


def test_evaluate_returns_nan_when_all_batches_nonfinite(tmp_path):
    trainer = _build_trainer(tmp_path)
    trainer.abort_on_nonfinite_loss = False

    trainer._eval_step = lambda batch: _loss_dict(float("nan"))

    result = trainer.evaluate(_loader(), epoch=0)

    assert result["num_batches"] == 0
    assert result["avg_loss"]    != result["avg_loss"]


def test_nonfinite_train_loss_abort_names_the_term_and_clears_the_data(tmp_path):
    trainer = _build_trainer(tmp_path)

    trainer._compute_loss = lambda batch: _diverged_loss_dict("mse_curve")

    with pytest.raises(FloatingPointError, match=r"non-finite terms: \['mse_curve'\]; inputs and targets finite"):
        trainer.train_epoch(_loader(), epoch=0)


def test_nonfinite_train_loss_abort_names_corrupt_inputs(tmp_path):
    trainer = _build_trainer(tmp_path)
    loader  = _loader()

    loader.dataset.tensors[0][0, 0, 0, 0] = float("nan")
    trainer._compute_loss                 = lambda batch: _loss_dict(float("nan"))

    with pytest.raises(FloatingPointError, match=r"non-finite terms: none \(every weighted term is finite\); non-finite inputs in the batch"):
        trainer.train_epoch(loader, epoch=0)


def test_evaluate_skip_warning_names_the_nonfinite_term(tmp_path):
    model, model_cfg = tiny_model(in_channels=2, n_gaussians=2)
    config           = tiny_trainer_config(n_gaussians=2, epochs=1)
    logger           = Logger(log_dir=str(tmp_path / "logs"), name="trainer_warn", level="WARNING")
    trainer          = Trainer(model, model_cfg, x_axis_numpy(), config, tmp_path, logger, norm_stats=identity_normalizer(6), emit_docs=False)

    trainer.abort_on_nonfinite_loss = False
    trainer._eval_step              = lambda batch: _diverged_loss_dict("cosine_curve")

    result = trainer.evaluate(_loader(), epoch=0)
    logger.close()

    text = (tmp_path / "logs" / "trainer_warn.log").read_text(encoding="utf-8")

    assert result["num_batches"] == 0
    assert "val loss is non-finite at epoch 1; non-finite terms: ['cosine_curve']; inputs and targets finite" in text


def test_checkpoint_save_restore_round_trip(tmp_path):
    trainer = _build_trainer(tmp_path, epochs=1)
    loader  = _loader()

    trainer.train(loader, loader, loader)

    saved = torch.load(tmp_path / "best_model.pt", map_location="cpu", weights_only=False)

    assert "params"        in saved
    assert "z_axis"        in saved
    assert "best_val_loss" in saved

    fresh_model, model_cfg = tiny_model(in_channels=2, n_gaussians=2)
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


def test_slot_diagnostics_adopt_the_label_run_threshold(tmp_path):
    model, model_cfg = tiny_model(in_channels=2, n_gaussians=2)
    config           = tiny_trainer_config(n_gaussians=2, epochs=1)

    config.gaussian.activity_threshold = 2.5e-3
    config.loss                        = LossConfig(use_param_l1=True, weight_param_l1=1.0, use_active_normalization=True)

    logger  = Logger(log_dir=str(tmp_path / "logs"), name="thr", level="ERROR")
    trainer = Trainer(model, model_cfg, x_axis_numpy(), config, tmp_path, logger, norm_stats=identity_normalizer(6), emit_docs=False)

    trainer.tracker.writer = object()
    val_loader             = _loader()

    trainer._before_validation(0, val_loader)

    assert trainer.param_sampler.thr == pytest.approx(2.5e-3)
    assert trainer.slot_vitals.thr   == pytest.approx(2.5e-3)
