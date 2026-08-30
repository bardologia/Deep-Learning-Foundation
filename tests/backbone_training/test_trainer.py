"""Tests for the backbone Trainer and its curriculum controller.

Covers component wiring and named parameter groups, a single optimisation step, one- and
multi-epoch fits, TensorBoard throughput and diagnostic scalars, non-finite handling in
evaluation, checkpoint save and restore, early-stopping and scheduler resets, and the
curriculum swap with its learning-rate and diagnostic-threshold effects.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from configuration.training.general.loss import LossConfig
from pipelines.backbone.training.loss    import Loss
from pipelines.backbone.training.trainer import Trainer
from pipelines.shared.training.curriculum import CurriculumController

from tests.backbone_training._helpers import identity_normalizer, tiny_model, tiny_trainer_config, x_axis_numpy

from tools.monitoring.logger import Logger


pytestmark = pytest.mark.usefixtures("force_cpu")


def _loader(in_channels: int = 2, n_gaussians: int = 2, n: int = 6, hw: int = 16) -> DataLoader:
    """Returns a loader of random (n, in_channels, hw, hw) inputs and (n, 3*n_gaussians, hw, hw) targets with a few hand-set pixels covering amplitude, mu separation and slot-order cases."""
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


def _build_trainer(tmp_path, epochs: int = 1, n_gaussians: int = 2) -> Trainer:
    """Returns a Trainer over a tiny backbone with an identity output normalizer."""
    model, model_cfg = tiny_model(in_channels=2, n_gaussians=n_gaussians)
    config           = tiny_trainer_config(n_gaussians=n_gaussians, epochs=epochs)
    logger           = Logger(log_dir=str(tmp_path / "logs"), name="trainer", level="ERROR")
    norm_stats       = identity_normalizer(n_gaussians * 3)

    trainer = Trainer(model, model_cfg, x_axis_numpy(), config, tmp_path, logger, norm_stats=norm_stats, emit_docs=False)
    trainer.warmup.resolve(200)
    return trainer


def test_trainer_builds_on_cpu_with_wired_components(tmp_path):
    """Verifies the trainer lands on CPU with criterion, optimizer, scheduler, stopper, warmup and curriculum controller wired."""
    trainer = _build_trainer(tmp_path)

    assert trainer.device.type == "cpu"
    assert isinstance(trainer.criterion, Loss)
    assert trainer.optimizer      is not None
    assert trainer.lr_scheduler   is not None
    assert trainer.early_stopping is not None
    assert trainer.warmup         is not None
    assert isinstance(trainer.curriculum_controller, CurriculumController)


def test_trainer_optimizer_has_named_param_groups(tmp_path):
    """Verifies the optimizer carries named encoder and output-head groups with a base learning rate each."""
    trainer = _build_trainer(tmp_path)
    names   = {group.get("name") for group in trainer.optimizer.param_groups}

    assert "encoder"     in names
    assert "output_head" in names
    assert len(trainer.base_lrs) == len(trainer.optimizer.param_groups)


def test_single_train_step_updates_parameters(tmp_path):
    """Verifies one training epoch returns a finite average loss and changes model parameters."""
    trainer = _build_trainer(tmp_path)
    loader  = _loader()

    before  = [p.detach().clone() for p in trainer.model.parameters()]
    avg     = trainer.train_epoch(loader, epoch=0)
    after   = list(trainer.model.parameters())

    assert isinstance(avg, float)
    assert torch.isfinite(torch.tensor(avg)).item()
    assert any(not torch.equal(a, b) for a, b in zip(before, after))


def test_compute_loss_returns_finite_total(tmp_path):
    """Verifies the per-batch loss computation returns a finite total."""
    trainer = _build_trainer(tmp_path)
    batch   = next(iter(_loader()))

    out     = trainer._compute_loss(batch)

    assert "total_loss" in out
    assert torch.isfinite(out["total_loss"]).item()


def test_one_epoch_fit_writes_checkpoint(tmp_path):
    """Verifies a one-epoch fit writes the best checkpoint and a finite best validation loss."""
    trainer = _build_trainer(tmp_path, epochs=1)
    loader  = _loader()

    train_losses, val_losses, best_val = trainer.train(loader, loader, loader)

    assert len(train_losses) == 1
    assert (tmp_path / "best_model.pt").exists()
    assert torch.isfinite(torch.tensor(best_val)).item()


def test_train_writes_final_model_checkpoint(tmp_path):
    """Verifies training always writes the final-epoch weights to final_model.pt with the same schema as best_model.pt."""
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
    """Verifies a three-epoch fit records one finite training loss per epoch and at least one validation loss."""
    trainer = _build_trainer(tmp_path, epochs=3)
    loader  = _loader()

    train_losses, val_losses, _ = trainer.train(loader, loader, loader)

    assert len(train_losses) == 3
    assert all(torch.isfinite(torch.tensor(loss)).item() for loss in train_losses)
    assert len(val_losses) >= 1


class RecordingWriter:
    """Minimal TensorBoard writer stub that records the scalars, histograms and figures it receives.

    Attributes:
        scalars    : Recorded (tag, value, step) triples.
        histograms : Recorded (tag, values, step) triples.
        figures    : Recorded (tag, figure, step) triples.
    """
    def __init__(self):
        """Initialises empty scalar, histogram and figure records."""
        self.scalars    = []
        self.histograms = []
        self.figures    = []

    def add_scalar(self, tag, value, step):
        """Records a scalar entry."""
        self.scalars.append((tag, value, step))

    def add_histogram(self, tag, values, step, bins="auto"):
        """Records a histogram entry."""
        self.histograms.append((tag, values, step))

    def add_figure(self, tag, figure, step, close=True):
        """Records a figure entry."""
        self.figures.append((tag, figure, step))

    def flush(self):
        """Accepts the flush call without doing anything."""
        pass

    def close(self):
        """Accepts the close call without doing anything."""
        pass


def test_train_logs_throughput_and_diagnostics(tmp_path):
    """Verifies training emits throughput, non-finite-batch and gradient-norm scalars plus twelve validation reconstruction figures."""
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
    assert all(tag.startswith("reconstruction/") and tag.endswith("/val") for tag, _, _ in writer.figures)


def _loss_dict(total: float) -> dict:
    """Returns a loss output dict carrying only the given scalar total."""
    return {"total_loss": torch.tensor(total), "components": {}, "monitor": {}, "occupancy": {}, "physical": {}}


def test_evaluate_raises_on_nonfinite_loss(tmp_path):
    """Verifies evaluation aborts with FloatingPointError on a non-finite batch loss."""
    trainer = _build_trainer(tmp_path)
    loader  = _loader()

    trainer._eval_step = lambda batch: _loss_dict(float("nan"))

    with pytest.raises(FloatingPointError):
        trainer.evaluate(loader, epoch=0)


def test_evaluate_skips_nonfinite_batches_when_abort_disabled(tmp_path):
    """Verifies evaluation skips non-finite batches and averages the rest when aborting is disabled."""
    trainer = _build_trainer(tmp_path)
    trainer.abort_on_nonfinite_loss = False

    loader  = _loader()
    batches = iter(range(len(loader)))

    trainer._eval_step = lambda batch: _loss_dict(float("inf") if next(batches) == 0 else 2.0)

    result = trainer.evaluate(loader, epoch=0)

    assert result["num_batches"] == len(loader) - 1
    assert result["avg_loss"]    == pytest.approx(2.0)


def test_evaluate_returns_nan_when_all_batches_nonfinite(tmp_path):
    """Verifies evaluation reports zero batches and a NaN average when every batch is non-finite."""
    trainer = _build_trainer(tmp_path)
    trainer.abort_on_nonfinite_loss = False

    trainer._eval_step = lambda batch: _loss_dict(float("nan"))

    result = trainer.evaluate(_loader(), epoch=0)

    assert result["num_batches"] == 0
    assert result["avg_loss"]    != result["avg_loss"]


def test_checkpoint_save_restore_round_trip(tmp_path):
    """Verifies the best checkpoint stores params, x axis and best loss, and reloads into a fresh model."""
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
    """Verifies the best checkpoint's epoch equals its best epoch while the last checkpoint holds the final epoch index."""
    trainer = _build_trainer(tmp_path, epochs=2)
    loader  = _loader()

    trainer.train(loader, loader, loader)

    best = torch.load(tmp_path / "best_model.pt", map_location="cpu", weights_only=False)
    last = torch.load(tmp_path / "last.pt",       map_location="cpu", weights_only=False)

    assert best["epoch"] == best["best_epoch"]
    assert best["epoch"] in (0, 1)
    assert last["epoch"] == 1


def test_restore_best_loads_best_parameters(tmp_path):
    """Verifies the trainer's weights after fitting match the saved best checkpoint."""
    trainer = _build_trainer(tmp_path, epochs=2)
    loader  = _loader()

    trainer.train(loader, loader, loader)

    saved = torch.load(tmp_path / "best_model.pt", map_location="cpu", weights_only=False)

    for name, param in trainer.model.state_dict().items():
        assert torch.equal(param, saved["params"][name])


def test_early_stopping_reset_clears_state(tmp_path):
    """Verifies resetting early stopping clears the best loss, counter and trigger flag."""
    trainer = _build_trainer(tmp_path)

    trainer.early_stopping(val_loss=1.0, epoch=0)
    trainer.early_stopping.reset()

    assert trainer.early_stopping.best_loss is None
    assert trainer.early_stopping.counter   == 0
    assert trainer.early_stopping.triggered is False


def test_scheduler_reset_restores_base_lrs(tmp_path):
    """Verifies resetting the scheduler returns the current learning rates to the base rates."""
    trainer = _build_trainer(tmp_path)

    trainer.lr_scheduler.step(epoch=5)
    trainer.lr_scheduler.reset(epoch_offset=0)

    assert trainer.lr_scheduler.current_lrs == trainer.lr_scheduler.base_lrs


def test_curriculum_swap_disabled_is_noop(tmp_path):
    """Verifies a disabled curriculum never swaps and leaves the loss generation at zero."""
    trainer = _build_trainer(tmp_path)

    swapped = trainer.curriculum_controller.maybe_swap(epoch=0)

    assert swapped is False
    assert trainer.criterion.loss_generation == 0


def test_curriculum_swap_replaces_loss_config(tmp_path):
    """Verifies reaching the swap epoch installs the complete loss config and bumps the loss generation."""
    model, model_cfg = tiny_model(in_channels=2, n_gaussians=2)
    config           = tiny_trainer_config(n_gaussians=2, epochs=1)

    config.curriculum.enabled    = True
    config.curriculum.swap_epoch = 0
    config.curriculum.complete   = LossConfig(use_mse_curve=True, weight_mse_curve=1.0)

    logger     = Logger(log_dir=str(tmp_path / "logs"), name="curr", level="ERROR")
    norm_stats = identity_normalizer(6)
    trainer    = Trainer(model, model_cfg, x_axis_numpy(), config, tmp_path, logger, norm_stats=norm_stats, emit_docs=False)

    swapped = trainer.curriculum_controller.maybe_swap(epoch=0)

    assert swapped is True
    assert trainer.criterion.loss_generation == 1


def test_slot_diagnostics_follow_the_curriculum_threshold(tmp_path):
    """Verifies the parameter sampler and slot vitals adopt the amplitude-zero threshold of the active stage."""
    model, model_cfg = tiny_model(in_channels=2, n_gaussians=2)
    config           = tiny_trainer_config(n_gaussians=2, epochs=1)

    config.curriculum.enabled    = True
    config.curriculum.swap_epoch = 0
    config.curriculum.warmup     = LossConfig(use_param_l1=True, weight_param_l1=1.0, amp_zero_thr=1e-4)
    config.curriculum.complete   = LossConfig(use_param_l1=True, weight_param_l1=1.0, amp_zero_thr=1e-1)

    logger  = Logger(log_dir=str(tmp_path / "logs"), name="thr", level="ERROR")
    trainer = Trainer(model, model_cfg, x_axis_numpy(), config, tmp_path, logger, norm_stats=identity_normalizer(6), emit_docs=False)

    trainer.tracker.writer = object()
    val_loader             = _loader()

    trainer._before_validation(0, val_loader)

    assert trainer.param_sampler.thr == pytest.approx(1e-4)
    assert trainer.slot_vitals.thr   == pytest.approx(1e-4)

    trainer.curriculum_controller.maybe_swap(epoch=0)
    trainer._before_validation(1, val_loader)

    assert trainer.param_sampler.thr == pytest.approx(1e-1)
    assert trainer.slot_vitals.thr   == pytest.approx(1e-1)


def test_curriculum_swap_applies_the_scheduled_learning_rate(tmp_path):
    """Verifies a swap without an LR reset keeps the scheduler's decayed learning rates."""
    model, model_cfg = tiny_model(in_channels=2, n_gaussians=2)
    config           = tiny_trainer_config(n_gaussians=2, epochs=4)

    config.curriculum.enabled      = True
    config.curriculum.swap_epoch   = 2
    config.curriculum.reset_lr     = False
    config.curriculum.reset_warmup = True
    config.warmup.warmup_enabled   = False

    logger  = Logger(log_dir=str(tmp_path / "logs"), name="swaplr", level="ERROR")
    trainer = Trainer(model, model_cfg, x_axis_numpy(), config, tmp_path, logger, norm_stats=identity_normalizer(6), emit_docs=False)

    decayed = trainer.lr_scheduler.step(epoch=2)

    trainer.curriculum_controller.maybe_swap(epoch=2)

    assert [group["lr"] for group in trainer.optimizer.param_groups] == pytest.approx(decayed)
    assert decayed != pytest.approx(trainer.lr_scheduler.base_lrs)
