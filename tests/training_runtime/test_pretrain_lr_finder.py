"""Tests for the pre-training LR range test.

Covers the rewind that makes every rung start from the same state, the paired
measurement batches, the minimum-loss selection over the safety factor, the
edge and noise refusals that keep a non-measurement out of the training config,
and the validation of the ladder definition.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch.utils.data import TensorDataset

from tools.training.pretraining.lr_finder import LrRangeTest, TrainerStateSnapshot


class _Logger:
    """Logger stub recording the subsection lines and table titles of the range test.

    Attributes:
        lines  : Subsection messages in emission order.
        tables : (title, rows) pairs of the metrics tables in emission order.
    """
    def __init__(self):
        """Starts with no recorded output."""
        self.lines  = []
        self.tables = []

    def subsection(self, message, *args, **kwargs):
        """Records a subsection line."""
        self.lines.append(message)

    def metrics_table(self, rows, columns=None, title=None):
        """Records a metrics table's title and rows."""
        self.tables.append((title, list(rows)))


class _Clipper:
    """Gradient-clipper stub recording norms without ever rescaling a gradient.

    Attributes:
        history : Gradient norms recorded so far.
        raises  : Whether maybe_clip reports a non-finite gradient norm.
    """

    def __init__(self, raises: bool = False) -> None:
        """Builds an empty history and stores whether the clipper should fail."""
        self.history = []
        self.raises  = bool(raises)

    def maybe_clip(self, model, global_step: int) -> float:
        """Returns the global gradient norm, raising when the stub is set to fail."""
        if self.raises:
            raise FloatingPointError(f"Gradient norm is non-finite at step {global_step}")

        total = sum(float(p.grad.pow(2).sum()) for p in model.parameters() if p.grad is not None)

        return math.sqrt(total)

    def record(self, grad_norm_value: float, global_step: int) -> None:
        """Appends a gradient norm to the history."""
        self.history.append(float(grad_norm_value))

    def state_dict(self) -> dict:
        """Returns the recorded gradient-norm history."""
        return {"history": list(self.history)}

    def load_state_dict(self, state: dict) -> None:
        """Restores the recorded gradient-norm history."""
        self.history = list(state["history"])


class _StubTrainer:
    """Trainer stub exposing the surface the range test drives.

    Implements ``micro_step`` and ``apply_gradients`` exactly as ``BaseTrainer``
    does, so the probe under test runs the same shape of step it runs in
    production.

    Attributes:
        model              : Module holding the single fitted parameter.
        optimizer          : SGD optimizer over that parameter.
        base_lrs           : Warmup-free base learning rate of the single group.
        use_amp            : Always False, keeping the probe on the plain forward path.
        accumulation_steps : Micro-batches per optimizer step.
        grad_clipper       : Clipper stub the probe snapshots and restores.
        optimizer_steps    : Optimizer steps taken, counted outside the snapshotted state.
        batches            : Batches seen by micro_step, in order.
    """

    def __init__(self, base_lr: float, start: float = 10.0, warmup_factor: float = 1.0, accumulation_steps: int = 1, clipper_raises: bool = False) -> None:
        """Builds the parameter, its optimizer and the clipper stub."""
        self.model     = torch.nn.Linear(1, 1, bias=False)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=base_lr * warmup_factor)

        with torch.no_grad():
            self.model.weight.fill_(float(start))

        self.base_lrs           = [float(base_lr)]
        self.use_amp            = False
        self.accumulation_steps = int(accumulation_steps)
        self.grad_clipper       = _Clipper(raises=clipper_raises)

        self.optimizer_steps = 0
        self.batches         = []

    @property
    def weight(self) -> float:
        """Returns the fitted scalar as a float."""
        return float(self.model.weight.detach().reshape(()))

    def _compute_loss(self, batch) -> dict:
        """Returns the mean squared distance of the weight to the batch targets."""
        return {"total_loss": ((self.model.weight.reshape(()) - batch[0]) ** 2).mean()}

    def _eval_step(self, batch) -> dict:
        """Returns the loss dictionary of one measurement batch."""
        return self._compute_loss(batch)

    def micro_step(self, batch, window_len: int) -> tuple[dict, float]:
        """Runs one micro-batch, scales its loss by the window and backpropagates it."""
        self.batches.append(batch)

        loss_dict = self._compute_loss(batch)
        loss      = loss_dict["total_loss"] / window_len

        if not torch.isfinite(loss):
            return loss_dict, float("nan")

        loss.backward()

        return loss_dict, float(loss.detach())

    def apply_gradients(self, global_step: int) -> float:
        """Clips, steps and clears the gradients, counting the step outside the snapshot."""
        self.optimizer_steps += 1

        grad_norm = self.grad_clipper.maybe_clip(self.model, global_step)
        self.grad_clipper.record(grad_norm, global_step)

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        return grad_norm


class _FlatTrainer(_StubTrainer):
    """Trainer stub whose loss never moves, so no rung can beat the first."""

    def _compute_loss(self, batch) -> dict:
        """Returns a constant loss still attached to the graph."""
        return {"total_loss": self.model.weight.reshape(()) * 0.0 + 1.0}


class _NanTrainer(_StubTrainer):
    """Trainer stub returning a non-finite loss on every step."""

    def _compute_loss(self, batch) -> dict:
        """Returns a NaN loss attached to the graph."""
        return {"total_loss": self.model.weight.reshape(()) * 0.0 + float("nan")}


class _ResponseTrainer(_StubTrainer):
    """Trainer stub whose measured loss is a known function of the applied rate.

    The loss is a parabola in log-multiplier with its minimum at ``optimum``,
    plus per-batch noise that does not depend on the rate. It is the shape a
    working range test must recover and the shape the previous selector lost to
    noise at the bottom of the ladder.

    Attributes:
        optimum : Multiplier the loss is minimised at.
        noise   : Amplitude of the per-batch noise added to every measurement.
    """

    def __init__(self, base_lr: float, optimum: float = 10.0, noise: float = 0.0, seed: int = 0, **kwargs) -> None:
        """Stores the response shape and seeds the noise generator."""
        super().__init__(base_lr=base_lr, **kwargs)

        self.optimum   = float(optimum)
        self.noise     = float(noise)
        self.generator = torch.Generator().manual_seed(int(seed))

    def _compute_loss(self, batch) -> dict:
        """Returns the parabolic response at the current group rate plus per-batch noise."""
        rate       = self.optimizer.param_groups[0]["lr"]
        multiplier = max(rate / self.base_lrs[0], 1e-12)
        offset     = math.log(multiplier / self.optimum)
        draw       = float(torch.randn(1, generator=self.generator)) * self.noise

        return {"total_loss": self.model.weight.reshape(()) * 0.0 + 1.0 + offset ** 2 + draw}


def _dataset(n: int = 512) -> TensorDataset:
    """Returns a dataset of fixed unit-variance targets the probe loader draws from."""
    generator = torch.Generator().manual_seed(7)
    return TensorDataset(torch.randn(n, 1, generator=generator))


def _test(trainer, **overrides) -> LrRangeTest:
    """Builds a range test over the stub trainer with short-ladder defaults."""
    settings = dict(
        trainer           = trainer,
        dataset           = _dataset(),
        device            = torch.device("cpu"),
        logger            = _Logger(),
        scale_min         = 1e-2,
        scale_max         = 1e2,
        rungs             = 40,
        rung_steps        = 2,
        probe_batches     = 3,
        divergence_factor = 4.0,
        safety_factor     = 10.0,
        min_improvement   = 0.02,
        noise_sigmas      = 3.0,
        seed              = 42,
    )
    settings.update(overrides)
    return LrRangeTest(**settings)


def test_selection_is_the_minimum_loss_rung_over_the_safety_factor():
    """Verifies the pick is the loss-minimum multiplier divided by the safety factor."""
    trainer = _ResponseTrainer(base_lr=0.05, optimum=10.0)
    result  = _test(trainer).run(batch_size=4)

    assert result is not None
    assert result["multiplier"] == pytest.approx(result["min_loss_multiplier"] / 10.0)
    assert result["min_loss_multiplier"] == pytest.approx(10.0, rel=0.15)
    assert len(result["losses"])  == result["rungs_run"]
    assert len(result["spreads"]) == result["rungs_run"]


def test_noise_does_not_pull_the_pick_to_the_ramp_floor():
    """Verifies per-batch noise cannot make the first rungs win, across several seeds."""
    ladder = _test(_ResponseTrainer(base_lr=0.05))._multipliers()
    floor  = ladder[2]

    for seed in range(6):
        trainer = _ResponseTrainer(base_lr=0.05, optimum=10.0, noise=0.3, seed=seed)
        result  = _test(trainer).run(batch_size=4)

        assert result is not None
        assert result["min_loss_multiplier"] > floor
        assert result["multiplier"] != pytest.approx(ladder[1] / 10.0)


def test_every_rung_starts_from_the_same_state():
    """Verifies the rewind restores the model before every rung and after the ramp."""
    trainer = _ResponseTrainer(base_lr=0.05, optimum=10.0)
    before  = trainer.weight

    starts = []
    test   = _test(trainer)

    original = test._rung

    def _record(snapshot, base_lrs, train_batches, probe_batches, multiplier, offset):
        """Records the weight the rung starts from before delegating to the real rung."""
        snapshot.restore()
        starts.append(trainer.weight)
        return original(snapshot, base_lrs, train_batches, probe_batches, multiplier, offset)

    test._rung = _record
    test.run(batch_size=4)

    assert len(set(starts)) == 1
    assert starts[0] == pytest.approx(before)
    assert trainer.weight == pytest.approx(before)


def test_every_rung_sees_the_same_batches():
    """Verifies the training batches are drawn once and reused verbatim by every rung."""
    trainer = _ResponseTrainer(base_lr=0.05, optimum=10.0)
    test    = _test(trainer, rungs=12, rung_steps=2)

    test.run(batch_size=4)

    per_rung = [trainer.batches[i * 2:(i + 1) * 2] for i in range(len(trainer.batches) // 2)]
    first    = per_rung[0]

    for rung in per_rung[1:]:
        for seen, expected in zip(rung, first):
            assert torch.equal(seen[0], expected[0])


def test_gradient_accumulation_is_mirrored():
    """Verifies a rung consumes accumulation_steps micro-batches per optimizer step."""
    trainer = _ResponseTrainer(base_lr=0.05, optimum=10.0, accumulation_steps=3)
    test    = _test(trainer, rungs=12, rung_steps=2)

    test.run(batch_size=4)

    assert trainer.optimizer_steps == 12 * 2
    assert len(trainer.batches)    == 3 * trainer.optimizer_steps


def test_accepted_pick_keeps_its_curve_and_selection_evidence():
    """Verifies an accepted pick logs every rung in one table, logs the drop it cleared, and keeps the curve and statistics in the record."""
    logger = _Logger()
    test   = _test(_ResponseTrainer(base_lr=0.05, optimum=10.0), logger=logger)
    result = test.run(batch_size=4)

    title, rows = logger.tables[-1]
    evidence    = test.record["evidence"]

    assert result is not None
    assert len(rows) == result["rungs_run"]
    assert f"{result['rungs_run']} completed rung(s)" in title
    assert evidence["outcome"] == "accepted"
    assert evidence["drop"] >= evidence["threshold"] > 0.0
    assert evidence["min_loss_rung"] == result["losses"].index(min(result["losses"])) + 1
    assert any("clearing the" in line and "standard errors" in line for line in logger.lines)
    assert test.record["multiplier"] == result["multiplier"]
    assert test.record["losses"]     == result["losses"]
    assert test.record["settings"]["rungs"] == 40
    assert test.record["batch_size"] == 4


def test_refused_pick_still_keeps_its_curve_and_the_reason():
    """Verifies a refusal leaves the full curve and the refusal reason in the record."""
    test = _test(_FlatTrainer(base_lr=0.05))

    assert test.run(batch_size=4) is None
    assert test.record["multiplier"] is None
    assert test.record["evidence"]["outcome"] == "minimum on the first rung"
    assert test.record["rungs_run"] == len(test.record["losses"]) == len(test.record["spreads"]) > 0


def test_monotone_increasing_response_is_rejected():
    """Verifies a minimum on the first rung is refused instead of returned."""
    trainer = _ResponseTrainer(base_lr=0.05, optimum=1e-6)

    assert _test(trainer).run(batch_size=4) is None


def test_undiverged_monotone_decreasing_response_is_rejected():
    """Verifies a minimum on the last rung without divergence is refused."""
    trainer = _ResponseTrainer(base_lr=0.05, optimum=1e6)

    assert _test(trainer).run(batch_size=4) is None


def test_flat_response_below_min_improvement_is_rejected():
    """Verifies a loss without any response yields no selection."""
    assert _test(_FlatTrainer(base_lr=0.05)).run(batch_size=4) is None


def test_non_finite_loss_stops_immediately_and_keeps_the_configuration():
    """Verifies a NaN loss ends the ramp before enough rungs exist."""
    assert _test(_NanTrainer(base_lr=0.05)).run(batch_size=4) is None


def test_clipper_gradient_explosion_marks_the_rung_diverged():
    """Verifies a non-finite gradient norm ends the ramp rather than propagating."""
    trainer = _StubTrainer(base_lr=0.05, clipper_raises=True)

    assert _test(trainer).run(batch_size=4) is None


def test_range_test_restores_the_configured_learning_rate():
    """Verifies the optimizer leaves the ramp at the rate the trainer entered it with."""
    trainer = _ResponseTrainer(base_lr=0.05, optimum=10.0, warmup_factor=0.1)

    _test(trainer).run(batch_size=4)

    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.005)


def test_warmup_scaled_group_rates_do_not_offset_the_ramp():
    """Verifies the ramp runs on the warmup-free base rates, not the warmup-shrunk group rates."""
    plain  = _test(_ResponseTrainer(base_lr=0.05, optimum=10.0)).run(batch_size=4)
    warmed = _test(_ResponseTrainer(base_lr=0.05, optimum=10.0, warmup_factor=0.1)).run(batch_size=4)

    assert plain  is not None
    assert warmed is not None
    assert warmed["multiplier"]          == pytest.approx(plain["multiplier"])
    assert warmed["min_loss_multiplier"] == pytest.approx(plain["min_loss_multiplier"])
    assert warmed["losses"]              == pytest.approx(plain["losses"])


def test_snapshot_does_not_alias_live_parameters():
    """Verifies the captured model state survives a later mutation of the trainer."""
    trainer  = _StubTrainer(base_lr=0.05, start=10.0)
    snapshot = TrainerStateSnapshot(trainer)

    with torch.no_grad():
        trainer.model.weight.fill_(-3.0)

    snapshot.restore()

    assert trainer.weight == pytest.approx(10.0)


def test_dataset_too_small_for_one_rung_is_rejected():
    """Verifies a dataset yielding fewer batches than one rung needs fails loudly."""
    test = _test(_StubTrainer(base_lr=0.05), rung_steps=8, probe_batches=4)
    test.dataset = _dataset(n=8)

    with pytest.raises(RuntimeError, match="measurement batches"):
        test.run(batch_size=4)


def test_ramp_definition_is_validated():
    """Verifies that malformed ladder settings are rejected loudly."""
    trainer = _StubTrainer(base_lr=0.05)

    with pytest.raises(ValueError, match="scale_min"):
        _test(trainer, scale_min=2.0, scale_max=1.0)
    with pytest.raises(ValueError, match="lr_rungs"):
        _test(trainer, rungs=3)
    with pytest.raises(ValueError, match="lr_rung_steps"):
        _test(trainer, rung_steps=0)
    with pytest.raises(ValueError, match="lr_probe_batches"):
        _test(trainer, probe_batches=0)
    with pytest.raises(ValueError, match="lr_divergence_factor"):
        _test(trainer, divergence_factor=1.0)
    with pytest.raises(ValueError, match="lr_safety_factor"):
        _test(trainer, safety_factor=0.5)
    with pytest.raises(ValueError, match="lr_min_improvement"):
        _test(trainer, min_improvement=1.0)
    with pytest.raises(ValueError, match="lr_noise_sigmas"):
        _test(trainer, noise_sigmas=-1.0)


def test_multiplier_ladder_is_geometric():
    """Verifies the ladder spans the bounds geometrically with the configured rungs."""
    ladder = _test(_StubTrainer(base_lr=0.05))._multipliers()

    assert len(ladder) == 40
    assert ladder[0] == pytest.approx(1e-2)
    assert ladder[-1] == pytest.approx(1e2)

    ratios = [math.log(ladder[i + 1] / ladder[i]) for i in range(len(ladder) - 1)]

    assert max(ratios) == pytest.approx(min(ratios))
