"""Learning-rate range test measuring the loss response of a rewound trainer to a geometric ladder of multipliers."""

from __future__ import annotations

import copy
import math
import statistics
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

from tools.monitoring.logger       import Logger
from tools.runtime.reproducibility import Reproducibility


class TrainerStateSnapshot:
    """Captures the trainer state one ramp rung is rewound to before it is measured.

    The snapshot covers everything a burst of optimizer steps mutates: the model
    parameters and buffers, the optimizer moments, the gradient clipper's
    adaptive window and the CPU and CUDA random states. Restoring all of it is
    what makes two rungs answer the same question about different learning
    rates rather than about different amounts of training.

    Attributes:
        trainer         : Trainer whose state the snapshot belongs to.
        model_state     : Detached CPU copy of the model state dict.
        optimizer_state : Detached CPU copy of the optimizer state dict.
        clipper_state   : Gradient-clipper history and recorded-step count.
        cpu_rng         : CPU random generator state.
        cuda_rng        : CUDA random generator states, or None outside CUDA.
    """

    def __init__(self, trainer) -> None:
        """Captures the model, optimizer, clipper and random state of a trainer.

        Args:
            trainer : Trainer exposing model, optimizer and grad_clipper.
        """
        self.trainer = trainer

        self.model_state     = self._to_cpu(trainer.model.state_dict())
        self.optimizer_state = self._to_cpu(trainer.optimizer.state_dict())
        self.clipper_state   = trainer.grad_clipper.state_dict()

        self.cpu_rng  = torch.get_rng_state()
        self.cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    def _to_cpu(self, value):
        """Returns a detached CPU deep copy of a state mapping, so no captured tensor aliases a live parameter.

        Args:
            value : State dict, nested mapping, sequence or leaf value to copy.

        Returns:
            The same structure with every tensor detached and moved to CPU.
        """
        if torch.is_tensor(value):
            return value.detach().to("cpu", copy=True)

        if isinstance(value, dict):
            return {key: self._to_cpu(item) for key, item in value.items()}

        if isinstance(value, (list, tuple)):
            return type(value)(self._to_cpu(item) for item in value)

        return copy.deepcopy(value)

    def restore(self) -> None:
        """Rewinds the trainer to the captured state and clears any pending gradients.

        ``optimizer.load_state_dict`` also rewrites the per-group learning
        rates, so a multiplier must be applied after this call, never before.
        """
        self.trainer.model.load_state_dict(self.model_state)
        self.trainer.optimizer.load_state_dict(self.optimizer_state)
        self.trainer.optimizer.zero_grad(set_to_none=True)
        self.trainer.grad_clipper.load_state_dict(self.clipper_state)

        torch.set_rng_state(self.cpu_rng)
        if self.cuda_rng is not None:
            torch.cuda.set_rng_state_all(self.cuda_rng)


class LrRangeTest:
    """Measures the loss a rewound trainer reaches under each multiplier of a geometric ladder.

    Every rung restores the same trainer state, applies one shared multiplier to
    the warmup-free base learning rates the trainer stores as ``base_lrs``, runs
    a fixed burst of faithful training steps through the trainer's own step code
    and measures the loss on a fixed set of held-out batches that is identical
    at every rung. The rungs therefore differ only in the learning rate, and the
    selection is a level statistic: the multiplier at the loss minimum, divided
    by a safety factor. A minimum sitting at either edge of the ladder, a ramp
    that never destabilised, and a response smaller than the measurement noise
    are all refusals rather than answers.

    Attributes:
        trainer           : Trainer whose optimizer, step code and loss path the rungs run through.
        dataset           : Dataset the training and measurement batches are drawn from.
        device            : Device the training steps run on.
        logger            : Logger for the ramp progress and the selection.
        scale_min         : Smallest multiplier of the ladder.
        scale_max         : Largest multiplier of the ladder.
        rungs             : Number of multipliers on the ladder.
        rung_steps        : Optimizer steps taken at each rung's multiplier.
        probe_batches     : Held-out batches whose loss is measured at every rung.
        divergence_factor : Multiple of the best rung loss that ends the ramp.
        safety_factor     : Divisor applied to the minimum-loss multiplier.
        min_improvement   : Smallest relative drop below the first rung that counts as a response.
        noise_sigmas      : Standard errors of the measurement the drop must also clear.
        seed              : Seed of the shuffle generator drawing the probe batches.
        record            : Evidence of the last run, accepted or refused: the settings, the per-rung curve and the statistics the selection rests on; None before the first run.
    """

    MIN_RUNGS    = 10
    LOG_INTERVAL = 10

    def __init__(
        self,
        trainer,
        dataset           : Dataset,
        device            : torch.device,
        logger            : Logger,
        scale_min         : float = 1e-2,
        scale_max         : float = 1e4,
        rungs             : int   = 60,
        rung_steps        : int   = 8,
        probe_batches     : int   = 4,
        divergence_factor : float = 4.0,
        safety_factor     : float = 10.0,
        min_improvement   : float = 0.02,
        noise_sigmas      : float = 3.0,
        seed              : int   = 42,
    ) -> None:
        """Stores the test inputs and validates the ladder definition.

        Args:
            trainer           : Trainer supplying base_lrs, the optimizer, the step code, the clipper and the loss.
            dataset           : Dataset the batches come from.
            device            : Device the steps run on.
            logger            : Logger for the test output.
            scale_min         : Smallest multiplier of the ladder, strictly positive.
            scale_max         : Largest multiplier of the ladder, above scale_min.
            rungs             : Multipliers on the ladder, at least MIN_RUNGS.
            rung_steps        : Optimizer steps taken at each rung, at least one.
            probe_batches     : Held-out batches measured at every rung, at least one.
            divergence_factor : Divergence threshold as a multiple of the best rung loss, above 1.
            safety_factor     : Divisor pulling the pick back from the loss minimum, at least 1.
            min_improvement   : Smallest relative drop below the first rung counting as a response, in [0, 1).
            noise_sigmas      : Standard errors of the rung measurement the drop must also clear, at least 0.
            seed              : Seed of the shuffle generator.

        Raises:
            ValueError : If the ladder bounds, rung count, burst length, batch count, divergence factor, safety factor, improvement threshold or noise threshold are out of range.
        """
        if not 0.0 < float(scale_min) < float(scale_max):
            raise ValueError(f"the LR ladder needs 0 < scale_min < scale_max, got scale_min={scale_min} scale_max={scale_max}")
        if int(rungs) < self.MIN_RUNGS:
            raise ValueError(f"lr_rungs={rungs} is below {self.MIN_RUNGS}; the range test needs enough rungs to locate a minimum away from both edges")
        if int(rung_steps) < 1:
            raise ValueError(f"lr_rung_steps={rung_steps} must be at least 1; a rung with no optimizer step measures the starting state")
        if int(probe_batches) < 1:
            raise ValueError(f"lr_probe_batches={probe_batches} must be at least 1; a rung with no measurement batch has no loss")
        if float(divergence_factor) <= 1.0:
            raise ValueError(f"lr_divergence_factor={divergence_factor} must exceed 1, or the ramp stops on any loss increase")
        if float(safety_factor) < 1.0:
            raise ValueError(f"lr_safety_factor={safety_factor} must be at least 1; values below 1 would pick a rate above the loss minimum")
        if not 0.0 <= float(min_improvement) < 1.0:
            raise ValueError(f"lr_min_improvement={min_improvement} must lie in [0, 1)")
        if float(noise_sigmas) < 0.0:
            raise ValueError(f"lr_noise_sigmas={noise_sigmas} must not be negative")

        self.trainer           = trainer
        self.dataset           = dataset
        self.device            = device
        self.logger            = logger
        self.scale_min         = float(scale_min)
        self.scale_max         = float(scale_max)
        self.rungs             = int(rungs)
        self.rung_steps        = int(rung_steps)
        self.probe_batches     = int(probe_batches)
        self.divergence_factor = float(divergence_factor)
        self.safety_factor     = float(safety_factor)
        self.min_improvement   = float(min_improvement)
        self.noise_sigmas      = float(noise_sigmas)
        self.seed              = int(seed)
        self.record            = None

    def _multipliers(self) -> list[float]:
        """Returns the geometric ladder of multipliers from ``scale_min`` to ``scale_max``."""
        span = math.log(self.scale_max / self.scale_min)

        return [self.scale_min * math.exp(span * i / (self.rungs - 1)) for i in range(self.rungs)]

    def _batches(self, batch_size: int) -> tuple[list, list]:
        """Draws the training and measurement batches every rung reuses verbatim.

        Args:
            batch_size : Batch size of the probe loader.

        Returns:
            Tuple of the training batches consumed by one rung's burst and the
            held-out batches the rung loss is measured on.

        Raises:
            RuntimeError : If the dataset yields fewer full batches than one rung and its measurement need.
        """
        accumulation = int(getattr(self.trainer, "accumulation_steps", 1))
        train_needed = self.rung_steps * accumulation
        total_needed = train_needed + self.probe_batches

        generator = Reproducibility.generator(self.seed)
        loader    = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0, generator=generator)

        if len(loader) < total_needed:
            raise RuntimeError(f"dataset holds {len(self.dataset)} samples, yielding {len(loader)} full batches of {batch_size}, fewer than the {train_needed} training and {self.probe_batches} measurement batches one rung needs")

        drawn = []
        for batch in loader:
            drawn.append(batch)
            if len(drawn) == total_needed:
                break

        return drawn[:train_needed], drawn[train_needed:]

    def _apply_multiplier(self, base_lrs: list[float], multiplier: float) -> None:
        """Writes the scaled learning rates into the optimizer groups."""
        for group, base_lr in zip(self.trainer.optimizer.param_groups, base_lrs):
            group["lr"] = base_lr * multiplier

    def _burn(self, train_batches: list, offset: int) -> bool:
        """Runs one rung's burst of optimizer steps through the trainer's own step code.

        Args:
            train_batches : Training batches of the burst, ordered by accumulation window.
            offset        : Step index the clipper logs the burst under.

        Returns:
            True when every step completed, False as soon as a micro-batch loss
            or a gradient norm went non-finite, which is the rung's divergence
            signal.
        """
        accumulation = int(getattr(self.trainer, "accumulation_steps", 1))

        for step in range(self.rung_steps):
            window = train_batches[step * accumulation:(step + 1) * accumulation]

            for batch in window:
                _, scaled = self.trainer.micro_step(batch, len(window))

                if not math.isfinite(scaled):
                    self.trainer.optimizer.zero_grad(set_to_none=True)
                    return False

            try:
                self.trainer.apply_gradients(offset + step)
            except FloatingPointError:
                self.trainer.optimizer.zero_grad(set_to_none=True)
                return False

        return True

    def _measure(self, probe_batches: list) -> tuple[float, float]:
        """Returns the rung's loss level and the spread of the batch losses behind it.

        The model is left in train mode: after a handful of steps the batch-norm
        running statistics are not representative of anything, and the restored
        random state makes the dropout draws identical across rungs, so the
        comparison stays paired. The level is the median because the composite
        losses of this codebase are heavy-tailed.

        Args:
            probe_batches : Held-out batches, the same ones at every rung.

        Returns:
            Tuple of the median batch loss and its sample standard deviation,
            both infinite when any measured loss was non-finite.
        """
        losses = []

        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.trainer.use_amp):
                for batch in probe_batches:
                    losses.append(float(self.trainer._eval_step(batch)["total_loss"]))

        if not all(math.isfinite(value) for value in losses):
            return math.inf, math.inf

        spread = statistics.stdev(losses) if len(losses) > 1 else 0.0

        return statistics.median(losses), spread

    def _rung(self, snapshot: TrainerStateSnapshot, base_lrs: list[float], train_batches: list, probe_batches: list, multiplier: float, offset: int) -> tuple[float, float]:
        """Rewinds the trainer, trains it at one multiplier and measures the loss it reached.

        Args:
            snapshot      : State every rung is rewound to.
            base_lrs      : Warmup-free base learning rates the multiplier scales.
            train_batches : Training batches of the burst.
            probe_batches : Held-out batches the loss is measured on.
            multiplier    : Multiplier this rung applies.
            offset        : Step index the clipper logs the burst under.

        Returns:
            Tuple of the rung's loss level and its spread, infinite when the burst diverged.
        """
        snapshot.restore()
        self._apply_multiplier(base_lrs, multiplier)

        if not self._burn(train_batches, offset):
            return math.inf, math.inf

        return self._measure(probe_batches)

    def _log_curve(self, multipliers: list[float], losses: list[float], spreads: list[float], divergence: Optional[float]) -> None:
        """Logs every completed rung, so the curve behind the selection stays in the preflight log.

        Args:
            multipliers : Multiplier of each completed rung, in ladder order.
            losses      : Loss level of each completed rung.
            spreads     : Spread of the batch losses behind each rung level.
            divergence  : Multiplier of the rung that ended the ramp, or None when the ladder ran to its end.
        """
        ending = f"ramp ended at x{divergence:.4g}" if divergence is not None else "ladder ran to its end"
        rows   = [{"Rung": index + 1, "Multiplier": f"x{multiplier:.4g}", "Loss": f"{loss:.6g}", "Spread": f"{spread:.6g}"} for index, (multiplier, loss, spread) in enumerate(zip(multipliers, losses, spreads))]

        self.logger.metrics_table(rows, columns=["Rung", "Multiplier", "Loss", "Spread"], title=f"LR range test curve: {len(rows)} completed rung(s), {ending}")

    def _select(self, multipliers: list[float], losses: list[float], spreads: list[float], diverged: bool) -> tuple[Optional[dict], dict]:
        """Picks the multiplier at the loss minimum, or refuses when the ramp measured nothing.

        The statistics the decision rests on are returned and logged whether
        the pick is accepted or refused, so an accepted multiplier can be
        checked afterwards against the noise it had to clear.

        Args:
            multipliers : Multiplier of each completed rung, in ladder order.
            losses      : Loss level of each completed rung.
            spreads     : Spread of the batch losses behind each rung level.
            diverged    : Whether the ramp ended on a diverging rung.

        Returns:
            Tuple of the selection and its evidence. The selection is a dict
            with the chosen and the minimum-loss multipliers, or None when too
            few rungs completed, the minimum sits at either edge of the searched
            ladder, or the response did not clear the measurement noise. The
            evidence names the outcome and carries the minimum-loss rung, the
            first and minimum losses, the drop, the relative and noise
            thresholds and the standard error, None where the decision ended
            before computing them.
        """
        evidence = {"outcome": None, "min_loss_rung": None, "first_loss": None, "min_loss": None, "drop": None, "relative_threshold": None, "deviation": None, "threshold": None}

        if len(losses) < self.MIN_RUNGS:
            self.logger.subsection(f"LR range test completed only {len(losses)} rungs, fewer than {self.MIN_RUNGS}; keeping the configured learning rate")
            return None, {**evidence, "outcome": "too few rungs"}

        best = min(range(len(losses)), key=lambda i: losses[i])

        evidence.update({"min_loss_rung": best + 1, "first_loss": losses[0], "min_loss": losses[best]})

        if best == 0:
            self.logger.subsection(f"LR range test found its minimum on the first rung x{multipliers[0]:.4g}: the response only increases, so the useful range lies below lr_scale_min. Keeping the configured learning rate; lower lr_scale_min to search there.")
            return None, {**evidence, "outcome": "minimum on the first rung"}

        if best == len(losses) - 1 and not diverged:
            self.logger.subsection(f"LR range test found its minimum on the last rung x{multipliers[best]:.4g} without ever destabilising: the top of the useful range was never bracketed. Keeping the configured learning rate; raise lr_scale_max to search there.")
            return None, {**evidence, "outcome": "minimum on the last rung without divergence"}

        drop      = losses[0] - losses[best]
        relative  = self.min_improvement * abs(losses[0])
        deviation = max(spreads[0], spreads[best]) / math.sqrt(self.probe_batches)
        threshold = max(relative, self.noise_sigmas * deviation)

        evidence.update({"drop": drop, "relative_threshold": relative, "deviation": deviation, "threshold": threshold})

        if drop < threshold:
            self.logger.subsection(f"LR range test measured a drop of {drop:.6g} from the first rung, below the {threshold:.6g} needed ({self.min_improvement:g} relative, {self.noise_sigmas:g} standard errors of {deviation:.6g}); the ramp shows no response above its own noise. Keeping the configured learning rate.")
            return None, {**evidence, "outcome": "drop below the noise threshold"}

        minimum = multipliers[best]

        self.logger.subsection(f"LR range test minimum on rung {best + 1} at x{minimum:.4g}: loss {losses[best]:.6g} against {losses[0]:.6g} on the first rung, a drop of {drop:.6g} clearing the {threshold:.6g} needed ({self.min_improvement:g} relative = {relative:.6g}, {self.noise_sigmas:g} standard errors of {deviation:.6g})")

        return {"multiplier": minimum / self.safety_factor, "min_loss_multiplier": minimum}, {**evidence, "outcome": "accepted"}

    def _settings(self) -> dict:
        """Returns the ladder and selection settings the run used, for the evidence record."""
        return {
            "scale_min"         : self.scale_min,
            "scale_max"         : self.scale_max,
            "rungs"             : self.rungs,
            "rung_steps"        : self.rung_steps,
            "probe_batches"     : self.probe_batches,
            "divergence_factor" : self.divergence_factor,
            "safety_factor"     : self.safety_factor,
            "min_improvement"   : self.min_improvement,
            "noise_sigmas"      : self.noise_sigmas,
            "seed"              : self.seed,
        }

    def run(self, batch_size: int) -> Optional[dict]:
        """Walks the ladder from one rewound state and returns the selected multiplier.

        Args:
            batch_size : Batch size the learning rate is probed at.

        Every completed rung is logged as one table once the ramp ends, and
        ``record`` holds the settings, the curve and the selection evidence
        whether or not the pick is accepted.

        Returns:
            Dict with "multiplier", "min_loss_multiplier",
            "divergence_multiplier", "rungs_run", "diverged", the per-rung
            "multipliers", "losses" and "spreads" curves and the "evidence" of
            the selection; None when the ramp measured no usable response and
            the configured learning rate should be kept.
        """
        train_batches, probe_batches = self._batches(batch_size)

        base_lrs = [float(lr) for lr in self.trainer.base_lrs]
        snapshot = TrainerStateSnapshot(self.trainer)

        multipliers = []
        losses      = []
        spreads     = []

        best        = math.inf
        diverged    = False
        divergence  = None

        for i, multiplier in enumerate(self._multipliers()):
            level, spread = self._rung(snapshot, base_lrs, train_batches, probe_batches, multiplier, i * self.rung_steps)

            if not math.isfinite(level):
                self.logger.subsection(f"rung {i + 1:>3}  x{multiplier:<10.4g} -> non-finite loss or gradient, ending the ramp")
                diverged   = True
                divergence = multiplier
                break

            if level > self.divergence_factor * best:
                self.logger.subsection(f"rung {i + 1:>3}  x{multiplier:<10.4g} -> loss {level:.6f} exceeds {self.divergence_factor:g}x the best {best:.6f}, ending the ramp")
                diverged   = True
                divergence = multiplier
                break

            best = min(best, level)

            multipliers.append(multiplier)
            losses.append(level)
            spreads.append(spread)

            if (i + 1) % self.LOG_INTERVAL == 0:
                self.logger.subsection(f"rung {i + 1:>3}  x{multiplier:<10.4g} -> loss {level:.6f} (spread {spread:.6f})")

        snapshot.restore()

        self._log_curve(multipliers, losses, spreads, divergence)

        selection, evidence = self._select(multipliers, losses, spreads, diverged)

        curve = {
            "divergence_multiplier" : divergence,
            "rungs_run"             : len(losses),
            "diverged"              : diverged,
            "multipliers"           : multipliers,
            "losses"                : losses,
            "spreads"               : spreads,
            "evidence"              : evidence,
        }

        self.record = {
            "batch_size"          : int(batch_size),
            "base_lrs"            : base_lrs,
            "settings"            : self._settings(),
            "multiplier"          : selection["multiplier"] if selection is not None else None,
            "min_loss_multiplier" : selection["min_loss_multiplier"] if selection is not None else None,
            **curve,
        }

        if selection is None:
            return None

        return {**selection, **curve}
