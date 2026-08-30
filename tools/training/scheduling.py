"""Learning-rate warmup and epoch-level decay schedules applied on top of base LRs."""

from __future__ import annotations

import math


class Warmup:
    """Ramps a learning-rate multiplier from a start factor to 1.0 over a fraction of the run.

    The ramp length is configured as a fraction of the run's total optimizer
    steps and resolved into a concrete step count via `resolve` once the
    trainer knows its step budget; stepping an unresolved active ramp raises.

    Attributes:
        config              : Training configuration exposing a warmup block.
        logger              : Logger for the configuration table and the completion message.
        warmup_fraction     : Fraction of the total optimizer steps the ramp spans.
        warmup_steps        : Optimizer steps of the ramp, None until `resolve` is called.
        warmup_start_factor : Multiplier at step zero.
        enabled             : Whether warmup is applied at all.
        mode                : Ramp shape, one of "linear", "cosine", "exponential", "polynomial".
        poly_power          : Exponent of the polynomial ramp.
        current_step        : Optimizer steps consumed by the ramp so far.
        warmup_finished     : Whether the ramp has completed.
    """

    def __init__(self, config, logger):
        """Reads the warmup settings from the config and reports them.

        Args:
            config : Training configuration exposing a warmup block.
            logger : Logger for the configuration table.
        """
        self.config = config
        self.logger = logger

        self.warmup_fraction     = self.config.warmup.warmup_fraction
        self.warmup_start_factor = self.config.warmup.warmup_start_factor
        self.enabled             = self.config.warmup.warmup_enabled
        self.mode                = self.config.warmup.warmup_mode
        self.poly_power          = self.config.warmup.warmup_poly_power

        self.warmup_steps       = None
        self.current_step       = 0
        self.warmup_finished    = False
        self._logged_completion = False

        active = self.enabled and self.warmup_fraction > 0

        info = {
            "Enabled"      : self.enabled,
            "Effective"    : "active" if active else "inactive (no warmup applied)",
            "Fraction"     : f"{self.warmup_fraction:g} of the total optimizer steps (resolved at train start)",
            "Mode"         : self.mode,
            "Start Factor" : self.warmup_start_factor,
        }

        if self.mode == "polynomial":
            info["Poly Power"] = self.poly_power

        if self.mode == "exponential" and self.warmup_start_factor <= 0:
            info["Mode"] = "exponential (start factor <= 0, falls back to a linear ramp)"

        self.logger.section("[Warmup Scheduler]")
        self.logger.kv_table(info)

    def resolve(self, total_steps: int) -> None:
        """Converts the configured fraction into a concrete step count over the run's step budget.

        Args:
            total_steps : Total optimizer steps the training run will take.
        """
        self.warmup_steps = int(round(self.warmup_fraction * int(total_steps)))

        if self.enabled and self.warmup_fraction > 0:
            if self.warmup_steps <= 0:
                self.warmup_finished = True

            self.logger.subsection(f"Warmup resolved: fraction {self.warmup_fraction:g} of {int(total_steps)} optimizer steps -> {self.warmup_steps} steps.")

    def factor(self) -> float:
        """Returns the warmup multiplier at the current step, 1.0 once the ramp is over."""
        if not self.enabled or self.warmup_fraction <= 0:
            return 1.0

        if self.warmup_steps is not None and (self.warmup_steps <= 0 or self.current_step >= self.warmup_steps):
            return 1.0

        progress = 0.0 if self.warmup_steps is None else self.current_step / self.warmup_steps
        s        = self.warmup_start_factor

        if self.mode == "cosine":
            cos_factor = (1.0 - math.cos(math.pi * progress)) / 2.0
            return s + (1.0 - s) * cos_factor

        elif self.mode == "exponential":
            if s <= 0:
                return progress
            return s ** (1.0 - progress)

        elif self.mode == "polynomial":
            return s + (1.0 - s) * (progress ** self.poly_power)

        else:
            return s + (1.0 - s) * progress

    def step(self) -> float:
        """Advances the ramp by one optimizer step and returns the multiplier to apply.

        Raises:
            RuntimeError : If the ramp is active but `resolve` has not been called.
        """
        if not self.enabled or self.warmup_fraction <= 0:
            self.warmup_finished = True
            return 1.0

        if self.warmup_finished:
            return 1.0

        if self.warmup_steps is None:
            raise RuntimeError("Warmup fraction is not resolved into a step count; call resolve(total_steps) before stepping")

        self.current_step += 1
        factor             = self.factor()

        if self.current_step >= self.warmup_steps and not self.warmup_finished:
            self.warmup_finished = True
            if not self._logged_completion:
                self.logger.info(f"Warmup completed after {self.current_step} optimizer steps.")
                self._logged_completion = True

        return factor

    def reset(self) -> None:
        """Restarts the ramp from step zero, as at the start of a new curriculum stage."""
        self.current_step       = 0
        self.warmup_finished    = False
        self._logged_completion = False

    def is_finished(self) -> bool:
        """Returns whether the ramp is over or was never active."""
        return self.warmup_finished or not self.enabled or self.warmup_fraction <= 0

    def state_dict(self) -> dict:
        """Returns the ramp position and completion flags for checkpointing."""
        return {
            "current_step"      : self.current_step,
            "warmup_finished"   : self.warmup_finished,
            "logged_completion" : self._logged_completion,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restores the ramp position and completion flags from a saved state."""
        self.current_step       = int(state["current_step"])
        self.warmup_finished    = bool(state["warmup_finished"])
        self._logged_completion = bool(state["logged_completion"])


class Scheduler:
    """Scales the base learning rates by an epoch-level decay factor.

    Every schedule decays from the base LR towards eta_min, expressed as a ratio
    of the first parameter group's base LR so all groups keep their relative size.

    Attributes:
        config         : Training configuration exposing a scheduler block.
        warmup         : Warmup whose factor multiplies the schedule, or None.
        logger         : Logger for the configuration table and restart messages.
        base_lrs       : Base learning rate per optimizer parameter group.
        current_lrs    : Learning rate per group after the last step().
        scheduler_type : Decay shape selected in the configuration.
    """

    def __init__(self, base_lrs, warmup, config, logger):
        """Stores the base learning rates and reports the schedule settings.

        Args:
            base_lrs : Base learning rate per optimizer parameter group.
            warmup   : Warmup applied on top of the schedule, or None.
            config   : Training configuration exposing a scheduler block.
            logger   : Logger for the configuration table.
        """
        self.config = config
        self.warmup = warmup
        self.logger = logger

        self.base_lrs       = list(base_lrs)
        self.current_lrs    = list(base_lrs)
        self.scheduler_type = self.config.scheduler.type

        self._epoch_offset = 0

        self._log_scheduler_info()

    def _eta_min_ratio(self) -> float:
        """Returns eta_min as a fraction of the first group's base rate before ``optimizer.lr_scale``.

        Dividing by the scaled rate would let the batch-size rule and the
        pre-training LR probe reshape the decay curve: a multiplier well below
        one lifts the floor towards the base rate and, once the scaled rate
        falls under eta_min, inverts the annealing. Measuring the floor against
        the unscaled rate keeps the schedule's shape fixed and moves the
        absolute floor with the rates, and the ratio is capped at one so no
        schedule can anneal upward.
        """
        unscaled = max(self.base_lrs[0], 1e-12) / max(float(self.config.optimizer.lr_scale), 1e-12)

        return min(1.0, float(self.config.scheduler.eta_min) / unscaled)

    def _progress(self, epoch: int) -> float:
        """Returns the schedule progress in [0, 1] for an epoch within the horizon."""
        return min(1.0, epoch / max(1, self.config.scheduler.epochs))

    def _cosine_annealing(self, epoch: int) -> float:
        """Returns the cosine-annealed learning-rate factor for an epoch."""
        ratio    = self._eta_min_ratio()
        progress = self._progress(epoch)
        return ratio + 0.5 * (1.0 - ratio) * (1.0 + math.cos(math.pi * progress))

    def _linear(self, epoch: int) -> float:
        """Returns the linearly decayed learning-rate factor for an epoch."""
        ratio    = self._eta_min_ratio()
        progress = self._progress(epoch)
        return 1.0 - (1.0 - ratio) * progress

    def _polynomial(self, epoch: int) -> float:
        """Returns the polynomially decayed learning-rate factor for an epoch."""
        ratio    = self._eta_min_ratio()
        progress = self._progress(epoch)
        return ratio + (1.0 - ratio) * (1.0 - progress) ** self.config.scheduler.power

    def _exponential(self, epoch: int) -> float:
        """Returns the exponentially decayed learning-rate factor for an epoch."""
        ratio    = max(self._eta_min_ratio(), 1e-8)
        progress = self._progress(epoch)
        return ratio ** progress

    def _step(self, epoch: int) -> float:
        """Returns the staircase learning-rate factor for an epoch, floored at the eta_min ratio."""
        ratio     = self._eta_min_ratio()
        step_size = max(1, self.config.scheduler.step_size)
        decayed   = self.config.scheduler.gamma ** (epoch // step_size)
        return max(decayed, ratio)

    def _constant(self) -> float:
        """Returns the unit factor of the constant schedule."""
        return 1.0

    def _factor_for(self, epoch: int) -> float:
        """Returns the decay factor of the configured schedule at an epoch.

        Args:
            epoch : Epoch index relative to the schedule offset.

        Returns:
            Multiplier applied to every base learning rate.

        Raises:
            ValueError : If the configured scheduler type is unknown.
        """
        if self.scheduler_type == "cosine_annealing":
            return self._cosine_annealing(epoch)

        if self.scheduler_type == "constant":
            return self._constant()

        if self.scheduler_type == "linear":
            return self._linear(epoch)

        if self.scheduler_type == "polynomial":
            return self._polynomial(epoch)

        if self.scheduler_type == "exponential":
            return self._exponential(epoch)

        if self.scheduler_type == "step":
            return self._step(epoch)

        raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")

    def reset(self, epoch_offset: int = 0) -> None:
        """Restarts the schedule from the base learning rates at a new epoch origin.

        Args:
            epoch_offset : Epoch index treated as the start of the horizon.
        """
        self._epoch_offset = epoch_offset
        self.current_lrs   = list(self.base_lrs)

        self.logger.subsection(f"LR schedule restarted from its base LRs: epoch offset {epoch_offset}, horizon {self.config.scheduler.epochs} epochs from the offset.")

    def step(self, epoch: int) -> list[float]:
        """Applies the schedule at an epoch and returns the learning rate of each group.

        Args:
            epoch : Absolute epoch index; the schedule offset is subtracted.

        Returns:
            Learning rate per optimizer parameter group, before warmup.
        """
        factor           = self._factor_for(epoch - self._epoch_offset)
        self.current_lrs = [lr * factor for lr in self.base_lrs]

        return list(self.current_lrs)

    def effective_lrs(self) -> list[float]:
        """Returns the current learning rates with the warmup factor applied while it is active."""
        if self.warmup is not None and not self.warmup.is_finished():
            f = self.warmup.factor()
            return [lr * f for lr in self.current_lrs]

        return list(self.current_lrs)

    def state_dict(self) -> dict:
        """Returns the current learning rates and the epoch offset for checkpointing."""
        return {
            "current_lrs"  : list(self.current_lrs),
            "epoch_offset" : self._epoch_offset,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restores the current learning rates and the epoch offset from a saved state."""
        self.current_lrs   = list(state["current_lrs"])
        self._epoch_offset = int(state["epoch_offset"])

    def _log_scheduler_info(self):
        """Logs the schedule type, base learning rates and the settings the type uses."""
        self.logger.section("[Learning Rate Scheduler]")
        info = {
            "Scheduler Type"           : self.scheduler_type,
            "Base LRs (after lr_scale)": self.base_lrs,
        }

        if self.scheduler_type in ("cosine_annealing", "linear", "polynomial", "exponential"):
            info["T_max (scheduler.epochs)"] = self.config.scheduler.epochs
            info["Eta Min (param group 0)"]  = f"{self.config.scheduler.eta_min} measured against the base rate before lr_scale={self.config.optimizer.lr_scale:g}, so the floor sits at {self._eta_min_ratio() * self.base_lrs[0]:.3g} (other groups floor at the same ratio of their own rate)"

        if self.scheduler_type == "polynomial":
            info["Power"] = self.config.scheduler.power

        if self.scheduler_type == "step":
            info["Step Size"]               = self.config.scheduler.step_size
            info["Gamma"]                   = self.config.scheduler.gamma
            info["Eta Min (param group 0)"] = f"{self.config.scheduler.eta_min} (decay floor)"

        warmup_active          = bool(self.warmup) and self.warmup.enabled and self.warmup.warmup_fraction > 0
        info["Warmup Active"]  = warmup_active
        self.logger.kv_table(info)
