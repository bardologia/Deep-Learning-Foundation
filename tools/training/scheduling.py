from __future__ import annotations

import math


class Warmup:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        self.warmup_steps        = self.config.warmup.warmup_steps
        self.warmup_start_factor = self.config.warmup.warmup_start_factor
        self.enabled             = self.config.warmup.warmup_enabled
        self.mode                = self.config.warmup.warmup_mode
        self.poly_power          = self.config.warmup.warmup_poly_power

        self.current_step       = 0
        self.warmup_finished    = False
        self._logged_completion = False

        active = self.enabled and self.warmup_steps > 0

        info = {
            "Enabled"      : self.enabled,
            "Effective"    : "active" if active else "inactive (no warmup applied)",
            "Steps"        : f"{self.warmup_steps} optimizer steps",
            "Mode"         : self.mode,
            "Start Factor" : self.warmup_start_factor,
        }

        if self.mode == "polynomial":
            info["Poly Power"] = self.poly_power

        if self.mode == "exponential" and self.warmup_start_factor <= 0:
            info["Mode"] = "exponential (start factor <= 0, falls back to a linear ramp)"

        self.logger.section("[Warmup Scheduler]")
        self.logger.kv_table(info)

    def factor(self) -> float:
        if not self.enabled or self.warmup_steps <= 0:
            return 1.0

        if self.current_step >= self.warmup_steps:
            return 1.0

        progress = self.current_step / self.warmup_steps
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
        if not self.enabled or self.warmup_steps <= 0:
            self.warmup_finished = True
            return 1.0

        if self.warmup_finished:
            return 1.0

        self.current_step += 1
        factor             = self.factor()

        if self.current_step >= self.warmup_steps and not self.warmup_finished:
            self.warmup_finished = True
            if not self._logged_completion:
                self.logger.info(f"Warmup completed after {self.current_step} optimizer steps.")
                self._logged_completion = True

        return factor

    def reset(self) -> None:
        self.current_step       = 0
        self.warmup_finished    = False
        self._logged_completion = False

    def is_finished(self) -> bool:
        return self.warmup_finished or not self.enabled or self.warmup_steps <= 0

    def state_dict(self) -> dict:
        return {
            "current_step"      : self.current_step,
            "warmup_finished"   : self.warmup_finished,
            "logged_completion" : self._logged_completion,
        }

    def load_state_dict(self, state: dict) -> None:
        self.current_step       = int(state["current_step"])
        self.warmup_finished    = bool(state["warmup_finished"])
        self._logged_completion = bool(state["logged_completion"])


class Scheduler:
    def __init__(self, base_lrs, warmup, config, logger):
        self.config = config
        self.warmup = warmup
        self.logger = logger

        self.base_lrs       = list(base_lrs)
        self.current_lrs    = list(base_lrs)
        self.scheduler_type = self.config.scheduler.type

        self._epoch_offset = 0

        self._log_scheduler_info()

    def _eta_min_ratio(self) -> float:
        return float(self.config.scheduler.eta_min) / max(self.base_lrs[0], 1e-12)

    def _progress(self, epoch: int) -> float:
        return min(1.0, epoch / max(1, self.config.scheduler.epochs))

    def _cosine_annealing(self, epoch: int) -> float:
        ratio    = self._eta_min_ratio()
        progress = self._progress(epoch)
        return ratio + 0.5 * (1.0 - ratio) * (1.0 + math.cos(math.pi * progress))

    def _linear(self, epoch: int) -> float:
        ratio    = self._eta_min_ratio()
        progress = self._progress(epoch)
        return 1.0 - (1.0 - ratio) * progress

    def _polynomial(self, epoch: int) -> float:
        ratio    = self._eta_min_ratio()
        progress = self._progress(epoch)
        return ratio + (1.0 - ratio) * (1.0 - progress) ** self.config.scheduler.power

    def _exponential(self, epoch: int) -> float:
        ratio    = max(self._eta_min_ratio(), 1e-8)
        progress = self._progress(epoch)
        return ratio ** progress

    def _step(self, epoch: int) -> float:
        ratio     = self._eta_min_ratio()
        step_size = max(1, self.config.scheduler.step_size)
        decayed   = self.config.scheduler.gamma ** (epoch // step_size)
        return max(decayed, ratio)

    def _constant(self) -> float:
        return 1.0

    def _factor_for(self, epoch: int) -> float:
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
        self._epoch_offset = epoch_offset
        self.current_lrs   = list(self.base_lrs)

        self.logger.subsection(f"LR schedule restarted from its base LRs: epoch offset {epoch_offset}, horizon {self.config.scheduler.epochs} epochs from the offset.")

    def step(self, epoch: int) -> list[float]:
        factor           = self._factor_for(epoch - self._epoch_offset)
        self.current_lrs = [lr * factor for lr in self.base_lrs]

        return list(self.current_lrs)

    def effective_lrs(self) -> list[float]:
        if self.warmup is not None and not self.warmup.is_finished():
            f = self.warmup.factor()
            return [lr * f for lr in self.current_lrs]

        return list(self.current_lrs)

    def state_dict(self) -> dict:
        return {
            "current_lrs"  : list(self.current_lrs),
            "epoch_offset" : self._epoch_offset,
        }

    def load_state_dict(self, state: dict) -> None:
        self.current_lrs   = list(state["current_lrs"])
        self._epoch_offset = int(state["epoch_offset"])

    def _log_scheduler_info(self):
        self.logger.section("[Learning Rate Scheduler]")
        info = {
            "Scheduler Type"           : self.scheduler_type,
            "Base LRs (after lr_scale)": self.base_lrs,
        }

        if self.scheduler_type in ("cosine_annealing", "linear", "polynomial", "exponential"):
            info["T_max (scheduler.epochs)"] = self.config.scheduler.epochs
            info["Eta Min (param group 0)"]  = f"{self.config.scheduler.eta_min} (other groups floor at eta_min * lr_i / lr_0)"

        if self.scheduler_type == "polynomial":
            info["Power"] = self.config.scheduler.power

        if self.scheduler_type == "step":
            info["Step Size"]               = self.config.scheduler.step_size
            info["Gamma"]                   = self.config.scheduler.gamma
            info["Eta Min (param group 0)"] = f"{self.config.scheduler.eta_min} (decay floor)"

        warmup_active          = bool(self.warmup) and self.warmup.enabled and self.warmup.warmup_steps > 0
        info["Warmup Active"]  = warmup_active
        self.logger.kv_table(info)
