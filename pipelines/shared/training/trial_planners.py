
from __future__ import annotations


class StabilityTrialPlanner:

    def __init__(self, trials, warmup_modes: tuple, clip_modes: tuple) -> None:
        self.trials       = trials
        self.warmup_modes = tuple(warmup_modes)
        self.clip_modes   = tuple(clip_modes)

    def summary(self) -> dict:
        return {
            "Ladder"     : "none -> +warmup per fraction -> +clip per norm -> +both",
            "Warmup"     : f"{', '.join(f'{fraction:g}' for fraction in self.trials.warmup_fractions)} of total steps, {self.trials.warmup_mode}",
            "Clipping"   : f"{self.trials.clip_mode} @ {', '.join(f'{norm:g}' for norm in self.trials.max_grad_norms)}",
            "Total runs" : 1 + len(self.trials.warmup_fractions) + 2 * len(self.trials.max_grad_norms),
        }

    def _overrides(self, warmup_fraction: float | None, max_grad_norm: float | None) -> dict:
        overrides = {
            "training.warmup_enabled" : warmup_fraction is not None,
            "training.clip_mode"      : self.trials.clip_mode if max_grad_norm is not None else "disabled",
        }

        if warmup_fraction is not None:
            overrides["training.warmup_fraction"] = warmup_fraction
            overrides["training.warmup_mode"]     = self.trials.warmup_mode

        if max_grad_norm is not None:
            overrides["training.max_grad_norm"] = max_grad_norm

        return overrides

    def _clip_tag(self, norm: float) -> str:
        return f"n{norm:g}" if self.trials.clip_mode == "fixed" else self.trials.clip_mode

    def plan(self) -> list[tuple[str, dict]]:
        first = self.trials.warmup_fractions[0]
        rungs = [("stab-0-none", self._overrides(None, None))]
        for fraction in self.trials.warmup_fractions:
            rungs.append((f"stab-1-warmup-w{fraction:g}", self._overrides(fraction, None)))
        for norm in self.trials.max_grad_norms:
            rungs.append((f"stab-2-clip-{self._clip_tag(norm)}", self._overrides(None, norm)))
        for norm in self.trials.max_grad_norms:
            rungs.append((f"stab-3-both-w{first:g}-{self._clip_tag(norm)}", self._overrides(first, norm)))

        return rungs


class EmaTrialPlanner:

    def __init__(self, trials) -> None:
        self.trials = trials

    def summary(self) -> dict:
        return {
            "Ladder"     : "off -> raw per decay -> decay warmup per decay",
            "Decays"     : ", ".join(f"{decay:g}" for decay in self.trials.decays),
            "Total runs" : 1 + 2 * len(self.trials.decays),
        }

    def plan(self) -> list[tuple[str, dict]]:
        rungs = [("ema-0-off", {"training.use_ema": False})]
        for decay in self.trials.decays:
            rungs.append((f"ema-1-raw-d{decay:g}", {"training.use_ema": True, "training.ema_decay": decay, "training.ema_warmup": False}))
        for decay in self.trials.decays:
            rungs.append((f"ema-2-warmup-d{decay:g}", {"training.use_ema": True, "training.ema_decay": decay, "training.ema_warmup": True}))

        return rungs


class AmpTrialPlanner:

    def summary(self) -> dict:
        return {
            "Ladder"     : "off -> on",
            "Total runs" : 2,
        }

    def plan(self) -> list[tuple[str, dict]]:
        return [
            ("amp-0-off", {"training.use_amp": False}),
            ("amp-1-on",  {"training.use_amp": True}),
        ]
