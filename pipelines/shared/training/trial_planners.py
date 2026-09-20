"""Trial planners for the optimization axes every training family shares.

Weight EMA, mixed-precision autocast and the warmup-plus-clipping stability
ladder are settings of the training loop itself, so the backbone, dual, JEPA
and profile-autoencoder sweeps all plan them from the same three planners.
Each planner validates one axis and turns it into a list of ``(run_name,
config_overrides)`` pairs consumed by the trial scheduler.
"""

from __future__ import annotations


class StabilityTrialPlanner:
    """Plans the cumulative training-stability ladder over warmup and gradient clipping.

    The ladder starts with neither learning-rate warmup nor gradient clipping,
    then plans one warmup-only rung per configured fraction of the total
    optimizer steps, one clipping-only rung per configured norm threshold, and
    one combined rung pairing the first fraction with each threshold, each rung
    on the otherwise unchanged base config.

    Attributes:
        trials       : Stability trial configuration with `warmup_fractions`, `warmup_mode`, `clip_mode` and `max_grad_norms`.
        warmup_modes : Valid warmup ramp shape names.
        clip_modes   : Valid non-disabled clipping strategy names.
    """

    def __init__(self, trials, warmup_modes: tuple, clip_modes: tuple) -> None:
        """Stores and validates the stability ladder configuration.

        Args:
            trials       : Configuration object carrying the warmup and clipping settings.
            warmup_modes : Registered warmup ramp shape names.
            clip_modes   : Registered clipping strategy names, "disabled" included.

        Raises:
            ValueError : On empty or out-of-range step or norm lists, an unknown warmup or clip mode, a disabled clip mode, or several norms under a non-fixed clip mode.
        """
        self.trials       = trials
        self.warmup_modes = tuple(warmup_modes)
        self.clip_modes   = tuple(clip_modes)

        self._validate()

    def _validate(self) -> None:
        """Checks the warmup fractions and mode and the clipping mode and thresholds.

        Raises:
            ValueError : On any empty, out-of-range, duplicated or unknown setting.
        """
        fractions = list(self.trials.warmup_fractions)
        norms     = list(self.trials.max_grad_norms)

        if not fractions:
            raise ValueError("stability_trials.warmup_fractions must name at least one fraction for the warmup rungs")

        for fraction in fractions:
            if not 0.0 < fraction <= 1.0:
                raise ValueError(f"stability_trials.warmup_fractions entry {fraction} must lie in (0, 1]")

        if len(set(fractions)) != len(fractions):
            raise ValueError(f"stability_trials.warmup_fractions {fractions} must be unique, one warmup rung each")

        if self.trials.warmup_mode not in self.warmup_modes:
            raise ValueError(f"Unknown stability_trials.warmup_mode '{self.trials.warmup_mode}'; allowed modes are {self.warmup_modes}")

        if self.trials.clip_mode not in self.clip_modes:
            raise ValueError(f"Unknown stability_trials.clip_mode '{self.trials.clip_mode}'; allowed modes are {self.clip_modes}")

        if self.trials.clip_mode == "disabled":
            raise ValueError("stability_trials.clip_mode must name an active clipping strategy; 'disabled' would make the clipping rungs identical to the warmup rungs")

        if not norms:
            raise ValueError("stability_trials.max_grad_norms must name at least one threshold for the clipping rungs")

        for norm in norms:
            if norm <= 0:
                raise ValueError(f"stability_trials.max_grad_norms entry {norm} must be positive")

        if len(set(norms)) != len(norms):
            raise ValueError(f"stability_trials.max_grad_norms {norms} must be unique, one clipping rung each")

        if self.trials.clip_mode != "fixed" and len(norms) > 1:
            raise ValueError(f"stability_trials.max_grad_norms lists {len(norms)} thresholds but clip_mode '{self.trials.clip_mode}' ignores the threshold, so the clipping rungs would be identical; sweep several norms only under the fixed strategy")

    def summary(self) -> dict:
        """Returns the ladder order, warmup and clipping settings and run count for logging."""
        return {
            "Ladder"     : "none -> +warmup per fraction -> +clip per norm -> +both",
            "Warmup"     : f"{', '.join(f'{fraction:g}' for fraction in self.trials.warmup_fractions)} of total steps, {self.trials.warmup_mode}",
            "Clipping"   : f"{self.trials.clip_mode} @ {', '.join(f'{norm:g}' for norm in self.trials.max_grad_norms)}",
            "Total runs" : 1 + len(self.trials.warmup_fractions) + 2 * len(self.trials.max_grad_norms),
        }

    def _overrides(self, warmup_fraction: float | None, max_grad_norm: float | None) -> dict:
        """Returns the `training.*` overrides of one ladder rung.

        Args:
            warmup_fraction : Fraction of the total optimizer steps of the rung's warmup ramp, or None for no warmup.
            max_grad_norm   : Norm threshold of the rung's gradient clipping, or None for no clipping.

        Returns:
            Overrides setting the warmup and clipping state explicitly, with
            the ramp and threshold settings added on the rungs that use them.
        """
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
        """Returns the run-name token of one clipping arm: the norm under the fixed strategy, the strategy name otherwise."""
        return f"n{norm:g}" if self.trials.clip_mode == "fixed" else self.trials.clip_mode

    def plan(self) -> list[tuple[str, dict]]:
        """Returns the bare rung, one warmup-only rung per fraction, one clipping-only rung per norm, and one combined rung per norm on the first fraction."""
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
    """Plans the weight-EMA arms: EMA off, then raw and decay-warmup EMA at each configured decay.

    Every rung trains the same model on the otherwise unchanged base config;
    only the weight-averaging state changes, so the axes isolate whether
    evaluating and checkpointing an exponential moving average of the weights
    beats the raw weights, at which decay, and whether ramping the decay as
    min(decay, (1+t)/(10+t)) over the update count t helps or hurts.

    Attributes:
        trials : EMA trial configuration with the `decays` list.
    """

    def __init__(self, trials) -> None:
        """Stores and validates the EMA ladder configuration.

        Args:
            trials : Configuration object carrying the EMA-on rung decays.

        Raises:
            ValueError : If the decay list is empty, holds a value outside the open interval (0, 1), or repeats a value.
        """
        self.trials = trials

        self._validate()

    def _validate(self) -> None:
        """Checks that the decays are non-empty, inside (0, 1) and unique.

        Raises:
            ValueError : On an empty, out-of-range or duplicated decay list.
        """
        decays = list(self.trials.decays)

        if not decays:
            raise ValueError("ema_trials.decays must name at least one decay for the EMA-on rungs")

        for decay in decays:
            if not 0.0 < decay < 1.0:
                raise ValueError(f"ema_trials.decays entry {decay} must lie strictly between 0 and 1")

        if len(set(decays)) != len(decays):
            raise ValueError(f"ema_trials.decays {decays} must be unique, one EMA-on rung each")

    def summary(self) -> dict:
        """Returns the ladder order, decays and run count for logging."""
        return {
            "Ladder"     : "off -> raw per decay -> decay warmup per decay",
            "Decays"     : ", ".join(f"{decay:g}" for decay in self.trials.decays),
            "Total runs" : 1 + 2 * len(self.trials.decays),
        }

    def plan(self) -> list[tuple[str, dict]]:
        """Returns the EMA-off rung, one raw EMA rung per decay, and one decay-warmup rung per decay."""
        rungs = [("ema-0-off", {"training.use_ema": False})]
        for decay in self.trials.decays:
            rungs.append((f"ema-1-raw-d{decay:g}", {"training.use_ema": True, "training.ema_decay": decay, "training.ema_warmup": False}))
        for decay in self.trials.decays:
            rungs.append((f"ema-2-warmup-d{decay:g}", {"training.use_ema": True, "training.ema_decay": decay, "training.ema_warmup": True}))

        return rungs


class AmpTrialPlanner:
    """Plans the mixed-precision pair: autocast off against autocast on.

    Both rungs train the same model on the otherwise unchanged base config;
    only `training.use_amp` flips, so the pair isolates what mixed-precision
    autocast costs or buys in accuracy and stability.
    """

    def summary(self) -> dict:
        """Returns the pair order and run count for logging."""
        return {
            "Ladder"     : "off -> on",
            "Total runs" : 2,
        }

    def plan(self) -> list[tuple[str, dict]]:
        """Returns the autocast-off and autocast-on rungs."""
        return [
            ("amp-0-off", {"training.use_amp": False}),
            ("amp-1-on",  {"training.use_amp": True}),
        ]
