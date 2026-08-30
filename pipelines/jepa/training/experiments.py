"""Trial planners expanding JEPA sweep configurations into named runs.

Each planner validates one experiment axis (latent-loss terms, decoded-curve
penalties, context masking, the loss curriculum, the autoencoder coupling and
the coupled autoencoder run) and turns it into a list of
``(run_name, config_overrides)`` pairs consumed by the training scheduler.

Validation covers the base configuration the arms are stacked on, so a
combination the trainer refuses is rejected here or left unplanned instead of
holding a GPU until it raises.
"""

from __future__ import annotations

import re

from dataclasses import fields

from configuration.training.jepa import EmbeddingLossConfig


class EmbeddingLossTrialPlanner:
    """Plans one run per latent-loss arm, every arm fully defining the latent terms.

    The decoded-curve settings stay at the base configuration, so the sweep
    isolates the choice of latent penalty and collapse regularizer.

    Attributes:
        embedding_trials : Mapping of run label to the `embedding_loss` fields the arm enables.
        TERMS            : Enable-flag and weight field pairs of every latent term, disabled before each arm.
        SHAPE_FIELDS     : Latent fields shaping a term rather than enabling one, which an arm may also set.
    """

    TERMS = (
        ("use_embedding_mse",         "weight_embedding_mse"),
        ("use_embedding_cosine",      "weight_embedding_cosine"),
        ("use_embedding_smoothl1",    "weight_embedding_smoothl1"),
        ("use_embedding_huber",       "weight_embedding_huber"),
        ("use_embedding_charbonnier", "weight_embedding_charbonnier"),
        ("use_embedding_variance",    "weight_embedding_variance"),
        ("use_embedding_covariance",  "weight_embedding_covariance"),
    )

    SHAPE_FIELDS = ("smoothl1_beta", "embedding_huber_delta", "embedding_charbonnier_eps", "variance_gamma")

    def __init__(self, embedding_trials: dict) -> None:
        """Stores and validates the latent-loss arms.

        Args:
            embedding_trials : Mapping of run label to embedding-loss field overrides.

        Raises:
            ValueError : If no arm is given, an arm sets an unknown or non-latent field, or an arm enables no latent term.
        """
        self.embedding_trials = embedding_trials

        self._validate()

    def _validate(self) -> None:
        """Checks that every arm sets known latent fields and enables at least one term.

        Raises:
            ValueError : On an empty mapping, unknown or non-latent fields, or an arm with every latent term disabled.
        """
        if not self.embedding_trials:
            raise ValueError("embedding_trials must list at least one arm")

        declared = {config_field.name for config_field in fields(EmbeddingLossConfig)}
        latent   = {key for pair in self.TERMS for key in pair} | set(self.SHAPE_FIELDS)

        for label, spec in self.embedding_trials.items():
            unknown = sorted(key for key in spec if key not in declared)
            if unknown:
                raise ValueError(f"Embedding arm '{label}' sets unknown embedding_loss fields {unknown}")

            outside = sorted(key for key in spec if key not in latent)
            if outside:
                raise ValueError(f"Embedding arm '{label}' sets the non-latent embedding_loss fields {outside}; the decoded-curve settings belong to the curve sweep, which is what keeps this sweep a comparison of latent losses alone")

            if not any(spec.get(use_key) for use_key, _ in self.TERMS):
                raise ValueError(f"Embedding arm '{label}' enables no latent term")

    def summary(self) -> dict:
        """Returns the arm labels and run count for logging."""
        return {
            "Embedding arms" : list(self.embedding_trials),
            "Total runs"     : len(self.embedding_trials),
        }

    def _base_overrides(self) -> dict:
        """Returns the overrides disabling every latent term, so each arm states its latent loss completely."""
        overrides = {}

        for use_key, weight_key in self.TERMS:
            overrides[f"embedding_loss.{use_key}"]    = False
            overrides[f"embedding_loss.{weight_key}"] = 0.0

        return overrides

    def plan(self) -> list[tuple[str, dict]]:
        """Returns one `(run_name, overrides)` pair per latent-loss arm."""
        plans = []

        for label, spec in self.embedding_trials.items():
            overrides = self._base_overrides()
            overrides.update({f"embedding_loss.{key}": value for key, value in spec.items()})
            plans.append((f"el-{label}", overrides))

        return plans


class CurveTrialPlanner:
    """Plans the decoded-curve arms: penalty kinds by weight, plus Sobolev arms on an mse anchor.

    Attributes:
        trials          : Curve trial configuration carrying `kinds`, `weights`, `sobolev_weights` and `include_baseline`.
        target_provider : Target provider of the base configuration, which decides whether the unanchored baseline arm is admissible.
    """

    KINDS = ("mse", "l1", "huber", "charbonnier")

    def __init__(self, trials, target_provider: str) -> None:
        """Stores and validates the decoded-curve sweep configuration.

        Args:
            trials          : Configuration object with `kinds`, `weights`, `sobolev_weights` and `include_baseline`.
            target_provider : Target provider of the base configuration.

        Raises:
            ValueError : If a kind is unknown, an axis is empty, non-positive or duplicated, or the baseline arm is planned under a live target.
        """
        self.trials          = trials
        self.target_provider = target_provider

        self._validate()

    def _validate(self) -> None:
        """Checks the penalty kinds, both weight axes and the admissibility of the baseline arm.

        Raises:
            ValueError : On unknown kinds, empty, non-positive or duplicated axis values, or a baseline arm the live target rejects.
        """
        if not self.trials.kinds:
            raise ValueError("curve_trials.kinds must list at least one penalty kind")

        unknown = [kind for kind in self.trials.kinds if kind not in self.KINDS]
        if unknown:
            raise ValueError(f"Unknown curve_trials.kinds {unknown}; allowed kinds are {self.KINDS}")

        if len(set(self.trials.kinds)) != len(self.trials.kinds):
            raise ValueError(f"curve_trials.kinds must be unique, got {self.trials.kinds}")

        for name, values in (("weights", self.trials.weights), ("sobolev_weights", self.trials.sobolev_weights)):
            if not values:
                raise ValueError(f"curve_trials.{name} must list at least one weight")

            if any(weight <= 0 for weight in values):
                raise ValueError(f"curve_trials.{name} must all be positive, got {values}")

            if len(set(values)) != len(values):
                raise ValueError(f"curve_trials.{name} must be unique, got {values}")

        if self.trials.include_baseline and self.target_provider == "live":
            raise ValueError("curve_trials.include_baseline plans the 'curve-off' arm without a curve reconstruction term, which target_provider 'live' rejects: a differentiable target branch needs that term to anchor it or the embedding-match loss collapses onto a constant embedding. Set curve_trials.include_baseline=false or target_provider 'stopgrad'.")

    def summary(self) -> dict:
        """Returns the swept kinds, weights and run count for logging."""
        return {
            "Curve kinds"     : list(self.trials.kinds),
            "Curve weights"   : list(self.trials.weights),
            "Sobolev weights" : list(self.trials.sobolev_weights),
            "Baseline"        : self.trials.include_baseline,
            "Total runs"      : len(self.plan()),
        }

    def plan(self) -> list[tuple[str, dict]]:
        """Returns the baseline, the kind-by-weight arms and the Sobolev arms."""
        plans = []

        if self.trials.include_baseline:
            plans.append(("curve-off", {"embedding_loss.use_curve_recon": False, "embedding_loss.use_curve_sobolev": False}))

        for kind in self.trials.kinds:
            for weight in self.trials.weights:
                plans.append((f"curve-{kind}{weight:g}", {
                    "embedding_loss.use_curve_recon"    : True,
                    "embedding_loss.curve_kind"         : kind,
                    "embedding_loss.weight_curve_recon" : weight,
                    "embedding_loss.use_curve_sobolev"  : False,
                }))

        for weight in self.trials.sobolev_weights:
            plans.append((f"curve-sob{weight:g}", {
                "embedding_loss.use_curve_recon"      : True,
                "embedding_loss.curve_kind"           : "mse",
                "embedding_loss.weight_curve_recon"   : 1.0,
                "embedding_loss.use_curve_sobolev"    : True,
                "embedding_loss.weight_curve_sobolev" : weight,
            }))

        return plans


class MaskingTrialPlanner:
    """Plans the context-masking grid of the JEPA embedding regime.

    Attributes:
        trials                  : Masking trial configuration carrying the strategies, ratios, fill and loss target.
        profile_autoencoder_run : Profile-autoencoder run of the base configuration, which the masked arms require.
    """

    STRATEGIES = ("block", "random")
    FILLS      = ("zeros", "noise")
    LOSS_ON    = ("masked", "all")

    def __init__(self, trials, profile_autoencoder_run: str | None) -> None:
        """Stores and validates the masking sweep configuration.

        Args:
            trials                  : Configuration object with `strategies`, `ratios`, `fill`, `loss_on` and `include_baseline`.
            profile_autoencoder_run : Profile-autoencoder run folder of the base configuration, empty when none is coupled.

        Raises:
            ValueError : If no profile autoencoder is coupled, a strategy, fill or loss target is unknown, or the ratios leave (0, 1) or repeat.
        """
        self.trials                  = trials
        self.profile_autoencoder_run = profile_autoencoder_run

        self._validate()

    def _validate(self) -> None:
        """Checks the coupled autoencoder, the strategies, the ratio axis, the fill and the loss target.

        Raises:
            ValueError : On a missing profile autoencoder, unknown categorical values or bad ratios.
        """
        if not self.profile_autoencoder_run:
            raise ValueError("masking_trials plan arms with masking.enabled, which the trainer accepts only with a coupled profile autoencoder: the mask restricts the embedding loss to the hidden pixels, and the Gaussian parameter loss does not support it. Set profile_autoencoder_run or sweep another trials_mode.")

        if not self.trials.strategies:
            raise ValueError("masking_trials.strategies must list at least one strategy")

        unknown = [strategy for strategy in self.trials.strategies if strategy not in self.STRATEGIES]
        if unknown:
            raise ValueError(f"Unknown masking_trials.strategies {unknown}; allowed strategies are {self.STRATEGIES}")

        if len(set(self.trials.strategies)) != len(self.trials.strategies):
            raise ValueError(f"masking_trials.strategies must be unique, got {self.trials.strategies}")

        if not self.trials.ratios:
            raise ValueError("masking_trials.ratios must list at least one ratio")

        if any(not 0.0 < ratio < 1.0 for ratio in self.trials.ratios):
            raise ValueError(f"masking_trials.ratios must lie in (0, 1), got {self.trials.ratios}")

        if len(set(self.trials.ratios)) != len(self.trials.ratios):
            raise ValueError(f"masking_trials.ratios must be unique, got {self.trials.ratios}")

        if self.trials.fill not in self.FILLS:
            raise ValueError(f"Unknown masking_trials.fill '{self.trials.fill}'; allowed fills are {self.FILLS}")

        if self.trials.loss_on not in self.LOSS_ON:
            raise ValueError(f"Unknown masking_trials.loss_on '{self.trials.loss_on}'; allowed targets are {self.LOSS_ON}")

    def summary(self) -> dict:
        """Returns the swept grid and run count for logging."""
        n_grid = len(self.trials.strategies) * len(self.trials.ratios)

        return {
            "Strategies" : list(self.trials.strategies),
            "Ratios"     : list(self.trials.ratios),
            "Fill"       : self.trials.fill,
            "Loss on"    : self.trials.loss_on,
            "Baseline"   : self.trials.include_baseline,
            "Total runs" : n_grid + (1 if self.trials.include_baseline else 0),
        }

    def plan(self) -> list[tuple[str, dict]]:
        """Returns the unmasked baseline and one arm per strategy-by-ratio cell."""
        plans = []

        if self.trials.include_baseline:
            plans.append(("mask-off", {"masking.enabled": False}))

        for strategy in self.trials.strategies:
            for ratio in self.trials.ratios:
                plans.append((f"mask-{strategy}{ratio:g}", {
                    "masking.enabled"    : True,
                    "masking.strategy"   : strategy,
                    "masking.mask_ratio" : ratio,
                    "masking.fill"       : self.trials.fill,
                    "masking.loss_on"    : self.trials.loss_on,
                }))

        return plans


class JepaCurriculumTrialPlanner:
    """Plans the curriculum grid over the swap epoch and the state resets at the swap.

    Attributes:
        trials : Curriculum trial configuration carrying `swap_epochs`, `resets` and `include_baseline`.
        epochs : Training length of the base configuration, which every swap epoch must stay below.
    """

    RESETS = {
        "none"      : {"reset_lr": False, "reset_warmup": False, "reset_optimizer": False},
        "lr"        : {"reset_lr": True,  "reset_warmup": False, "reset_optimizer": False},
        "warmup"    : {"reset_lr": False, "reset_warmup": True,  "reset_optimizer": False},
        "optimizer" : {"reset_lr": False, "reset_warmup": False, "reset_optimizer": True},
        "all"       : {"reset_lr": True,  "reset_warmup": True,  "reset_optimizer": True},
    }

    def __init__(self, trials, epochs: int) -> None:
        """Stores and validates the curriculum sweep configuration.

        Args:
            trials : Configuration object with `swap_epochs`, `resets` and `include_baseline`.
            epochs : Number of epochs the base configuration trains for.

        Raises:
            ValueError : If a reset is unknown or the swap epochs are empty, non-positive, duplicated or not below the training length.
        """
        self.trials = trials
        self.epochs = epochs

        self._validate()

    def _validate(self) -> None:
        """Checks the swap epochs against the training length and the reset labels.

        Raises:
            ValueError : On bad swap epochs, swap epochs the training length never reaches, or unknown or duplicated resets.
        """
        if not self.trials.swap_epochs:
            raise ValueError("curriculum_trials.swap_epochs must list at least one epoch")

        if any(epoch <= 0 for epoch in self.trials.swap_epochs):
            raise ValueError(f"curriculum_trials.swap_epochs must all be positive, got {self.trials.swap_epochs}")

        if len(set(self.trials.swap_epochs)) != len(self.trials.swap_epochs):
            raise ValueError(f"curriculum_trials.swap_epochs must be unique, got {self.trials.swap_epochs}")

        late = [epoch for epoch in self.trials.swap_epochs if epoch >= self.epochs]
        if late:
            raise ValueError(f"curriculum_trials.swap_epochs {late} must be below training.epochs={self.epochs}; the main loss stage would never run on those arms")

        if not self.trials.resets:
            raise ValueError("curriculum_trials.resets must list at least one reset")

        unknown = [reset for reset in self.trials.resets if reset not in self.RESETS]
        if unknown:
            raise ValueError(f"Unknown curriculum_trials.resets {unknown}; allowed resets are {tuple(self.RESETS)}")

        if len(set(self.trials.resets)) != len(self.trials.resets):
            raise ValueError(f"curriculum_trials.resets must be unique, got {self.trials.resets}")

    def summary(self) -> dict:
        """Returns the swept grid and run count for logging."""
        n_grid = len(self.trials.swap_epochs) * len(self.trials.resets)

        return {
            "Swap epochs" : list(self.trials.swap_epochs),
            "Resets"      : list(self.trials.resets),
            "Baseline"    : self.trials.include_baseline,
            "Total runs"  : n_grid + (1 if self.trials.include_baseline else 0),
        }

    def plan(self) -> list[tuple[str, dict]]:
        """Returns the curriculum-free baseline and one arm per swap-epoch-by-reset cell."""
        plans = []

        if self.trials.include_baseline:
            plans.append(("cur-off", {"curriculum.enabled": False}))

        for epoch in self.trials.swap_epochs:
            for reset in self.trials.resets:
                overrides = {"curriculum.enabled": True, "curriculum.swap_epoch": epoch}
                overrides.update({f"curriculum.{key}": value for key, value in self.RESETS[reset].items()})
                plans.append((f"cur-e{epoch}-{reset}", overrides))

        return plans


class CouplingTrialPlanner:
    """Plans the coupling arms over the freeze mode, the fine-tune rate and the target provider.

    A live target is differentiable, so it is planned only on the fine-tune arms
    and only when the base embedding loss anchors it with a curve reconstruction
    term; the frozen cell of that provider is skipped rather than planned into a
    run the trainer rejects.

    Attributes:
        trials         : Coupling trial configuration carrying `modes`, `finetune_lrs` and `target_providers`.
        providers      : Target providers the arms may select.
        embedding_loss : Embedding-loss configuration of the base run, which decides whether a live target is anchored.
    """

    MODES = ("frozen", "finetune")

    def __init__(self, trials, providers: tuple, embedding_loss) -> None:
        """Stores and validates the coupling sweep configuration.

        Args:
            trials         : Configuration object with `modes`, `finetune_lrs` and `target_providers`.
            providers      : Valid target-provider names.
            embedding_loss : Embedding-loss configuration of the base run.

        Raises:
            ValueError : If a mode or provider is unknown, the fine-tune rates are empty, non-positive or duplicated while `finetune` is swept, or a live target is swept without a fine-tune arm or without a curve reconstruction term anchoring it.
        """
        self.trials         = trials
        self.providers      = tuple(providers)
        self.embedding_loss = embedding_loss

        self._validate()

    @property
    def anchored(self) -> bool:
        """Whether the base embedding loss carries the curve reconstruction term a differentiable target branch needs."""
        return self.embedding_loss.use_curve_recon and self.embedding_loss.weight_curve_recon > 0.0

    def _validate(self) -> None:
        """Checks the modes, the providers, the fine-tune rates and the admissibility of a live target.

        Raises:
            ValueError : On unknown or duplicated categorical values, bad fine-tune rates, or a live target with no fine-tune arm or no anchoring curve term.
        """
        if not self.trials.modes:
            raise ValueError("coupling_trials.modes must list at least one mode")

        unknown = [mode for mode in self.trials.modes if mode not in self.MODES]
        if unknown:
            raise ValueError(f"Unknown coupling_trials.modes {unknown}; allowed modes are {self.MODES}")

        if len(set(self.trials.modes)) != len(self.trials.modes):
            raise ValueError(f"coupling_trials.modes must be unique, got {self.trials.modes}")

        if not self.trials.target_providers:
            raise ValueError("coupling_trials.target_providers must list at least one provider")

        unknown = [provider for provider in self.trials.target_providers if provider not in self.providers]
        if unknown:
            raise ValueError(f"Unknown coupling_trials.target_providers {unknown}; allowed providers are {self.providers}")

        if len(set(self.trials.target_providers)) != len(self.trials.target_providers):
            raise ValueError(f"coupling_trials.target_providers must be unique, got {self.trials.target_providers}")

        if "finetune" in self.trials.modes:
            if not self.trials.finetune_lrs:
                raise ValueError("coupling_trials.finetune_lrs must list at least one rate when 'finetune' is swept")

            if any(rate <= 0 for rate in self.trials.finetune_lrs):
                raise ValueError(f"coupling_trials.finetune_lrs must all be positive, got {self.trials.finetune_lrs}")

            if len(set(self.trials.finetune_lrs)) != len(self.trials.finetune_lrs):
                raise ValueError(f"coupling_trials.finetune_lrs must be unique, got {self.trials.finetune_lrs}")

        if "live" in self.trials.target_providers:
            if "finetune" not in self.trials.modes:
                raise ValueError(f"coupling_trials.target_providers includes 'live', which requires a trainable profile autoencoder, but coupling_trials.modes is {list(self.trials.modes)}; add 'finetune' or drop the 'live' provider")

            if not self.anchored:
                raise ValueError("coupling_trials.target_providers includes 'live', which keeps the target branch differentiable, so the embedding-match loss can collapse to a constant embedding; enable embedding_loss.use_curve_recon with a positive embedding_loss.weight_curve_recon to anchor it, or drop the 'live' provider")

    def summary(self) -> dict:
        """Returns the swept arms and run count for logging."""
        return {
            "Modes"        : list(self.trials.modes),
            "Finetune lrs" : list(self.trials.finetune_lrs),
            "Providers"    : list(self.trials.target_providers),
            "Total runs"   : len(self.plan()),
        }

    def plan(self) -> list[tuple[str, dict]]:
        """Returns one frozen arm per provider that admits it and one fine-tune arm per provider and rate."""
        plans = []

        for provider in self.trials.target_providers:
            if "frozen" in self.trials.modes and provider != "live":
                plans.append((f"cpl-{provider}-frozen", {"profile_autoencoder_mode": "frozen", "target_provider": provider}))

            if "finetune" in self.trials.modes:
                for rate in self.trials.finetune_lrs:
                    plans.append((f"cpl-{provider}-ft{rate:g}", {"profile_autoencoder_mode": "finetune", "target_provider": provider, "ae_finetune_lr": rate}))

        return plans


class AeRunTrialPlanner:
    """Plans one JEPA training per pretrained profile-autoencoder run.

    Every arm is named after the run it couples, so reordering the runs leaves
    the run directory of each arm untouched and a resumed sweep keeps every
    result bound to the autoencoder that produced it.

    Attributes:
        trials : Autoencoder-run trial configuration carrying the `runs` list.
    """

    def __init__(self, trials) -> None:
        """Stores and validates the autoencoder-run arms.

        Args:
            trials : Configuration object with the `runs` list.

        Raises:
            ValueError : If no run is given, a run repeats, or two runs share an arm name once sanitized.
        """
        self.trials = trials

        self._validate()

    def _validate(self) -> None:
        """Checks that the runs are present, unique and named apart once sanitized.

        Raises:
            ValueError : On an empty or duplicated run list, or on runs collapsing onto the same arm name.
        """
        if not self.trials.runs:
            raise ValueError("ae_run_trials.runs must list at least one profile-autoencoder run folder")

        if len(set(self.trials.runs)) != len(self.trials.runs):
            raise ValueError(f"ae_run_trials.runs must be unique, got {self.trials.runs}")

        names      = [self.run_name(run) for run in self.trials.runs]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"ae_run_trials.runs must stay distinct once sanitized into arm names, but {duplicates} name more than one run of {self.trials.runs}")

    @staticmethod
    def run_name(run: str) -> str:
        """Returns the arm name of one coupled autoencoder run, sanitized into a directory name.

        Args:
            run : Run folder of the coupled profile autoencoder, possibly a nested path.

        Returns:
            The arm name, every character outside letters, digits, dot, dash and
            underscore replaced by a single underscore.
        """
        return f"pae-{re.sub(r'[^A-Za-z0-9._-]+', '_', run)}"

    def summary(self) -> dict:
        """Returns the arm-to-run mapping and run count for logging."""
        return {
            "AE runs"    : {self.run_name(run): run for run in self.trials.runs},
            "Total runs" : len(self.trials.runs),
        }

    def plan(self) -> list[tuple[str, dict]]:
        """Returns one `(run_name, overrides)` pair per coupled autoencoder run."""
        return [(self.run_name(run), {"profile_autoencoder_run": run}) for run in self.trials.runs]
