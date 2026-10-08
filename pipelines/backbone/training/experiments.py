"""Trial planners that expand backbone sweep configurations into named runs.

Each planner validates one experiment axis (curriculum stages, physics loss
terms, patch sizes, secondary-track selections, ...) and turns it into a list of
``(run_name, config_overrides)`` pairs consumed by the training scheduler.
"""

from __future__ import annotations

from fractions  import Fraction
from itertools  import combinations
from math       import comb

import numpy as np

from models                                   import get_backbone
from pipelines.backbone.training.loss_terms   import LOSS_TERMS
from pipelines.shared.dataset.dataset_spatial import Layout
from pipelines.shared.model.model_builder     import ModelBuilder


class TrialPlanner:
    """Base planner holding the warmup-stage loss configurations of a sweep.

    Attributes:
        warmup_losses : Mapping of trial label to a dict of loss-config field names and values applied to the warmup curriculum stage.
    """

    def __init__(self, warmup_losses: dict) -> None:
        """Stores the warmup loss variants keyed by trial label."""
        self.warmup_losses = warmup_losses

    @staticmethod
    def _stage_overrides(stage: str, loss: dict) -> dict:
        """Returns the loss fields rewritten as `curriculum.<stage>.<field>` override paths."""
        return {f"curriculum.{stage}.{key}": value for key, value in loss.items()}


class CurriculumTrialPlanner(TrialPlanner):
    """Plans the full cross product of warmup and complete curriculum losses.

    Attributes:
        warmup_losses   : Loss variants applied to the warmup stage.
        complete_losses : Loss variants applied to the complete stage.
    """

    def __init__(self, warmup_losses: dict, complete_losses: dict) -> None:
        """Stores the warmup and complete loss variants of the curriculum grid."""
        super().__init__(warmup_losses)
        self.complete_losses = complete_losses

    def summary(self) -> dict:
        """Returns the counts of warmup and complete loss variants for logging."""
        return {"Warmup losses": len(self.warmup_losses), "Complete losses": len(self.complete_losses)}

    def plan(self) -> list[tuple[str, dict]]:
        """Returns one `(run_name, overrides)` pair per warmup-by-complete loss combination."""
        plans = []

        for warmup_label, warmup_loss in self.warmup_losses.items():
            for complete_label, complete_loss in self.complete_losses.items():
                run_name  = f"w-{warmup_label}_c-{complete_label}"
                overrides = {"curriculum.enabled": True, "curriculum.inherit": False}
                overrides.update(self._stage_overrides("warmup",   warmup_loss))
                overrides.update(self._stage_overrides("complete", complete_loss))
                plans.append((run_name, overrides))

        return plans


class WarmupTrialPlanner(TrialPlanner):
    """Plans one curriculum-free run per warmup loss variant."""

    def summary(self) -> dict:
        """Returns the number of loss variants for logging."""
        return {"Warmup losses": len(self.warmup_losses)}

    def plan(self) -> list[tuple[str, dict]]:
        """Returns one `(run_name, overrides)` pair per loss variant, curriculum disabled."""
        plans = []

        for label, loss in self.warmup_losses.items():
            run_name  = f"nc-{label}"
            overrides = {"curriculum.enabled": False}
            overrides.update(self._stage_overrides("complete", loss))
            plans.append((run_name, overrides))

        return plans


class SlotPresenceTrialPlanner:
    """Plans runs toggling the slot-presence flags of the loss.

    Attributes:
        presence_trials : Mapping of trial label to a dict setting every flag in `FLAG_KEYS` to a boolean.
    """

    FLAG_KEYS = ("use_active_normalization", "presence_balance")

    def __init__(self, presence_trials: dict) -> None:
        """Stores and validates the presence-flag trial specifications.

        Args:
            presence_trials : Mapping of trial label to boolean flag settings.

        Raises:
            ValueError : If the mapping is empty or any trial omits, adds or mistypes a flag.
        """
        self.presence_trials = presence_trials

        self._validate()

    def _validate(self) -> None:
        """Checks that every trial sets exactly the known flags to booleans.

        Raises:
            ValueError : If no trial is given, or a trial has missing, unknown or non-boolean flags.
        """
        if not self.presence_trials:
            raise ValueError("presence_trials must list at least one trial")

        for label, spec in self.presence_trials.items():
            missing = [key for key in self.FLAG_KEYS if key not in spec]
            if missing:
                raise ValueError(f"Presence trial '{label}' must set {missing}; every trial defines all of {self.FLAG_KEYS}")

            unknown = [key for key in spec if key not in self.FLAG_KEYS]
            if unknown:
                raise ValueError(f"Presence trial '{label}' has unknown keys {unknown}; allowed keys are {self.FLAG_KEYS}")

            non_bool = [key for key in self.FLAG_KEYS if not isinstance(spec[key], bool)]
            if non_bool:
                raise ValueError(f"Presence trial '{label}' must set {non_bool} to booleans, got {spec}")

    def summary(self) -> dict:
        """Returns the trial labels and run count for logging."""
        return {
            "Presence trials" : list(self.presence_trials),
            "Total runs"      : len(self.presence_trials),
        }

    def _overrides(self, spec: dict) -> dict:
        """Returns the flag settings as complete-stage overrides with curriculum disabled."""
        overrides = {"curriculum.enabled": False}

        for key, value in spec.items():
            overrides[f"curriculum.complete.{key}"] = value

        return overrides

    def plan(self) -> list[tuple[str, dict]]:
        """Returns one `(run_name, overrides)` pair per presence trial."""
        plans = []

        for trial_label, spec in self.presence_trials.items():
            run_name = f"pr-{trial_label}"
            plans.append((run_name, self._overrides(spec)))

        return plans


class PairLossTrialPlanner:
    """Plans runs pairing a fixed base loss term with one weighted partner term.

    Attributes:
        trials : Pair-trial configuration with `base_component`, `base_weight`, `components`, `weights` and `include_baseline`.
        terms  : Registered loss terms keyed by component name.
    """

    def __init__(self, trials) -> None:
        """Stores and validates the pair sweep configuration against the loss term registry.

        Args:
            trials : Configuration object describing the base term and the partner terms and weights to sweep.

        Raises:
            ValueError : If any component or weight is unknown, duplicated or non-positive.
        """
        self.trials = trials
        self.terms  = {term.name: term for term in LOSS_TERMS}

        self._validate()

    def _validate(self) -> None:
        """Checks the base and partner components and weights against the loss registry.

        Raises:
            ValueError : On unknown components, a repeated base component, empty or duplicated lists, or non-positive weights.
        """
        names = tuple(self.terms)

        if self.trials.base_component not in self.terms:
            raise ValueError(f"Unknown pair_trials.base_component '{self.trials.base_component}'; allowed components are {names}")

        if self.trials.base_weight <= 0:
            raise ValueError(f"pair_trials.base_weight={self.trials.base_weight} must be positive")

        if not self.trials.components:
            raise ValueError("pair_trials.components must list at least one loss component to test")

        unknown = [component for component in self.trials.components if component not in self.terms]
        if unknown:
            raise ValueError(f"Unknown pair_trials.components {unknown}; allowed components are {names}")

        if self.trials.base_component in self.trials.components:
            raise ValueError(f"pair_trials.components must not repeat the base component '{self.trials.base_component}'")

        duplicates = sorted({component for component in self.trials.components if self.trials.components.count(component) > 1})
        if duplicates:
            raise ValueError(f"pair_trials.components must be unique, duplicated: {duplicates}")

        if not self.trials.weights:
            raise ValueError("pair_trials.weights must list at least one weight to test")

        if any(weight <= 0 for weight in self.trials.weights):
            raise ValueError(f"pair_trials.weights must all be positive, got {self.trials.weights}")

        if len(set(self.trials.weights)) != len(self.trials.weights):
            raise ValueError(f"pair_trials.weights must be unique, got {self.trials.weights}")

    def summary(self) -> dict:
        """Returns the base term, swept partners, weights and run count for logging."""
        return {
            "Base"       : f"{self.trials.base_component} @ {self.trials.base_weight:g}",
            "Components" : list(self.trials.components),
            "Weights"    : list(self.trials.weights),
            "Baseline"   : self.trials.include_baseline,
            "Total runs" : len(self.trials.components) * len(self.trials.weights) + (1 if self.trials.include_baseline else 0),
        }

    def _base_overrides(self) -> dict:
        """Returns overrides that switch every loss term off except the weighted base term."""
        overrides = {"curriculum.enabled": False, "curriculum.inherit": False}

        for term in self.terms.values():
            overrides[f"curriculum.complete.{term.use_flag}"]   = False
            overrides[f"curriculum.complete.{term.weight_key}"] = 0.0

        base = self.terms[self.trials.base_component]
        overrides[f"curriculum.complete.{base.use_flag}"]   = True
        overrides[f"curriculum.complete.{base.weight_key}"] = self.trials.base_weight

        return overrides

    def plan(self) -> list[tuple[str, dict]]:
        """Returns the optional base-only baseline plus one run per partner term and weight."""
        plans = []

        if self.trials.include_baseline:
            plans.append(("pair-baseline", self._base_overrides()))

        for component in self.trials.components:
            term = self.terms[component]
            for weight in self.trials.weights:
                run_name  = f"pair-{component}-w{weight:g}"
                overrides = self._base_overrides()
                overrides[f"curriculum.complete.{term.use_flag}"]   = True
                overrides[f"curriculum.complete.{term.weight_key}"] = weight
                plans.append((run_name, overrides))

        return plans


class LogEnergyTrialPlanner:
    """Plans a weight sweep of the log-energy curve term across both stages.

    Attributes:
        trials : Log-energy trial configuration with `weights`, `replace_covariance` and `include_baseline`.
    """

    STAGES = ("warmup", "complete")

    def __init__(self, trials) -> None:
        """Stores and validates the log-energy sweep configuration.

        Args:
            trials : Configuration object carrying the weights to sweep.

        Raises:
            ValueError : If the weights are empty, duplicated or non-positive.
        """
        self.trials = trials

        self._validate()

    def _validate(self) -> None:
        """Checks that the swept weights are present, unique and positive.

        Raises:
            ValueError : On an empty, duplicated or non-positive weight list.
        """
        if not self.trials.weights:
            raise ValueError("log_energy_trials.weights must list at least one weight to test")

        if any(weight <= 0 for weight in self.trials.weights):
            raise ValueError(f"log_energy_trials.weights must all be positive, got {self.trials.weights}")

        if len(set(self.trials.weights)) != len(self.trials.weights):
            raise ValueError(f"log_energy_trials.weights must be unique, got {self.trials.weights}")

    def summary(self) -> dict:
        """Returns the swept weights, moments replacement flag and run count."""
        n_extras = 1 if self.trials.include_baseline else 0

        return {
            "Weights"            : list(self.trials.weights),
            "Replace covariance" : self.trials.replace_covariance,
            "Baseline"           : self.trials.include_baseline,
            "Total runs"         : len(self.trials.weights) + n_extras,
        }

    def _neutral_overrides(self) -> dict:
        """Returns overrides switching the log-energy term (and optionally the moments term) off in both stages."""
        overrides = {"curriculum.inherit": False}

        for stage in self.STAGES:
            overrides[f"curriculum.{stage}.use_log_energy_curve"]    = False
            overrides[f"curriculum.{stage}.weight_log_energy_curve"] = 0.0

        if self.trials.replace_covariance:
            for stage in self.STAGES:
                overrides[f"curriculum.{stage}.use_moments"]    = False
                overrides[f"curriculum.{stage}.weight_moments"] = 0.0

        return overrides

    def plan(self) -> list[tuple[str, dict]]:
        """Returns the optional baseline plus one run per log-energy weight."""
        plans = []

        if self.trials.include_baseline:
            plans.append(("logE-baseline", self._neutral_overrides()))

        for weight in self.trials.weights:
            overrides = self._neutral_overrides()
            for stage in self.STAGES:
                overrides[f"curriculum.{stage}.use_log_energy_curve"]    = True
                overrides[f"curriculum.{stage}.weight_log_energy_curve"] = weight
            plans.append((f"logE-w{weight:g}", overrides))

        return plans


class AblationTrialPlanner:
    """Plans a cumulative ablation ladder that degrades features one at a time.

    Attributes:
        features     : Ordered feature specs, each with a `label`, an `enable` override dict and a `degrade` override dict.
        include_full : Whether the fully enabled model is planned as rung zero.
    """

    def __init__(self, features: list, include_full: bool) -> None:
        """Stores and validates the ordered ablation ladder.

        Args:
            features     : Ordered feature specifications to ablate.
            include_full : Whether to plan the all-features-enabled rung.

        Raises:
            ValueError : If the list is empty, a feature lacks a required key or labels repeat.
        """
        self.features     = list(features)
        self.include_full = include_full

        self._validate()

    def _validate(self) -> None:
        """Checks that every feature defines label, enable and degrade with unique labels.

        Raises:
            ValueError : On an empty ladder, missing keys or duplicated labels.
        """
        if not self.features:
            raise ValueError("ablation_features must list at least one feature to ablate")

        for index, feature in enumerate(self.features):
            missing = [key for key in ("label", "enable", "degrade") if key not in feature]
            if missing:
                raise ValueError(f"Ablation feature #{index} must define {missing}, got {feature}")

        labels     = [feature["label"] for feature in self.features]
        duplicates = sorted({label for label in labels if labels.count(label) > 1})
        if duplicates:
            raise ValueError(f"Ablation feature labels must be unique, duplicated: {duplicates}; duplicate labels collapse distinct trials onto one run directory")

    def summary(self) -> dict:
        """Returns the ablation order and run count for logging."""
        return {
            "Ablation features" : len(self.features),
            "Order"             : " -> ".join(feature["label"] for feature in self.features),
            "Total runs"        : len(self.features) + (1 if self.include_full else 0),
        }

    def _enabled_overrides(self) -> dict:
        """Returns the merged enable overrides of every feature."""
        merged = {}
        for feature in self.features:
            merged.update(feature["enable"])
        return merged

    def _degraded_prefix(self, count: int) -> dict:
        """Returns the merged degrade overrides of the first `count` features."""
        merged = {}
        for feature in self.features[:count]:
            merged.update(feature["degrade"])
        return merged

    def _run_name(self, step: int) -> str:
        """Returns the run name of the ladder rung that degraded `step` features."""
        if step == 0:
            return "abl-0-full"
        if step == len(self.features):
            return f"abl-{step}-baseline"
        return f"abl-{step}-no_{self.features[step - 1]['label']}"

    def plan(self) -> list[tuple[str, dict]]:
        """Returns the ladder rungs, each degrading one more feature than the last."""
        enabled = self._enabled_overrides()
        plans   = []

        if self.include_full:
            plans.append((self._run_name(0), dict(enabled)))

        for step in range(1, len(self.features) + 1):
            overrides = {**enabled, **self._degraded_prefix(step)}
            plans.append((self._run_name(step), overrides))

        return plans


class ContextTrialPlanner:
    """Plans a context ladder of backbone rungs with per-rung model overrides.

    Attributes:
        raw_trials     : The rungs exactly as configured, used for error reporting.
        trials         : The dict-shaped rungs kept for planning.
        registry_names : Backbone names registered in the model registry.
    """

    RESERVED_KEYS = ("label", "backbone")

    def __init__(self, trials: list, registry_names: tuple) -> None:
        """Stores and validates the context ladder rungs.

        Args:
            trials         : Rungs, each a dict with `label`, `backbone` and optional model override keys.
            registry_names : Names of the registered backbones.

        Raises:
            ValueError : If a rung is not a dict, lacks a required key, repeats a label or names an unregistered backbone.
        """
        self.raw_trials     = list(trials)
        self.trials         = [dict(trial) for trial in trials if isinstance(trial, dict)]
        self.registry_names = tuple(registry_names)

        self._validate()

    def _validate(self) -> None:
        """Checks rung shape, required keys, label uniqueness and backbone registration.

        Raises:
            ValueError : On malformed rungs, an empty ladder, missing keys, duplicate labels or unknown backbones.
        """
        malformed = [trial for trial in self.raw_trials if not isinstance(trial, dict)]
        if malformed:
            raise ValueError(f"context_trials rungs must be dicts of label, backbone and model overrides, got {malformed}; bare backbone names are the pre-rung format and must be rewritten")

        if not self.trials:
            raise ValueError("context_trials must list at least one rung")

        for trial in self.trials:
            missing = [key for key in self.RESERVED_KEYS if key not in trial]
            if missing:
                raise ValueError(f"context_trials rung {trial} is missing required keys {missing}")

        labels     = [trial["label"] for trial in self.trials]
        duplicates = sorted({label for label in labels if labels.count(label) > 1})
        if duplicates:
            raise ValueError(f"context_trials labels must be unique, duplicated: {duplicates}")

        unknown = sorted({trial["backbone"] for trial in self.trials if trial["backbone"] not in self.registry_names})
        if unknown:
            raise ValueError(f"Unknown context_trials backbones {unknown}; registered backbones are {sorted(self.registry_names)}")

    def _overrides(self, trial: dict) -> dict:
        """Returns the rung keys that are model overrides rather than reserved metadata."""
        return {key: value for key, value in trial.items() if key not in self.RESERVED_KEYS}

    def _settings(self, trial: dict) -> dict:
        """Returns the entry-config settings selecting the rung backbone and its overrides."""
        overrides = self._overrides(trial)
        settings  = {"backbone_name": trial["backbone"]}

        if overrides:
            settings["model_overrides"] = overrides

        return settings

    def summary(self) -> dict:
        """Returns the ladder rungs and run count for logging."""
        return {
            "Rungs"      : [f"{trial['label']}:{trial['backbone']}" for trial in self.trials],
            "Total runs" : len(self.trials),
        }

    def plan(self) -> list[tuple[str, dict]]:
        """Returns one `(run_name, settings)` pair per context rung."""
        return [(f"ctx-{trial['label']}", self._settings(trial)) for trial in self.trials]


class ReachTrialPlanner:
    """Plans capacity-matched architecture rungs for the spatial-reach comparison.

    Every rung is instantiated at build time so its parameter count can be
    verified to sit inside the configured match tolerance; otherwise the
    comparison would confound architecture with capacity.

    Attributes:
        trials               : Reach trial configuration with `rungs`, `patch_size`, `patch_stride`, `in_channels` and `match_tolerance`.
        rungs                : The dict-shaped rungs kept for planning.
        registry_names       : Names of the registered backbones.
        head                 : Output head every rung is built with.
        expected_in_channels : Input channel count implied by the dataset input configuration.
        out_channels         : Output channel count implied by the Gaussian head.
        parameter_counts     : Parameter count per rung label.
    """

    RESERVED_KEYS = ("label", "backbone")

    def __init__(self, trials, registry_names: tuple, head: str, expected_in_channels: int, out_channels: int) -> None:
        """Validates the reach rungs and verifies that their parameter counts match.

        Args:
            trials               : Reach trial configuration object.
            registry_names       : Names of the registered backbones.
            head                 : Output head used when instantiating every rung.
            expected_in_channels : Input width the dataset actually yields.
            out_channels         : Output width the Gaussian head actually yields.

        Raises:
            ValueError : On malformed rungs, fewer than two rungs, unknown backbones, non-positive patch geometry, a channel-count mismatch with the dataset, or parameter counts outside the tolerance.
        """
        self.trials               = trials
        self.rungs                = [dict(rung) for rung in trials.rungs if isinstance(rung, dict)]
        self.registry_names       = tuple(registry_names)
        self.head                 = head
        self.expected_in_channels = expected_in_channels
        self.out_channels         = out_channels

        self._validate()

        self.parameter_counts = {rung["label"]: self._parameters(rung) for rung in self.rungs}

        self._verify_capacity_match()

    def _validate(self) -> None:
        """Checks rung shape, backbone registration, patch geometry and channel widths.

        Raises:
            ValueError : On any malformed, incomplete or inconsistent rung or sweep setting.
        """
        malformed = [rung for rung in self.trials.rungs if not isinstance(rung, dict)]
        if malformed:
            raise ValueError(f"reach_trials.rungs must be dicts of label, backbone and model overrides, got {malformed}")

        if len(self.rungs) < 2:
            raise ValueError("reach_trials.rungs must list at least two architectures; a reach comparison needs something to compare against")

        for rung in self.rungs:
            missing = [key for key in self.RESERVED_KEYS if key not in rung]
            if missing:
                raise ValueError(f"reach_trials rung {rung} is missing required keys {missing}")

        labels     = [rung["label"] for rung in self.rungs]
        duplicates = sorted({label for label in labels if labels.count(label) > 1})
        if duplicates:
            raise ValueError(f"reach_trials labels must be unique, duplicated: {duplicates}")

        unknown = sorted({rung["backbone"] for rung in self.rungs if rung["backbone"] not in self.registry_names})
        if unknown:
            raise ValueError(f"Unknown reach_trials backbones {unknown}; registered backbones are {sorted(self.registry_names)}")

        if any(size < 1 for size in self.trials.patch_size) or any(stride < 1 for stride in self.trials.patch_stride):
            raise ValueError(f"reach_trials.patch_size={tuple(self.trials.patch_size)} and patch_stride={tuple(self.trials.patch_stride)} must be positive")

        if self.trials.match_tolerance <= 0:
            raise ValueError(f"reach_trials.match_tolerance={self.trials.match_tolerance} must be positive; the arms are only comparable when their capacity is matched to a stated tolerance")

        if self.trials.in_channels != self.expected_in_channels:
            raise ValueError(f"reach_trials.in_channels={self.trials.in_channels} but the input configuration yields {self.expected_in_channels} channels; training builds every arm at the dataset width, so the capacity check would count the wrong first layer")

        if self.out_channels < 1:
            raise ValueError(f"reach_trials capacity check needs the resolved output width, got out_channels={self.out_channels}; training builds every arm at the dataset Gaussian count, so the capacity check would count the wrong output stage")

    def _overrides(self, rung: dict) -> dict:
        """Returns the rung keys that are model overrides rather than reserved metadata."""
        return {key: value for key, value in rung.items() if key not in self.RESERVED_KEYS}

    def _parameters(self, rung: dict) -> int:
        """Returns the parameter count of the rung backbone built at the sweep widths."""
        sizing   = ModelBuilder.image_size_override(rung["backbone"], self.trials.patch_size)
        model, _ = get_backbone(rung["backbone"], in_channels=self.trials.in_channels, out_channels=self.out_channels, head=self.head, **sizing, **self._overrides(rung))

        return sum(parameter.numel() for parameter in model.parameters())

    def _verify_capacity_match(self) -> None:
        """Checks that the rung parameter counts spread no wider than the match tolerance.

        Raises:
            ValueError : If the relative spread between the smallest and largest rung exceeds `trials.match_tolerance`.
        """
        low, high = min(self.parameter_counts.values()), max(self.parameter_counts.values())
        deviation = (high - low) / low

        if deviation > self.trials.match_tolerance:
            counts = {label: f"{count:,}" for label, count in self.parameter_counts.items()}
            raise ValueError(f"reach_trials arms differ by {100 * deviation:.2f} % in parameter count ({counts}) at head '{self.head}' with {self.out_channels} output channels, exceeding the {100 * self.trials.match_tolerance:.1f} % match tolerance; the comparison only isolates architecture from capacity when the arms are size matched, so retune the rung widths")

    def _settings(self, rung: dict) -> dict:
        """Returns the entry-config settings pinning the rung backbone, overrides and patch geometry."""
        overrides = self._overrides(rung)
        settings  = {
            "backbone_name"         : rung["backbone"],
            "training.patch_size"   : tuple(self.trials.patch_size),
            "training.patch_stride" : tuple(self.trials.patch_stride),
        }

        if overrides:
            settings["model_overrides"] = overrides

        return settings

    def summary(self) -> dict:
        """Returns the rungs, their parameter counts, counting widths and patch geometry."""
        return {
            "Rungs"      : [f"{rung['label']}:{rung['backbone']}" for rung in self.rungs],
            "Parameters" : {label: f"{count:,}" for label, count in self.parameter_counts.items()},
            "Counted at" : f"head '{self.head}', {self.trials.in_channels} in / {self.out_channels} out channels",
            "Patch"      : f"{tuple(self.trials.patch_size)} stride {tuple(self.trials.patch_stride)}",
            "Total runs" : len(self.rungs),
        }

    def plan(self) -> list[tuple[str, dict]]:
        """Returns one `(run_name, settings)` pair per capacity-matched reach rung."""
        return [(f"reach-{rung['label']}", self._settings(rung)) for rung in self.rungs]


class HeadMatchingTrialPlanner:
    """Plans the cross product of output heads and parameter-matching strategies.

    The backbone comes from the entry that launches the sweep, not from the trial
    configuration, so the block that names a backbone is the one the runs use.

    Attributes:
        trials         : Head trial configuration with `heads` and `matchings`.
        backbone       : Backbone family held fixed across the trials, taken from the entry configuration.
        registry_names : Names of the registered backbones.
        head_names     : Registered output head names.
        matching_names : Allowed parameter-matching strategy names.
    """

    def __init__(self, trials, backbone: str, registry_names: tuple, head_names: tuple, matching_names: tuple) -> None:
        """Stores and validates the head-by-matching grid.

        Args:
            trials         : Configuration object naming the heads and matchings.
            backbone       : Backbone family every trial trains, as the entry configuration names it.
            registry_names : Names of the registered backbones.
            head_names     : Registered output head names.
            matching_names : Allowed parameter-matching strategy names.

        Raises:
            ValueError : On an unknown backbone, head or matching, or on empty or duplicated head/matching lists.
        """
        self.trials         = trials
        self.backbone       = backbone
        self.registry_names = tuple(registry_names)
        self.head_names     = tuple(head_names)
        self.matching_names = tuple(matching_names)

        self._validate()

    def _validate(self) -> None:
        """Checks the backbone, heads and matchings against the registries.

        Raises:
            ValueError : On unknown, empty or duplicated entries.
        """
        t = self.trials

        if self.backbone not in self.registry_names:
            raise ValueError(f"Unknown backbone '{self.backbone}'; registered backbones are {sorted(self.registry_names)}")

        if not t.heads:
            raise ValueError("head_trials.heads must list at least one head")

        unknown = [head for head in t.heads if head not in self.head_names]
        if unknown:
            raise ValueError(f"Unknown head_trials.heads {unknown}; registered heads are {self.head_names}")

        duplicates = sorted({head for head in t.heads if t.heads.count(head) > 1})
        if duplicates:
            raise ValueError(f"head_trials.heads must be unique, duplicated: {duplicates}")

        if not t.matchings:
            raise ValueError("head_trials.matchings must list at least one matching strategy")

        unknown = [matching for matching in t.matchings if matching not in self.matching_names]
        if unknown:
            raise ValueError(f"Unknown head_trials.matchings {unknown}; allowed matchings are {self.matching_names}")

        duplicates = sorted({matching for matching in t.matchings if t.matchings.count(matching) > 1})
        if duplicates:
            raise ValueError(f"head_trials.matchings must be unique, duplicated: {duplicates}")

    def summary(self) -> dict:
        """Returns the backbone, swept heads and matchings and the run count."""
        return {
            "Backbone"   : self.backbone,
            "Heads"      : list(self.trials.heads),
            "Matchings"  : list(self.trials.matchings),
            "Total runs" : len(self.trials.heads) * len(self.trials.matchings),
        }

    def plan(self) -> list[tuple[str, dict]]:
        """Returns one `(run_name, overrides)` pair per head and matching strategy."""
        plans = []

        for head in self.trials.heads:
            for matching in self.trials.matchings:
                run_name  = f"hm-{head}-{matching}"
                overrides = {
                    "backbone_name"                      : self.backbone,
                    "backbone_head"                      : head,
                    "curriculum.warmup.param_matching"   : matching,
                    "curriculum.complete.param_matching" : matching,
                }
                plans.append((run_name, overrides))

        return plans


class InputTrialPlanner:
    """Plans runs over input-channel variants and the track scope they use.

    Attributes:
        input_trials : Mapping of trial label to a spec setting `tracks` plus any of the input toggles in `INPUT_KEYS`.
        candidates   : Secondary track labels available in the dataset.
    """

    INPUT_KEYS   = ("use_primary", "use_secondaries", "use_derived", "use_aux")
    TRACKS_KEY   = "tracks"
    TRACK_SCOPES = ("all", "reduced")

    def __init__(self, input_trials: dict, candidates: list[str]) -> None:
        """Stores and validates the input variants against the available secondaries.

        Args:
            input_trials : Mapping of trial label to input specification.
            candidates   : Secondary track labels available in the dataset.

        Raises:
            ValueError : If no trial is given, a trial omits a valid `tracks` scope, or a trial carries unknown keys.
        """
        self.input_trials = input_trials
        self.candidates   = list(candidates)

        self._validate()

    @classmethod
    def from_dataset(cls, input_trials: dict, dataset_path) -> "InputTrialPlanner":
        """Builds a planner whose channel candidates are every secondary channel the dataset records.

        Args:
            input_trials : Mapping of trial label to input specification.
            dataset_path : Root of the dataset whose data/dataset.json lists the channel labels.

        Returns:
            A planner with every secondary channel of the dataset as candidate, so the 'all' scope reaches past the configured selection.

        Raises:
            FileNotFoundError : If the dataset has no data/dataset.json.
            ValueError        : If the dataset records no channel labels or no secondary channel.
        """
        labels = Layout.secondary_channel_labels(dataset_path)
        if not labels:
            raise ValueError(f"Input trials need at least one secondary channel to enumerate channel scopes, but the dataset at {dataset_path} records only its reference channel")

        return cls(input_trials, labels)

    def _validate(self) -> None:
        """Checks that every trial names a valid track scope and only known input keys.

        Raises:
            ValueError : On an empty mapping, an invalid scope or unknown keys.
        """
        if not self.input_trials:
            raise ValueError("input_trials must list at least one input variant")

        for label, spec in self.input_trials.items():
            if spec.get(self.TRACKS_KEY) not in self.TRACK_SCOPES:
                raise ValueError(f"Input trial '{label}' must set '{self.TRACKS_KEY}' to one of {self.TRACK_SCOPES}, got {spec.get(self.TRACKS_KEY)!r}")

            unknown = [key for key in spec if key != self.TRACKS_KEY and key not in self.INPUT_KEYS]
            if unknown:
                raise ValueError(f"Input trial '{label}' has unknown keys {unknown}; allowed keys are ('{self.TRACKS_KEY}',) + {self.INPUT_KEYS}")

    def summary(self) -> dict:
        """Returns the variant count and the split between all-track and reduced-track trials."""
        scopes = [spec[self.TRACKS_KEY] for spec in self.input_trials.values()]

        return {
            "Input variants" : len(self.input_trials),
            "Tracks"         : f"{scopes.count('all')} all ({len(self.candidates)} secondaries), {scopes.count('reduced')} reduced (configured selection)",
        }

    def _overrides(self, spec: dict) -> dict:
        """Returns the input toggles as `input.*` overrides, widening the secondaries when the scope is 'all'."""
        overrides = {f"input.{key}": value for key, value in spec.items() if key != self.TRACKS_KEY}

        if spec[self.TRACKS_KEY] == "all":
            overrides["paths.secondary_labels"] = tuple(self.candidates)

        return overrides

    def plan(self) -> list[tuple[str, dict]]:
        """Returns one `(run_name, overrides)` pair per input variant."""
        plans = []

        for label, spec in self.input_trials.items():
            run_name = f"in-{label}"
            plans.append((run_name, self._overrides(spec)))

        return plans


class AugmentationTrialPlanner:
    """Plans the cumulative augmentation ladder from no augmentation to flips plus noise.

    The ladder adds one augmentation per rung: horizontal flips, then vertical
    flips on top, then additive Gaussian input noise on top of both flips, one
    final rung per configured noise strength.

    Attributes:
        trials : Augmentation trial configuration with `flip_probability`, `noise_probability` and `noise_stds`.
    """

    def __init__(self, trials) -> None:
        """Stores and validates the augmentation ladder configuration.

        Args:
            trials : Configuration object carrying the flip and noise probabilities and the noise strengths.

        Raises:
            ValueError : If a probability falls outside (0, 1] or the noise strengths are empty, non-positive or duplicated.
        """
        self.trials = trials

        self._validate()

    def _validate(self) -> None:
        """Checks the probabilities and the noise strength list.

        Raises:
            ValueError : On an out-of-range probability, or an empty, non-positive or duplicated noise strength list.
        """
        if not 0 < self.trials.flip_probability <= 1:
            raise ValueError(f"augmentation_trials.flip_probability={self.trials.flip_probability} must be in (0, 1]")

        if not 0 < self.trials.noise_probability <= 1:
            raise ValueError(f"augmentation_trials.noise_probability={self.trials.noise_probability} must be in (0, 1]")

        if not self.trials.noise_stds:
            raise ValueError("augmentation_trials.noise_stds must list at least one noise strength")

        if any(std <= 0 for std in self.trials.noise_stds):
            raise ValueError(f"augmentation_trials.noise_stds must all be positive, got {self.trials.noise_stds}")

        if len(set(self.trials.noise_stds)) != len(self.trials.noise_stds):
            raise ValueError(f"augmentation_trials.noise_stds must be unique, got {self.trials.noise_stds}")

    def summary(self) -> dict:
        """Returns the ladder order, probabilities and run count for logging."""
        return {
            "Ladder"      : "none -> +hflip -> +vflip -> +noise",
            "Flip p"      : self.trials.flip_probability,
            "Noise p"     : self.trials.noise_probability,
            "Noise stds"  : list(self.trials.noise_stds),
            "Total runs"  : 3 + len(self.trials.noise_stds),
        }

    def _overrides(self, flip_h: bool, flip_v: bool, noise_std: float | None) -> dict:
        """Returns the `augmentation.*` overrides of one ladder rung.

        Args:
            flip_h    : Whether the rung trains with horizontal flips.
            flip_v    : Whether the rung trains with vertical flips.
            noise_std : Noise standard deviation of the rung, or None for no noise.

        Returns:
            Overrides setting every augmentation probability explicitly, plus
            the noise strength on noise rungs.
        """
        overrides = {
            "augmentation.p_flip_h" : self.trials.flip_probability if flip_h else 0.0,
            "augmentation.p_flip_v" : self.trials.flip_probability if flip_v else 0.0,
            "augmentation.p_rot90"  : 0.0,
            "augmentation.p_noise"  : self.trials.noise_probability if noise_std is not None else 0.0,
        }

        if noise_std is not None:
            overrides["augmentation.noise_std"] = noise_std

        return overrides

    def plan(self) -> list[tuple[str, dict]]:
        """Returns the ladder rungs: none, horizontal flips, both flips, then both flips plus each noise strength."""
        plans = [
            ("aug-0-none",  self._overrides(False, False, None)),
            ("aug-1-hflip", self._overrides(True,  False, None)),
            ("aug-2-vflip", self._overrides(True,  True,  None)),
        ]

        for std in self.trials.noise_stds:
            plans.append((f"aug-3-noise{std:g}", self._overrides(True, True, std)))

        return plans


class ClampTrialPlanner:
    """Plans the output-clamp ladder: no clamp, hard clamp, one leaky rung per slope pair.

    Every rung trains the same model; only the shape of the output clamp used
    on denormalization changes. The hard rung zeroes both leak slopes, each
    leaky rung uses one configured slope pair, and the clamp bounds always
    come from the run's normalization config.

    Attributes:
        trials : Clamp trial configuration with the `leaky_slopes` and `param_leaky_slopes` lists.
    """

    def __init__(self, trials) -> None:
        """Stores and validates the clamp ladder configuration.

        Args:
            trials : Configuration object carrying the leaky-rung slope lists.

        Raises:
            ValueError : On empty, mismatched, non-positive or duplicated slope lists.
        """
        self.trials = trials

        self._validate()

    def _validate(self) -> None:
        """Checks that the leaky-rung slope lists pair up, stay positive and stay unique.

        Raises:
            ValueError : On an empty, length-mismatched, non-positive or duplicated slope list.
        """
        slopes       = list(self.trials.leaky_slopes)
        param_slopes = list(self.trials.param_leaky_slopes)

        if not slopes:
            raise ValueError("clamp_trials.leaky_slopes must name at least one slope for the leaky rungs")

        if len(param_slopes) != len(slopes):
            raise ValueError(f"clamp_trials.param_leaky_slopes has {len(param_slopes)} entries but leaky_slopes has {len(slopes)}; the lists pair up by position, one leaky rung each")

        for slope in slopes + param_slopes:
            if slope <= 0:
                raise ValueError(f"clamp_trials slope {slope} must be positive; a zero slope makes a leaky rung identical to the hard rung")

        if len(set(slopes)) != len(slopes):
            raise ValueError(f"clamp_trials.leaky_slopes {slopes} must be unique; the curve slope names each leaky rung")

    def summary(self) -> dict:
        """Returns the ladder order, leaky-rung slope pairs and run count for logging."""
        pairs = ", ".join(f"{slope:g}/{param:g}" for slope, param in zip(self.trials.leaky_slopes, self.trials.param_leaky_slopes))
        return {
            "Ladder"       : "none -> hard -> leaky per slope",
            "Leaky slopes" : f"{pairs} (curve/param)",
            "Total runs"   : 2 + len(self.trials.leaky_slopes),
        }

    def _overrides(self, enabled: bool, leaky_slope: float, param_leaky_slope: float) -> dict:
        """Returns the `normalization.*` clamp overrides of one ladder rung."""
        return {
            "normalization.clamp_output"            : enabled,
            "normalization.clamp_leaky_slope"       : leaky_slope,
            "normalization.param_clamp_leaky_slope" : param_leaky_slope,
        }

    def plan(self) -> list[tuple[str, dict]]:
        """Returns the disabled and hard rungs followed by one leaky rung per slope pair."""
        rungs = [
            ("clamp-0-none", self._overrides(False, 0.0, 0.0)),
            ("clamp-1-hard", self._overrides(True,  0.0, 0.0)),
        ]
        for slope, param_slope in zip(self.trials.leaky_slopes, self.trials.param_leaky_slopes):
            rungs.append((f"clamp-2-leaky-s{slope:g}", self._overrides(True, slope, param_slope)))

        return rungs


class ArchitecturePatchStep:
    """Derives the admissible patch step from a U-Net encoder pyramid.

    Attributes:
        backbone_name   : Registry name of the backbone.
        backbone_head   : Head name applied on top of the backbone.
        model_overrides : Model configuration overrides applied to the registry defaults.
    """

    PYRAMID_BACKBONES = frozenset({"unet", "unet_skip", "resunet", "attention_unet"})

    def __init__(self, backbone_name: str, backbone_head: str, model_overrides: dict) -> None:
        """Stores the backbone identity whose pyramid depth fixes the patch step.

        Args:
            backbone_name   : Registry name of the backbone.
            backbone_head   : Head name applied on top of the backbone.
            model_overrides : Model configuration overrides.
        """
        self.backbone_name   = backbone_name
        self.backbone_head   = backbone_head
        self.model_overrides = model_overrides

    def resolve(self) -> int:
        """Returns the admissible patch step in pixels, 2 raised to the pyramid depth.

        Returns:
            The smallest patch length increment that survives every downsampling stage.

        Raises:
            ValueError : If the backbone is not a verified one-halving-per-stage U-Net, or if its encoder feature pyramid is empty.
        """
        if self.backbone_name not in self.PYRAMID_BACKBONES:
            raise ValueError(f"Backbone '{self.backbone_name}' is not a verified one-halving-per-stage U-Net ({sorted(self.PYRAMID_BACKBONES)}); the admissible patch step cannot be derived from the architecture, set patch_trials.step explicitly")

        model_config = ModelBuilder.config_from_registry(self.backbone_name, self.model_overrides, head=self.backbone_head)
        features     = model_config.features

        if not features:
            raise ValueError(f"Backbone '{self.backbone_name}' has an empty encoder feature pyramid; set patch_trials.step explicitly")

        return 2 ** len(features)


class PatchSizeTrialPlanner:
    """Plans runs over a rectangular grid of patch shapes under a constant pixel budget.

    The admissible patch step follows the backbone's encoder pyramid, so every
    patch length stays divisible through all downsampling stages; each trial
    overrides the patch geometry and, under the constant pixel budget, rescales
    the batch size and its learning-rate reference together so every shape sees
    the same pixel count per optimizer step.

    Attributes:
        trials          : Patch trial configuration with the grid bounds, step, stride ratio and the pixel-budget switch.
        backbone_name   : Backbone whose pyramid derives the admissible step.
        backbone_head   : Output head attached to the backbone.
        model_overrides : Architecture overrides applied to the backbone.
        training        : Training queue config supplying the reference patch size, batch size and learning-rate reference batch size.
    """

    def __init__(self, trials, backbone_name: str, backbone_head: str, model_overrides: dict, training) -> None:
        """Stores and validates the patch-grid sweep configuration.

        Args:
            trials          : Configuration object carrying the grid bounds and stride ratio.
            backbone_name   : Registry name of the swept backbone.
            backbone_head   : Head name applied on top of the backbone.
            model_overrides : Model configuration overrides.
            training        : Training queue config with the pixel-budget reference values.

        Raises:
            ValueError : If the stride ratio falls outside (0, 1].
        """
        self.trials          = trials
        self.backbone_name   = backbone_name
        self.backbone_head   = backbone_head
        self.model_overrides = model_overrides
        self.training        = training

        self._patch_step  = None
        self._patch_sizes = None

        self._validate()

    def _validate(self) -> None:
        """Checks the stride ratio.

        Raises:
            ValueError : If the stride ratio falls outside (0, 1].
        """
        if not 0 < self.trials.stride_ratio <= 1:
            raise ValueError(f"patch_trials.stride_ratio={self.trials.stride_ratio} must be in (0, 1]")

    def patch_step(self) -> int:
        """Returns the admissible patch step in pixels, explicit or derived from the backbone."""
        if self._patch_step is None:
            explicit         = self.trials.step
            self._patch_step = explicit if explicit > 0 else ArchitecturePatchStep(self.backbone_name, self.backbone_head, self.model_overrides).resolve()

        return self._patch_step

    def _axis_sizes(self, axis: int, label: str) -> list[int]:
        """Returns the admissible patch lengths in pixels along one axis.

        Args:
            axis  : 0 for rows, 1 for cols.
            label : Axis name quoted in error messages.

        Returns:
            Patch lengths from the configured minimum to maximum in steps of the
            admissible patch step.

        Raises:
            ValueError : If the minimum is not a multiple of the step, or if the maximum lies below the first admissible size.
        """
        step  = self.patch_step()
        start = self.trials.minimum[axis] if self.trials.minimum[axis] > 0 else step

        if start % step != 0:
            raise ValueError(f"patch_trials.minimum {label}={start} is not a multiple of the admissible step {step}")
        if self.trials.maximum[axis] < start:
            raise ValueError(f"patch_trials.maximum {label}={self.trials.maximum[axis]} is below the first admissible size {start}")

        return list(range(start, self.trials.maximum[axis] + 1, step))

    def patch_sizes(self) -> tuple[list[int], list[int]]:
        """Returns the admissible row and column patch lengths in pixels.

        Returns:
            Tuple of the row lengths and the column lengths.

        Raises:
            ValueError : If their product yields fewer than two patch shapes.
        """
        if self._patch_sizes is not None:
            return self._patch_sizes

        row_sizes = self._axis_sizes(0, "row")
        col_sizes = self._axis_sizes(1, "col")

        if len(row_sizes) * len(col_sizes) < 2:
            raise ValueError(f"The grid rows {row_sizes} x cols {col_sizes} yields {len(row_sizes) * len(col_sizes)} patch shape; a sweep needs at least 2")

        self._patch_sizes = (row_sizes, col_sizes)

        return self._patch_sizes

    def _budget_scale(self, patch_size: tuple[int, int]) -> Fraction:
        """Returns the shared batch rescaling factor of one trial under the constant pixel budget.

        The factor is the ratio between the batch size whose pixel count matches
        the reference budget at this patch shape and the configured batch size.
        Applying the one factor to both the batch size and its learning-rate
        reference keeps their ratio exact, so the batch-linear learning-rate
        scale is identical across patch shapes.

        Args:
            patch_size : Patch geometry of this trial as (rows, cols) in pixels.

        Returns:
            The rescaling factor as an exact fraction.

        Raises:
            ValueError : If the patch is large enough that not even a single sample fits the reference pixel budget.
        """
        reference = self.training.patch_size
        rescaled  = self.training.batch_size * reference[0] * reference[1] // (patch_size[0] * patch_size[1])

        if rescaled < 1:
            raise ValueError(f"patch shape {patch_size} exceeds the pixel budget of training.batch_size={self.training.batch_size} at the reference patch {tuple(reference)}: not a single sample fits, so no batch size holds the budget")

        return Fraction(rescaled, self.training.batch_size)

    def _pixel_rescaled(self, base: int, scale: Fraction, label: str) -> int:
        """Applies the shared budget scale to one batch quantity.

        Args:
            base  : Value quoted at the reference patch size.
            scale : Shared rescaling factor of the trial.
            label : Training config field name quoted in error messages.

        Returns:
            The rescaled value in samples.

        Raises:
            ValueError : If the scale does not carry the value onto a positive whole number, which would drift the batch-linear learning-rate scale away from its configured value.
        """
        rescaled = base * scale

        if rescaled.denominator != 1 or rescaled < 1:
            raise ValueError(f"the constant pixel budget rescales training.{label}={base} by {scale} to {float(rescaled)}, which is not a positive whole number of samples; quote it so it rescales in lockstep with training.batch_size={self.training.batch_size}")

        return int(rescaled)

    @staticmethod
    def run_name(patch_size: tuple[int, int]) -> str:
        """Returns the trial run name encoding one (rows, cols) patch shape."""
        return f"p-{patch_size[0]:03d}x{patch_size[1]:03d}"

    def _batch_label(self) -> str:
        """Returns the batch size and learning-rate reference each trial trains with, per trial under the pixel budget."""
        if not self.trials.constant_pixel_budget:
            return f"{self.training.batch_size} / {self.training.lr_reference_batch_size} in every trial (constant pixel budget off)"

        return ", ".join(f"{name} {overrides['training.batch_size']} / {overrides['training.lr_reference_batch_size']}" for name, overrides in self.plan())

    def summary(self) -> dict:
        """Returns the patch step, both axis grids, the stride ratio, the budget switch, the pinned batch-size probe and the batch sizes each trial trains with."""
        row_sizes, col_sizes = self.patch_sizes()

        return {
            "Patch step"                : self.patch_step(),
            "Row sizes"                 : row_sizes,
            "Col sizes"                 : col_sizes,
            "Stride ratio"              : self.trials.stride_ratio,
            "Constant pixel budget"     : self.trials.constant_pixel_budget,
            "Batch-size probe"          : "off in every trial (pretrain.find_batch_size pinned to false)",
            "Batch size / LR reference" : self._batch_label(),
        }

    def plan(self) -> list[tuple[str, dict]]:
        """Returns one `(run_name, overrides)` pair per (rows, cols) patch shape.

        Every trial pins the patch geometry and disables the max-batch probe so
        the comparison stays controlled; under the constant pixel budget the
        batch size and its learning-rate reference are rescaled by one shared
        factor, which keeps the batch-linear learning-rate scale identical
        across trials.

        Raises:
            ValueError : If the grid yields fewer than two shapes, or if a shape cannot hold the pixel budget with whole batch sizes.
        """
        row_sizes, col_sizes = self.patch_sizes()

        plans = []
        for row_size in row_sizes:
            for col_size in col_sizes:
                size   = (row_size, col_size)
                stride = tuple(max(1, int(round(edge * self.trials.stride_ratio))) for edge in size)

                overrides = {
                    "training.patch_size"      : size,
                    "training.patch_stride"    : stride,
                    "pretrain.find_batch_size" : False,
                }

                if self.trials.constant_pixel_budget:
                    scale = self._budget_scale(size)

                    overrides["training.batch_size"]              = self._pixel_rescaled(self.training.batch_size, scale, "batch_size")
                    overrides["training.lr_reference_batch_size"] = self._pixel_rescaled(self.training.lr_reference_batch_size, scale, "lr_reference_batch_size")

                plans.append((self.run_name(size), overrides))

        return plans


class SecondaryTrialPlanner:
    """Plans runs over subsets of secondary channels drawn by a selection strategy.

    Subsets are drawn randomly (`uniform`, `gaussian`), swept as sliding blocks
    of the channel order (`consecutive`, `spaced`), or enumerated in full
    (`exhaustive`). Channels named in `trials.fixed` join every subset and leave
    the swept pool, so `n_secondaries` counts only the channels the strategy picks
    on top of them: one fixed channel and `n_secondaries` of 1 under `exhaustive`
    pairs that channel with every other secondary of the stack, one run each.

    Attributes:
        trials     : Secondary trial configuration carrying the strategy and its parameters.
        candidates : Secondary channel labels available in the dataset, in acquisition order.
        fixed      : Channel labels held in every subset, in acquisition order.
        pool       : Channel labels the strategy draws from, the candidates minus the fixed ones.
    """

    STRATEGIES   = ("uniform", "gaussian", "consecutive", "spaced", "exhaustive")
    MAX_ATTEMPTS = 1000

    def __init__(self, trials, candidates: list[str]) -> None:
        """Stores and validates the secondary-selection sweep configuration.

        Args:
            trials     : Configuration object naming the strategy and its parameters.
            candidates : Secondary channel labels available in the dataset.

        Raises:
            ValueError : On a repeated candidate label, an unknown or repeated fixed label, an unknown strategy or a parameter that is out of range for the chosen strategy.
        """
        self.trials     = trials
        self.candidates = list(candidates)
        self.fixed      = [label for label in self.candidates if label in tuple(trials.fixed)]
        self.pool       = [label for label in self.candidates if label not in tuple(trials.fixed)]

        self._validate()

    @classmethod
    def from_dataset(cls, trials, dataset_path) -> "SecondaryTrialPlanner":
        """Builds a planner whose candidates are every secondary channel the dataset records.

        Args:
            trials       : Configuration object naming the strategy and its parameters.
            dataset_path : Root of the dataset whose data/dataset.json lists the channel labels.

        Returns:
            A planner over every secondary channel of the dataset, independent of the configured selection.

        Raises:
            FileNotFoundError : If the dataset has no data/dataset.json.
            ValueError        : If the dataset records no channel labels or no secondary channel.
        """
        labels = Layout.secondary_channel_labels(dataset_path)
        if not labels:
            raise ValueError(f"Secondary trials need at least one secondary channel to sample from, but the dataset at {dataset_path} records only its reference channel")

        return cls(trials, labels)

    def _validate(self) -> None:
        """Checks the strategy name and the parameters it requires.

        Raises:
            ValueError : On a candidate list naming one channel twice, a fixed label that is unknown or repeated, an unknown strategy, a secondary count outside [1, len(pool)], a missing Gaussian mean or sigma, or a non-positive trial count, block step or spacing.
        """
        t = self.trials

        repeated = sorted({label for label in self.candidates if self.candidates.count(label) > 1})
        if repeated:
            raise ValueError(f"Secondary candidates name {repeated} more than once, so a subset could pair a channel with itself; every channel must appear once, regenerate the dataset's channel labels")

        unknown = [label for label in t.fixed if label not in self.candidates]
        if unknown:
            raise ValueError(f"secondary_trials.fixed names {unknown}, which are not secondary channels of this dataset; candidates are {self.candidates}")

        repeated_fixed = sorted({label for label in t.fixed if list(t.fixed).count(label) > 1})
        if repeated_fixed:
            raise ValueError(f"secondary_trials.fixed names {repeated_fixed} more than once; every fixed channel must appear once")

        if t.strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown secondary_trials.strategy '{t.strategy}', expected one of {self.STRATEGIES}")

        if not 1 <= t.n_secondaries <= len(self.pool):
            raise ValueError(f"secondary_trials.n_secondaries={t.n_secondaries} counts the swept channels and must be in [1, {len(self.pool)}], the channels left to sweep are {self.pool} and the fixed ones are {self.fixed}")

        if t.strategy in ("uniform", "gaussian") and t.n_trials < 1:
            raise ValueError(f"secondary_trials.n_trials={t.n_trials} must be >= 1 for strategy '{t.strategy}'")

        if t.strategy == "gaussian" and (t.mean is None or t.sigma is None):
            raise ValueError("secondary_trials strategy 'gaussian' requires explicit mean and sigma (index units over the secondary list)")

        if t.strategy in ("consecutive", "spaced") and t.block_step < 1:
            raise ValueError(f"secondary_trials.block_step={t.block_step} must be >= 1")

        if t.strategy == "spaced" and t.spacing < 1:
            raise ValueError(f"secondary_trials.spacing={t.spacing} must be >= 1")

    def summary(self) -> dict:
        """Returns the strategy, swept subset size, fixed channels and candidate pool, plus the run count when exhaustive."""
        exhaustive = self.trials.strategy == "exhaustive"

        summary = {
            "Strategy"    : self.trials.strategy,
            "Secondaries" : f"{self.trials.n_secondaries} (all combinations)" if exhaustive else self.trials.n_secondaries,
            "Fixed"       : f"{len(self.fixed)} -> {self.fixed}, in every trial on top of the swept ones" if self.fixed else "none",
            "Candidates"  : f"{len(self.pool)} -> {self.pool}",
        }

        if exhaustive:
            summary["Total runs"] = self._exhaustive_count()

        return summary

    def _pick(self, indices) -> list[str]:
        """Returns the fixed labels plus the pool labels at the given indices, in channel order."""
        chosen = set(self.fixed).union(self.pool[index] for index in indices)

        return [label for label in self.candidates if label in chosen]

    def _uniform(self) -> list[list[str]]:
        """Returns distinct subsets drawn uniformly at random without replacement.

        Returns:
            One label list per trial, each holding `n_secondaries` labels.

        Raises:
            RuntimeError : If too few distinct subsets are found within the attempt budget.
        """
        rng = np.random.default_rng(self.trials.seed)

        seen, selections = set(), []
        for _ in range(self.MAX_ATTEMPTS):
            if len(selections) == self.trials.n_trials:
                break

            indices = tuple(sorted(rng.choice(len(self.pool), size=self.trials.n_secondaries, replace=False).tolist()))
            if indices in seen:
                continue

            seen.add(indices)
            selections.append(self._pick(indices))

        self._ensure_complete(selections)
        return selections

    def _gaussian(self) -> list[list[str]]:
        """Returns distinct subsets drawn from a Gaussian over the candidate index axis.

        Returns:
            One label list per trial, each holding `n_secondaries` labels.

        Raises:
            RuntimeError : If too few distinct subsets are found within the attempt budget.
        """
        rng = np.random.default_rng(self.trials.seed)

        seen, selections = set(), []
        for _ in range(self.MAX_ATTEMPTS):
            if len(selections) == self.trials.n_trials:
                break

            indices = self._gaussian_draw(rng)
            if indices is None or indices in seen:
                continue

            seen.add(indices)
            selections.append(self._pick(indices))

        self._ensure_complete(selections)
        return selections

    def _gaussian_draw(self, rng) -> tuple | None:
        """Draws one Gaussian-distributed subset of candidate indices.

        Args:
            rng : NumPy generator supplying the normal draws.

        Returns:
            The sorted index tuple, or None if the attempt budget ran out before
            `n_secondaries` distinct in-range indices were collected.
        """
        indices = set()

        for _ in range(self.MAX_ATTEMPTS):
            if len(indices) == self.trials.n_secondaries:
                return tuple(sorted(indices))

            index = int(round(rng.normal(self.trials.mean, self.trials.sigma)))
            if 0 <= index < len(self.pool):
                indices.add(index)

        return None

    def _consecutive(self) -> list[list[str]]:
        """Returns every contiguous block of `n_secondaries` pool candidates, advanced by `block_step`."""
        n, step, total = self.trials.n_secondaries, self.trials.block_step, len(self.pool)

        return [self._pick(range(start, start + n)) for start in range(0, total - n + 1, step)]

    def _spaced(self) -> list[list[str]]:
        """Returns sliding blocks whose members are `spacing` pool candidates apart.

        Returns:
            One label list per block start, advanced by `block_step`.

        Raises:
            ValueError : If the span of one block exceeds the candidate count.
        """
        n, step, spacing, total = self.trials.n_secondaries, self.trials.block_step, self.trials.spacing, len(self.pool)

        span = (n - 1) * spacing + 1
        if span > total:
            raise ValueError(f"Spaced selection spans {span} candidates but only {total} are available; reduce n_secondaries or spacing")

        return [self._pick(range(start, start + span, spacing)) for start in range(0, total - span + 1, step)]

    def _exhaustive_count(self) -> int:
        """Returns the number of subsets an exhaustive enumeration would produce."""
        return comb(len(self.pool), self.trials.n_secondaries)

    def _exhaustive(self) -> list[list[str]]:
        """Returns every combination of `n_secondaries` pool candidates."""
        return [self._pick(indices) for indices in combinations(range(len(self.pool)), self.trials.n_secondaries)]

    def _ensure_complete(self, selections: list) -> None:
        """Checks that the requested number of distinct subsets was drawn.

        Args:
            selections : Subsets collected so far.

        Raises:
            RuntimeError : If fewer than `n_trials` subsets were found.
        """
        if len(selections) < self.trials.n_trials:
            raise RuntimeError(f"Found only {len(selections)} of {self.trials.n_trials} distinct secondary sets after {self.MAX_ATTEMPTS} attempts; lower n_trials, widen sigma or change strategy")

    def plan(self) -> list[tuple[str, dict]]:
        """Returns one `(run_name, overrides)` pair per secondary-channel subset of the strategy, each subset carrying the fixed channels."""
        selections = getattr(self, f"_{self.trials.strategy}")()

        plans = []
        for index, labels in enumerate(selections):
            run_name = f"sec-{self.trials.strategy}-t{index:02d}_{'-'.join(labels)}"
            plans.append((run_name, {"paths.secondary_labels": tuple(labels)}))

        return plans
