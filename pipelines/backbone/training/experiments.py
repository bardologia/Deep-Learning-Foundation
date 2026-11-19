
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

    def __init__(self, warmup_losses: dict) -> None:
        self.warmup_losses = warmup_losses

    @staticmethod
    def _stage_overrides(stage: str, loss: dict) -> dict:
        return {f"curriculum.{stage}.{key}": value for key, value in loss.items()}


class CurriculumTrialPlanner(TrialPlanner):

    def __init__(self, warmup_losses: dict, complete_losses: dict) -> None:
        super().__init__(warmup_losses)
        self.complete_losses = complete_losses

    def summary(self) -> dict:
        return {"Warmup losses": len(self.warmup_losses), "Complete losses": len(self.complete_losses)}

    def plan(self) -> list[tuple[str, dict]]:
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

    def summary(self) -> dict:
        return {"Warmup losses": len(self.warmup_losses)}

    def plan(self) -> list[tuple[str, dict]]:
        plans = []

        for label, loss in self.warmup_losses.items():
            run_name  = f"nc-{label}"
            overrides = {"curriculum.enabled": False}
            overrides.update(self._stage_overrides("complete", loss))
            plans.append((run_name, overrides))

        return plans


class SlotPresenceTrialPlanner:

    FLAG_KEYS = ("use_active_normalization", "presence_balance")

    def __init__(self, presence_trials: dict) -> None:
        self.presence_trials = presence_trials

        self._validate()

    def _validate(self) -> None:
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
        return {
            "Presence trials" : list(self.presence_trials),
            "Total runs"      : len(self.presence_trials),
        }

    def _overrides(self, spec: dict) -> dict:
        overrides = {"curriculum.enabled": False}

        for key, value in spec.items():
            overrides[f"curriculum.complete.{key}"] = value

        return overrides

    def plan(self) -> list[tuple[str, dict]]:
        plans = []

        for trial_label, spec in self.presence_trials.items():
            run_name = f"pr-{trial_label}"
            plans.append((run_name, self._overrides(spec)))

        return plans


class PairLossTrialPlanner:

    def __init__(self, trials) -> None:
        self.trials = trials
        self.terms  = {term.name: term for term in LOSS_TERMS}

        self._validate()

    def _validate(self) -> None:
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
        return {
            "Base"       : f"{self.trials.base_component} @ {self.trials.base_weight:g}",
            "Components" : list(self.trials.components),
            "Weights"    : list(self.trials.weights),
            "Baseline"   : self.trials.include_baseline,
            "Total runs" : len(self.trials.components) * len(self.trials.weights) + (1 if self.trials.include_baseline else 0),
        }

    def _base_overrides(self) -> dict:
        overrides = {"curriculum.enabled": False, "curriculum.inherit": False}

        for term in self.terms.values():
            overrides[f"curriculum.complete.{term.use_flag}"]   = False
            overrides[f"curriculum.complete.{term.weight_key}"] = 0.0

        base = self.terms[self.trials.base_component]
        overrides[f"curriculum.complete.{base.use_flag}"]   = True
        overrides[f"curriculum.complete.{base.weight_key}"] = self.trials.base_weight

        return overrides

    def plan(self) -> list[tuple[str, dict]]:
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

    STAGES = ("warmup", "complete")

    def __init__(self, trials) -> None:
        self.trials = trials

        self._validate()

    def _validate(self) -> None:
        if not self.trials.weights:
            raise ValueError("log_energy_trials.weights must list at least one weight to test")

        if any(weight <= 0 for weight in self.trials.weights):
            raise ValueError(f"log_energy_trials.weights must all be positive, got {self.trials.weights}")

        if len(set(self.trials.weights)) != len(self.trials.weights):
            raise ValueError(f"log_energy_trials.weights must be unique, got {self.trials.weights}")

    def summary(self) -> dict:
        n_extras = 1 if self.trials.include_baseline else 0

        return {
            "Weights"            : list(self.trials.weights),
            "Replace covariance" : self.trials.replace_covariance,
            "Baseline"           : self.trials.include_baseline,
            "Total runs"         : len(self.trials.weights) + n_extras,
        }

    def _neutral_overrides(self) -> dict:
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

    def __init__(self, features: list, include_full: bool) -> None:
        self.features     = list(features)
        self.include_full = include_full

        self._validate()

    def _validate(self) -> None:
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
        return {
            "Ablation features" : len(self.features),
            "Order"             : " -> ".join(feature["label"] for feature in self.features),
            "Total runs"        : len(self.features) + (1 if self.include_full else 0),
        }

    def _enabled_overrides(self) -> dict:
        merged = {}
        for feature in self.features:
            merged.update(feature["enable"])
        return merged

    def _degraded_prefix(self, count: int) -> dict:
        merged = {}
        for feature in self.features[:count]:
            merged.update(feature["degrade"])
        return merged

    def _run_name(self, step: int) -> str:
        if step == 0:
            return "abl-0-full"
        if step == len(self.features):
            return f"abl-{step}-baseline"
        return f"abl-{step}-no_{self.features[step - 1]['label']}"

    def plan(self) -> list[tuple[str, dict]]:
        enabled = self._enabled_overrides()
        plans   = []

        if self.include_full:
            plans.append((self._run_name(0), dict(enabled)))

        for step in range(1, len(self.features) + 1):
            overrides = {**enabled, **self._degraded_prefix(step)}
            plans.append((self._run_name(step), overrides))

        return plans


class ContextTrialPlanner:

    RESERVED_KEYS = ("label", "backbone")

    def __init__(self, trials: list, registry_names: tuple) -> None:
        self.raw_trials     = list(trials)
        self.trials         = [dict(trial) for trial in trials if isinstance(trial, dict)]
        self.registry_names = tuple(registry_names)

        self._validate()

    def _validate(self) -> None:
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
        return {key: value for key, value in trial.items() if key not in self.RESERVED_KEYS}

    def _settings(self, trial: dict) -> dict:
        overrides = self._overrides(trial)
        settings  = {"backbone_name": trial["backbone"]}

        if overrides:
            settings["model_overrides"] = overrides

        return settings

    def summary(self) -> dict:
        return {
            "Rungs"      : [f"{trial['label']}:{trial['backbone']}" for trial in self.trials],
            "Total runs" : len(self.trials),
        }

    def plan(self) -> list[tuple[str, dict]]:
        return [(f"ctx-{trial['label']}", self._settings(trial)) for trial in self.trials]


class ReachTrialPlanner:

    RESERVED_KEYS = ("label", "backbone")

    def __init__(self, trials, registry_names: tuple, head: str, expected_in_channels: int, out_channels: int) -> None:
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
        return {key: value for key, value in rung.items() if key not in self.RESERVED_KEYS}

    def _parameters(self, rung: dict) -> int:
        sizing   = ModelBuilder.image_size_override(rung["backbone"], self.trials.patch_size)
        model, _ = get_backbone(rung["backbone"], in_channels=self.trials.in_channels, out_channels=self.out_channels, head=self.head, **sizing, **self._overrides(rung))

        return sum(parameter.numel() for parameter in model.parameters())

    def _verify_capacity_match(self) -> None:
        low, high = min(self.parameter_counts.values()), max(self.parameter_counts.values())
        deviation = (high - low) / low

        if deviation > self.trials.match_tolerance:
            counts = {label: f"{count:,}" for label, count in self.parameter_counts.items()}
            raise ValueError(f"reach_trials arms differ by {100 * deviation:.2f} % in parameter count ({counts}) at head '{self.head}' with {self.out_channels} output channels, exceeding the {100 * self.trials.match_tolerance:.1f} % match tolerance; the comparison only isolates architecture from capacity when the arms are size matched, so retune the rung widths")

    def _settings(self, rung: dict) -> dict:
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
        return {
            "Rungs"      : [f"{rung['label']}:{rung['backbone']}" for rung in self.rungs],
            "Parameters" : {label: f"{count:,}" for label, count in self.parameter_counts.items()},
            "Counted at" : f"head '{self.head}', {self.trials.in_channels} in / {self.out_channels} out channels",
            "Patch"      : f"{tuple(self.trials.patch_size)} stride {tuple(self.trials.patch_stride)}",
            "Total runs" : len(self.rungs),
        }

    def plan(self) -> list[tuple[str, dict]]:
        return [(f"reach-{rung['label']}", self._settings(rung)) for rung in self.rungs]


class HeadMatchingTrialPlanner:

    def __init__(self, trials, backbone: str, registry_names: tuple, head_names: tuple, matching_names: tuple) -> None:
        self.trials         = trials
        self.backbone       = backbone
        self.registry_names = tuple(registry_names)
        self.head_names     = tuple(head_names)
        self.matching_names = tuple(matching_names)

        self._validate()

    def _validate(self) -> None:
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
        return {
            "Backbone"   : self.backbone,
            "Heads"      : list(self.trials.heads),
            "Matchings"  : list(self.trials.matchings),
            "Total runs" : len(self.trials.heads) * len(self.trials.matchings),
        }

    def plan(self) -> list[tuple[str, dict]]:
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

    INPUT_KEYS   = ("use_primary", "use_secondaries", "use_derived", "use_aux")
    TRACKS_KEY   = "tracks"
    TRACK_SCOPES = ("all", "reduced")

    def __init__(self, input_trials: dict, candidates: list[str]) -> None:
        self.input_trials = input_trials
        self.candidates   = list(candidates)

        self._validate()

    @classmethod
    def from_dataset(cls, input_trials: dict, dataset_path) -> "InputTrialPlanner":
        labels = Layout.secondary_channel_labels(dataset_path)
        if not labels:
            raise ValueError(f"Input trials need at least one secondary channel to enumerate channel scopes, but the dataset at {dataset_path} records only its reference channel")

        return cls(input_trials, labels)

    def _validate(self) -> None:
        if not self.input_trials:
            raise ValueError("input_trials must list at least one input variant")

        for label, spec in self.input_trials.items():
            if spec.get(self.TRACKS_KEY) not in self.TRACK_SCOPES:
                raise ValueError(f"Input trial '{label}' must set '{self.TRACKS_KEY}' to one of {self.TRACK_SCOPES}, got {spec.get(self.TRACKS_KEY)!r}")

            unknown = [key for key in spec if key != self.TRACKS_KEY and key not in self.INPUT_KEYS]
            if unknown:
                raise ValueError(f"Input trial '{label}' has unknown keys {unknown}; allowed keys are ('{self.TRACKS_KEY}',) + {self.INPUT_KEYS}")

    def summary(self) -> dict:
        scopes = [spec[self.TRACKS_KEY] for spec in self.input_trials.values()]

        return {
            "Input variants" : len(self.input_trials),
            "Tracks"         : f"{scopes.count('all')} all ({len(self.candidates)} secondaries), {scopes.count('reduced')} reduced (configured selection)",
        }

    def _overrides(self, spec: dict) -> dict:
        overrides = {f"input.{key}": value for key, value in spec.items() if key != self.TRACKS_KEY}

        if spec[self.TRACKS_KEY] == "all":
            overrides["paths.secondary_labels"] = tuple(self.candidates)

        return overrides

    def plan(self) -> list[tuple[str, dict]]:
        plans = []

        for label, spec in self.input_trials.items():
            run_name = f"in-{label}"
            plans.append((run_name, self._overrides(spec)))

        return plans


class AugmentationTrialPlanner:

    def __init__(self, trials) -> None:
        self.trials = trials

        self._validate()

    def _validate(self) -> None:
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
        return {
            "Ladder"      : "none -> +hflip -> +vflip -> +noise",
            "Flip p"      : self.trials.flip_probability,
            "Noise p"     : self.trials.noise_probability,
            "Noise stds"  : list(self.trials.noise_stds),
            "Total runs"  : 3 + len(self.trials.noise_stds),
        }

    def _overrides(self, flip_h: bool, flip_v: bool, noise_std: float | None) -> dict:
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
        plans = [
            ("aug-0-none",  self._overrides(False, False, None)),
            ("aug-1-hflip", self._overrides(True,  False, None)),
            ("aug-2-vflip", self._overrides(True,  True,  None)),
        ]

        for std in self.trials.noise_stds:
            plans.append((f"aug-3-noise{std:g}", self._overrides(True, True, std)))

        return plans


class ClampTrialPlanner:

    def __init__(self, trials) -> None:
        self.trials = trials

        self._validate()

    def _validate(self) -> None:
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
        pairs = ", ".join(f"{slope:g}/{param:g}" for slope, param in zip(self.trials.leaky_slopes, self.trials.param_leaky_slopes))
        return {
            "Ladder"       : "none -> hard -> leaky per slope",
            "Leaky slopes" : f"{pairs} (curve/param)",
            "Total runs"   : 2 + len(self.trials.leaky_slopes),
        }

    def _overrides(self, enabled: bool, leaky_slope: float, param_leaky_slope: float) -> dict:
        return {
            "normalization.clamp_output"            : enabled,
            "normalization.clamp_leaky_slope"       : leaky_slope,
            "normalization.param_clamp_leaky_slope" : param_leaky_slope,
        }

    def plan(self) -> list[tuple[str, dict]]:
        rungs = [
            ("clamp-0-none", self._overrides(False, 0.0, 0.0)),
            ("clamp-1-hard", self._overrides(True,  0.0, 0.0)),
        ]
        for slope, param_slope in zip(self.trials.leaky_slopes, self.trials.param_leaky_slopes):
            rungs.append((f"clamp-2-leaky-s{slope:g}", self._overrides(True, slope, param_slope)))

        return rungs


class ArchitecturePatchStep:

    PYRAMID_BACKBONES = frozenset({"unet", "unet_skip", "resunet", "attention_unet"})

    def __init__(self, backbone_name: str, backbone_head: str, model_overrides: dict) -> None:
        self.backbone_name   = backbone_name
        self.backbone_head   = backbone_head
        self.model_overrides = model_overrides

    def resolve(self) -> int:
        if self.backbone_name not in self.PYRAMID_BACKBONES:
            raise ValueError(f"Backbone '{self.backbone_name}' is not a verified one-halving-per-stage U-Net ({sorted(self.PYRAMID_BACKBONES)}); the admissible patch step cannot be derived from the architecture, set patch_trials.step explicitly")

        model_config = ModelBuilder.config_from_registry(self.backbone_name, self.model_overrides, head=self.backbone_head)
        features     = model_config.features

        if not features:
            raise ValueError(f"Backbone '{self.backbone_name}' has an empty encoder feature pyramid; set patch_trials.step explicitly")

        return 2 ** len(features)


class PatchSizeTrialPlanner:

    def __init__(self, trials, backbone_name: str, backbone_head: str, model_overrides: dict, training) -> None:
        self.trials          = trials
        self.backbone_name   = backbone_name
        self.backbone_head   = backbone_head
        self.model_overrides = model_overrides
        self.training        = training

        self._patch_step  = None
        self._patch_sizes = None

        self._validate()

    def _validate(self) -> None:
        if not 0 < self.trials.stride_ratio <= 1:
            raise ValueError(f"patch_trials.stride_ratio={self.trials.stride_ratio} must be in (0, 1]")

    def patch_step(self) -> int:
        if self._patch_step is None:
            explicit         = self.trials.step
            self._patch_step = explicit if explicit > 0 else ArchitecturePatchStep(self.backbone_name, self.backbone_head, self.model_overrides).resolve()

        return self._patch_step

    def _axis_sizes(self, axis: int, label: str) -> list[int]:
        step  = self.patch_step()
        start = self.trials.minimum[axis] if self.trials.minimum[axis] > 0 else step

        if start % step != 0:
            raise ValueError(f"patch_trials.minimum {label}={start} is not a multiple of the admissible step {step}")
        if self.trials.maximum[axis] < start:
            raise ValueError(f"patch_trials.maximum {label}={self.trials.maximum[axis]} is below the first admissible size {start}")

        return list(range(start, self.trials.maximum[axis] + 1, step))

    def patch_sizes(self) -> tuple[list[int], list[int]]:
        if self._patch_sizes is not None:
            return self._patch_sizes

        row_sizes = self._axis_sizes(0, "row")
        col_sizes = self._axis_sizes(1, "col")

        if len(row_sizes) * len(col_sizes) < 2:
            raise ValueError(f"The grid rows {row_sizes} x cols {col_sizes} yields {len(row_sizes) * len(col_sizes)} patch shape; a sweep needs at least 2")

        self._patch_sizes = (row_sizes, col_sizes)

        return self._patch_sizes

    def _budget_scale(self, patch_size: tuple[int, int]) -> Fraction:
        reference = self.training.patch_size
        rescaled  = self.training.batch_size * reference[0] * reference[1] // (patch_size[0] * patch_size[1])

        if rescaled < 1:
            raise ValueError(f"patch shape {patch_size} exceeds the pixel budget of training.batch_size={self.training.batch_size} at the reference patch {tuple(reference)}: not a single sample fits, so no batch size holds the budget")

        return Fraction(rescaled, self.training.batch_size)

    def _pixel_rescaled(self, base: int, scale: Fraction, label: str) -> int:
        rescaled = base * scale

        if rescaled.denominator != 1 or rescaled < 1:
            raise ValueError(f"the constant pixel budget rescales training.{label}={base} by {scale} to {float(rescaled)}, which is not a positive whole number of samples; quote it so it rescales in lockstep with training.batch_size={self.training.batch_size}")

        return int(rescaled)

    @staticmethod
    def run_name(patch_size: tuple[int, int]) -> str:
        return f"p-{patch_size[0]:03d}x{patch_size[1]:03d}"

    def _batch_label(self) -> str:
        if not self.trials.constant_pixel_budget:
            return f"{self.training.batch_size} / {self.training.lr_reference_batch_size} in every trial (constant pixel budget off)"

        return ", ".join(f"{name} {overrides['training.batch_size']} / {overrides['training.lr_reference_batch_size']}" for name, overrides in self.plan())

    def summary(self) -> dict:
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

    STRATEGIES   = ("uniform", "gaussian", "consecutive", "spaced", "exhaustive")
    MAX_ATTEMPTS = 1000

    def __init__(self, trials, candidates: list[str]) -> None:
        self.trials     = trials
        self.candidates = list(candidates)
        self.fixed      = [label for label in self.candidates if label in tuple(trials.fixed)]
        self.pool       = [label for label in self.candidates if label not in tuple(trials.fixed)]

        self._validate()

    @classmethod
    def from_dataset(cls, trials, dataset_path) -> "SecondaryTrialPlanner":
        labels = Layout.secondary_channel_labels(dataset_path)
        if not labels:
            raise ValueError(f"Secondary trials need at least one secondary channel to sample from, but the dataset at {dataset_path} records only its reference channel")

        return cls(trials, labels)

    def _validate(self) -> None:
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
        chosen = set(self.fixed).union(self.pool[index] for index in indices)

        return [label for label in self.candidates if label in chosen]

    def _uniform(self) -> list[list[str]]:
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
        indices = set()

        for _ in range(self.MAX_ATTEMPTS):
            if len(indices) == self.trials.n_secondaries:
                return tuple(sorted(indices))

            index = int(round(rng.normal(self.trials.mean, self.trials.sigma)))
            if 0 <= index < len(self.pool):
                indices.add(index)

        return None

    def _consecutive(self) -> list[list[str]]:
        n, step, total = self.trials.n_secondaries, self.trials.block_step, len(self.pool)

        return [self._pick(range(start, start + n)) for start in range(0, total - n + 1, step)]

    def _spaced(self) -> list[list[str]]:
        n, step, spacing, total = self.trials.n_secondaries, self.trials.block_step, self.trials.spacing, len(self.pool)

        span = (n - 1) * spacing + 1
        if span > total:
            raise ValueError(f"Spaced selection spans {span} candidates but only {total} are available; reduce n_secondaries or spacing")

        return [self._pick(range(start, start + span, spacing)) for start in range(0, total - span + 1, step)]

    def _exhaustive_count(self) -> int:
        return comb(len(self.pool), self.trials.n_secondaries)

    def _exhaustive(self) -> list[list[str]]:
        return [self._pick(indices) for indices in combinations(range(len(self.pool)), self.trials.n_secondaries)]

    def _ensure_complete(self, selections: list) -> None:
        if len(selections) < self.trials.n_trials:
            raise RuntimeError(f"Found only {len(selections)} of {self.trials.n_trials} distinct secondary sets after {self.MAX_ATTEMPTS} attempts; lower n_trials, widen sigma or change strategy")

    def plan(self) -> list[tuple[str, dict]]:
        selections = getattr(self, f"_{self.trials.strategy}")()

        plans = []
        for index, labels in enumerate(selections):
            run_name = f"sec-{self.trials.strategy}-t{index:02d}_{'-'.join(labels)}"
            plans.append((run_name, {"paths.secondary_labels": tuple(labels)}))

        return plans
