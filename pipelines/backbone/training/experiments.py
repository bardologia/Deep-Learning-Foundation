
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

        self.parameter_counts = {rung["label"]: self._parameters(rung) for rung in self.rungs}

    def _overrides(self, rung: dict) -> dict:
        return {key: value for key, value in rung.items() if key not in self.RESERVED_KEYS}

    def _parameters(self, rung: dict) -> int:
        sizing   = ModelBuilder.image_size_override(rung["backbone"], self.trials.patch_size)
        model, _ = get_backbone(rung["backbone"], in_channels=self.trials.in_channels, out_channels=self.out_channels, head=self.head, **sizing, **self._overrides(rung))

        return sum(parameter.numel() for parameter in model.parameters())

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

    @classmethod
    def from_dataset(cls, input_trials: dict, dataset_path) -> "InputTrialPlanner":
        labels = Layout.secondary_channel_labels(dataset_path)

        return cls(input_trials, labels)

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
        model_config = ModelBuilder.config_from_registry(self.backbone_name, self.model_overrides, head=self.backbone_head)
        features     = model_config.features

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

    def patch_step(self) -> int:
        if self._patch_step is None:
            explicit         = self.trials.step
            self._patch_step = explicit if explicit > 0 else ArchitecturePatchStep(self.backbone_name, self.backbone_head, self.model_overrides).resolve()

        return self._patch_step

    def _axis_sizes(self, axis: int, label: str) -> list[int]:
        step  = self.patch_step()
        start = self.trials.minimum[axis] if self.trials.minimum[axis] > 0 else step

        return list(range(start, self.trials.maximum[axis] + 1, step))

    def patch_sizes(self) -> tuple[list[int], list[int]]:
        if self._patch_sizes is not None:
            return self._patch_sizes

        row_sizes = self._axis_sizes(0, "row")
        col_sizes = self._axis_sizes(1, "col")

        self._patch_sizes = (row_sizes, col_sizes)

        return self._patch_sizes

    def _budget_scale(self, patch_size: tuple[int, int]) -> Fraction:
        reference = self.training.patch_size
        rescaled  = self.training.batch_size * reference[0] * reference[1] // (patch_size[0] * patch_size[1])

        return Fraction(rescaled, self.training.batch_size)

    def _pixel_rescaled(self, base: int, scale: Fraction, label: str) -> int:
        rescaled = base * scale

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

    @classmethod
    def from_dataset(cls, trials, dataset_path) -> "SecondaryTrialPlanner":
        labels = Layout.secondary_channel_labels(dataset_path)

        return cls(trials, labels)

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

        return [self._pick(range(start, start + span, spacing)) for start in range(0, total - span + 1, step)]

    def _exhaustive_count(self) -> int:
        return comb(len(self.pool), self.trials.n_secondaries)

    def _exhaustive(self) -> list[list[str]]:
        return [self._pick(indices) for indices in combinations(range(len(self.pool)), self.trials.n_secondaries)]

    def plan(self) -> list[tuple[str, dict]]:
        selections = getattr(self, f"_{self.trials.strategy}")()

        plans = []
        for index, labels in enumerate(selections):
            run_name = f"sec-{self.trials.strategy}-t{index:02d}_{'-'.join(labels)}"
            plans.append((run_name, {"paths.secondary_labels": tuple(labels)}))

        return plans
