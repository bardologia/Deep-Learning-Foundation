
from __future__ import annotations

from fractions import Fraction

from models                               import get_backbone
from pipelines.shared.model.model_builder import ModelBuilder


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

    def __init__(self, trials, registry_names: tuple, head: str, out_channels: int) -> None:
        self.trials         = trials
        self.rungs          = [dict(rung) for rung in trials.rungs if isinstance(rung, dict)]
        self.registry_names = tuple(registry_names)
        self.head           = head
        self.out_channels   = out_channels

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
                    "training.patch_size"   : size,
                    "training.patch_stride" : stride,
                }

                if self.trials.constant_pixel_budget:
                    scale = self._budget_scale(size)

                    overrides["training.batch_size"]              = self._pixel_rescaled(self.training.batch_size, scale, "batch_size")
                    overrides["training.lr_reference_batch_size"] = self._pixel_rescaled(self.training.lr_reference_batch_size, scale, "lr_reference_batch_size")

                plans.append((self.run_name(size), overrides))

        return plans
