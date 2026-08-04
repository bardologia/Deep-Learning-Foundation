from __future__ import annotations

import gc
from dataclasses import dataclass, field

from configuration.benchmark              import BenchmarkConfig
from configuration.sar.gaussian_config    import GaussianConfig
from models                               import BACKBONE_CONFIG_REGISTRY, get_backbone
from pipelines.shared.model.model_builder import ModelBuilder
from tools.data.gaussians                 import GaussianHead
from tools.monitoring.logger              import Logger



@dataclass
class WidthRule:
    attribute : str
    divisor   : float
    is_float  : bool = False
    unlock    : bool = False


class WidthScaler:
    def __init__(self, locked: tuple[str, ...]) -> None:
        self.locked   = frozenset(locked)
        feature_rules = [WidthRule(attribute="features", divisor=8)]

        self.rules: dict[str, list[WidthRule]] = {
            "unet"           : feature_rules,
            "unet_skip"      : feature_rules,
            "resunet"        : feature_rules,
            "attention_unet" : feature_rules,
            "unetplusplus"   : feature_rules,
            "linknet"        : feature_rules,
            "swin_unet"      : [WidthRule(attribute="mlp_ratio", divisor=0.25, is_float=True)],
            "transunet"      : [WidthRule(attribute="cnn_features",  divisor=8)],
            "unetr"          : [WidthRule(attribute="decoder_features", divisor=8)],
            "deeplabv3plus"  : feature_rules,
            "segformer"      : [WidthRule(attribute="embedding_dims", divisor=8, unlock=True), WidthRule(attribute="decoder_channels", divisor=8)],
            "convnext_unet"  : feature_rules,
            "dense_unet"     : [WidthRule(attribute="growth_rate", divisor=2)],
            "hrnet"          : [WidthRule(attribute="base_channels", divisor=8)],
            "multires_unet"  : feature_rules,
            "fpn"            : feature_rules + [WidthRule(attribute="pyramid_channels", divisor=8)],
            "u2net"          : feature_rules,
            "pixel_mlp"      : feature_rules,
            "local_cnn"      : feature_rules,
            "nafnet"         : [WidthRule(attribute="width", divisor=1)],
        }

        self._enforce_locked()

    def _enforce_locked(self) -> None:
        for model_name, rules in self.rules.items():
            if not rules:
                raise ValueError(f"Model '{model_name}' has no width rule; size matching has nothing to scale once locked parameters are removed.")

            for rule in rules:
                if rule.attribute in self.locked and not rule.unlock:
                    raise ValueError(f"Width rule for '{model_name}' targets locked hyperparameter '{rule.attribute}'. Locked parameters may not be scaled: {sorted(self.locked)}.")

    def rules_for(self, model_key: str) -> list[WidthRule]:
        name, _head = ModelBuilder.split_key(model_key)
        if name not in self.rules:
            raise ValueError(f"No width rule registered for model '{name}'. Available: {list(self.rules.keys())}")

        return self.rules[name]

    def overrides(self, model_key: str, scale: float) -> dict:
        rules       = self.rules_for(model_key)
        name, _head = ModelBuilder.split_key(model_key)
        defaults    = BACKBONE_CONFIG_REGISTRY[name]()
        overrides   = {}

        for rule in rules:
            base = getattr(defaults, rule.attribute)

            if isinstance(base, list):
                overrides[rule.attribute] = [self._round(value, scale, rule) for value in base]
            else:
                overrides[rule.attribute] = self._round(base, scale, rule)

        return overrides

    def scaled_config(self, model_key: str, scale: float):
        name, head = ModelBuilder.split_key(model_key)
        config     = BACKBONE_CONFIG_REGISTRY[name]()
        config.head = head

        for attribute, value in self.overrides(model_key, scale).items():
            setattr(config, attribute, value)

        return config

    def _round(self, value: float, scale: float, rule: WidthRule) -> float:
        scaled  = round(value * scale / rule.divisor) * rule.divisor
        floored = max(rule.divisor, scaled)

        return floored if rule.is_float else int(floored)


@dataclass
class SizeMatchResult:
    model         : str
    scale         : float
    overrides     : dict
    parameters    : int
    target        : int
    deviation_pct : float
    iterations    : int
    history       : list[dict] = field(default_factory=list)
    flags         : list[str]  = field(default_factory=list)


class DegeneracyAuditor:
    def __init__(self, config, scaler: WidthScaler) -> None:
        self.config = config
        self.scaler = scaler

    def _scale_flags(self, result: SizeMatchResult) -> list[str]:
        flags = []

        if result.scale <= self.config.scale_low * 1.1:
            flags.append(f"scale {result.scale:.4f} converged at the lower search bound {self.config.scale_low}")

        if result.scale >= self.config.scale_high:
            flags.append(f"scale {result.scale:.4f} reached the configured upper bound {self.config.scale_high}")

        return flags

    def _width_flags(self, result: SizeMatchResult) -> list[str]:
        flags = []

        for rule in self.scaler.rules_for(result.model):
            value   = result.overrides[rule.attribute]
            minimum = min(value) if isinstance(value, list) else value

            if minimum <= rule.divisor:
                flags.append(f"{rule.attribute} clamped at the rounding minimum {rule.divisor}")

        return flags

    def _convergence_flags(self, result: SizeMatchResult) -> list[str]:
        if abs(result.deviation_pct) > 100.0 * self.config.tolerance:
            return [f"deviation {result.deviation_pct:+.3f} % exceeds the {100.0 * self.config.tolerance:.1f} % tolerance after {result.iterations} iterations"]

        return []

    def audit(self, result: SizeMatchResult) -> list[str]:
        return self._scale_flags(result) + self._width_flags(result) + self._convergence_flags(result)


class SizeMatcher:
    def __init__(self, config: BenchmarkConfig, logger: Logger) -> None:
        self.config      = config
        self.logger      = logger
        self.scaler      = WidthScaler(locked=config.size_match.locked_params)
        self.auditor     = DegeneracyAuditor(config=config.size_match, scaler=self.scaler)
        self.patch_size  = config.training.patch_size
        self.in_channels = config.size_match.in_channels

        self._validate_in_channels()

        gaussian_cfg      = GaussianConfig.from_dataset(config.paths.dataset_path, config.paths.parameters_path)
        self.out_channels = GaussianHead.total_channels(gaussian_cfg.params_per_gaussian, gaussian_cfg.n_default_gaussians)

    def _validate_in_channels(self) -> None:
        labels = self.config.paths.secondary_labels
        if not labels:
            return

        expected = self.config.input.total_channels(len(labels), len(labels))
        if self.in_channels != expected:
            raise SystemExit(f"size_match.in_channels={self.in_channels} but the input configuration yields {expected} channels for {len(labels)} secondaries; capacity matching would count the wrong width")

    def reference_count(self) -> int:
        reference   = self.config.size_match.reference_model
        name, head  = ModelBuilder.split_key(reference)
        config      = BACKBONE_CONFIG_REGISTRY[name]()
        config.head = head

        return self._count(reference, config)

    def match(self, model_name: str, target: int) -> SizeMatchResult:
        size_match = self.config.size_match

        low  = size_match.scale_low
        high = size_match.scale_high
        while self._count_at(model_name, high) < target and high < 64.0:
            high *= 2.0

        best    : tuple | None = None
        history : list[dict]   = []

        for iteration in range(1, size_match.max_iterations + 1):
            scale      = (low * high) ** 0.5
            parameters = self._count_at(model_name, scale)
            deviation  = (parameters - target) / max(target, 1)

            history.append({"iteration": iteration, "scale": scale, "parameters": parameters, "deviation_pct": 100.0 * deviation})

            if best is None or abs(deviation) < abs(best[1]):
                best = (scale, deviation, parameters)

            if abs(deviation) <= size_match.tolerance:
                break

            if parameters < target:
                low = scale
            else:
                high = scale

        scale, deviation, parameters = best

        result = SizeMatchResult(
            model         = model_name,
            scale         = scale,
            overrides     = self.scaler.overrides(model_name, scale),
            parameters    = parameters,
            target        = target,
            deviation_pct = 100.0 * deviation,
            iterations    = len(history),
            history       = history,
        )

        result.flags = self.auditor.audit(result)

        return result

    def _count_at(self, model_name: str, scale: float) -> int:
        return self._count(model_name, self.scaler.scaled_config(model_name, scale))

    def _count(self, model_key: str, model_config) -> int:
        name, _head = ModelBuilder.split_key(model_key)

        overrides = {"in_channels": self.in_channels, "out_channels": self.out_channels}
        overrides.update(ModelBuilder.image_size_override(name, self.patch_size))

        model, _   = get_backbone(name, config=model_config, **overrides)
        parameters = sum(p.numel() for p in model.parameters())

        del model
        gc.collect()

        return parameters
