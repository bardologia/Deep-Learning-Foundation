"""Capacity matching that scales each architecture to a reference parameter count.

Every backbone declares which of its hyperparameters carry its width; a bisection
over a global scale factor drives the parameter count of each architecture onto
the reference count, so the benchmark compares architecture rather than size.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field

from configuration.benchmark              import BenchmarkConfig
from configuration.dataset.gaussian    import GaussianConfig
from models                               import BACKBONE_CONFIG_REGISTRY, get_backbone
from pipelines.shared.model.model_builder import ModelBuilder
from tools.data.gaussians                 import GaussianHead
from tools.monitoring.logger              import Logger


@dataclass
class WidthRule:
    """Scaling rule for one width-carrying hyperparameter of an architecture.

    Attributes:
        attribute : Name of the model-config field to scale.
        divisor   : Granularity the scaled value is rounded to, and its minimum.
        is_float  : Whether the scaled value stays a float instead of becoming an int.
        unlock    : Whether this rule may scale an attribute the lock list names.
    """

    attribute : str
    divisor   : float
    is_float  : bool = False
    unlock    : bool = False


class WidthScaler:
    """Maps a scale factor onto the width hyperparameters of each architecture.

    Attributes:
        locked : Hyperparameters that must stay at their default, being part of the architecture rather than its width.
        rules  : Width rules per backbone name.
    """

    def __init__(self, locked: tuple[str, ...]) -> None:
        """Builds the per-architecture width rules and checks them against the lock list.

        Args:
            locked : Hyperparameter names that may not be scaled.

        Raises:
            ValueError : If an architecture has no width rule, or a rule targets a locked hyperparameter without declaring `unlock`.
        """
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
        """Checks that every architecture keeps a scalable width after locking.

        Raises:
            ValueError : On an architecture without rules, or a rule targeting a locked hyperparameter.
        """
        for model_name, rules in self.rules.items():
            if not rules:
                raise ValueError(f"Model '{model_name}' has no width rule; size matching has nothing to scale once locked parameters are removed.")

            for rule in rules:
                if rule.attribute in self.locked and not rule.unlock:
                    raise ValueError(f"Width rule for '{model_name}' targets locked hyperparameter '{rule.attribute}'. Locked parameters may not be scaled: {sorted(self.locked)}.")

    def rules_for(self, model_key: str) -> list[WidthRule]:
        """Returns the width rules of the architecture named by `model_key`.

        Args:
            model_key : Model key, with or without a head suffix.

        Returns:
            The width rules registered for the architecture.

        Raises:
            ValueError : If the architecture has no registered rule.
        """
        name, _head = ModelBuilder.split_key(model_key)
        if name not in self.rules:
            raise ValueError(f"No width rule registered for model '{name}'. Available: {list(self.rules.keys())}")

        return self.rules[name]

    def overrides(self, model_key: str, scale: float) -> dict:
        """Returns the architecture's width attributes scaled and rounded by `scale`.

        Args:
            model_key : Model key naming the architecture.
            scale     : Multiplier applied to the registry default of each width attribute.

        Returns:
            Mapping of attribute name to the scaled value, list-valued where the
            default is a per-stage list.
        """
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

    def config_from_overrides(self, model_key: str, overrides: dict):
        """Returns the registry model config of `model_key` with `overrides` applied."""
        name, head = ModelBuilder.split_key(model_key)

        return ModelBuilder.config_from_registry(name, overrides, head=head)

    def _round(self, value: float, scale: float, rule: WidthRule) -> float:
        """Returns `value` scaled, snapped to the rule's granularity and floored at it."""
        scaled  = round(value * scale / rule.divisor) * rule.divisor
        floored = max(rule.divisor, scaled)

        return floored if rule.is_float else int(floored)


@dataclass
class SizeMatchResult:
    """Outcome of matching one architecture onto the reference parameter count.

    Attributes:
        model         : Model key that was matched.
        scale         : Width scale factor the search settled on.
        overrides     : Width attribute values implied by that scale.
        parameters    : Parameter count reached.
        target        : Reference parameter count aimed at.
        deviation_pct : Signed relative deviation from the target, in percent.
        iterations    : Bisection iterations performed.
        settings      : Size-match settings the result was produced under.
        history       : Per-iteration scale, parameter count and deviation.
        flags         : Degeneracy warnings raised by the auditor.
    """

    model         : str
    scale         : float
    overrides     : dict
    parameters    : int
    target        : int
    deviation_pct : float
    iterations    : int
    settings      : dict       = field(default_factory=dict)
    history       : list[dict] = field(default_factory=list)
    flags         : list[str]  = field(default_factory=list)


class DegeneracyAuditor:
    """Flags size-match results whose widths hit a search bound or a rounding floor.

    Attributes:
        config : Size-match configuration holding the bounds and tolerance.
        scaler : Width scaler supplying the rules of each architecture.
    """

    def __init__(self, config, scaler: WidthScaler) -> None:
        """Stores the size-match configuration and the width scaler to audit against."""
        self.config = config
        self.scaler = scaler

    def _scale_flags(self, result: SizeMatchResult) -> list[str]:
        """Returns warnings for a scale that converged onto a search bound."""
        flags = []

        if result.scale <= self.config.scale_low * 1.1:
            flags.append(f"scale {result.scale:.4f} converged at the lower search bound {self.config.scale_low}")

        if result.scale >= self.config.scale_high:
            flags.append(f"scale {result.scale:.4f} reached the configured upper bound {self.config.scale_high}")

        return flags

    def _width_flags(self, result: SizeMatchResult) -> list[str]:
        """Returns warnings for width attributes clamped at their rounding minimum."""
        flags = []

        for rule in self.scaler.rules_for(result.model):
            value   = result.overrides[rule.attribute]
            minimum = min(value) if isinstance(value, list) else value

            if minimum <= rule.divisor:
                flags.append(f"{rule.attribute} clamped at the rounding minimum {rule.divisor}")

        return flags

    def _convergence_flags(self, result: SizeMatchResult) -> list[str]:
        """Returns a warning when the deviation still exceeds the configured tolerance."""
        if abs(result.deviation_pct) > 100.0 * self.config.tolerance:
            return [f"deviation {result.deviation_pct:+.3f} % exceeds the {100.0 * self.config.tolerance:.1f} % tolerance after {result.iterations} iterations"]

        return []

    def audit(self, result: SizeMatchResult) -> list[str]:
        """Returns every scale, width and convergence warning raised by a result."""
        return self._scale_flags(result) + self._width_flags(result) + self._convergence_flags(result)


class SizeMatcher:
    """Bisects the width scale of each architecture onto the reference parameter count.

    Attributes:
        config       : Benchmark configuration.
        logger       : Logger for the matching pass.
        scaler       : Width scaler mapping scale factors onto model overrides.
        auditor      : Degeneracy auditor inspecting each result.
        patch_size   : Patch size models are counted at.
        in_channels  : Input channel count models are counted at.
        count_cache  : Parameter counts keyed by model and override set.
        out_channels : Output channel count implied by the dataset Gaussian head.
    """

    def __init__(self, config: BenchmarkConfig, logger: Logger) -> None:
        """Prepares the scaler and auditor and resolves the counting widths.

        Args:
            config : Benchmark configuration supplying the size-match settings.
            logger : Logger for the matching pass.

        Raises:
            SystemExit : If the configured input channel count contradicts the input configuration and secondary track list.
        """
        self.config      = config
        self.logger      = logger
        self.scaler      = WidthScaler(locked=config.size_match.locked_params)
        self.auditor     = DegeneracyAuditor(config=config.size_match, scaler=self.scaler)
        self.patch_size  = config.training.patch_size
        self.in_channels = config.size_match.in_channels
        self.count_cache = {}

        self._validate_in_channels()

        gaussian_cfg      = GaussianConfig.from_dataset(config.paths.dataset_path, config.paths.parameters_path)
        self.out_channels = GaussianHead.total_channels(gaussian_cfg.params_per_gaussian, gaussian_cfg.n_default_gaussians)

    def _validate_in_channels(self) -> None:
        """Checks the configured input width against the one the input configuration yields.

        Raises:
            SystemExit : On a mismatch, which would have every model counted at the wrong first layer.
        """
        labels = self.config.paths.secondary_labels
        if not labels:
            return

        expected = self.config.input.total_channels(len(labels), len(labels))
        if self.in_channels != expected:
            raise SystemExit(f"size_match.in_channels={self.in_channels} but the input configuration yields {expected} channels for {len(labels)} secondaries; capacity matching would count the wrong width")

    def reference_count(self) -> int:
        """Returns the parameter count of the reference model at its registry defaults."""
        reference  = self.config.size_match.reference_model
        name, head = ModelBuilder.split_key(reference)

        return self._count(reference, ModelBuilder.config_from_registry(name, {}, head=head))

    def match(self, model_name: str, target: int) -> SizeMatchResult:
        """Bisects the width scale of one model onto the target parameter count.

        The upper bound is doubled until it brackets the target, then a geometric
        bisection narrows the interval until the deviation falls inside the
        tolerance or the iteration budget runs out; the best iterate is returned.

        Args:
            model_name : Model key to match.
            target     : Reference parameter count to reach.

        Returns:
            The matching result, with the degeneracy flags already attached.

        Raises:
            SystemExit : If the model cannot reach the target below the scale ceiling, or already exceeds it at the lower bound.
        """
        size_match = self.config.size_match

        low  = size_match.scale_low
        high = size_match.scale_high

        high_count = self._count_at(model_name, high)
        while high_count < target and high < size_match.scale_ceiling:
            high      *= 2.0
            high_count = self._count_at(model_name, high)

        if high_count < target:
            raise SystemExit(f"{model_name} reaches only {high_count:,} parameters at scale {high:g}, the ceiling size_match.scale_ceiling={size_match.scale_ceiling:g}, and cannot match the target {target:,}")

        low_count = self._count_at(model_name, low)
        if low_count > target:
            raise SystemExit(f"{model_name} already holds {low_count:,} parameters at the lower bound size_match.scale_low={low:g} and cannot be shrunk to the target {target:,}")

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
            settings      = self.config.size_match_settings(),
            history       = history,
        )

        result.flags = self.auditor.audit(result)

        return result

    def _count_at(self, model_name: str, scale: float) -> int:
        """Returns the parameter count of `model_name` at width `scale`, memoized per override set."""
        overrides = self.scaler.overrides(model_name, scale)
        key       = (model_name, repr(sorted(overrides.items())))

        if key not in self.count_cache:
            self.count_cache[key] = self._count(model_name, self.scaler.config_from_overrides(model_name, overrides))

        return self.count_cache[key]

    def _count(self, model_key: str, model_config) -> int:
        """Instantiates the model at the counting widths and returns its parameter count.

        Args:
            model_key    : Model key naming the architecture.
            model_config : Resolved model configuration to build.

        Returns:
            The total number of parameters; the model is freed before returning.
        """
        name, _head = ModelBuilder.split_key(model_key)

        overrides = {"in_channels": self.in_channels, "out_channels": self.out_channels}
        overrides.update(ModelBuilder.image_size_override(name, self.patch_size))

        model, _   = get_backbone(name, config=model_config, **overrides)
        parameters = sum(p.numel() for p in model.parameters())

        del model
        gc.collect()

        return parameters
