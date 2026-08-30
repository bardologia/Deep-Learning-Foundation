"""Trial planners that expand dual-trunk sweep configurations into named runs.

The context, reach and head planners specialise their backbone counterparts so
that a rung applies to both trunks at once, while the routing and ratio planners
cover the two axes that only exist for the dual model: which input channel
groups feed each trunk, and how the parameter budget is split between the
Gaussian-parameter arm and the existence arm.
"""

from __future__ import annotations

from models.dual                             import get_dual
from pipelines.backbone.training.experiments import ContextTrialPlanner, HeadMatchingTrialPlanner, ReachTrialPlanner
from pipelines.shared.model.model_builder    import ModelBuilder


class DualContextTrialPlanner(ContextTrialPlanner):
    """Context ladder planner that applies each rung to both dual trunks."""

    def _settings(self, trial: dict) -> dict:
        """Returns the entry-config settings placing the rung backbone, features and overrides on both trunks."""
        overrides = self._overrides(trial)
        features  = overrides.pop("features", None)
        features  = list(features) if features else []

        settings = {
            "params_backbone"    : trial["backbone"],
            "existence_backbone" : trial["backbone"],
            "model_overrides"    : {"params_features": features, "existence_features": features},
        }

        if overrides:
            settings["params_overrides"]    = dict(overrides)
            settings["existence_overrides"] = dict(overrides)

        return settings


class DualHeadMatchingTrialPlanner(HeadMatchingTrialPlanner):
    """Head-by-matching planner that sets the sweep backbone on both dual trunks."""

    def plan(self) -> list[tuple[str, dict]]:
        """Returns one `(run_name, overrides)` pair per head and matching strategy combination."""
        plans = []

        for head in self.trials.heads:
            for matching in self.trials.matchings:
                run_name  = f"hm-{head}-{matching}"
                overrides = {
                    "params_backbone"                    : self.trials.backbone,
                    "existence_backbone"                 : self.trials.backbone,
                    "curriculum.warmup.param_matching"   : matching,
                    "curriculum.complete.param_matching" : matching,
                }
                plans.append((run_name, overrides))

        return plans


class DualReachTrialPlanner(ReachTrialPlanner):
    """Reach ladder planner that counts parameters on the assembled dual model.

    Attributes:
        model_name     : Registered dual model name instantiated for the counts.
        trunk_channels : Input channel index tuples of the params and existence trunks, in that order.
    """

    def __init__(self, trials, registry_names: tuple, expected_in_channels: int, out_channels: int, model_name: str, trunk_channels: tuple) -> None:
        """Stores the dual model identity and defers rung validation to the backbone planner.

        Args:
            trials               : Reach trial configuration object.
            registry_names       : Names of the registered backbones.
            expected_in_channels : Input width the dataset actually yields.
            out_channels         : Output width the Gaussian heads actually yield.
            model_name           : Registered dual model name.
            trunk_channels       : Channel index tuples for the params and existence trunks.
        """
        self.model_name     = model_name
        self.trunk_channels = tuple(trunk_channels)

        super().__init__(trials, registry_names, "set_pred", expected_in_channels, out_channels)

    def _trunk_features(self, rung: dict) -> list[int]:
        """Returns the per-trunk feature widths of a rung, empty when it keeps the backbone default."""
        features = rung.get("features")
        return list(features) if features else []

    def _parameters(self, rung: dict) -> int:
        """Returns the total parameter count of the dual model built from the rung.

        Args:
            rung : Reach rung dict with `backbone`, optional `features` and model override keys.

        Returns:
            Number of parameters of the assembled dual model, both trunks and
            heads included.
        """
        overrides = self._overrides(rung)
        overrides.pop("features", None)
        features = self._trunk_features(rung)
        sizing   = ModelBuilder.image_size_override(rung["backbone"], self.trials.patch_size)

        model, _ = get_dual(
            self.model_name,
            in_channels         = self.trials.in_channels,
            out_channels        = self.out_channels,
            params_backbone     = rung["backbone"],
            existence_backbone  = rung["backbone"],
            params_channels     = self.trunk_channels[0],
            existence_channels  = self.trunk_channels[1],
            params_features     = features,
            existence_features  = features,
            params_overrides    = {**overrides, **sizing},
            existence_overrides = {**overrides, **sizing},
        )

        return sum(parameter.numel() for parameter in model.parameters())

    def _settings(self, rung: dict) -> dict:
        """Returns the entry-config settings placing the rung on both trunks along with the sweep patch geometry."""
        overrides = self._overrides(rung)
        overrides.pop("features", None)
        features = self._trunk_features(rung)

        settings = {
            "params_backbone"       : rung["backbone"],
            "existence_backbone"    : rung["backbone"],
            "training.patch_size"   : tuple(self.trials.patch_size),
            "training.patch_stride" : tuple(self.trials.patch_stride),
            "model_overrides"       : {"params_features": features, "existence_features": features},
        }

        if overrides:
            settings["params_overrides"]    = dict(overrides)
            settings["existence_overrides"] = dict(overrides)

        return settings


class DualRoutingTrialPlanner:
    """Plans runs that vary which input channel groups each dual trunk reads.

    Attributes:
        trials          : Routing trial configuration with `params_features`, `existence_features` and a `trials` mapping of label to per-trunk channel-group selections.
        model_overrides : Model overrides shared by every routing variant.
        groups          : Channel group names the dataset input configuration exposes.
    """

    TRUNK_KEYS   = ("params", "existence")
    FEATURE_KEYS = ("params_features", "existence_features")

    def __init__(self, trials_config, model_overrides: dict, groups: tuple) -> None:
        """Stores and validates the routing variants against the available channel groups.

        Args:
            trials_config   : Routing trial configuration object.
            model_overrides : Model overrides applied to every variant.
            groups          : Channel group names available for trunk routing.

        Raises:
            ValueError : On empty feature ladders, feature widths duplicated in `model_overrides`, or malformed trunk selections.
        """
        self.trials          = trials_config
        self.model_overrides = dict(model_overrides)
        self.groups          = tuple(groups)

        self._validate()

    def _validate(self) -> None:
        """Checks the feature ladders and that every variant selects both trunks with known keys.

        Raises:
            ValueError : If a feature ladder is empty or non-positive, a feature key is also set in `model_overrides`, no variant is listed, or a variant misses or adds a trunk key.
        """
        for key in self.FEATURE_KEYS:
            features = getattr(self.trials, key)
            if not features:
                raise ValueError(f"routing_trials.{key} must list at least one feature width")

            invalid = [width for width in features if not isinstance(width, int) or width < 1]
            if invalid:
                raise ValueError(f"routing_trials.{key} must be positive integers, got {features}")

        duplicated = [key for key in self.FEATURE_KEYS if key in self.model_overrides]
        if duplicated:
            raise ValueError(f"Set {duplicated} via the routing_trials fields, not model_overrides")

        if not self.trials.trials:
            raise ValueError("routing_trials.trials must list at least one trunk-input variant")

        for label, spec in self.trials.trials.items():
            missing = [key for key in self.TRUNK_KEYS if key not in spec]
            if missing:
                raise ValueError(f"Routing trial '{label}' must set {missing}; every trial defines all of {self.TRUNK_KEYS}")

            unknown = [key for key in spec if key not in self.TRUNK_KEYS]
            if unknown:
                raise ValueError(f"Routing trial '{label}' has unknown keys {unknown}; allowed keys are {self.TRUNK_KEYS}")

            for key in self.TRUNK_KEYS:
                self._validate_selection(label, key, spec[key])

    def _validate_selection(self, label: str, key: str, selection) -> None:
        """Checks one trunk's channel-group selection.

        Args:
            label     : Routing variant label used in error messages.
            key       : Trunk key being validated, `params` or `existence`.
            selection : Channel group names routed into that trunk.

        Raises:
            ValueError : If the selection is empty, names an unknown group or repeats a group.
        """
        if not selection:
            raise ValueError(f"Routing trial '{label}' selects no channel groups for '{key}'")

        unknown = [group for group in selection if group not in self.groups]
        if unknown:
            raise ValueError(f"Routing trial '{label}' has unknown '{key}' groups {unknown}; available groups are {list(self.groups)}")

        duplicates = sorted({group for group in selection if list(selection).count(group) > 1})
        if duplicates:
            raise ValueError(f"Routing trial '{label}' repeats '{key}' groups {duplicates}")

    def summary(self) -> dict:
        """Returns the trunk feature ladders, the routing variants and the run count for logging."""
        return {
            "Trunk features" : f"params {self.trials.params_features}, existence {self.trials.existence_features}",
            "Routing variants" : {label: f"{'+'.join(spec['params'])} | {'+'.join(spec['existence'])}" for label, spec in self.trials.trials.items()},
            "Total runs"     : len(self.trials.trials),
        }

    def _overrides(self, spec: dict) -> dict:
        """Returns the entry-config overrides routing the variant's channel groups into the two trunks.

        The DEM input channel is switched on exactly when a trunk selects the
        `dem` group, so a selected group always carries its channels and a
        DEM-free variant cannot inherit the channel from the base config.
        """
        features = {
            "params_features"    : list(self.trials.params_features),
            "existence_features" : list(self.trials.existence_features),
        }

        return {
            "params_input"    : list(spec["params"]),
            "existence_input" : list(spec["existence"]),
            "input.use_aux"   : "aux" in spec["params"] or "aux" in spec["existence"],
            "model_overrides" : {**self.model_overrides, **features},
        }

    def plan(self) -> list[tuple[str, dict]]:
        """Returns one `(run_name, overrides)` pair per routing variant, named `di-<label>`."""
        plans = []

        for label, spec in self.trials.trials.items():
            run_name = f"di-{label}"
            plans.append((run_name, self._overrides(spec)))

        return plans


class DualRatioTrialPlanner:
    """Plans budget-matched runs that vary the parameter split between the two arms.

    Every variant is instantiated at build time so the realized share of each arm
    and the total budget can be checked against the declared split; otherwise the
    sweep would confound the arm ratio with total capacity.

    Attributes:
        trials          : Ratio trial configuration with a `trials` mapping of `<params>-<existence>` share label to per-arm feature ladders, and a `match_tolerance`.
        model_overrides : Model overrides shared by every variant.
        model_name      : Registered dual model name instantiated for the counts.
        backbones       : Params and existence trunk backbone names, in that order.
        trunk_channels  : Input channel index tuples of the two trunks.
        in_channels     : Input width the dataset yields.
        out_channels    : Output width the Gaussian heads yield.
        shares          : Declared percentage shares per label as `(params, existence)`.
        arm_counts      : Realized parameter count per arm and label.
    """

    ARM_KEYS     = ("params", "existence")
    FEATURE_KEYS = ("params_features", "existence_features")

    ARM_PREFIXES = {
        "params"    : ("trunk_params.", "gaussian_heads."),
        "existence" : ("trunk_existence.", "existence_head.", "amp_off"),
    }

    def __init__(self, trials_config, model_overrides: dict, model_name: str, backbones: tuple, trunk_channels: tuple, in_channels: int, out_channels: int) -> None:
        """Validates the arm ladders and verifies that the realized splits and budgets match.

        Args:
            trials_config   : Ratio trial configuration object.
            model_overrides : Model overrides applied to every variant.
            model_name      : Registered dual model name.
            backbones       : Params and existence trunk backbone names.
            trunk_channels  : Channel index tuples for the params and existence trunks.
            in_channels     : Input width the dataset yields.
            out_channels    : Output width the Gaussian heads yield.

        Raises:
            ValueError : On malformed ladders or labels, a realized share outside `match_tolerance`, or total budgets differing by more than `match_tolerance`.
        """
        self.trials          = trials_config
        self.model_overrides = dict(model_overrides)
        self.model_name      = model_name
        self.backbones       = tuple(backbones)
        self.trunk_channels  = tuple(trunk_channels)
        self.in_channels     = in_channels
        self.out_channels    = out_channels

        self._validate()

        self.shares     = {label: self._shares(label) for label in self.trials.trials}
        self.arm_counts = {label: self._parameters(spec) for label, spec in self.trials.trials.items()}

        self._verify_share_match()
        self._verify_budget_match()

    def _validate(self) -> None:
        """Checks the variant set, tolerance, output width, and per-arm ladder shape.

        Raises:
            ValueError : If no variant is listed, the tolerance or output width is non-positive, a feature key is also set in `model_overrides`, a variant misses or adds an arm key, or the arms mix ladder depths.
        """
        if not self.trials.trials:
            raise ValueError("ratio_trials.trials must list at least one arm-split variant")

        if self.trials.match_tolerance <= 0:
            raise ValueError(f"ratio_trials.match_tolerance={self.trials.match_tolerance} must be positive; the splits are only comparable when every trial holds the same total budget to a stated tolerance")

        if self.out_channels < 1:
            raise ValueError(f"ratio_trials arm accounting needs the resolved output width, got out_channels={self.out_channels}; training builds the gaussian heads at the dataset Gaussian count, so the arm counts would come from a model that is never trained")

        duplicated = [key for key in self.FEATURE_KEYS if key in self.model_overrides]
        if duplicated:
            raise ValueError(f"Set {duplicated} via the ratio_trials ladders, not model_overrides")

        for label, spec in self.trials.trials.items():
            missing = [key for key in self.ARM_KEYS if key not in spec]
            if missing:
                raise ValueError(f"Ratio trial '{label}' must set {missing}; every trial defines all of {self.ARM_KEYS}")

            unknown = [key for key in spec if key not in self.ARM_KEYS]
            if unknown:
                raise ValueError(f"Ratio trial '{label}' has unknown keys {unknown}; allowed keys are {self.ARM_KEYS}")

            for key in self.ARM_KEYS:
                self._validate_ladder(label, key, spec[key])

        depths = {len(spec[key]) for spec in self.trials.trials.values() for key in self.ARM_KEYS}
        if len(depths) != 1:
            raise ValueError(f"Ratio trials mix ladder depths {sorted(depths)}; every arm keeps the same layer count so the split only varies width")

    def _validate_ladder(self, label: str, key: str, ladder) -> None:
        """Checks one arm's feature ladder.

        Args:
            label  : Ratio variant label used in error messages.
            key    : Arm key being validated, `params` or `existence`.
            ladder : Feature widths of that arm's trunk stages.

        Raises:
            ValueError : If the ladder is empty or holds a non-positive integer.
        """
        if not ladder:
            raise ValueError(f"Ratio trial '{label}' lists no feature widths for '{key}'")

        invalid = [width for width in ladder if not isinstance(width, int) or width < 1]
        if invalid:
            raise ValueError(f"Ratio trial '{label}' ladder '{key}' must be positive integers, got {list(ladder)}")

    def _shares(self, label: str) -> tuple[int, int]:
        """Returns the declared percentage shares parsed from a variant label.

        Args:
            label : Variant label of the form `<params>-<existence>`, e.g. `70-30`.

        Returns:
            The params and existence percentage shares.

        Raises:
            ValueError : If the label is not two integers, the shares do not sum to 100, or the existence share exceeds the params share.
        """
        parts = label.split("-")

        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            raise ValueError(f"Ratio trial label '{label}' must be '<params>-<existence>' percentage shares, e.g. '70-30'")

        params_share, existence_share = int(parts[0]), int(parts[1])

        if params_share + existence_share != 100:
            raise ValueError(f"Ratio trial '{label}' shares must sum to 100, got {params_share + existence_share}")

        if existence_share > params_share:
            raise ValueError(f"Ratio trial '{label}' puts the larger share on the existence arm; the detection arm is always the smaller one, so state the params share first")

        return params_share, existence_share

    def _parameters(self, spec: dict) -> dict:
        """Returns the parameter count of each arm of the model built from a variant.

        Args:
            spec : Variant spec mapping each arm key to its feature ladder.

        Returns:
            Parameter counts keyed by arm, attributed through `ARM_PREFIXES`.

        Raises:
            ValueError : If a parameter name matches no arm prefix or more than one.
        """
        model, _ = get_dual(
            self.model_name,
            in_channels        = self.in_channels,
            out_channels       = self.out_channels,
            params_backbone    = self.backbones[0],
            existence_backbone = self.backbones[1],
            params_channels    = self.trunk_channels[0],
            existence_channels = self.trunk_channels[1],
            params_features    = list(spec["params"]),
            existence_features = list(spec["existence"]),
            **self.model_overrides,
        )

        counts = dict.fromkeys(self.ARM_KEYS, 0)
        for name, parameter in model.named_parameters():
            arms = [arm for arm, prefixes in self.ARM_PREFIXES.items() if name.startswith(prefixes)]
            if len(arms) != 1:
                raise ValueError(f"Parameter '{name}' of {self.model_name} does not belong to exactly one arm; extend ARM_PREFIXES before running ratio trials")
            counts[arms[0]] += parameter.numel()

        return counts

    def _verify_share_match(self) -> None:
        """Checks that every variant realizes its declared arm split.

        Raises:
            ValueError : If a realized params share deviates from the declared one by more than `match_tolerance`.
        """
        for label, counts in self.arm_counts.items():
            total     = sum(counts.values())
            share     = counts["params"] / total
            deviation = abs(share - self.shares[label][0] / 100)

            if deviation > self.trials.match_tolerance:
                raise ValueError(f"Ratio trial '{label}' realizes a {100 * share:.2f}/{100 * (1 - share):.2f} split ({counts['params']:,} / {counts['existence']:,} parameters), {100 * deviation:.2f} points away from its declared share and beyond the {100 * self.trials.match_tolerance:.1f} point match tolerance; retune the ladders")

    def _verify_budget_match(self) -> None:
        """Checks that every variant spends the same total parameter budget.

        Raises:
            ValueError : If the largest and smallest totals differ by more than `match_tolerance` relative to the smallest.
        """
        totals    = {label: sum(counts.values()) for label, counts in self.arm_counts.items()}
        low, high = min(totals.values()), max(totals.values())
        deviation = (high - low) / low

        if deviation > self.trials.match_tolerance:
            counts = {label: f"{total:,}" for label, total in totals.items()}
            raise ValueError(f"Ratio trials differ by {100 * deviation:.2f} % in total parameter count ({counts}), exceeding the {100 * self.trials.match_tolerance:.1f} % match tolerance; the sweep only isolates the arm split when every trial spends the same budget, so retune the ladders")

    def _overrides(self, spec: dict) -> dict:
        """Returns the entry-config overrides carrying the variant's per-arm feature ladders."""
        features = {
            "params_features"    : list(spec["params"]),
            "existence_features" : list(spec["existence"]),
        }

        return {"model_overrides": {**self.model_overrides, **features}}

    def summary(self) -> dict:
        """Returns the realized arm budgets, the input stack, the counting width and the run count."""
        splits = {}
        for label, counts in self.arm_counts.items():
            total         = sum(counts.values())
            splits[label] = f"{counts['params']:,} / {counts['existence']:,} = {total:,} ({100 * counts['params'] / total:.2f}/{100 * counts['existence'] / total:.2f})"

        return {
            "Arm budgets (params / existence)" : splits,
            "Input stack"                      : f"{self.in_channels} channels, params {list(self.trunk_channels[0])}, existence {list(self.trunk_channels[1])}",
            "Counted at"                       : f"{self.out_channels} output channels",
            "Total runs"                       : len(self.trials.trials),
        }

    def plan(self) -> list[tuple[str, dict]]:
        """Returns one `(run_name, overrides)` pair per ratio variant, named `dr-<label>`."""
        return [(f"dr-{label}", self._overrides(spec)) for label, spec in self.trials.trials.items()]
