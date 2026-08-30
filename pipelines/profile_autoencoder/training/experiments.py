"""Trial planners expanding profile-autoencoder sweep configurations into named runs.

Each planner validates one experiment axis (architecture family, latent width,
capacity, reconstruction loss, smoothness levers, profile augmentation, data
sampling) and turns it into a list of ``(run_name, config_overrides)`` pairs
consumed by the training scheduler.
"""

from __future__ import annotations

from dataclasses import fields

from configuration.training.profile_autoencoder  import ProfileAeLossConfig
from pipelines.profile_autoencoder.training.loss import Loss


class ArchTrialPlanner:
    """Plans one run per profile-autoencoder family rung.

    Attributes:
        trials          : Rung dicts holding a `label`, an `ae_model_name` and optional model overrides.
        registry_keys   : Registered autoencoder names the rungs may select.
        model_overrides : Entry-level model overrides, which this sweep refuses since they belong to one family only.
    """

    RESERVED = ("label", "ae_model_name")

    def __init__(self, trials: list, registry_keys: tuple, model_overrides: dict) -> None:
        """Stores and validates the architecture rungs.

        Args:
            trials          : Rung dicts of the sweep.
            registry_keys   : Names registered in the profile-autoencoder registry.
            model_overrides : Model overrides from the entry configuration.

        Raises:
            ValueError : If no rung is given, a rung misses a key, a label repeats, a model name is unknown, or the entry sets model overrides.
        """
        self.trials          = trials
        self.registry_keys   = tuple(registry_keys)
        self.model_overrides = dict(model_overrides)

        self._validate()

    def _validate(self) -> None:
        """Checks that every rung names a unique label and a registered autoencoder, and that the entry pins no architecture field.

        Raises:
            ValueError : On empty rungs, missing keys, duplicate labels, unknown model names or entry-level model overrides.
        """
        if not self.trials:
            raise ValueError("arch_trials must list at least one rung")

        if self.model_overrides:
            raise ValueError(f"model_overrides sets {sorted(self.model_overrides)}, which are architecture fields of a single family and cannot be applied across the families the arch sweep walks; move them into the arch_trials rung of the family they belong to")

        labels = []
        for rung in self.trials:
            missing = [key for key in self.RESERVED if key not in rung]
            if missing:
                raise ValueError(f"Arch rung {rung} must set {missing}")

            if rung["ae_model_name"] not in self.registry_keys:
                raise ValueError(f"Arch rung '{rung['label']}' selects unknown autoencoder '{rung['ae_model_name']}'; registered: {sorted(self.registry_keys)}")

            labels.append(rung["label"])

        duplicates = sorted({label for label in labels if labels.count(label) > 1})
        if duplicates:
            raise ValueError(f"arch_trials labels must be unique, duplicated: {duplicates}")

    def summary(self) -> dict:
        """Returns the rung labels and run count for logging."""
        return {
            "Families"   : [rung["label"] for rung in self.trials],
            "Total runs" : len(self.trials),
        }

    def plan(self) -> list[tuple[str, dict]]:
        """Returns one `(run_name, overrides)` pair per family rung."""
        plans = []

        for rung in self.trials:
            overrides = {key: value for key, value in rung.items() if key not in self.RESERVED}
            plans.append((f"arch-{rung['label']}", {"ae_model_name": rung["ae_model_name"], "model_overrides": overrides}))

        return plans


class EmbeddingDimTrialPlanner:
    """Plans one run per latent width of the embedding ladder.

    Attributes:
        trials          : Embedding trial configuration carrying the `dims` ladder.
        model_overrides : Entry-level model overrides each rung extends.
    """

    def __init__(self, trials, model_overrides: dict) -> None:
        """Stores and validates the latent-width ladder.

        Args:
            trials          : Configuration object with the `dims` list.
            model_overrides : Model overrides from the entry configuration.

        Raises:
            ValueError : If the ladder is empty, repeats or holds non-positive widths, or the entry overrides already pin the embedding dimension.
        """
        self.trials          = trials
        self.model_overrides = dict(model_overrides)

        self._validate()

    def _validate(self) -> None:
        """Checks that the widths are positive, unique and not pinned by the entry overrides.

        Raises:
            ValueError : On empty, non-positive or duplicated widths, or a pinned embedding dimension.
        """
        if not self.trials.dims:
            raise ValueError("embedding_trials.dims must list at least one latent width")

        if any(dim <= 0 for dim in self.trials.dims):
            raise ValueError(f"embedding_trials.dims must all be positive, got {self.trials.dims}")

        if len(set(self.trials.dims)) != len(self.trials.dims):
            raise ValueError(f"embedding_trials.dims must be unique, got {self.trials.dims}")

        if "embedding_dim" in self.model_overrides:
            raise ValueError("model_overrides pins embedding_dim, which the embedding sweep varies; drop it from the overrides")

    def summary(self) -> dict:
        """Returns the swept widths and run count for logging."""
        return {
            "Embedding dims" : list(self.trials.dims),
            "Total runs"     : len(self.trials.dims),
        }

    def plan(self) -> list[tuple[str, dict]]:
        """Returns one `(run_name, overrides)` pair per latent width."""
        return [(f"emb-d{dim}", {"model_overrides": {**self.model_overrides, "embedding_dim": dim}}) for dim in self.trials.dims]


class CapacityTrialPlanner:
    """Plans the width-by-depth capacity grid of the configured autoencoder family.

    Attributes:
        trials            : Capacity trial configuration carrying `widths` and `depths`.
        ae_model_name     : Family the grid is swept within.
        model_overrides   : Entry-level model overrides each rung extends.
        config_registry   : Mapping from registered name to the family's config class.
        WIDTH_FIELDS      : Field carrying the width of the blocks `depth` counts, per family; `cnn_attn_ae` sweeps the attention token width, since its `seq_channels` only sizes the single convolutional stem outside the attention stack.
        LAYER_LIST_FIELDS : Explicit per-layer width list of the families owning one, which replaces both swept axes when set.
    """

    WIDTH_FIELDS = {
        "mlp_ae"           : "hidden_dim",
        "resmlp_ae"        : "hidden_dim",
        "transformer1d_ae" : "hidden_dim",
        "gru_ae"           : "hidden_dim",
        "conv1d_ae"        : "seq_channels",
        "tcn_ae"           : "seq_channels",
        "cnn_attn_ae"      : "hidden_dim",
    }

    LAYER_LIST_FIELDS = {
        "mlp_ae"    : "hidden_dims",
        "conv1d_ae" : "seq_channels_list",
    }

    def __init__(self, trials, ae_model_name: str, model_overrides: dict, config_registry: dict) -> None:
        """Stores the grid settings and resolves the family's width field and default choices.

        Args:
            trials          : Configuration object with `widths` and `depths`.
            ae_model_name   : Registered name of the swept family.
            model_overrides : Model overrides from the entry configuration.
            config_registry : Profile-autoencoder config registry.

        Raises:
            ValueError : If the family is unknown, a resolved axis is empty or duplicated, or the entry overrides pin a swept field or the family's per-layer width list.
        """
        self.trials          = trials
        self.ae_model_name   = ae_model_name
        self.model_overrides = dict(model_overrides)
        self.config_registry = config_registry

        if ae_model_name not in self.WIDTH_FIELDS:
            raise ValueError(f"Unknown autoencoder family '{ae_model_name}' for the capacity sweep; known: {sorted(self.WIDTH_FIELDS)}")

        self.width_field      = self.WIDTH_FIELDS[ae_model_name]
        self.layer_list_field = self.LAYER_LIST_FIELDS.get(ae_model_name)
        self.widths           = list(trials.widths) or self._tunable_choices(self.width_field)
        self.depths           = list(trials.depths) or self._tunable_choices("depth")

        self._validate()

    def _tunable_choices(self, field_name: str) -> list[int]:
        """Returns the family's tuner choices for a field.

        Args:
            field_name : Architecture field the choices are read for.

        Returns:
            The categorical choices of the family's tunable search space.

        Raises:
            ValueError : If the family declares no tunable choices for the field.
        """
        space = self.config_registry[self.ae_model_name].tunable_arch_params()

        if field_name not in space or "choices" not in space[field_name]:
            raise ValueError(f"Family '{self.ae_model_name}' declares no tunable choices for '{field_name}'; set capacity_trials.widths and capacity_trials.depths explicitly")

        return list(space[field_name]["choices"])

    def _validate(self) -> None:
        """Checks that both axes are positive, unique and neither pinned nor overridden by the entry overrides.

        Raises:
            ValueError : On empty, non-positive or duplicated axis values, a pinned swept field, or a pinned per-layer width list.
        """
        for name, values in (("widths", self.widths), ("depths", self.depths)):
            if not values:
                raise ValueError(f"capacity_trials.{name} resolved to no values")

            if any(value <= 0 for value in values):
                raise ValueError(f"capacity_trials.{name} must all be positive, got {values}")

            if len(set(values)) != len(values):
                raise ValueError(f"capacity_trials.{name} must be unique, got {values}")

        pinned = [key for key in (self.width_field, "depth") if key in self.model_overrides]
        if pinned:
            raise ValueError(f"model_overrides pins {pinned}, which the capacity sweep varies; drop them from the overrides")

        if self.layer_list_field is not None and self.layer_list_field in self.model_overrides:
            raise ValueError(f"model_overrides pins '{self.layer_list_field}', which replaces the swept '{self.width_field}' and 'depth' on every rung and would train {len(self.widths) * len(self.depths)} identical models; drop it from the overrides")

    def summary(self) -> dict:
        """Returns the swept axes and run count for logging."""
        return {
            "Width field" : self.width_field,
            "Widths"      : list(self.widths),
            "Depths"      : list(self.depths),
            "Total runs"  : len(self.widths) * len(self.depths),
        }

    def plan(self) -> list[tuple[str, dict]]:
        """Returns one `(run_name, overrides)` pair per width-by-depth cell."""
        plans = []

        for width in self.widths:
            for depth in self.depths:
                overrides = {**self.model_overrides, self.width_field: width, "depth": depth}
                plans.append((f"cap-w{width}-d{depth}", {"model_overrides": overrides}))

        return plans


class ProfileLossTrialPlanner:
    """Plans one run per reconstruction-loss arm, every arm fully defining the loss.

    Attributes:
        loss_trials : Mapping of run label to the `ae_loss` fields the arm enables.
    """

    TERMS = (
        ("use_mse_curve",         "weight_mse_curve"),
        ("use_l1_curve",          "weight_l1_curve"),
        ("use_huber_curve",       "weight_huber_curve"),
        ("use_charbonnier_curve", "weight_charbonnier_curve"),
        ("use_cosine_curve",      "weight_cosine_curve"),
        ("use_log_energy_curve",  "weight_log_energy_curve"),
        ("use_tv_curve",          "weight_tv_curve"),
        ("use_sobolev",           "weight_sobolev"),
        ("use_latent_noise",      "weight_latent_noise"),
    )

    def __init__(self, loss_trials: dict) -> None:
        """Stores and validates the loss arms.

        Args:
            loss_trials : Mapping of run label to loss-config field overrides.

        Raises:
            ValueError : If no arm is given, an arm sets an unknown field, an arm enables no term, or an arm enables a term without its weight.
        """
        self.loss_trials = loss_trials

        self._validate()

    def _validate(self) -> None:
        """Checks that every arm sets known loss fields, enables at least one term and weights every term it enables.

        Raises:
            ValueError : On an empty mapping, unknown fields, an arm with every term disabled, or an enabled term whose weight the arm leaves out.
        """
        if not self.loss_trials:
            raise ValueError("loss_trials must list at least one arm")

        known = {config_field.name for config_field in fields(ProfileAeLossConfig)}

        for label, spec in self.loss_trials.items():
            unknown = sorted(key for key in spec if key not in known)
            if unknown:
                raise ValueError(f"Loss arm '{label}' sets unknown ae_loss fields {unknown}")

            if not any(spec.get(use_key) for use_key, _ in self.TERMS):
                raise ValueError(f"Loss arm '{label}' enables no loss term")

            unweighted = [(use_key, weight_key) for use_key, weight_key in self.TERMS if spec.get(use_key) and weight_key not in spec]
            if unweighted:
                raise ValueError(f"Loss arm '{label}' enables {[use_key for use_key, _ in unweighted]} without setting {[weight_key for _, weight_key in unweighted]}; every weight is zeroed before an arm, so those terms would train at weight 0.0")

    def summary(self) -> dict:
        """Returns the arm labels and run count for logging."""
        return {
            "Loss arms"  : list(self.loss_trials),
            "Total runs" : len(self.loss_trials),
        }

    def _base_overrides(self) -> dict:
        """Returns the overrides disabling every loss term, so each arm states its loss completely."""
        overrides = {}

        for use_key, weight_key in self.TERMS:
            overrides[f"ae_loss.{use_key}"]    = False
            overrides[f"ae_loss.{weight_key}"] = 0.0

        return overrides

    def plan(self) -> list[tuple[str, dict]]:
        """Returns one `(run_name, overrides)` pair per loss arm."""
        plans = []

        for label, spec in self.loss_trials.items():
            overrides = self._base_overrides()
            overrides.update({f"ae_loss.{key}": value for key, value in spec.items()})
            plans.append((f"loss-{label}", overrides))

        return plans


class SmoothnessTrialPlanner:
    """Plans the Sobolev and latent-noise smoothness arms of the profile autoencoder.

    Attributes:
        trials  : Smoothness trial configuration carrying the weights, noise levels and grid flags.
        ae_loss : Base loss configuration the arms only add smoothness levers to.
    """

    def __init__(self, trials, ae_loss: ProfileAeLossConfig) -> None:
        """Stores and validates the smoothness sweep configuration.

        Args:
            trials  : Configuration object with `sobolev_weights`, `latent_noise_stds`, `combine` and `include_baseline`.
            ae_loss : Loss configuration of the entry, supplying the reconstruction terms the arms keep.

        Raises:
            ValueError : If an axis is empty, non-positive or duplicated, or the base loss holds no reconstruction term besides the swept Sobolev penalty.
        """
        self.trials  = trials
        self.ae_loss = ae_loss

        self._validate()

    def _validate(self) -> None:
        """Checks that both axes are positive and unique, and that every arm builds a non-empty reconstruction loss.

        Raises:
            ValueError : On empty, non-positive or duplicated axis values, or a base loss whose only term is the swept Sobolev penalty.
        """
        for name, values in (("sobolev_weights", self.trials.sobolev_weights), ("latent_noise_stds", self.trials.latent_noise_stds)):
            if not values:
                raise ValueError(f"smoothness_trials.{name} must list at least one value")

            if any(value <= 0 for value in values):
                raise ValueError(f"smoothness_trials.{name} must all be positive, got {values}")

            if len(set(values)) != len(values):
                raise ValueError(f"smoothness_trials.{name} must be unique, got {values}")

        kept_terms = [use_key for _, use_key, _ in Loss.TERM_SPECS if use_key != "use_sobolev"]
        if not any(getattr(self.ae_loss, use_key) for use_key in kept_terms):
            raise ValueError(f"ae_loss enables none of {kept_terms}, so every arm leaving use_sobolev off builds an empty reconstruction loss and aborts at launch; enable one reconstruction term before sweeping smoothness")

    def summary(self) -> dict:
        """Returns the swept axes and run count for logging."""
        return {
            "Sobolev weights"   : list(self.trials.sobolev_weights),
            "Latent noise stds" : list(self.trials.latent_noise_stds),
            "Combined grid"     : self.trials.combine,
            "Baseline"          : self.trials.include_baseline,
            "Total runs"        : len(self.plan()),
        }

    def _arm(self, sobolev_weight: float | None, latent_noise_std: float | None) -> dict:
        """Returns the overrides of one arm, disabling the lever whose value is None."""
        overrides = {
            "ae_loss.use_sobolev"      : sobolev_weight is not None,
            "ae_loss.use_latent_noise" : latent_noise_std is not None,
        }

        if sobolev_weight is not None:
            overrides["ae_loss.weight_sobolev"] = sobolev_weight

        if latent_noise_std is not None:
            overrides["ae_loss.latent_noise_std"]    = latent_noise_std
            overrides["ae_loss.weight_latent_noise"] = 1.0

        return overrides

    def plan(self) -> list[tuple[str, dict]]:
        """Returns the baseline, the one-at-a-time arms and, when combined, the full grid."""
        plans = []

        if self.trials.include_baseline:
            plans.append(("smooth-baseline", self._arm(None, None)))

        for weight in self.trials.sobolev_weights:
            plans.append((f"smooth-sob{weight:g}", self._arm(weight, None)))

        for std in self.trials.latent_noise_stds:
            plans.append((f"smooth-ln{std:g}", self._arm(None, std)))

        if self.trials.combine:
            for weight in self.trials.sobolev_weights:
                for std in self.trials.latent_noise_stds:
                    plans.append((f"smooth-sob{weight:g}-ln{std:g}", self._arm(weight, std)))

        return plans


class ProfileAugmentationTrialPlanner:
    """Plans the cumulative profile-augmentation ladder from none to flips, shifts, scaling and noise.

    Attributes:
        trials : Augmentation trial configuration carrying the rung probabilities and noise levels.
    """

    def __init__(self, trials) -> None:
        """Stores and validates the augmentation ladder configuration.

        Args:
            trials : Configuration object with the rung probabilities, `max_shift` and `noise_stds`.

        Raises:
            ValueError : If a probability leaves [0, 1], the shift is not positive, or the noise levels are empty, non-positive or duplicated.
        """
        self.trials = trials

        self._validate()

    def _validate(self) -> None:
        """Checks the rung probabilities, the shift range and the noise levels.

        Raises:
            ValueError : On out-of-range probabilities, a non-positive shift, or bad noise levels.
        """
        probabilities = {
            "flip_probability"      : self.trials.flip_probability,
            "shift_probability"     : self.trials.shift_probability,
            "amp_scale_probability" : self.trials.amp_scale_probability,
            "noise_probability"     : self.trials.noise_probability,
        }

        bad = sorted(name for name, value in probabilities.items() if not 0.0 < value <= 1.0)
        if bad:
            raise ValueError(f"augmentation_trials {bad} must lie in (0, 1]")

        if self.trials.max_shift <= 0:
            raise ValueError(f"augmentation_trials.max_shift must be positive, got {self.trials.max_shift}")

        if not self.trials.noise_stds:
            raise ValueError("augmentation_trials.noise_stds must list at least one level")

        if any(std <= 0 for std in self.trials.noise_stds):
            raise ValueError(f"augmentation_trials.noise_stds must all be positive, got {self.trials.noise_stds}")

        if len(set(self.trials.noise_stds)) != len(self.trials.noise_stds):
            raise ValueError(f"augmentation_trials.noise_stds must be unique, got {self.trials.noise_stds}")

    def summary(self) -> dict:
        """Returns the ladder settings and run count for logging."""
        return {
            "Flip / shift / scale p" : (self.trials.flip_probability, self.trials.shift_probability, self.trials.amp_scale_probability),
            "Noise stds"             : list(self.trials.noise_stds),
            "Total runs"             : 4 + len(self.trials.noise_stds),
        }

    def _rung(self, flip: bool, shift: bool, scale: bool, noise_std: float | None) -> dict:
        """Returns the augmentation overrides of one cumulative rung."""
        return {
            "profile_augmentation.p_flip"      : self.trials.flip_probability if flip else 0.0,
            "profile_augmentation.p_shift"     : self.trials.shift_probability if shift else 0.0,
            "profile_augmentation.max_shift"   : self.trials.max_shift,
            "profile_augmentation.p_amp_scale" : self.trials.amp_scale_probability if scale else 0.0,
            "profile_augmentation.p_noise"     : self.trials.noise_probability if noise_std is not None else 0.0,
            "profile_augmentation.noise_std"   : noise_std if noise_std is not None else 0.0,
        }

    def plan(self) -> list[tuple[str, dict]]:
        """Returns the cumulative ladder, ending in one rung per noise level."""
        plans = [
            ("aug-0-none",  self._rung(False, False, False, None)),
            ("aug-1-flip",  self._rung(True,  False, False, None)),
            ("aug-2-shift", self._rung(True,  True,  False, None)),
            ("aug-3-scale", self._rung(True,  True,  True,  None)),
        ]

        for std in self.trials.noise_stds:
            plans.append((f"aug-4-noise{std:g}", self._rung(True, True, True, std)))

        return plans


class SamplingTrialPlanner:
    """Plans the data-mix grid over the pixel subsample and the kept empty-profile fraction.

    Attributes:
        trials : Sampling trial configuration carrying `subsamples` and `empty_fracs`.
    """

    def __init__(self, trials) -> None:
        """Stores and validates the sampling grid configuration.

        Args:
            trials : Configuration object with `subsamples` and `empty_fracs`.

        Raises:
            ValueError : If an axis is empty, out of range or duplicated.
        """
        self.trials = trials

        self._validate()

    def _validate(self) -> None:
        """Checks that the subsamples lie in (0, 1], the empty fractions in [0, 1], and both are unique.

        Raises:
            ValueError : On empty, out-of-range or duplicated axis values.
        """
        if not self.trials.subsamples:
            raise ValueError("sampling_trials.subsamples must list at least one fraction")

        if any(not 0.0 < value <= 1.0 for value in self.trials.subsamples):
            raise ValueError(f"sampling_trials.subsamples must lie in (0, 1], got {self.trials.subsamples}")

        if len(set(self.trials.subsamples)) != len(self.trials.subsamples):
            raise ValueError(f"sampling_trials.subsamples must be unique, got {self.trials.subsamples}")

        if not self.trials.empty_fracs:
            raise ValueError("sampling_trials.empty_fracs must list at least one fraction")

        if any(not 0.0 <= value <= 1.0 for value in self.trials.empty_fracs):
            raise ValueError(f"sampling_trials.empty_fracs must lie in [0, 1], got {self.trials.empty_fracs}")

        if len(set(self.trials.empty_fracs)) != len(self.trials.empty_fracs):
            raise ValueError(f"sampling_trials.empty_fracs must be unique, got {self.trials.empty_fracs}")

    def summary(self) -> dict:
        """Returns the swept axes and run count for logging."""
        return {
            "Subsamples"      : list(self.trials.subsamples),
            "Empty fractions" : list(self.trials.empty_fracs),
            "Total runs"      : len(self.trials.subsamples) * len(self.trials.empty_fracs),
        }

    def plan(self) -> list[tuple[str, dict]]:
        """Returns one `(run_name, overrides)` pair per subsample-by-empty-fraction cell."""
        plans = []

        for subsample in self.trials.subsamples:
            for empty_frac in self.trials.empty_fracs:
                plans.append((f"smp-p{subsample:g}-e{empty_frac:g}", {"pixel_subsample": subsample, "keep_empty_frac": empty_frac}))

        return plans
