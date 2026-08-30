"""Per-channel normalization strategies, presets and fitted statistics.

Every input and output channel is normalized as (x - loc) / scale, optionally
after a log1p compression. A strategy decides how loc and scale are obtained:
from percentiles or moments of a sampled pool, or from fixed physical bounds.
The default "per_slot" mapping assigns a strategy to each channel slot key.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum        import Enum
from typing      import Optional

import numpy as np


class NormMethod(Enum):
    """Rule deciding how a channel's location and scale are obtained."""

    MIN_MAX_P999   = "min_max_p999"
    ROBUST_IQR     = "robust_iqr"
    FIXED_DIV_PI   = "fixed_div_pi"
    ZSCORE         = "zscore"
    FIXED_LOG1P    = "fixed_log1p"
    FIXED_ANGLE_01 = "fixed_angle_01"
    FIXED_BOUNDS   = "fixed_bounds"


@dataclass
class ChannelStrategy:
    """Normalization rule applied to one channel group.

    Attributes:
        norm_method : Rule deciding how location and scale are obtained.
        apply_log1p : Whether log1p compression precedes the statistics fit.
        bounds      : For FIXED_BOUNDS, the (min_tuple, max_tuple) pair of per-output-channel physical bounds; None otherwise.
    """

    norm_method : NormMethod
    apply_log1p : bool = False
    bounds      : Optional[tuple] = None

    @property
    def data_free(self) -> bool:
        """True when the statistics are constants and need no data pool."""
        return self.norm_method in (NormMethod.FIXED_DIV_PI, NormMethod.FIXED_LOG1P, NormMethod.FIXED_ANGLE_01)

    def fit(self, flat: np.ndarray) -> tuple[float, float]:
        """Returns the (loc, scale) pair for this strategy.

        Args:
            flat : Flattened sample pool of values for the channel group, of shape (n_samples,). Ignored by the data-free methods.

        Returns:
            The location subtracted from the channel and the strictly positive scale it is divided by. For phase methods the values are expressed in radians.

        Raises:
            ValueError : If the method is FIXED_BOUNDS, if the sample pool is empty for a data-driven method, or if the method is unknown.
        """
        if self.norm_method is NormMethod.FIXED_DIV_PI:
            return 0.0, float(np.pi)

        if self.norm_method is NormMethod.FIXED_LOG1P:
            return 0.0, 1.0

        if self.norm_method is NormMethod.FIXED_ANGLE_01:
            return -float(np.pi), 2.0 * float(np.pi)

        if self.norm_method is NormMethod.FIXED_BOUNDS:
            raise ValueError("fixed_bounds is resolved per output channel by the stats computer; it cannot be fitted from data and is not valid for input channels.")

        if flat.size == 0:
            raise ValueError(f"Cannot fit {self.norm_method.value} normalization on an empty sample pool; the collection mask left no pixels for this channel group.")

        data = (
            np.log1p(np.maximum(flat.astype(np.float64), 0.0))
            if self.apply_log1p
            else flat.astype(np.float64)
        )

        if self.norm_method is NormMethod.MIN_MAX_P999:
            lo = float(np.percentile(data, 0.1))
            hi = float(np.percentile(data, 99.9))
            return lo, max(hi - lo, 1e-8)

        if self.norm_method is NormMethod.ROBUST_IQR:
            med = float(np.percentile(data, 50))
            iqr = float(np.percentile(data, 75)) - float(np.percentile(data, 25))
            return med, max(iqr, 1e-8)

        if self.norm_method is NormMethod.ZSCORE:
            m = float(data.mean())
            s = float(data.std())
            return m, max(s, 1e-8)

        raise ValueError(f"Unknown norm method: {self.norm_method}")

    def as_dict(self) -> dict:
        """Returns the JSON-serializable form of this strategy."""
        bounds = [list(side) for side in self.bounds] if self.bounds is not None else None

        return {"norm_method": self.norm_method.value, "apply_log1p": self.apply_log1p, "bounds": bounds}

    @classmethod
    def from_dict(cls, d: dict) -> "ChannelStrategy":
        """Rebuilds a strategy from its serialized form.

        Args:
            d : Payload previously produced by `as_dict`.

        Returns:
            The reconstructed ChannelStrategy.
        """
        bounds = d["bounds"]

        return cls(
            norm_method = NormMethod(d["norm_method"]),
            apply_log1p = bool(d["apply_log1p"]),
            bounds      = tuple(tuple(side) for side in bounds) if bounds is not None else None,
        )

    @classmethod
    def from_slot(cls, full_key: str) -> "ChannelStrategy":
        """Returns the default strategy registered for a channel slot key."""
        return _SLOT_STRATEGIES[full_key]


class Presets:
    """Named channel strategies selectable from configuration by string."""

    MIN_MAX          = ChannelStrategy(NormMethod.MIN_MAX_P999,   apply_log1p=False)
    MIN_MAX_LOG1P    = ChannelStrategy(NormMethod.MIN_MAX_P999,   apply_log1p=True)
    ROBUST_IQR       = ChannelStrategy(NormMethod.ROBUST_IQR,     apply_log1p=False)
    ROBUST_IQR_LOG1P = ChannelStrategy(NormMethod.ROBUST_IQR,     apply_log1p=True)
    FIXED_DIV_PI     = ChannelStrategy(NormMethod.FIXED_DIV_PI,   apply_log1p=False)
    ZSCORE           = ChannelStrategy(NormMethod.ZSCORE,         apply_log1p=False)
    ZSCORE_LOG1P     = ChannelStrategy(NormMethod.ZSCORE,         apply_log1p=True)
    FIXED_LOG1P      = ChannelStrategy(NormMethod.FIXED_LOG1P,    apply_log1p=True)
    FIXED_ANGLE_01   = ChannelStrategy(NormMethod.FIXED_ANGLE_01, apply_log1p=False)
    FIXED_BOUNDS     = ChannelStrategy(NormMethod.FIXED_BOUNDS,   apply_log1p=False)

    @classmethod
    def names(cls) -> list[str]:
        """Returns the preset names accepted by `by_name`."""
        return ["min_max", "min_max_log1p", "robust_iqr", "robust_iqr_log1p", "fixed_div_pi", "zscore", "zscore_log1p", "fixed_log1p", "fixed_angle_01", "fixed_bounds"]

    @classmethod
    def by_name(cls, name: str) -> ChannelStrategy:
        """Resolves a preset name to its strategy.

        Args:
            name : Preset name, matched case-insensitively after stripping.

        Returns:
            The ChannelStrategy registered under that name.

        Raises:
            ValueError : If the name is not a known preset.
        """
        table = {
            "min_max"          : cls.MIN_MAX,
            "min_max_log1p"    : cls.MIN_MAX_LOG1P,
            "robust_iqr"       : cls.ROBUST_IQR,
            "robust_iqr_log1p" : cls.ROBUST_IQR_LOG1P,
            "fixed_div_pi"     : cls.FIXED_DIV_PI,
            "zscore"           : cls.ZSCORE,
            "zscore_log1p"     : cls.ZSCORE_LOG1P,
            "fixed_log1p"      : cls.FIXED_LOG1P,
            "fixed_angle_01"   : cls.FIXED_ANGLE_01,
            "fixed_bounds"     : cls.FIXED_BOUNDS,
        }

        key = name.strip().lower()
        if key not in table:
            raise ValueError(f"Unknown normalization preset '{name}'; valid presets are {cls.names()} or 'per_slot'")

        return table[key]


_SLOT_STRATEGIES: dict[str, ChannelStrategy] = {
    "chan/mag"           : Presets.ROBUST_IQR_LOG1P,
    "chan/raw_re_im"     : Presets.ZSCORE,
    "chan/norm_re_im"    : Presets.ZSCORE,
    "chan/phase"         : Presets.ZSCORE,

    "derived/mag"        : Presets.ROBUST_IQR_LOG1P,
    "derived/raw_re_im"  : Presets.ZSCORE,
    "derived/norm_re_im" : Presets.ZSCORE,
    "derived/phase"      : Presets.FIXED_DIV_PI,

    "out/amp"            : Presets.ROBUST_IQR_LOG1P,
    "out/mu"             : Presets.ZSCORE,
    "out/sigma"          : Presets.ROBUST_IQR_LOG1P,

    "aux/value"          : Presets.ZSCORE,
}


@dataclass
class OutputClampConfig:
    """Leaky clamp applied to the network outputs in physical units.

    Attributes:
        enabled           : Whether the clamp is applied at all.
        floor             : Lower bound of the clamped range.
        ceil              : Upper bound of the clamped range.
        leaky_slope       : Slope kept outside the range for curve outputs.
        param_leaky_slope : Slope kept outside the range for Gaussian parameter outputs.
        amp_max           : Upper bound applied to predicted scatterer amplitudes.
    """

    enabled           : bool  = True
    floor             : float = 0.0
    ceil              : float = 200.0
    leaky_slope       : float = 0.1
    param_leaky_slope : float = 0.1
    amp_max           : float = 200.0

    def as_dict(self) -> dict:
        """Returns the JSON-serializable form of this clamp configuration."""
        return {"enabled": self.enabled, "floor": self.floor, "ceil": self.ceil, "leaky_slope": self.leaky_slope, "param_leaky_slope": self.param_leaky_slope, "amp_max": self.amp_max}

    @classmethod
    def from_dict(cls, payload: dict) -> "OutputClampConfig":
        """Rebuilds a clamp configuration from its serialized form.

        Args:
            payload : Payload previously produced by `as_dict`.

        Returns:
            The reconstructed OutputClampConfig.
        """
        return cls(
            enabled           = bool(payload["enabled"]),
            floor             = float(payload["floor"]),
            ceil              = float(payload["ceil"]),
            leaky_slope       = float(payload["leaky_slope"]),
            param_leaky_slope = float(payload["param_leaky_slope"]),
            amp_max           = float(payload["amp_max"]),
        )


@dataclass
class NormalizationConfig:
    """User-facing normalization choices for a training or inference run.

    A slot-specific field left at "default" falls back to the blanket input or
    output strategy, which in turn may be "per_slot" to use the built-in slot
    table.

    Attributes:
        input_strategy          : Preset name applied to every input slot, or "per_slot".
        output_strategy         : Preset name applied to every output slot, or "per_slot".
        pass_mag                : Preset override for single-pass magnitude channels.
        pass_phase              : Preset override for single-pass phase channels.
        ifg_mag                 : Preset override for interferogram magnitude channels.
        ifg_phase               : Preset override for interferogram phase channels.
        out_amp                 : Preset override for predicted Gaussian amplitudes.
        out_mu                  : Preset override for predicted Gaussian means in metres.
        out_sigma               : Preset override for predicted Gaussian widths in metres.
        dem                     : Preset override for the DEM elevation channel in metres.
        fixed_out_bounds_min    : Per-output-channel lower bounds used by the fixed_bounds preset.
        fixed_out_bounds_max    : Per-output-channel upper bounds used by the fixed_bounds preset.
        clamp_output            : Whether the output clamp is enabled.
        clamp_floor             : Lower bound of the output clamp.
        clamp_ceil              : Upper bound of the output clamp.
        clamp_leaky_slope       : Leak slope outside the clamped range for curve outputs.
        param_clamp_leaky_slope : Leak slope outside the clamped range for parameter outputs.
        amp_max                 : Upper bound applied to predicted scatterer amplitudes.
        SLOT_FIELDS             : Mapping from channel slot key to the field holding its override.
    """

    input_strategy  : str = "per_slot"
    output_strategy : str = "per_slot"

    chan_mag      : str = "default"
    channel_phase : str = "default"
    derived_mag   : str = "default"
    derived_phase : str = "default"
    out_amp       : str = "default"
    out_mu        : str = "default"
    out_sigma     : str = "default"
    aux           : str = "default"

    fixed_out_bounds_min : tuple = (1e-05, -20.0, 0.01, 1e-05, -20.0, 0.01)
    fixed_out_bounds_max : tuple = (10.0, 80.0, 50.0, 10.0, 80.0, 50.0)

    clamp_output            : bool  = True
    clamp_floor             : float = 0.0
    clamp_ceil              : float = 200.0
    clamp_leaky_slope       : float = 0.1
    param_clamp_leaky_slope : float = 0.1
    amp_max                 : float = 200.0

    SLOT_FIELDS = {
        "chan/mag"      : "chan_mag",
        "chan/phase"    : "channel_phase",
        "derived/mag"   : "derived_mag",
        "derived/phase" : "derived_phase",
        "out/amp"       : "out_amp",
        "out/mu"        : "out_mu",
        "out/sigma"     : "out_sigma",
        "aux/value"     : "aux",
    }

    def strategy(self, which: str, slot_key: str) -> ChannelStrategy:
        """Resolves the strategy governing one channel slot.

        A per-slot override takes precedence; otherwise the blanket input or
        output strategy applies, with "per_slot" deferring to the slot table.

        Args:
            which    : Either "input" or "output", selecting the blanket strategy.
            slot_key : Channel slot key such as "ifg/phase" or "out/sigma".

        Returns:
            The ChannelStrategy for that slot, with fixed bounds filled in.
        """
        field_name = self.SLOT_FIELDS.get(slot_key)
        if field_name is not None and getattr(self, field_name) != "default":
            return self._with_bounds(Presets.by_name(getattr(self, field_name)))

        name = self.input_strategy if which == "input" else self.output_strategy
        if name == "per_slot":
            return ChannelStrategy.from_slot(slot_key)
        return self._with_bounds(Presets.by_name(name))

    def _with_bounds(self, strategy: ChannelStrategy) -> ChannelStrategy:
        """Returns the strategy with the configured fixed output bounds attached."""
        if strategy.norm_method is not NormMethod.FIXED_BOUNDS:
            return strategy
        return ChannelStrategy(NormMethod.FIXED_BOUNDS, bounds=(tuple(self.fixed_out_bounds_min), tuple(self.fixed_out_bounds_max)))

    def clamp(self) -> OutputClampConfig:
        """Returns the output clamp configuration built from the clamp fields."""
        return OutputClampConfig(enabled=self.clamp_output, floor=self.clamp_floor, ceil=self.clamp_ceil, leaky_slope=self.clamp_leaky_slope, param_leaky_slope=self.param_clamp_leaky_slope, amp_max=self.amp_max)


@dataclass
class ChannelStats:
    """Fitted normalization statistics for an ordered list of channels.

    Attributes:
        loc        : Per-channel location subtracted before scaling.
        scale      : Per-channel strictly positive scale divided out.
        names      : Per-channel slot names in tensor channel order.
        strategies : Per-channel strategy that produced the statistics.
        clampable  : Per-channel flag marking outputs the leaky clamp applies to.
    """

    loc        : list[float]
    scale      : list[float]
    names      : Optional[list[str]]             = None
    strategies : Optional[list[ChannelStrategy]] = None
    clampable  : Optional[list[bool]]            = None

    @property
    def n_channels(self) -> int:
        """Number of channels these statistics cover."""
        return len(self.loc)

    def as_dict(self) -> dict:
        """Returns the JSON-serializable per-channel statistics payload.

        Returns:
            A dict with a "channels" list, one entry per channel carrying its name, location, scale, clampable flag and serialized strategy.

        Raises:
            ValueError : If the per-channel clampable flags are missing.
        """
        if self.clampable is None:
            raise ValueError("ChannelStats is missing per-channel clampable flags; cannot serialize.")

        entries = []
        for i, (m, s) in enumerate(zip(self.loc, self.scale)):
            entry: dict = {"name": self.names[i], "loc": m, "scale": s, "clampable": bool(self.clampable[i])}
            if self.strategies and i < len(self.strategies):
                entry.update(self.strategies[i].as_dict())

            entries.append(entry)

        return {"channels": entries}

    @classmethod
    def from_dict(cls, payload: dict) -> "ChannelStats":
        """Rebuilds channel statistics from their serialized form.

        Args:
            payload : Payload previously produced by `as_dict`.

        Returns:
            The reconstructed ChannelStats, in the stored channel order.
        """
        entries    = payload["channels"]
        strategies = [ChannelStrategy.from_dict(e) for e in entries]

        return cls(
            loc        = [e["loc"]   for e in entries],
            scale      = [e["scale"] for e in entries],
            names      = [e["name"]  for e in entries],
            strategies = strategies,
            clampable  = [bool(e["clampable"]) for e in entries],
        )
