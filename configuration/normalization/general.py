from __future__ import annotations

from dataclasses import dataclass
from enum        import Enum
from typing      import Optional

import numpy as np


class NormMethod(Enum):
    MIN_MAX_P999   = "min_max_p999"
    ROBUST_IQR     = "robust_iqr"
    FIXED_DIV_PI   = "fixed_div_pi"
    ZSCORE         = "zscore"
    FIXED_LOG1P    = "fixed_log1p"
    FIXED_ANGLE_01 = "fixed_angle_01"
    FIXED_BOUNDS   = "fixed_bounds"


@dataclass
class ChannelStrategy:
    norm_method : NormMethod
    apply_log1p : bool = False
    bounds      : Optional[tuple] = None

    @property
    def data_free(self) -> bool:
        return self.norm_method in (NormMethod.FIXED_DIV_PI, NormMethod.FIXED_LOG1P, NormMethod.FIXED_ANGLE_01)

    def fit(self, flat: np.ndarray) -> tuple[float, float]:
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
        return {"norm_method": self.norm_method.value, "apply_log1p": self.apply_log1p}

    @classmethod
    def from_dict(cls, d: dict) -> "ChannelStrategy":
        return cls(
            norm_method = NormMethod(d["norm_method"]),
            apply_log1p = bool(d["apply_log1p"]),
        )

    @classmethod
    def from_slot(cls, full_key: str) -> "ChannelStrategy":
        return _SLOT_STRATEGIES[full_key]


class Presets:
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
        return ["min_max", "min_max_log1p", "robust_iqr", "robust_iqr_log1p", "fixed_div_pi", "zscore", "zscore_log1p", "fixed_log1p", "fixed_angle_01", "fixed_bounds"]

    @classmethod
    def by_name(cls, name: str) -> ChannelStrategy:
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
    "pass/mag"        : Presets.ROBUST_IQR_LOG1P,
    "pass/raw_re_im"  : Presets.ZSCORE,
    "pass/norm_re_im" : Presets.ZSCORE,
    "pass/phase"      : Presets.ZSCORE,

    "ifg/mag"         : Presets.ROBUST_IQR_LOG1P,
    "ifg/raw_re_im"   : Presets.ZSCORE,
    "ifg/norm_re_im"  : Presets.ZSCORE,
    "ifg/phase"       : Presets.FIXED_DIV_PI,

    "out/amp"         : Presets.ROBUST_IQR_LOG1P,
    "out/mu"          : Presets.ZSCORE,
    "out/sigma"       : Presets.ROBUST_IQR_LOG1P,

    "dem/elevation"   : Presets.ZSCORE,
}


@dataclass
class OutputClampConfig:
    enabled           : bool  = True
    floor             : float = 0.0
    ceil              : float = 200.0
    leaky_slope       : float = 0.1
    param_leaky_slope : float = 0.1
    amp_max           : float = 200.0

    def as_dict(self) -> dict:
        return {"enabled": self.enabled, "floor": self.floor, "ceil": self.ceil, "leaky_slope": self.leaky_slope, "param_leaky_slope": self.param_leaky_slope, "amp_max": self.amp_max}

    @classmethod
    def from_dict(cls, payload: dict) -> "OutputClampConfig":
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
    input_strategy  : str = "per_slot"
    output_strategy : str = "per_slot"

    pass_mag   : str = "default"
    pass_phase : str = "default"
    ifg_mag    : str = "default"
    ifg_phase  : str = "default"
    out_amp    : str = "default"
    out_mu     : str = "default"
    out_sigma  : str = "default"
    dem        : str = "default"

    fixed_out_bounds_min : tuple = (1e-05, -15.0, 0.01, 1e-05, 1.0, 1.0)
    fixed_out_bounds_max : tuple = (10.0, 5.0, 5.0, 10.0, 40.0, 20.0)

    clamp_output            : bool  = True
    clamp_floor             : float = 0.0
    clamp_ceil              : float = 200.0
    clamp_leaky_slope       : float = 0.1
    param_clamp_leaky_slope : float = 0.1
    amp_max                 : float = 200.0

    SLOT_FIELDS = {
        "pass/mag"      : "pass_mag",
        "pass/phase"    : "pass_phase",
        "ifg/mag"       : "ifg_mag",
        "ifg/phase"     : "ifg_phase",
        "out/amp"       : "out_amp",
        "out/mu"        : "out_mu",
        "out/sigma"     : "out_sigma",
        "dem/elevation" : "dem",
    }

    def strategy(self, which: str, slot_key: str) -> ChannelStrategy:
        field_name = self.SLOT_FIELDS.get(slot_key)
        if field_name is not None and getattr(self, field_name) != "default":
            return self._with_bounds(Presets.by_name(getattr(self, field_name)))

        name = self.input_strategy if which == "input" else self.output_strategy
        if name == "per_slot":
            return ChannelStrategy.from_slot(slot_key)
        return self._with_bounds(Presets.by_name(name))

    def _with_bounds(self, strategy: ChannelStrategy) -> ChannelStrategy:
        if strategy.norm_method is not NormMethod.FIXED_BOUNDS:
            return strategy
        return ChannelStrategy(NormMethod.FIXED_BOUNDS, bounds=(tuple(self.fixed_out_bounds_min), tuple(self.fixed_out_bounds_max)))

    def clamp(self) -> OutputClampConfig:
        return OutputClampConfig(enabled=self.clamp_output, floor=self.clamp_floor, ceil=self.clamp_ceil, leaky_slope=self.clamp_leaky_slope, param_leaky_slope=self.param_clamp_leaky_slope, amp_max=self.amp_max)


@dataclass
class ChannelStats:
    loc        : list[float]
    scale      : list[float]
    names      : Optional[list[str]]             = None
    strategies : Optional[list[ChannelStrategy]] = None
    clampable  : Optional[list[bool]]            = None

    @property
    def n_channels(self) -> int:
        return len(self.loc)

    def as_dict(self) -> dict:
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
        entries    = payload["channels"]
        strategies = [ChannelStrategy.from_dict(e) for e in entries]

        return cls(
            loc        = [e["loc"]   for e in entries],
            scale      = [e["scale"] for e in entries],
            names      = [e["name"]  for e in entries],
            strategies = strategies,
            clampable  = [bool(e["clampable"]) for e in entries],
        )
