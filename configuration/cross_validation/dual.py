"""Dual-trunk specific settings for cross-validation runs."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DualCvConfig:
    """Dual-trunk model settings used when cross-validating the dual family.

    Attributes:
        model_name          : Dual model family instantiated per fold.
        params_backbone     : Backbone family of the parameter trunk.
        params_input        : Input representations concatenated for the parameter trunk.
        params_overrides    : Architecture fields overridden on the parameter trunk.
        existence_backbone  : Backbone family of the existence trunk.
        existence_input     : Input representations concatenated for the existence trunk.
        existence_overrides : Architecture fields overridden on the existence trunk.
    """

    model_name : str = "dual_resunet"

    params_backbone     : str       = "unet_skip"
    params_input        : list[str] = field(default_factory=lambda: ["chan", "derived", "aux"])
    params_overrides    : dict      = field(default_factory=dict)

    existence_backbone  : str       = "unet_skip"
    existence_input     : list[str] = field(default_factory=lambda: ["chan", "derived", "aux"])
    existence_overrides : dict      = field(default_factory=dict)
