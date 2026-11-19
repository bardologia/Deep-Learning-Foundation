
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DualCvConfig:

    model_name : str = "dual_resunet"

    params_backbone     : str       = "unet_skip"
    params_input        : list[str] = field(default_factory=lambda: ["chan", "derived", "aux"])
    params_overrides    : dict      = field(default_factory=dict)

    existence_backbone  : str       = "unet_skip"
    existence_input     : list[str] = field(default_factory=lambda: ["chan", "derived", "aux"])
    existence_overrides : dict      = field(default_factory=dict)
