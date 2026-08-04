from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path
from typing      import Literal, Optional, Sequence, Tuple

import numpy as np

from configuration.normalization.general import ChannelStrategy, NormalizationConfig
from tools.data.representation           import Representation
from tools.data.regions                  import SplitRegions


@dataclass
class InputConfig:
    use_primary            : bool           = True
    primary_representation : Representation = Representation.MAG_ONLY
 
    use_secondaries            : bool           = False
    secondaries_representation : Representation = Representation.MAG_ONLY
 
    use_derived            : bool           = True
    derived_representation : Representation = Representation.ANGLE_ONLY

    use_aux                       : bool           = False

    @classmethod
    def full_stack(cls) -> "InputConfig":
        return cls(
            use_primary     = True,  primary_representation        = Representation.MAG_ONLY,
            use_secondaries = True,  secondaries_representation    = Representation.MAG_ONLY,
            use_derived     = True,  derived_representation = Representation.ANGLE_ONLY,
        )

    @property
    def primary_channels_per_source(self) -> int:
        return self.primary_representation.channels_per_source if self.use_primary else 0

    @property
    def secondaries_channels_per_source(self) -> int:
        return self.secondaries_representation.channels_per_source if self.use_secondaries else 0

    @property
    def derived_channels_per_source(self) -> int:
        return self.derived_representation.channels_per_source if self.use_derived else 0

    def total_channels(self, n_secondaries: int, n_derived: int) -> int:
        n  = self.primary_channels_per_source
        n += n_secondaries    * self.secondaries_channels_per_source
        n += n_derived * self.derived_channels_per_source
        if self.use_aux:
            n += 1
        return n

    def channel_group_keys(self, n_secondaries: int, n_derived: int) -> list[str]:
        keys: list[str] = []

        if self.use_primary:
            slot_kinds = self.primary_representation.slot_kinds
            cpp        = len(slot_kinds)
            keys.extend(f"chan/{slot_kinds[i % cpp]}" for i in range(1 * cpp))

        if self.use_secondaries:
            slot_kinds = self.secondaries_representation.slot_kinds
            cpp        = len(slot_kinds)
            keys.extend(f"chan/{slot_kinds[i % cpp]}" for i in range(n_secondaries * cpp))

        if self.use_derived:
            slot_kinds = self.derived_representation.slot_kinds
            cpp        = len(slot_kinds)
            keys.extend(f"derived/{slot_kinds[i % cpp]}" for i in range(n_derived * cpp))

        if self.use_aux:
            keys.append("aux/value")

        return keys

    def as_dict(self) -> dict:
        return {
            "use_primary"                : self.use_primary,
            "primary_representation"     : self.primary_representation.value,
            "use_secondaries"            : self.use_secondaries,
            "secondaries_representation" : self.secondaries_representation.value,
            "use_derived"                : self.use_derived,
            "derived_representation"     : self.derived_representation.value,
            "use_aux"                    : self.use_aux,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "InputConfig":
        return cls(
            use_primary                = bool(payload["use_primary"]),
            primary_representation     = Representation(payload["primary_representation"]),
            use_secondaries            = bool(payload["use_secondaries"]),
            secondaries_representation = Representation(payload["secondaries_representation"]),
            use_derived                = bool(payload["use_derived"]),
            derived_representation     = Representation(payload["derived_representation"]),
            use_aux                    = bool(payload["use_aux"]),
        )

      
@dataclass
class OutputConfig:
    use_amplitude : bool = True
    use_mu        : bool = True
    use_sigma     : bool = True
    
    output_strategies : dict[str, ChannelStrategy] = field(default_factory=lambda: {
        "out/amp"   : ChannelStrategy.from_slot("out/amp"),
        "out/mu"    : ChannelStrategy.from_slot("out/mu"),
        "out/sigma" : ChannelStrategy.from_slot("out/sigma"),
    })

    def strategy_for(self, role_key: str) -> ChannelStrategy:
        return self.output_strategies[role_key]

    @property
    def role_names(self) -> list[str]:
        names: list[str] = []
        if self.use_amplitude:
            names.append("a")
        if self.use_mu:
            names.append("mu")
        if self.use_sigma:
            names.append("sig")
        return names

    @property
    def params_per_gaussian(self) -> int:
        return len(self.role_names)

    def selected_indices(self, n_gaussians: int) -> list[int]:
        role_to_idx = {"a": 0, "mu": 1, "sig": 2}
        local       = [role_to_idx[name] for name in self.role_names]

        indices: list[int] = []
        for g in range(n_gaussians):
            base = g * 3
            for i in local:
                indices.append(base + i)
     
        return indices

    def total_channels(self, n_gaussians: int) -> int:
        return n_gaussians * self.params_per_gaussian

    def as_dict(self) -> dict:
        return {
            "use_amplitude"     : self.use_amplitude,
            "use_mu"            : self.use_mu,
            "use_sigma"         : self.use_sigma,
            "output_strategies" : {
                k: v.as_dict() for k, v in self.output_strategies.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "OutputConfig":
        strategies = {key: ChannelStrategy.from_dict(value) for key, value in payload["output_strategies"].items()}

        return cls(
            use_amplitude     = bool(payload["use_amplitude"]),
            use_mu            = bool(payload["use_mu"]),
            use_sigma         = bool(payload["use_sigma"]),
            output_strategies = strategies,
        )

   
@dataclass
class PatchConfig:
    size                  : Tuple[int, int] = (64, 32)
    stride                : Tuple[int, int] = (32, 16)
    use_symmetric_padding : bool            = True


@dataclass
class AugmentationConfig:
    p_flip_h  : float = 0.5
    p_flip_v  : float = 0.5
    p_rot90   : float = 0.0

    noise_std : float = 0.01
    p_noise   : float = 0.0


@dataclass
class DatasetConfig:
    preprocessing_run_directory : Path
    split_regions               : SplitRegions
    parameters_path             : Optional[Path]            = None
    secondary_labels            : Optional[Tuple[str, ...]] = ("CH01", "CH02", "CH03", "CH04")
    patch                       : PatchConfig               = field(default_factory=PatchConfig)
    input_config                : InputConfig               = field(default_factory=InputConfig)
    output_config               : OutputConfig              = field(default_factory=OutputConfig)
    normalization               : NormalizationConfig       = field(default_factory=NormalizationConfig)
    augmentation                : AugmentationConfig        = field(default_factory=AugmentationConfig)

    batch_size      : int                  = 8
    num_workers     : int                  = 16
    prefetch_factor : int                  = 8
    shuffle_train   : bool                 = True
    pin_memory      : bool                 = True
    z_axis          : Optional[np.ndarray] = field(default=None, repr=False)
    n_gaussians     : int                  = 1

