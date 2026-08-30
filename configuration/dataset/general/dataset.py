"""Dataset configuration for the patch-based tomographic SAR training pipelines.

The dataclasses here describe which SAR channels form the network input, how
Gaussian scatterer parameters are laid out in the target, how patches are cut
from the scene, and how the loaders are parameterised.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path
from typing      import ClassVar, Optional, Tuple

import numpy as np

from configuration.normalization.general import ChannelStrategy, NormalizationConfig
from tools.data.representation           import Representation
from tools.data.regions                  import SplitRegions


@dataclass
class InputConfig:
    """Selection and complex representation of the SAR channels forming the input.

    Attributes:
        use_primary                   : Whether the primary pass is part of the input stack.
        primary_representation        : Complex-to-real representation of the primary pass.
        use_secondaries               : Whether the secondary passes are part of the input stack.
        secondaries_representation    : Complex-to-real representation of the secondary passes.
        use_interferograms            : Whether interferograms are part of the input stack.
        interferograms_representation : Complex-to-real representation of the interferograms.
        use_dem                       : Whether the DEM elevation channel is appended.
    """

    use_primary            : bool           = True
    primary_representation : Representation = Representation.MAG_ONLY
 
    use_secondaries            : bool           = False
    secondaries_representation : Representation = Representation.MAG_ONLY
 
    use_derived            : bool           = True
    derived_representation : Representation = Representation.ANGLE_ONLY

    use_aux                       : bool           = False

    @classmethod
    def full_stack(cls) -> "InputConfig":
        """Returns the standard configuration: primary and secondary magnitudes plus interferometric phase."""
        return cls(
            use_primary     = True,  primary_representation        = Representation.MAG_ONLY,
            use_secondaries = True,  secondaries_representation    = Representation.MAG_ONLY,
            use_derived     = True,  derived_representation = Representation.ANGLE_ONLY,
        )

    @classmethod
    def no_stack(cls) -> "InputConfig":
        """Returns a configuration with every input source disabled."""
        return cls(use_primary=False, use_secondaries=False, use_interferograms=False, use_dem=False)

    @property
    def primary_channels_per_source(self) -> int:
        """Number of real channels contributed by the primary chan, zero if unused."""
        return self.primary_representation.channels_per_source if self.use_primary else 0

    @property
    def secondaries_channels_per_source(self) -> int:
        """Number of real channels contributed by each secondary chan, zero if unused."""
        return self.secondaries_representation.channels_per_source if self.use_secondaries else 0

    @property
    def derived_channels_per_source(self) -> int:
        """Number of real channels contributed by each derived, zero if unused."""
        return self.derived_representation.channels_per_source if self.use_derived else 0

    def total_channels(self, n_secondaries: int, n_derived: int) -> int:
        """Returns the channel count of the assembled input stack.

        Args:
            n_secondaries    : Number of secondary channels available in the scene.
            n_derived : Number of derived available in the scene.

        Returns:
            Total number of real-valued input channels, Aux included when enabled.
        """
        n  = self.primary_channels_per_source
        n += n_secondaries    * self.secondaries_channels_per_source
        n += n_derived * self.derived_channels_per_source
        if self.use_aux:
            n += 1
        return n

    def channel_group_keys(self, n_secondaries: int, n_derived: int) -> list[str]:
        """Returns the normalisation group key of every input channel, in stack order.

        Args:
            n_secondaries    : Number of secondary channels available in the scene.
            n_derived : Number of derived available in the scene.

        Returns:
            One ``source/slot_kind`` key per channel of the assembled stack.
        """
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
        """Returns the JSON-serialisable form of the input selection."""
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
        """Rebuilds an input configuration from its serialised form.

        Args:
            payload : Mapping previously produced by :meth:`as_dict`.

        Returns:
            The reconstructed configuration.
        """
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
    """Layout and normalisation of the predicted Gaussian scatterer parameters.

    Attributes:
        ROLE_NAMES        : Parameter roles of one Gaussian: amplitude, mean elevation and standard deviation.
        output_strategies : Normalisation strategy per output slot.
    """

    ROLE_NAMES : ClassVar[list[str]] = ["a", "mu", "sig"]

    output_strategies : dict[str, ChannelStrategy] = field(default_factory=lambda: {
        "out/amp"   : ChannelStrategy.from_slot("out/amp"),
        "out/mu"    : ChannelStrategy.from_slot("out/mu"),
        "out/sigma" : ChannelStrategy.from_slot("out/sigma"),
    })

    def strategy_for(self, role_key: str) -> ChannelStrategy:
        """Returns the normalisation strategy registered for an output slot key.

        Args:
            role_key : Output slot key such as ``out/mu``.

        Returns:
            The channel normalisation strategy for that slot.

        Raises:
            KeyError : If no strategy is registered for the key.
        """
        return self.output_strategies[role_key]

    @property
    def role_names(self) -> list[str]:
        """Parameter role names of a single Gaussian component."""
        return list(self.ROLE_NAMES)

    @property
    def params_per_gaussian(self) -> int:
        """Number of predicted values describing one Gaussian component."""
        return len(self.ROLE_NAMES)

    def selected_indices(self, n_gaussians: int) -> list[int]:
        """Returns the target channel indices kept for the given Gaussian count.

        Args:
            n_gaussians : Number of Gaussian components predicted per pixel.

        Returns:
            Channel indices into the target tensor, in slot order.
        """
        return list(range(n_gaussians * self.params_per_gaussian))

    def total_channels(self, n_gaussians: int) -> int:
        """Returns the number of target channels for the given Gaussian count.

        Args:
            n_gaussians : Number of Gaussian components predicted per pixel.

        Returns:
            Total target channel count.
        """
        return n_gaussians * self.params_per_gaussian

    def as_dict(self) -> dict:
        """Returns the JSON-serialisable form of the output strategies."""
        return {
            "output_strategies" : {
                k: v.as_dict() for k, v in self.output_strategies.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "OutputConfig":
        """Rebuilds an output configuration from its serialised form.

        Args:
            payload : Mapping previously produced by :meth:`as_dict`.

        Returns:
            The reconstructed configuration.
        """
        strategies = {key: ChannelStrategy.from_dict(value) for key, value in payload["output_strategies"].items()}

        return cls(output_strategies = strategies)

   
@dataclass
class PatchConfig:
    """Geometry of the patches cut from the scene.

    Attributes:
        size                  : Patch extent in pixels as (azimuth, range).
        stride                : Step between consecutive patch origins in pixels, (azimuth, range).
        use_symmetric_padding : Whether the scene edges are padded symmetrically so every pixel is covered by a patch.
    """

    size                  : Tuple[int, int] = (64, 32)
    stride                : Tuple[int, int] = (32, 16)
    use_symmetric_padding : bool            = True


@dataclass
class AugmentationConfig:
    """Probabilities and strengths of the training-time patch augmentations.

    Attributes:
        p_flip_h  : Probability of a horizontal flip.
        p_flip_v  : Probability of a vertical flip.
        p_rot90   : Probability of a 90 degree rotation.
        noise_std : Standard deviation of the additive Gaussian noise, in normalised input units.
        p_noise   : Probability of applying the additive noise.
    """

    p_flip_h  : float = 0.5
    p_flip_v  : float = 0.5
    p_rot90   : float = 0.0

    noise_std : float = 0.01
    p_noise   : float = 0.0


@dataclass
class DatasetConfig:
    """Full specification of a patch dataset and its loaders.

    Attributes:
        preprocessing_run_directory : Directory holding the preprocessed scene.
        split_regions               : Azimuth and range regions defining train, val and test.
        parameters_path             : Directory holding the extracted Gaussian parameters.
        secondary_labels            : Track labels of the secondary passes to load.
        patch                       : Patch geometry.
        input_config                : Input channel selection.
        output_config               : Target layout and normalisation.
        normalization               : Input and target normalisation settings.
        augmentation                : Training-time augmentation settings.
        batch_size                  : Patches per batch.
        num_workers                 : Data loader worker processes.
        prefetch_factor             : Batches prefetched per worker.
        shuffle_train               : Whether the training split is shuffled.
        pin_memory                  : Whether loader batches are pinned for faster host-to-device copies.
        x_axis                      : Elevation axis samples in metres, of shape (n_elevation,).
        n_gaussians                 : Number of Gaussian components predicted per pixel.
    """

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

