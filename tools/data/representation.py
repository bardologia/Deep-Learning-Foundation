"""Channel representations of the complex SAR stack fed to the networks.

Each pass of the stack is complex; a representation states how many real
channels it becomes and how they are computed, from raw real/imaginary parts
to magnitude with unit-norm phasors.
"""
from __future__ import annotations

from enum import Enum

import numpy as np


_CHANNELS_PER_SOURCE: dict[str, int] = {
    "real_imag"     : 2,
    "mag_real_imag" : 3,
    "mag_angle"     : 2,
    "mag_ri_angle"  : 4,
    "angle_only"    : 1,
    "mag_only"      : 1,
}


class Representation(Enum):
    """Complex-to-real channel layout used to encode a SAR stack.

    Members name the channel content per pass: raw real and imaginary parts,
    magnitude with normalised real and imaginary parts, magnitude and phase angle,
    or magnitude respectively phase alone.
    """
    REAL_IMAG     = "real_imag"
    MAG_REAL_IMAG = "mag_real_imag"
    MAG_ANGLE     = "mag_angle"
    MAG_RI_ANGLE  = "mag_ri_angle"
    ANGLE_ONLY    = "angle_only"
    MAG_ONLY      = "mag_only"

    @property
    def channels_per_source(self) -> int:
        """Number of real channels this representation produces per source."""
        return _CHANNELS_PER_SOURCE[self.value]

    @property
    def slot_kinds(self) -> list[str]:
        """Per-channel kind tags naming what each channel of one pass holds."""
        return {
            Representation.REAL_IMAG     : ["raw_re_im",  "raw_re_im"],
            Representation.MAG_REAL_IMAG : ["mag",    "norm_re_im", "norm_re_im"],
            Representation.MAG_ANGLE     : ["mag",    "phase"],
            Representation.MAG_RI_ANGLE  : ["mag",    "norm_re_im", "norm_re_im", "phase"],
            Representation.ANGLE_ONLY    : ["phase"],
            Representation.MAG_ONLY      : ["mag"],
        }[self]

    def channel_values(self, data: np.ndarray) -> list[np.ndarray]:
        """Converts complex data into the real channels of this representation.

        Args:
            data : Complex array of any shape.

        Returns:
            One real array per channel, each of the same shape as data; magnitudes
            are linear, angles are in radians and normalised parts are unit-norm.

        Raises:
            ValueError : If the representation is not handled.
        """
        if self is Representation.MAG_ONLY:
            return [np.abs(data)]

        if self is Representation.ANGLE_ONLY:
            return [np.angle(data)]

        if self is Representation.REAL_IMAG:
            return [data.real, data.imag]

        if self is Representation.MAG_ANGLE:
            return [np.abs(data), np.angle(data)]

        if self is Representation.MAG_REAL_IMAG:
            magnitude = np.abs(data)
            safe_mag  = np.where(magnitude > 0, magnitude, 1.0)
            return [magnitude, data.real / safe_mag, data.imag / safe_mag]

        if self is Representation.MAG_RI_ANGLE:
            magnitude = np.abs(data)
            safe_mag  = np.where(magnitude > 0, magnitude, 1.0)
            return [magnitude, data.real / safe_mag, data.imag / safe_mag, np.angle(data)]

        raise ValueError(f"Unsupported representation: {self}")

    def convert_into(self, out: np.ndarray, data: np.ndarray) -> None:
        """Writes the channels of one source into an interleaved output array.

        Args:
            out  : Destination of shape (n_sources*channels_per_source, ...), source-major with the channels of a source strided by channels_per_source.
            data : Complex data of shape (n_sources, ...).
        """
        cpp      = _CHANNELS_PER_SOURCE[self.value]
        channels = self.channel_values(data)

        for c, arr in enumerate(channels):
            out[c::cpp] = arr
