from __future__ import annotations

from enum import Enum

import numpy as np


_CHANNELS_PER_PASS: dict[str, int] = {
    "real_imag"     : 2,
    "mag_real_imag" : 3,
    "mag_angle"     : 2,
    "mag_ri_angle"  : 4,
    "angle_only"    : 1,
    "mag_only"      : 1,
}


class Representation(Enum):
    REAL_IMAG     = "real_imag"
    MAG_REAL_IMAG = "mag_real_imag"
    MAG_ANGLE     = "mag_angle"
    MAG_RI_ANGLE  = "mag_ri_angle"
    ANGLE_ONLY    = "angle_only"
    MAG_ONLY      = "mag_only"

    @property
    def channels_per_pass(self) -> int:
        return _CHANNELS_PER_PASS[self.value]

    @property
    def slot_kinds(self) -> list[str]:
        return {
            Representation.REAL_IMAG     : ["raw_re_im",  "raw_re_im"],
            Representation.MAG_REAL_IMAG : ["mag",    "norm_re_im", "norm_re_im"],
            Representation.MAG_ANGLE     : ["mag",    "phase"],
            Representation.MAG_RI_ANGLE  : ["mag",    "norm_re_im", "norm_re_im", "phase"],
            Representation.ANGLE_ONLY    : ["phase"],
            Representation.MAG_ONLY      : ["mag"],
        }[self]

    def channel_values(self, data: np.ndarray) -> list[np.ndarray]:
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
        cpp      = _CHANNELS_PER_PASS[self.value]
        channels = self.channel_values(data)

        for c, arr in enumerate(channels):
            out[c::cpp] = arr
