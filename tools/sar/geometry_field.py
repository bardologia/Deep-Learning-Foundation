from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib     import Path
from typing      import ClassVar

import numpy as np

from tools.baselines.containers import SecondarySelection, TrackProfiles
from tools.sar.track_parameters import TrackParameters


@dataclass
class GeometryField:
    labels        : list
    reference     : str
    wavelength    : float
    azimuth_start : int
    range_start   : int
    look_angle    : np.ndarray
    slant_range   : np.ndarray
    baseline_h    : np.ndarray
    baseline_v    : np.ndarray

    FILENAME    : ClassVar[str]   = "geometry_field.npz"
    CONVENTIONS : ClassVar[tuple] = ("height", "slant")

    @property
    def n_tracks(self) -> int:
        return len(self.labels)

    @property
    def n_azimuth(self) -> int:
        return int(self.baseline_h.shape[1])

    @property
    def n_range(self) -> int:
        return int(self.slant_range.shape[0])

    def perpendicular_baseline(self) -> np.ndarray:
        cos_theta = np.cos(self.look_angle).reshape(1, 1, -1)
        sin_theta = np.sin(self.look_angle).reshape(1, 1, -1)

        horizontal = self.baseline_h[:, :, None]
        vertical   = self.baseline_v[:, :, None]

        return horizontal * cos_theta + vertical * sin_theta

    def kz(self, convention: str) -> np.ndarray:
        if convention not in self.CONVENTIONS:
            raise ValueError(f"Unknown height_axis_convention '{convention}', expected one of {self.CONVENTIONS}")

        scale     = 4.0 * math.pi / self.wavelength
        sin_theta = np.sin(self.look_angle).reshape(1, 1, -1)

        denominator = self.slant_range.reshape(1, 1, -1)
        if convention == "height":
            denominator = denominator * sin_theta

        return scale * self.perpendicular_baseline() / denominator

    def validate_extent(self, crop) -> None:
        actual   = (self.azimuth_start, self.azimuth_start + self.n_azimuth, self.range_start, self.range_start + self.n_range)
        expected = crop.as_tuple()

        if actual != expected:
            raise ValueError(f"Geometry field extent az[{actual[0]}, {actual[1]}) rg[{actual[2]}, {actual[3]}) does not match the preprocessing global crop az[{expected[0]}, {expected[1]}) rg[{expected[2]}, {expected[3]}); the stored geometry_field.npz belongs to a different crop, re-run preprocessing to regenerate it.")

    def subset(self, secondary_labels) -> "GeometryField":
        if secondary_labels is None:
            return self

        keep = [0] + [1 + index for index in SecondarySelection.indices(self.labels, secondary_labels)]

        return GeometryField(
            labels        = [self.labels[index] for index in keep],
            reference     = self.reference,
            wavelength    = self.wavelength,
            azimuth_start = self.azimuth_start,
            range_start   = self.range_start,
            look_angle    = self.look_angle,
            slant_range   = self.slant_range,
            baseline_h    = self.baseline_h[keep],
            baseline_v    = self.baseline_v[keep],
        )

    def slice(self, azimuth_slice: slice, range_slice: slice) -> "GeometryField":
        azimuth_offset = 0 if azimuth_slice.start is None else int(azimuth_slice.start)
        range_offset   = 0 if range_slice.start   is None else int(range_slice.start)

        return GeometryField(
            labels        = list(self.labels),
            reference     = self.reference,
            wavelength    = self.wavelength,
            azimuth_start = self.azimuth_start + azimuth_offset,
            range_start   = self.range_start + range_offset,
            look_angle    = self.look_angle[range_slice],
            slant_range   = self.slant_range[range_slice],
            baseline_h    = self.baseline_h[:, azimuth_slice],
            baseline_v    = self.baseline_v[:, azimuth_slice],
        )

    def describe(self) -> dict:
        look_deg = np.degrees(self.look_angle)

        return {
            "Tracks"           : self.n_tracks,
            "Reference"        : self.reference,
            "Azimuth extent"   : f"[{self.azimuth_start}, {self.azimuth_start + self.n_azimuth})",
            "Range extent"     : f"[{self.range_start}, {self.range_start + self.n_range})",
            "Wavelength [m]"   : f"{self.wavelength:.4f}",
            "Slant range [m]"  : f"{float(self.slant_range[0]):.1f} - {float(self.slant_range[-1]):.1f}",
            "Look angle [deg]" : f"{float(look_deg[0]):.2f} - {float(look_deg[-1]):.2f}",
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("wb") as handle:
            np.savez_compressed(
                handle,
                labels        = np.array(self.labels),
                reference     = np.array(self.reference),
                wavelength    = np.float64(self.wavelength),
                azimuth_start = np.int64(self.azimuth_start),
                range_start   = np.int64(self.range_start),
                look_angle    = np.asarray(self.look_angle,  dtype=np.float64),
                slant_range   = np.asarray(self.slant_range, dtype=np.float64),
                baseline_h    = np.asarray(self.baseline_h,  dtype=np.float64),
                baseline_v    = np.asarray(self.baseline_v,  dtype=np.float64),
            )

        return path

    @classmethod
    def load(cls, path: str | Path) -> "GeometryField":
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(
                labels        = [str(label) for label in data["labels"]],
                reference     = str(data["reference"]),
                wavelength    = float(data["wavelength"]),
                azimuth_start = int(data["azimuth_start"]),
                range_start   = int(data["range_start"]),
                look_angle    = np.asarray(data["look_angle"],  dtype=np.float64),
                slant_range   = np.asarray(data["slant_range"], dtype=np.float64),
                baseline_h    = np.asarray(data["baseline_h"],  dtype=np.float64),
                baseline_v    = np.asarray(data["baseline_v"],  dtype=np.float64),
            )


class GeometryFieldBuilder:
    def __init__(self, parameters: TrackParameters, profiles: TrackProfiles, crop) -> None:
        self.parameters = parameters
        self.profiles   = profiles
        self.crop       = crop

    def _validate_labels(self) -> list:
        if list(self.profiles.labels) != list(self.parameters.labels):
            raise ValueError(f"Track-profile labels {list(self.profiles.labels)} do not match track-parameter labels {list(self.parameters.labels)}; cannot align baselines to geometry.")

        return list(self.parameters.labels)

    def _range_geometry(self) -> tuple:
        reference = self.parameters.parameters[0]

        slant_full = np.asarray(reference["r"], dtype=np.float64)
        height     = float(reference["h0"]) - float(reference["terrain"])

        if self.crop.range_end > slant_full.shape[0]:
            raise ValueError(f"Crop range_end {self.crop.range_end} exceeds the slant-range vector length {slant_full.shape[0]} for reference track {self.parameters.reference}.")

        slant_range = slant_full[self.crop.range_start:self.crop.range_end]

        if height <= 0.0 or height >= float(slant_range.min()):
            raise ValueError(f"Sensor height above terrain {height:.2f} m must be positive and below the nearest slant range {float(slant_range.min()):.2f} m for reference track {self.parameters.reference}; h0/terrain in the track parameters are corrupt, and clipping would silently yield a zero look angle and infinite kz.")

        look_angle = np.arccos(np.clip(height / slant_range, -1.0, 1.0))

        return slant_range, look_angle

    def _azimuth_baselines(self) -> tuple:
        start = self.crop.azimuth_start - int(self.profiles.azimuth_start)
        stop  = self.crop.azimuth_end   - int(self.profiles.azimuth_start)

        if start < 0 or stop > self.profiles.n_samples:
            raise ValueError(f"Crop azimuth [{self.crop.azimuth_start}, {self.crop.azimuth_end}) is not covered by track profiles spanning [{self.profiles.azimuth_start}, {self.profiles.azimuth_start + self.profiles.n_samples}); regenerate track_profiles over the full crop.")

        horizontal = np.asarray(self.profiles.horizontal, dtype=np.float64)[:, start:stop]
        vertical   = np.asarray(self.profiles.vertical,   dtype=np.float64)[:, start:stop]

        baseline_h = horizontal - horizontal[0:1]
        baseline_v = vertical   - vertical[0:1]

        return baseline_h, baseline_v

    def build(self) -> GeometryField:
        self.parameters.validate_right_looking()

        labels                  = self._validate_labels()
        slant_range, look_angle = self._range_geometry()
        baseline_h, baseline_v  = self._azimuth_baselines()

        return GeometryField(
            labels        = labels,
            reference     = self.parameters.reference,
            wavelength    = float(self.parameters.parameters[0]["lambda"]),
            azimuth_start = int(self.crop.azimuth_start),
            range_start   = int(self.crop.range_start),
            look_angle    = look_angle,
            slant_range   = slant_range,
            baseline_h    = baseline_h,
            baseline_v    = baseline_v,
        )
