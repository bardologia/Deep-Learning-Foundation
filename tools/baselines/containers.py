from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib     import Path
from typing      import ClassVar

import numpy as np



class SecondarySelection:
    @staticmethod
    def indices(labels: list, secondary_labels) -> list:
        primary     = labels[0]
        secondaries = list(labels[1:])
        requested   = [str(label) for label in secondary_labels]

        if primary in requested:
            raise ValueError(f"Pass {primary} is the reference and is always included; remove it from secondary_labels")

        unknown = [label for label in requested if label not in secondaries]
        if unknown:
            raise ValueError(f"Unknown secondary labels {unknown}; secondaries are {secondaries}")

        if len(set(requested)) != len(requested):
            raise ValueError(f"secondary_labels contains duplicates: {requested}")

        return [secondaries.index(label) for label in requested]


@dataclass
class TrackBaselines:
    labels              : list
    vertical            : list
    horizontal          : list
    vertical_std        : list
    horizontal_std      : list
    vertical_absolute   : list         = field(default_factory=list)
    horizontal_absolute : list         = field(default_factory=list)
    track_files         : list         = field(default_factory=list)
    azimuth_window      : tuple | None = None

    FILENAME : ClassVar[str] = "baselines.json"

    @property
    def reference(self) -> str:
        return self.labels[0]

    @property
    def n_tracks(self) -> int:
        return len(self.labels)

    def subset(self, secondary_labels) -> "TrackBaselines":
        if secondary_labels is None:
            return self

        keep = [0] + [1 + index for index in SecondarySelection.indices(self.labels, secondary_labels)]

        def pick(values: list) -> list:
            return [values[index] for index in keep] if values else list(values)

        return TrackBaselines(
            labels              = pick(self.labels),
            vertical            = pick(self.vertical),
            horizontal          = pick(self.horizontal),
            vertical_std        = pick(self.vertical_std),
            horizontal_std      = pick(self.horizontal_std),
            vertical_absolute   = pick(self.vertical_absolute),
            horizontal_absolute = pick(self.horizontal_absolute),
            track_files         = pick(self.track_files),
            azimuth_window      = self.azimuth_window,
        )

    def baselines(self, component: str = "vertical", look_angle_deg: float | None = None) -> tuple:
        if component == "vertical":
            return tuple(self.vertical)
        if component == "horizontal":
            return tuple(self.horizontal)
        if component == "magnitude":
            return tuple(float(np.hypot(v, h)) for v, h in zip(self.vertical, self.horizontal))
        if component == "perpendicular":
            if look_angle_deg is None:
                raise ValueError("Baseline component 'perpendicular' requires look_angle_deg")
            theta = float(np.deg2rad(look_angle_deg))
            return tuple(float(h * np.cos(theta) + v * np.sin(theta)) for v, h in zip(self.vertical, self.horizontal))
        raise ValueError(f"Unknown baseline component '{component}', expected 'vertical', 'horizontal', 'magnitude' or 'perpendicular'")

    def describe(self) -> dict:
        window = "full track" if self.azimuth_window is None else f"[{self.azimuth_window[0]}, {self.azimuth_window[1]})"
        table  = {
            "Tracks"             : self.n_tracks,
            "Reference"          : self.reference,
            "Azimuth window"     : window,
            "Vertical [m]"       : ", ".join(f"{v:.2f}" for v in self.vertical),
            "Horizontal [m]"     : ", ".join(f"{h:.2f}" for h in self.horizontal),
            "Vertical std [m]"   : ", ".join(f"{s:.2f}" for s in self.vertical_std),
            "Horizontal std [m]" : ", ".join(f"{s:.2f}" for s in self.horizontal_std),
        }

        if self.vertical_absolute:
            table["Vertical absolute [m]"]   = ", ".join(f"{v:.2f}" for v in self.vertical_absolute)
            table["Horizontal absolute [m]"] = ", ".join(f"{h:.2f}" for h in self.horizontal_absolute)

        return table

    def to_payload(self) -> dict:
        return {
            "labels"              : list(self.labels),
            "reference"           : self.reference,
            "vertical"            : [float(v) for v in self.vertical],
            "horizontal"          : [float(h) for h in self.horizontal],
            "vertical_std"        : [float(s) for s in self.vertical_std],
            "horizontal_std"      : [float(s) for s in self.horizontal_std],
            "vertical_absolute"   : [float(v) for v in self.vertical_absolute],
            "horizontal_absolute" : [float(h) for h in self.horizontal_absolute],
            "track_files"         : [str(f) for f in self.track_files],
            "azimuth_window"      : list(self.azimuth_window) if self.azimuth_window is not None else None,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "TrackBaselines":
        window = payload["azimuth_window"]
        return cls(
            labels              = list(payload["labels"]),
            vertical            = [float(v) for v in payload["vertical"]],
            horizontal          = [float(h) for h in payload["horizontal"]],
            vertical_std        = [float(s) for s in payload["vertical_std"]],
            horizontal_std      = [float(s) for s in payload["horizontal_std"]],
            vertical_absolute   = [float(v) for v in payload["vertical_absolute"]],
            horizontal_absolute = [float(h) for h in payload["horizontal_absolute"]],
            track_files         = list(payload["track_files"]),
            azimuth_window      = tuple(window) if window is not None else None,
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_payload(), indent=4), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TrackBaselines":
        return cls.from_payload(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class TrackProfiles:
    labels        : list
    horizontal    : np.ndarray
    vertical      : np.ndarray
    azimuth_start : int
    track_files   : list = field(default_factory=list)

    FILENAME : ClassVar[str] = "track_profiles.npz"

    @property
    def n_tracks(self) -> int:
        return len(self.labels)

    @property
    def n_samples(self) -> int:
        return int(self.horizontal.shape[1])

    @property
    def azimuth_axis(self) -> np.ndarray:
        return np.arange(self.azimuth_start, self.azimuth_start + self.n_samples)

    def relative_to_reference(self, component: str = "vertical") -> np.ndarray:
        profiles = self.vertical if component == "vertical" else self.horizontal
        return profiles - profiles[0]

    def planar_deviation(self) -> np.ndarray:
        h_centered = self.horizontal - np.nanmean(self.horizontal, axis=1, keepdims=True)
        v_centered = self.vertical   - np.nanmean(self.vertical,   axis=1, keepdims=True)
        return np.sqrt(h_centered ** 2 + v_centered ** 2)

    def deviation_radii(self) -> np.ndarray:
        return np.sqrt(np.nanmean(self.planar_deviation() ** 2, axis=1))

    def position_summary(self) -> dict:
        deviation = self.planar_deviation()

        return {
            "labels"          : list(self.labels),
            "horizontal_mean" : [float(x) for x in np.nanmean(self.horizontal, axis=1)],
            "vertical_mean"   : [float(x) for x in np.nanmean(self.vertical,   axis=1)],
            "horizontal_span" : [float(x) for x in np.nanmax(self.horizontal, axis=1) - np.nanmin(self.horizontal, axis=1)],
            "vertical_span"   : [float(x) for x in np.nanmax(self.vertical,   axis=1) - np.nanmin(self.vertical,   axis=1)],
            "deviation_rms"   : [float(x) for x in np.sqrt(np.nanmean(deviation ** 2, axis=1))],
            "deviation_max"   : [float(x) for x in np.nanmax(deviation, axis=1)],
            "azimuth_start"   : int(self.azimuth_start),
            "n_samples"       : self.n_samples,
        }

    def subset(self, secondary_labels) -> "TrackProfiles":
        if secondary_labels is None:
            return self

        keep = [0] + [1 + index for index in SecondarySelection.indices(self.labels, secondary_labels)]

        return TrackProfiles(
            labels        = [self.labels[index] for index in keep],
            horizontal    = self.horizontal[keep],
            vertical      = self.vertical[keep],
            azimuth_start = self.azimuth_start,
            track_files   = [self.track_files[index] for index in keep] if self.track_files else [],
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("wb") as handle:
            np.savez_compressed(
                handle,
                labels        = np.array(self.labels),
                horizontal    = np.asarray(self.horizontal, dtype=np.float32),
                vertical      = np.asarray(self.vertical,   dtype=np.float32),
                azimuth_start = np.int64(self.azimuth_start),
                track_files   = np.array([str(f) for f in self.track_files]),
            )

        return path

    @classmethod
    def load(cls, path: str | Path) -> "TrackProfiles":
        with np.load(Path(path)) as data:
            return cls(
                labels        = [str(label) for label in data["labels"]],
                horizontal    = np.asarray(data["horizontal"], dtype=float),
                vertical      = np.asarray(data["vertical"],   dtype=float),
                azimuth_start = int(data["azimuth_start"]),
                track_files   = [str(f) for f in data["track_files"]],
            )

    @classmethod
    def profiles_file(cls, dataset_dir: str | Path) -> Path:
        return Path(dataset_dir) / "data" / cls.FILENAME
