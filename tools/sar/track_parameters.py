from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field
from pathlib     import Path
from typing      import ClassVar

from tools.baselines.reading import PassProductResolver


class StepParameterFile:
    INTEGER_TYPES = ("long", "int", "byte", "uint", "ulong", "short")
    FLOAT_TYPES   = ("double", "float")

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _object_node(self, root: ElementTree.Element) -> ElementTree.Element:
        node = root.find("object")

        if node is None:
            raise ValueError(f"{self.path} has no <object> parameter block under <idl2xml>")

        return node

    def _parameter(self, parameter_el: ElementTree.Element) -> tuple:
        name      = parameter_el.get("name")
        datatype  = parameter_el.find("datatype")
        value_el  = parameter_el.find("value")
        base_type = (datatype.text or "").strip().lower() if datatype is not None else "string"

        return name, self._value(value_el, base_type)

    def _value(self, value_el: ElementTree.Element, base_type: str):
        nested_objects    = value_el.findall("object")
        nested_parameters = value_el.findall("parameter")

        if nested_objects:
            return self._group(nested_objects[0])

        if nested_parameters:
            group = self._group(value_el)
            return group["ptr"] if list(group) == ["ptr"] else group

        return self._coerce_text(value_el.text or "", base_type)

    def _group(self, object_el: ElementTree.Element) -> dict:
        return {name: value for name, value in (self._parameter(p) for p in object_el.findall("parameter"))}

    def _coerce_text(self, text: str, base_type: str):
        stripped = text.strip()

        if stripped.startswith("[") and stripped.endswith("]"):
            tokens = [token for token in stripped[1:-1].split(",") if token.strip() != ""]
            return [self._coerce_scalar(token, base_type) for token in tokens]

        return self._coerce_scalar(stripped, base_type)

    def _coerce_scalar(self, token: str, base_type: str):
        token = token.strip()

        if base_type in self.INTEGER_TYPES:
            return int(token)

        if base_type in self.FLOAT_TYPES:
            return float(token)

        return token

    def parse(self) -> dict:
        root        = ElementTree.parse(self.path).getroot()
        object_node = self._object_node(root)

        return {name: value for name, value in (self._parameter(p) for p in object_node.findall("parameter"))}


class StepParameterResolver(PassProductResolver):
    PRODUCT_SUBDIR       = Path("INF") / "INF-RDP"
    PRODUCT_PATTERN      = "pp_*.xml"
    POLARISATION_PATTERN = re.compile(r"_[A-Za-z]*?([hvHV]{2})_[Tt][0-9A-Za-z]+$")

    def _polarisation_of(self, parameter_file: Path) -> str:
        match = self.POLARISATION_PATTERN.search(parameter_file.stem)

        if match is None:
            raise ValueError(f"Cannot read polarisation from parameter file name '{parameter_file.name}'")

        return match.group(1).lower()

    def resolve_for_polarisation(self, pass_directory: str | Path, polarisation: str) -> Path:
        directory  = Path(pass_directory) / self.PRODUCT_SUBDIR
        wanted     = polarisation.lower()
        candidates = sorted(directory.glob(self.PRODUCT_PATTERN))
        matches    = [candidate for candidate in candidates if self._polarisation_of(candidate) == wanted]

        if not matches:
            available = ", ".join(self._polarisation_of(candidate) for candidate in candidates) or "none"
            raise FileNotFoundError(f"No pp_*.xml for polarisation '{wanted}' under {directory} (available: {available})")

        if len(matches) > 1:
            raise ValueError(f"Multiple parameter files for polarisation '{wanted}' under {directory}: {[m.name for m in matches]}")

        return matches[0]


@dataclass
class TrackParameters:
    labels      : list
    parameters  : list
    track_files : list = field(default_factory=list)

    FILENAME : ClassVar[str] = "track_parameters.json"

    @property
    def reference(self) -> str:
        return self.labels[0]

    @property
    def n_tracks(self) -> int:
        return len(self.labels)

    def validate_right_looking(self) -> None:
        left = [label for label, params in zip(self.labels, self.parameters) if params["antdir"] <= 0]

        if left:
            raise ValueError(f"Tracks {left} are left-looking (antdir <= 0); the kz formula assumes a right-looking geometry, so left-looking stacks would train against sign-flipped physics. Left-looking data is not supported.")

    def derived(self) -> list:
        return [self._track_geometry(params) for params in self.parameters]

    def _track_geometry(self, params: dict) -> dict:
        slant_range = params["r"]
        height      = params["h0"] - params["terrain"]
        near        = float(slant_range[0])
        far         = float(slant_range[-1])

        return {
            "look_side"            : "right" if params["antdir"] > 0 else "left",
            "depression_angle_deg" : math.degrees(params["da"]),
            "wavelength_m"         : params["lambda"],
            "sensor_altitude_m"    : params["h0"],
            "terrain_height_m"     : params["terrain"],
            "slant_range_near_m"   : near,
            "slant_range_far_m"    : far,
            "slant_range_ref_m"    : params["rref"],
            "look_angle_near_deg"  : math.degrees(math.acos(min(1.0, height / near))),
            "look_angle_far_deg"   : math.degrees(math.acos(min(1.0, height / far))),
        }

    def describe(self) -> dict:
        geometry = self.derived()

        return {
            "Tracks"              : self.n_tracks,
            "Reference"           : self.reference,
            "Look side"           : geometry[0]["look_side"],
            "Wavelength [m]"      : f"{geometry[0]['wavelength_m']:.4f}",
            "Depression [deg]"    : ", ".join(f"{g['depression_angle_deg']:.2f}" for g in geometry),
            "Slant range ref [m]" : ", ".join(f"{g['slant_range_ref_m']:.1f}" for g in geometry),
            "Look angle [deg]"    : ", ".join(f"{g['look_angle_near_deg']:.1f}-{g['look_angle_far_deg']:.1f}" for g in geometry),
        }

    def _validate_key_sets(self) -> None:
        reference_keys = set(self.parameters[0])

        for label, params in zip(self.labels, self.parameters):
            missing = reference_keys - set(params)
            extra   = set(params) - reference_keys

            if missing or extra:
                raise ValueError(f"Track '{label}' parameter keys differ from reference '{self.reference}': missing {sorted(missing)}, extra {sorted(extra)}; a lossless shared/per-track partition is impossible.")

    def _partition(self) -> tuple:
        self._validate_key_sets()

        shared       = {}
        varying_keys = []

        for key in self.parameters[0]:
            values = [params[key] for params in self.parameters]

            if all(value == values[0] for value in values):
                shared[key] = values[0]
            else:
                varying_keys.append(key)

        per_track = {label: {key: params[key] for key in varying_keys} for label, params in zip(self.labels, self.parameters)}

        return shared, per_track

    def to_payload(self) -> dict:
        shared, per_track = self._partition()

        return {
            "labels"      : list(self.labels),
            "reference"   : self.reference,
            "shared"      : shared,
            "per_track"   : per_track,
            "track_files" : [str(f) for f in self.track_files],
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "TrackParameters":
        shared     = payload["shared"]
        parameters = [{**shared, **payload["per_track"][label]} for label in payload["labels"]]

        return cls(
            labels      = list(payload["labels"]),
            parameters  = parameters,
            track_files = list(payload["track_files"]),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_payload(), indent=4), encoding="utf-8")

        return path

    @classmethod
    def load(cls, path: str | Path) -> "TrackParameters":
        return cls.from_payload(json.loads(Path(path).read_text(encoding="utf-8")))


class TrackParameterCollector:
    def __init__(self, track_paths: dict) -> None:
        self.track_paths = {label: Path(path) for label, path in track_paths.items()}

    @classmethod
    def from_pass_directories(cls, pass_directories: list, polarisation: str) -> "TrackParameterCollector":
        resolver = StepParameterResolver()
        labels   = [resolver.label(str(directory)) for directory in pass_directories]

        duplicates = sorted({label for label in labels if labels.count(label) > 1})
        if duplicates:
            raise ValueError(f"Pass directories resolve to duplicate labels {duplicates}; every pass must map to a unique label or tracks would be silently dropped.")

        track_paths = {label: resolver.resolve_for_polarisation(directory, polarisation) for label, directory in zip(labels, pass_directories)}

        return cls(track_paths)

    def collect(self) -> TrackParameters:
        labels     = list(self.track_paths.keys())
        files      = [self.track_paths[label] for label in labels]
        parameters = [StepParameterFile(path).parse() for path in files]

        collected = TrackParameters(labels=labels, parameters=parameters, track_files=[str(f) for f in files])
        collected.validate_right_looking()

        return collected
