from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib     import Path

from tools.baselines import TrackBaselines


@dataclass
class GeometryConfig:
    wavelength             : float = 0.23
    slant_range            : float = 5000.0
    look_angle_deg         : float = 45.0
    baselines              : tuple = (0.0, 11.25, 22.5, 33.75, 45.0, 56.25, 67.5, 78.75, 90.0)
    kz_values              : tuple = ()
    baselines_source       : str   = "dataset"
    baseline_component     : str   = "perpendicular"
    baselines_origin       : str   = "config"
    height_axis_convention : str   = "height"

    def baselines_file(self, dataset_dir: str | Path) -> Path:
        return Path(dataset_dir) / "meta" / TrackBaselines.FILENAME

    def resolved(self, dataset_dir: str | Path, secondary_labels=None) -> "GeometryConfig":
        if self.baselines_source not in ("dataset", "manual"):
            raise ValueError(f"Unknown baselines_source '{self.baselines_source}', expected 'dataset' or 'manual'")

        if self.baselines_source == "manual" or len(self.kz_values) > 0:
            return self

        path = self.baselines_file(dataset_dir)
        if not path.exists():
            raise FileNotFoundError(f"baselines_source='dataset' but {path} does not exist; re-run preprocessing to extract baselines or set baselines_source='manual' to use the configured values.")

        table = TrackBaselines.load(path).subset(secondary_labels)
        return replace(self, baselines=table.baselines(self.baseline_component, look_angle_deg=self.look_angle_deg), baselines_origin=str(path))
