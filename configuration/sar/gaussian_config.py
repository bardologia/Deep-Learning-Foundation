from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib     import Path


@dataclass
class GaussianConfig:
    n_default_gaussians : int
    x_min               : float
    x_max               : float
    params_per_gaussian : int = 3

    @classmethod
    def from_dataset(cls, dataset_dir: str | Path, parameters_path: str | Path) -> "GaussianConfig":
        meta_dir     = Path(dataset_dir) / "meta"
        cfg          = json.loads((meta_dir / "config_state.json").read_text())
        height_range = cfg["tomogram_config"]["height_range"]

        extraction_meta_path = Path(parameters_path).parent / "param_extraction_meta.json"

        if not extraction_meta_path.is_file():
            raise FileNotFoundError(f"No param_extraction_meta.json next to {parameters_path}; the parameter run must be self-describing to derive n_gaussians, re-run the extraction for it.")

        extraction = json.loads(extraction_meta_path.read_text())

        return cls(
            n_default_gaussians = int(extraction["k_max"]),
            x_min               = float(height_range[0]),
            x_max               = float(height_range[1]),
        )
