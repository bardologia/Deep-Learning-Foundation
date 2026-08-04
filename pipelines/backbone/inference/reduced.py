from __future__ import annotations

from dataclasses import dataclass
from pathlib     import Path
from typing      import Dict, List

import numpy as np

from configuration.inference                         import InferenceConfig
from pipelines.backbone.inference.loader             import Run
from pipelines.backbone.inference.run_metadata_paths import InferenceMetadata
from tools.data.io                                   import FileIO
from tools.monitoring.logger                         import Logger
from tools.data.preprocessing                        import ProfileNormalizer
from tools.data.regions                              import CropRegion
from tools.sar                                       import TomogramLauncher
from tools.baselines                                 import SecondarySelection


@dataclass
class ReducedSynthesis:
    curves      : np.ndarray
    orientation : Dict[str, float]


class ReducedTomogramSynthesizer:
    CONFIG_STATE_FILENAME = "config_state.json"
    LAYOUT_FILENAME       = "dataset.json"

    def __init__(self, run: Run, meta: InferenceMetadata, cfg: InferenceConfig, logger: Logger) -> None:
        self._run   = run
        self.meta   = meta
        self.cfg    = cfg
        self.logger = logger

        self.preprocessing_dir = Path(run.dataset_config.preprocessing_run_directory)
        self.cache_directory   = Path(cfg.run_directory) / cfg.reduced_cache_subdir

    def _load_processing_state(self) -> dict:
        state_path = self.preprocessing_dir / "meta" / self.CONFIG_STATE_FILENAME
        if not state_path.is_file():
            raise FileNotFoundError(f"Preprocessing configuration state required to reproduce the Capon tomogram was not found: {state_path}")

        return FileIO.load_json(state_path)

    def _select_indices(self) -> List[int]:
        layout_path = self.preprocessing_dir / "data" / self.LAYOUT_FILENAME
        layout      = FileIO.load_json(layout_path)
        pass_labels = layout["pass_labels"]

        if not pass_labels:
            raise ValueError(f"Dataset layout {layout_path} records no pass labels; baseline extraction must succeed during pre-processing before secondaries can be selected by label.")

        indices = SecondarySelection.indices(pass_labels, self._run.secondary_labels)
        if not indices:
            raise ValueError(f"Resolved an empty secondary selection from labels {self._run.secondary_labels}; cannot build a reduced tomogram.")

        return indices

    def _cache_key(self, select: List[int], crop: CropRegion) -> str:
        select_token = "-".join(str(index) for index in sorted(select))
        return f"{crop.as_identifier_string()}_sel{len(select)}-{select_token}"

    def _build_spec(self, state: dict, select: List[int], crop: CropRegion, tomogram_path: Path, dem_path: Path) -> dict:
        tomogram_state = dict(state["tomogram_config"])
        tomogram_state["track_selection"] = select

        pyrat_directory = str(self.cfg.reduced_pyrat_dir) if self.cfg.reduced_pyrat_dir is not None else state["paths"]["pyrat_directory"]

        return {
            "tomogram_config"  : tomogram_state,
            "stack_identifier" : state["stack_identifier"],
            "dataset_type"     : state["dataset_type"],
            "pyrat_directory"  : pyrat_directory,
            "main_directory"   : str(self.cfg.run_directory),
            "run_subdirectory" : "reduced_work",
            "effort"           : self.cfg.reduced_effort,
            "crop"             : list(crop.as_tuple()),
            "tomogram_path"    : str(tomogram_path),
            "dem_path"         : str(dem_path),
        }

    def _generate(self, state: dict, select: List[int], crop: CropRegion, tomogram_path: Path, dem_path: Path, cache_key: str) -> None:
        spec      = self._build_spec(state, select, crop, tomogram_path, dem_path)
        spec_path = self.cache_directory / f"reduced_spec_{cache_key}.json"

        launcher = TomogramLauncher(self.cfg.reduced_env_name, logger=self.logger)
        launcher.generate(spec, spec_path)

    def _validate_alignment(self, reduced: np.ndarray) -> None:
        n_height, az, rg = reduced.shape
        region           = self._run.split_region

        if n_height != self._run.x_axis_length:
            raise ValueError(f"Reduced tomogram has {n_height} elevation bins but the run x_axis has {self._run.x_axis_length}; height_range must match the preprocessing run that produced the ground-truth tomogram.")

        if (az, rg) != (region.azimuth_size, region.range_size):
            raise ValueError(f"Reduced tomogram spatial shape {(az, rg)} does not match the split region {(region.azimuth_size, region.range_size)}.")

    def _report_orientation(self, reduced: np.ndarray, gt_curves: np.ndarray) -> Dict[str, float]:
        gt_profile  = ProfileNormalizer.unit_area(np.asarray(gt_curves)).mean(axis=(1, 2))
        red_profile = ProfileNormalizer.unit_area(reduced).mean(axis=(1, 2))

        gt_centered  = gt_profile  - gt_profile.mean()
        red_centered = red_profile - red_profile.mean()
        denom        = float(np.linalg.norm(gt_centered) * np.linalg.norm(red_centered)) + 1e-12

        corr_direct  = float(np.dot(gt_centered, red_centered)) / denom
        corr_flipped = float(np.dot(gt_centered, red_centered[::-1])) / denom

        self.logger.kv_table({
            "Mean-profile corr (aligned)" : f"{corr_direct:.4f}",
            "Mean-profile corr (flipped)" : f"{corr_flipped:.4f}",
        }, title="Reduced vs GT elevation orientation (verify capon_phase_sign)")

        if corr_flipped > corr_direct:
            self.logger.subsection("WARNING: flipped elevation orientation correlates better with GT; verify the Capon elevation sign on the server.")

        return {
            "reduced_orientation_corr_aligned" : corr_direct,
            "reduced_orientation_corr_flipped" : corr_flipped,
        }

    def run(self, gt_curves: np.ndarray) -> ReducedSynthesis:
        self.logger.section("[Inference: Reduced Capon Baseline]")

        state  = self._load_processing_state()
        select = self._select_indices()
        crop   = CropRegion(*self._run.split_region.as_tuple())

        self.logger.kv_table({
            "Secondary labels" : ", ".join(self._run.secondary_labels),
            "Select indices"   : ", ".join(str(index) for index in select),
            "Crop"             : crop.as_tuple(),
            "Stack id"         : state["stack_identifier"],
        }, title="Reduced selection")

        FileIO.ensure_dir(self.cache_directory)
        cache_key     = self._cache_key(select, crop)
        tomogram_path = self.cache_directory / f"reduced_tomogram_{cache_key}.npy"
        dem_path      = self.cache_directory / f"reduced_dem_{cache_key}.npy"

        if tomogram_path.is_file():
            self.logger.subsection(f"Reduced tomogram loaded from cache : {tomogram_path}")
        else:
            self._generate(state, select, crop, tomogram_path, dem_path, cache_key)

        reduced = np.load(str(tomogram_path), allow_pickle=False).astype(np.float32)

        self._validate_alignment(reduced)
        orientation = self._report_orientation(reduced, gt_curves)

        if self.cfg.save_cubes:
            np.save(self.meta.cube_dir / "reduced_curves.npy", reduced)

        self.logger.subsection(f"Reduced tomogram ready : shape {reduced.shape}")

        return ReducedSynthesis(curves=reduced, orientation=orientation)
